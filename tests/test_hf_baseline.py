"""CPU integration checks for the two HF scheduling modes and length invariant."""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any

import pytest
import torch
import uvicorn
from transformers import GPT2Config, GPT2LMHeadModel

from llmbench.baseline.hf_server import Baseline, make_app
from llmbench.loadtest.runner import run_load


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    bos_token_id = 1

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        return "x" if tokens else ""


def _model() -> GPT2LMHeadModel:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_positions=32,
            n_ctx=32,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=0,
            pad_token_id=0,
        )
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    return model.eval()


def _payload(index: int) -> dict[str, Any]:
    return {
        "model": "local-tiny-gpt2",
        "prompt": [1, 2 + index % 2, 3] + [4] * (index % 2),
        "max_tokens": 3,
        "stream": True,
        "stream_options": {"include_usage": True},
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "seed": 0,
    }


async def _run(
    mode: str, min_override: int | None = None
) -> tuple[dict[str, Any], list[Any], list[int]]:
    baseline = Baseline(
        _model(),
        TinyTokenizer(),
        mode=mode,
        batch_size=2,
        batch_wait_ms=50,
        min_new_tokens_override=min_override,
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    app = make_app(baseline)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        assert server.started
        summary, rows = await run_load(
            url=f"http://127.0.0.1:{port}/v1/completions",
            concurrency=2,
            timeout_s=10,
            drain_s=2,
            payload_factory=_payload,
            mode="closed",
            request_count=2,
        )
        return summary, rows, baseline.generated_batch_sizes
    finally:
        server.should_exit = True
        await task


def test_naive_exact_length() -> None:
    summary, rows, sizes = asyncio.run(_run("naive"))
    assert summary["on_time_completed_requests"] == 2
    assert all(row.status == "ok" and row.completion_tokens == 3 for row in rows)
    assert sizes == [1, 1]


def test_static_batch_exact_length() -> None:
    summary, rows, sizes = asyncio.run(_run("static"))
    assert summary["on_time_completed_requests"] == 2
    assert all(row.status == "ok" and row.completion_tokens == 3 for row in rows)
    assert sizes == [2]


@pytest.mark.parametrize("mode", ["naive", "static"])
def test_length_guard_catches_min_new_tokens_mutation(mode) -> None:
    _, rows, _ = asyncio.run(_run(mode, min_override=0))
    assert all(row.status == "error" for row in rows)
    assert all("completion_tokens 1 != max_tokens 3" in row.error for row in rows)


def test_model_generation_defaults_cannot_override_greedy_settings() -> None:
    from llmbench.baseline.hf_server import Job

    async def run():
        model = _model()
        model.generation_config.do_sample = True
        model.generation_config.num_beams = 2
        model.generation_config.forced_eos_token_id = 7
        model.generation_config.repetition_penalty = 2.0
        baseline = Baseline(model, TinyTokenizer(), mode="naive")
        job = Job([1, 2, 3], 3)
        await asyncio.to_thread(baseline._generate, [job], asyncio.get_running_loop())
        tokens = []
        while (token := await job.events.get()) is not None:
            tokens.append(token)
        # EOS=0 is suppressed; tied zero logits greedily choose token 1.
        assert tokens == [1, 1, 1]

    asyncio.run(run())


def test_actual_generate_batches_and_routes_distinct_rows() -> None:
    """Inspect the real generate boundary and compare streamed IDs to its output."""
    from unittest.mock import patch

    from llmbench.baseline.hf_server import Job

    async def run():
        model = _model()

        # Deterministic, row-dependent logits while retaining real HF generation.
        def row_logits(module, args, kwargs, output):
            ids = kwargs["input_ids"]
            output.logits.fill_(-100)
            next_ids = (ids[:, -1] % 6) + 1
            output.logits[:, -1].scatter_(1, next_ids[:, None], 100)
            return output

        hook = model.register_forward_hook(row_logits, with_kwargs=True)
        baseline = Baseline(
            model, TinyTokenizer(), mode="static", batch_size=2, batch_wait_ms=50
        )
        jobs = [Job([1, 2], 3), Job([1, 3, 4], 3)]
        original = model.generate
        calls = []

        def observe(**kwargs):
            result = original(**kwargs)
            calls.append((kwargs, result.tolist()))
            return result

        try:
            with patch.object(model, "generate", side_effect=observe):
                await baseline.start()
                for job in jobs:
                    baseline.pending.put_nowait(job)
                streamed = []
                for job in jobs:
                    tokens = []
                    while (
                        token := await asyncio.wait_for(job.events.get(), 5)
                    ) is not None:
                        assert isinstance(token, int)
                        tokens.append(token)
                    streamed.append(tokens)
                await baseline.stop()
            assert len(calls) == 1
            arguments, sequences = calls[0]
            assert arguments["input_ids"].tolist() == [[0, 1, 2], [1, 3, 4]]
            assert arguments["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
            assert streamed == [[3, 4, 5], [5, 6, 1]]
            assert streamed == [row[-3:] for row in sequences]
            import json
            import os
            from pathlib import Path

            if evidence_dir := os.environ.get("LLMBENCH_AUDIT_EVIDENCE_DIR"):
                path = Path(evidence_dir)
                path.mkdir(parents=True, exist_ok=True)
                (path / "actual_batch.json").write_text(
                    json.dumps(
                        {
                            "model": "local GPT2 with row-dependent forward hook",
                            "torch": torch.__version__,
                            "generate_calls": len(calls),
                            "input_ids": arguments["input_ids"].tolist(),
                            "attention_mask": arguments["attention_mask"].tolist(),
                            "generated_sequences": sequences,
                            "streamed_per_row": streamed,
                            "generation_config": arguments[
                                "generation_config"
                            ].to_dict(),
                            "use_model_defaults": arguments["use_model_defaults"],
                        },
                        indent=2,
                    )
                    + "\n"
                )
        finally:
            hook.remove()
            await baseline.stop()

    asyncio.run(run())


def test_streamer_delivers_before_generation_ends() -> None:
    from llmbench.baseline.hf_server import Job, TokenStreamer

    async def run():
        jobs = [Job([1], 2), Job([2], 2)]
        streamer = TokenStreamer(jobs, asyncio.get_running_loop())
        streamer.put(torch.tensor([[1], [2]]))
        streamer.put(torch.tensor([3, 4]))
        assert await asyncio.wait_for(jobs[0].events.get(), 1) == 3
        assert await asyncio.wait_for(jobs[1].events.get(), 1) == 4
        assert all(job.events.empty() for job in jobs)
        streamer.put(torch.tensor([5, 6]))
        streamer.end()
        for job, expected in zip(jobs, [5, 6], strict=True):
            assert await job.events.get() == expected
            assert await job.events.get() is None

    asyncio.run(run())


@pytest.mark.parametrize("setting", ["top_k", "repetition_penalty", "logit_bias"])
def test_api_rejects_silently_ignored_generation_settings(setting) -> None:
    from fastapi import HTTPException

    baseline = Baseline(_model(), TinyTokenizer(), mode="naive")
    app = make_app(baseline)
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/v1/completions"
    )
    payload = _payload(0) | {setting: 2}
    with pytest.raises(HTTPException) as error:
        asyncio.run(endpoint(payload))
    assert error.value.status_code == 400


def _completion_endpoint(baseline: Baseline) -> Any:
    app = make_app(baseline)
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/v1/completions"
    )


def test_api_accepts_ignore_eos_true_and_rejects_false() -> None:
    from fastapi import HTTPException

    endpoint = _completion_endpoint(Baseline(_model(), TinyTokenizer(), mode="naive"))
    with pytest.raises(HTTPException) as error:
        asyncio.run(endpoint(_payload(0) | {"ignore_eos": False}))
    assert error.value.status_code == 400
    response = asyncio.run(endpoint(_payload(0) | {"ignore_eos": True}))
    assert response.status_code == 200


@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_api_accepts_only_skip_special_tokens_false(value) -> None:
    from fastapi import HTTPException

    endpoint = _completion_endpoint(Baseline(_model(), TinyTokenizer(), mode="naive"))
    with pytest.raises(HTTPException) as error:
        asyncio.run(endpoint(_payload(0) | {"skip_special_tokens": value}))
    assert error.value.status_code == 400
    response = asyncio.run(endpoint(_payload(0) | {"skip_special_tokens": False}))
    assert response.status_code == 200


def test_stats_report_actual_batches_and_settings() -> None:
    baseline = Baseline(_model(), TinyTokenizer(), mode="static", batch_size=3)
    baseline.generated_batch_sizes.extend([3, 1])
    app = make_app(baseline)
    stats = next(
        route.endpoint for route in app.routes if getattr(route, "path", "") == "/stats"
    )
    result = asyncio.run(stats())
    assert result["generated_batch_sizes"] == [3, 1]
    assert result["mode"] == "static" and result["batch_size"] == 3
    assert result["dtype"] == "torch.float32"


def test_continuous_usage_is_cumulative_on_every_token_chunk() -> None:
    continuous = {"include_usage": True, "continuous_usage_stats": True}

    async def generate() -> list[dict[str, Any]]:
        baseline = Baseline(_model(), TinyTokenizer(), mode="naive")
        endpoint = _completion_endpoint(baseline)
        await baseline.start()
        try:
            response = await endpoint(_payload(0) | {"stream_options": continuous})
            events = []
            async for chunk in response.body_iterator:
                text = chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                data = text.removeprefix("data: ").strip()
                if data != "[DONE]":
                    events.append(json.loads(data))
            return events
        finally:
            await baseline.stop()

    events = asyncio.run(generate())
    progress = [e["usage"]["completion_tokens"] for e in events if e.get("choices")]
    assert progress == [1, 2, 3]
    assert events[-1] == {
        "choices": [],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
    }


def test_api_rejects_other_stream_options() -> None:
    from fastapi import HTTPException

    endpoint = _completion_endpoint(Baseline(_model(), TinyTokenizer(), mode="naive"))
    for options in (
        {"include_usage": True, "extra": True},
        {"include_usage": True, "continuous_usage_stats": False},
    ):
        with pytest.raises(HTTPException) as error:
            asyncio.run(endpoint(_payload(0) | {"stream_options": options}))
        assert error.value.status_code == 400


def test_disconnected_clients_are_skipped_and_stop_their_batch() -> None:
    from llmbench.baseline.hf_server import AllCancelled, Job

    jobs = [Job([1], 3), Job([1], 3)]
    ids = torch.zeros((2, 1), dtype=torch.long)
    assert not AllCancelled(jobs)(ids, torch.zeros(1)).any()
    jobs[0].cancelled = True
    assert not AllCancelled(jobs)(ids, torch.zeros(1)).any()  # one client left
    jobs[1].cancelled = True
    assert AllCancelled(jobs)(ids, torch.zeros(1)).all()

    async def run() -> tuple[int, list[int]]:
        baseline = Baseline(_model(), TinyTokenizer(), mode="naive")
        gone, live = Job([1, 2], 3, cancelled=True), Job([1, 2], 3)
        baseline.pending.put_nowait(gone)
        baseline.pending.put_nowait(live)
        await baseline.start()
        try:
            tokens = []
            while (token := await live.events.get()) is not None:
                tokens.append(token)
        finally:
            await baseline.stop()
        return baseline.skipped_cancelled_jobs, tokens

    skipped, tokens = asyncio.run(run())
    assert skipped == 1 and len(tokens) == 3


def test_closing_the_stream_marks_the_job_cancelled() -> None:
    async def run() -> bool:
        baseline = Baseline(_model(), TinyTokenizer(), mode="naive")
        endpoint = _completion_endpoint(baseline)
        response = await endpoint(_payload(0))
        job = baseline.pending.get_nowait()
        baseline.pending.put_nowait(job)
        job.events.put_nowait(1)
        iterator = response.body_iterator
        await iterator.__anext__()  # one token chunk, then the client leaves
        await iterator.aclose()  # type: ignore[attr-defined]
        return job.cancelled

    assert asyncio.run(run())
