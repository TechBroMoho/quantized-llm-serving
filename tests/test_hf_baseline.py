"""CPU integration checks for the two HF scheduling modes and length invariant."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

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


def test_length_guard_catches_min_new_tokens_mutation() -> None:
    _, rows, _ = asyncio.run(_run("naive", min_override=0))
    assert all(row.status == "error" for row in rows)
    assert all("completion_tokens 1 != max_tokens 3" in row.error for row in rows)
