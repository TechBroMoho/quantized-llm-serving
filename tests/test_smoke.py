"""CPU rehearsals of the exact smoke driver used inside Modal containers."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from llmbench.smoke import (
    SmokeWorkload,
    hf_stats_collector,
    pool_for_model,
    run_server_smoke,
)

WORKLOAD = SmokeWorkload(
    input_tokens=16,
    output_tokens=4,
    concurrency=2,
    warmup_requests=2,
    measured_requests=6,
    timeout_s=30,
    seed=11,
)


def _port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _mock_command(port: int, tokens: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "llmbench.mock.server",
        "--port",
        str(port),
        "--ttft-ms",
        "5",
        "--itl-ms",
        "1",
        "--tokens",
        str(tokens),
    ]


def _run_mock(tmp_path: Path, tokens: int) -> dict[str, Any]:
    from llmbench.loadtest.workloads import TokenPromptPool

    port = _port()
    pool = TokenPromptPool(
        allowed_ids=range(3, 1000),
        count=WORKLOAD.total_requests,
        length=WORKLOAD.input_tokens,
        seed=WORKLOAD.seed,
    )
    return asyncio.run(
        run_server_smoke(
            label="mock",
            command=_mock_command(port, tokens),
            base_url=f"http://127.0.0.1:{port}",
            model_name="mock-model",
            pool=pool,
            workload=WORKLOAD,
            out_dir=tmp_path,
            health_timeout_s=30,
        )
    )


def test_smoke_driver_passes_and_saves_raw_results(tmp_path: Path) -> None:
    summary = _run_mock(tmp_path, tokens=WORKLOAD.output_tokens)
    assert summary["passed"], summary["failures"]
    saved = json.loads((tmp_path / "smoke_summary.json").read_text())
    assert saved["passed"] and saved["state"] == "finished"
    assert saved["prompt_pool"]["issued"] == WORKLOAD.total_requests
    rows = [
        json.loads(line)
        for line in (tmp_path / "load_summary.jsonl").read_text().splitlines()
    ]
    assert len(rows) == WORKLOAD.measured_requests
    assert {row["prompt_tokens"] for row in rows} == {16}
    assert {row["completion_tokens"] for row in rows} == {4}
    assert saved["server_returncode_after_stop"] is not None


def test_smoke_driver_fails_on_short_output_but_still_saves(tmp_path: Path) -> None:
    summary = _run_mock(tmp_path, tokens=WORKLOAD.output_tokens - 1)
    assert not summary["passed"]
    assert any("completion_tokens 3 != max_tokens 4" in f for f in summary["failures"])
    saved = json.loads((tmp_path / "smoke_summary.json").read_text())
    assert saved["state"] == "finished" and not saved["passed"]


def test_smoke_driver_reports_server_that_exits_before_health(tmp_path: Path) -> None:
    from llmbench.loadtest.workloads import TokenPromptPool

    pool = TokenPromptPool(allowed_ids=range(3, 99), count=8, length=16, seed=0)
    summary = asyncio.run(
        run_server_smoke(
            label="crash",
            command=[sys.executable, "-c", "print('boom'); raise SystemExit(3)"],
            base_url=f"http://127.0.0.1:{_port()}",
            model_name="none",
            pool=pool,
            workload=WORKLOAD,
            out_dir=tmp_path,
            health_timeout_s=30,
        )
    )
    assert not summary["passed"]
    assert "exited with code 3" in summary["failures"][0]
    assert "boom" in summary["server_log_tail"]


def _save_tiny_model(path: Path) -> None:
    vocab = {"<eos>": 0, "<bos>": 1, "<unk>": 2} | {f"t{i}": i for i in range(3, 32)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()  # type: ignore[assignment]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
        pad_token="<eos>",
    )
    tokenizer.save_pretrained(path)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=64,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=0,
            pad_token_id=0,
        )
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)  # every step prefers EOS (token 0)
    model.save_pretrained(path)


@pytest.mark.parametrize(("mode", "batch_size"), [("naive", 1), ("static", 2)])
def test_hf_smoke_rehearsal_on_cpu(tmp_path: Path, mode: str, batch_size: int) -> None:
    model_path = tmp_path / "model"
    _save_tiny_model(model_path)
    port = _port()
    command = [
        sys.executable,
        "-m",
        "llmbench.baseline.hf_server",
        "--model",
        str(model_path),
        "--mode",
        mode,
        "--dtype",
        "float32",
        "--port",
        str(port),
    ]
    if mode == "static":
        command += ["--batch-size", str(batch_size), "--batch-wait-ms", "100"]
    pool = pool_for_model(str(model_path), WORKLOAD)
    assert all(token >= 3 for prompt in pool.prompts for token in prompt)
    summary = asyncio.run(
        run_server_smoke(
            label=f"hf_{mode}",
            command=command,
            base_url=f"http://127.0.0.1:{port}",
            model_name="tiny",
            pool=pool,
            workload=WORKLOAD,
            out_dir=tmp_path / "out",
            health_timeout_s=120,
            env={"HF_HUB_OFFLINE": "1"},
            collect=hf_stats_collector(mode, batch_size, WORKLOAD.total_requests),
        )
    )
    assert summary["passed"], summary["failures"]
    sizes = summary["collected"]["generated_batch_sizes"]
    assert sum(sizes) == WORKLOAD.total_requests
    assert max(sizes) == batch_size
    assert summary["collected"]["dtype"] == "torch.float32"


def test_hf_stats_collector_rejects_unbatched_static_mode() -> None:
    collect = hf_stats_collector("static", 4, 3)
    from unittest.mock import patch

    stats = {"text": json.dumps({"generated_batch_sizes": [1, 1, 1]}), "status": 200}
    with patch("llmbench.smoke.fetch_text", return_value=stats):
        _, failures = collect("http://unused")
    assert failures == ["static mode never batched: [1, 1, 1]"]
