"""Phase 4 helpers that run without llm-compressor (CPU, project env)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from llmbench.quantization import (
    calibration_summary,
    check_checkpoint,
    directory_manifest,
    inspect_tensors,
    preprocess_rows,
)
from llmbench.sanity import check_outputs, run_sanity

AWQ = {"expected_group_size": 128, "expected_symmetric": False}


def _summary(**weights: Any) -> dict[str, Any]:
    spec = {"num_bits": 4, "group_size": 128, "symmetric": False} | weights
    return {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {"group_0": {"weights": spec}},
            "ignore": ["lm_head"],
        },
        "checkpoint": {"safetensors_bytes": 10},
    }


def test_checkpoint_checks_accept_expected_and_reject_each_deviation() -> None:
    assert check_checkpoint(_summary(), AWQ) == []
    assert check_checkpoint(_summary(num_bits=8), AWQ) == ["num_bits 8"]
    assert check_checkpoint(_summary(group_size=64), AWQ) == ["group_size 64"]
    assert check_checkpoint(_summary(symmetric=True), AWQ) == ["symmetric True"]
    no_ignore = _summary()
    no_ignore["quantization_config"]["ignore"] = []
    assert check_checkpoint(no_ignore, AWQ) == ["lm_head not in ignore list"]


def _write_shard(path: Path, zero_points: bool, head: torch.dtype) -> None:
    tensors = {
        "model.layers.0.self_attn.q_proj.weight_packed": torch.zeros(
            8, 2, dtype=torch.int32
        ),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.ones(
            8, 1, dtype=torch.bfloat16
        ),
        "lm_head.weight": torch.zeros(4, 4, dtype=head),
    }
    if zero_points:
        tensors["model.layers.0.self_attn.q_proj.weight_zero_point"] = torch.zeros(
            1, 1, dtype=torch.int32
        )
    save_file(tensors, str(path / "model.safetensors"))


@pytest.mark.parametrize(
    ("zero_points", "symmetric", "head", "expected"),
    [
        (True, False, torch.bfloat16, []),
        (False, True, torch.bfloat16, []),
        (False, False, torch.bfloat16, ["asymmetric scheme stored no zero points"]),
        (True, True, torch.bfloat16, ["symmetric scheme stored zero points"]),
        (False, True, torch.float16, ["lm_head dtype F16"]),
    ],
)
def test_tensor_inspection(
    tmp_path: Path, zero_points, symmetric, head, expected
) -> None:
    _write_shard(tmp_path, zero_points, head)
    assert inspect_tensors(tmp_path, symmetric)["failures"] == expected


def test_directory_manifest_skips_hub_cache(tmp_path: Path) -> None:
    (tmp_path / "a.safetensors").write_bytes(b"12345")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / ".cache" / "huggingface").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "x").write_text("ignored")
    manifest = directory_manifest(tmp_path)
    assert [f["path"] for f in manifest["files"]] == ["a.safetensors", "config.json"]
    assert manifest["safetensors_bytes"] == 5 and manifest["total_bytes"] == 7


class ChatTokenizer:
    def apply_chat_template(
        self, messages: list[dict[str, str]], tokenize: bool
    ) -> str:
        assert tokenize is False
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        assert kwargs == {
            "padding": False,
            "max_length": 5,
            "truncation": True,
            "add_special_tokens": False,
        }
        ids = [ord(c) for c in text][: kwargs["max_length"]]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_preprocessing_follows_example_formats_and_truncates() -> None:
    user = {"format": "user_text", "text_field": "text", "max_seq_length": 5}
    rows = preprocess_rows(ChatTokenizer(), [{"text": "hello"}], user)
    assert rows[0]["input_ids"] == [ord(c) for c in "user:"]
    chat = {"format": "messages", "text_field": "messages", "max_seq_length": 5}
    rows = preprocess_rows(
        ChatTokenizer(), [{"messages": [{"role": "assistant", "content": "x"}]}], chat
    )
    assert rows[0]["input_ids"] == [ord(c) for c in "assis"]
    assert calibration_summary(rows)["tokens_max"] == 5
    with pytest.raises(ValueError):
        preprocess_rows(ChatTokenizer(), [{}], {**chat, "format": "other"})


FAKE_VLLM = r"""#!{python}
import json, sys
from aiohttp import web
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
async def health(_): return web.json_response({{}})
async def complete(request):
    body = await request.json()
    text = " sanity" if body["temperature"] == 0.0 else ""
    return web.json_response({{"choices": [{{"text": text, "finish_reason": "length"}}],
        "usage": {{"prompt_tokens": 3, "completion_tokens": body["max_tokens"]}}}})
print("INFO Model loading took 1.0 GiB", flush=True)
app = web.Application()
app.router.add_get("/health", health)
app.router.add_post("/v1/completions", complete)
web.run_app(app, host="127.0.0.1", port=port, print=None)
"""


def test_run_sanity_end_to_end_with_fake_vllm(tmp_path: Path, monkeypatch) -> None:
    import socket

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "vllm"
    fake.write_text(FAKE_VLLM.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"x" * 10)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    sanity = {"engine_args": [], "prompts": ["a", "b"], "max_tokens": 4}
    summary = run_sanity(
        label="fake",
        model_path=str(model),
        served_name="m",
        sanity=sanity | {"health_timeout_s": 30},
        out_dir=tmp_path / "out",
        port=port,
    )
    assert summary["passed"], summary["failures"]
    assert [o["text"] for o in summary["outputs"]] == [" sanity", " sanity"]
    assert summary["checkpoint_bytes"]["safetensors_bytes"] == 10
    assert any("Model loading took" in line for line in summary["log_excerpts"])
    assert json.loads((tmp_path / "out" / "sanity.json").read_text())["passed"]


def test_sanity_rejects_empty_completions() -> None:
    outputs = [{"text": " ", "usage": {"completion_tokens": 0}}]
    assert check_outputs(outputs, 2) == [
        "1 outputs, expected 2",
        "prompt 0: empty completion",
        "prompt 0: no completion tokens in usage",
    ]
