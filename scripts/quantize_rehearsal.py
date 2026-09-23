"""CPU rehearsal of the Phase 4 quantization code path (no GPU, $0).

Runs the exact `llmbench.quantization` functions the Modal job uses, with both
configured recipes, on a tiny randomly initialised Qwen3 that shares the real
Qwen3-8B tokenizer and chat template. Then it reloads each compressed
checkpoint through compressed-tensors, runs a forward pass, and checks that a
wrong expectation is rejected. Run in the pinned quantization environment:

    make quantize-rehearsal
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from llmbench.quantization import (
    check_checkpoint,
    inspect_tensors,
    prepare_calibration,
    run_quantization,
)

OUT = Path("results/validation/phase4/rehearsal")
WORK = Path(".cache/phase4-rehearsal")


def tiny_model(tokenizer_dir: Path, model_dir: Path) -> None:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=128,  # every quantized in_features is a multiple of 128
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.save_pretrained(model_dir)
    AutoTokenizer.from_pretrained(tokenizer_dir).save_pretrained(model_dir)


def synthetic_rows(kind: str, count: int) -> list[dict[str, Any]]:
    if kind == "user_text":
        return [{"text": f"Calibration passage {i}. " * 20} for i in range(count)]
    return [
        {
            "messages": [
                {"role": "user", "content": f"Question {i}: what is {i} plus {i}?"},
                {"role": "assistant", "content": f"The answer is {2 * i}."},
            ]
        }
        for i in range(count)
    ]


def main() -> int:
    config = yaml.safe_load(Path("configs/phase4_quantize.yaml").read_text())
    model_cfg = config["model"]
    WORK.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = Path(
        snapshot_download(
            model_cfg["id"],
            revision=model_cfg["revision"],
            allow_patterns=["tokenizer*", "vocab.json", "merges.txt"],
            local_dir=WORK / "tokenizer",
        )
    )
    model_dir = WORK / "tiny-qwen3"
    tiny_model(tokenizer_dir, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    report: dict[str, Any] = {"model": "tiny random Qwen3 (2 layers, hidden 128)"}
    failed = False
    for name, variant in config["variants"].items():
        calibration = copy.deepcopy(variant["calibration"])
        # Rehearsal overrides only the size of the workload, not the method.
        calibration.update({"num_samples": 8, "max_seq_length": 64})
        started = time.perf_counter()
        prep = prepare_calibration(
            tokenizer,
            calibration,
            WORK / f"calibration-{name}",
            rows=synthetic_rows(calibration["format"], 12),
        )
        summary = run_quantization(
            model_path=str(model_dir),
            calibration_dir=WORK / f"calibration-{name}",
            out_dir=WORK / f"quantized-{name}",
            variant=variant,
            calibration=calibration,
        )
        failures = check_checkpoint(summary, variant)
        wrong = dict(variant, expected_symmetric=not variant["expected_symmetric"])
        mutation_caught = bool(check_checkpoint(summary, wrong))
        tensors = inspect_tensors(
            WORK / f"quantized-{name}", bool(variant["expected_symmetric"])
        )
        failures += tensors["failures"]
        # compressed-tensors 0.11.0 cannot decompress packed zero points in
        # transformers, so only the symmetric checkpoint gets an HF forward;
        # vLLM loads both (Phase 4 sanity run).
        finite: bool | None = None
        if variant["expected_symmetric"]:
            reloaded = AutoModelForCausalLM.from_pretrained(WORK / f"quantized-{name}")
            ids = torch.tensor([tokenizer("Hello world")["input_ids"]])
            with torch.inference_mode():
                finite = bool(torch.isfinite(reloaded(input_ids=ids).logits).all())
        report[name] = {
            "calibration_samples": prep["samples"],
            "calibration_tokens_max": prep["tokens_max"],
            "structural_failures": failures,
            "wrong_symmetry_rejected": mutation_caught,
            "reload_forward_finite": finite,
            "tensors": tensors,
            "quantization_config": summary["quantization_config"],
            "safetensors_bytes": summary["checkpoint"]["safetensors_bytes"],
            "elapsed_s": time.perf_counter() - started,
        }
        failed |= bool(failures) or not mutation_caught or finite is False
        print(
            name,
            json.dumps(
                {k: v for k, v in report[name].items() if k != "quantization_config"}
            ),
        )
    OUT.mkdir(parents=True, exist_ok=True)
    report["passed"] = not failed
    (OUT / "rehearsal.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print("rehearsal passed" if not failed else "rehearsal FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
