"""Phase 4 W4A16 quantization with llm-compressor (AWQ and GPTQ).

Shared by the Modal functions and the local CPU rehearsal. `llmcompressor`,
`datasets` and friends live in a separate pinned environment
(`requirements/quantize.in`) because they need an older transformers than the
serving stack, so they are imported inside functions.

Calibration preprocessing follows each method's official llm-compressor 0.7.1
example (slice, shuffle(seed), chat template, truncate, no added special
tokens) but runs on CPU ahead of time, so the GPU job only loads token IDs.
"""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_manifest(path: Path) -> dict[str, Any]:
    """Every file with size and sha256, plus totals (on-disk checkpoint size)."""
    files: list[dict[str, Any]] = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = item.relative_to(path).as_posix()
        if relative.startswith(".cache/"):
            continue  # huggingface_hub bookkeeping, not part of the checkpoint
        files.append(
            {
                "path": relative,
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return {
        "files": files,
        "total_bytes": sum(entry["bytes"] for entry in files),
        "safetensors_bytes": sum(
            entry["bytes"] for entry in files if entry["path"].endswith(".safetensors")
        ),
    }


def preprocess_rows(
    tokenizer: Any, rows: list[dict[str, Any]], calibration: dict[str, Any]
) -> list[dict[str, list[int]]]:
    """Apply the example's chat template and truncation to already-sliced rows."""
    field = calibration["text_field"]
    max_length = int(calibration["max_seq_length"])
    examples = []
    for row in rows:
        if calibration["format"] == "user_text":
            messages = [{"role": "user", "content": row[field]}]
        elif calibration["format"] == "messages":
            messages = row[field]
        else:
            raise ValueError(f"unknown calibration format {calibration['format']}")
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        encoded = tokenizer(
            text,
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )
        examples.append(
            {
                "input_ids": list(encoded["input_ids"]),
                "attention_mask": list(encoded["attention_mask"]),
            }
        )
    return examples


def calibration_summary(examples: list[dict[str, list[int]]]) -> dict[str, Any]:
    lengths = [len(example["input_ids"]) for example in examples]
    return {
        "samples": len(examples),
        "tokens_total": sum(lengths),
        "tokens_min": min(lengths) if lengths else 0,
        "tokens_max": max(lengths) if lengths else 0,
        "input_ids_sha256": hashlib.sha256(
            json.dumps([e["input_ids"] for e in examples]).encode()
        ).hexdigest(),
    }


def prepare_calibration(
    tokenizer: Any,
    calibration: dict[str, Any],
    out_dir: Path,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load (or accept) the example's slice, preprocess it and save to disk."""
    from datasets import Dataset, load_dataset

    samples = int(calibration["num_samples"])
    if rows is None:
        source = load_dataset(
            calibration["dataset_id"],
            split=f"{calibration['split']}[:{samples}]",
            revision=calibration["revision"],
        )
    else:
        source = Dataset.from_list(rows[:samples])
    source = source.shuffle(seed=int(calibration["shuffle_seed"]))
    examples = preprocess_rows(tokenizer, list(source), calibration)
    if len(examples) != samples:
        raise ValueError(f"expected {samples} calibration samples, got {len(examples)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(examples).save_to_disk(str(out_dir / "dataset"))
    summary = {
        "calibration": calibration,
        **calibration_summary(examples),
        # Shows exactly how the chat template rendered the first sample.
        "first_sample_decoded_prefix": tokenizer.decode(examples[0]["input_ids"][:96]),
    }
    (out_dir / "calibration.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_recipe(variant: dict[str, Any]) -> list[Any]:
    """The official example's modifier, with every argument taken from config."""
    kwargs = {
        "targets": variant["targets"],
        "scheme": variant["scheme"],
        "ignore": variant["ignore"],
    }
    if variant["method"] == "awq":
        from llmcompressor.modifiers.awq import AWQModifier

        return [AWQModifier(**kwargs)]
    if variant["method"] == "gptq":
        from llmcompressor.modifiers.quantization import (
            GPTQModifier,
        )

        return [GPTQModifier(**kwargs)]
    raise ValueError(f"unknown method {variant['method']}")


class GpuMemorySampler:
    """Poll nvidia-smi so peak device memory includes non-PyTorch allocations."""

    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self.peak_mib = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                ).stdout.split()
                if out:
                    self.peak_mib = max(self.peak_mib, int(out[0]))
                    self.samples += 1
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuMemorySampler:
        if shutil.which("nvidia-smi"):
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=15)


def run_quantization(
    *,
    model_path: str,
    calibration_dir: Path,
    out_dir: Path,
    variant: dict[str, Any],
    calibration: dict[str, Any],
    progress: Callable[[str, dict[str, Any]], None] = lambda stage, data: None,
) -> dict[str, Any]:
    """Load BF16 weights, run oneshot, save compressed; returns timings/peaks."""
    import torch
    from datasets import load_from_disk
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    summary: dict[str, Any] = {"variant": variant, "calibration": calibration}
    started = time.perf_counter()
    with GpuMemorySampler() as sampler:
        tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore[no-untyped-call]
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
        summary["load_model_s"] = time.perf_counter() - started
        summary["model_dtype"] = str(next(model.parameters()).dtype)
        progress("model_loaded", summary)

        dataset = load_from_disk(str(calibration_dir / "dataset"))
        recipe = build_recipe(variant)
        oneshot_started = time.perf_counter()
        oneshot(
            model=model,
            dataset=dataset,
            recipe=recipe,
            max_seq_length=int(calibration["max_seq_length"]),
            num_calibration_samples=int(calibration["num_samples"]),
        )
        summary["oneshot_s"] = time.perf_counter() - oneshot_started
        progress("oneshot_done", summary)

        save_started = time.perf_counter()
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_dir), save_compressed=True)
        tokenizer.save_pretrained(str(out_dir))
        summary["save_s"] = time.perf_counter() - save_started
    summary["total_s"] = time.perf_counter() - started
    summary["peak_gpu_memory_nvidia_smi_mib"] = sampler.peak_mib or None
    summary["nvidia_smi_samples"] = sampler.samples
    if torch.cuda.is_available():
        summary["torch_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["torch_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved()
    # ru_maxrss is KiB on Linux, bytes on macOS; record the raw value and OS.
    summary["host_peak_rss_raw"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    config = json.loads((out_dir / "config.json").read_text())
    summary["quantization_config"] = config.get("quantization_config")
    summary["checkpoint"] = directory_manifest(out_dir)
    return summary


def check_checkpoint(summary: dict[str, Any], variant: dict[str, Any]) -> list[str]:
    """Structural checks that the saved checkpoint is what the recipe asked for."""
    failures = []
    config = summary.get("quantization_config") or {}
    if config.get("quant_method") != "compressed-tensors":
        failures.append(f"quant_method {config.get('quant_method')!r}")
    if config.get("format") != "pack-quantized":
        failures.append(f"format {config.get('format')!r}")
    groups = config.get("config_groups") or {}
    weights = [group.get("weights") or {} for group in groups.values()]
    if not weights:
        failures.append("no quantization config groups")
    for spec in weights:
        if spec.get("num_bits") != 4:
            failures.append(f"num_bits {spec.get('num_bits')}")
        if spec.get("group_size") != int(variant["expected_group_size"]):
            failures.append(f"group_size {spec.get('group_size')}")
        if spec.get("symmetric") != bool(variant["expected_symmetric"]):
            failures.append(f"symmetric {spec.get('symmetric')}")
    if "lm_head" not in (config.get("ignore") or []):
        failures.append("lm_head not in ignore list")
    if not summary.get("checkpoint", {}).get("safetensors_bytes"):
        failures.append("no safetensors written")
    return failures


def inspect_tensors(checkpoint_dir: Path, symmetric: bool) -> dict[str, Any]:
    """Check stored tensors: packed int4 linears, zero points iff asymmetric,
    and an unquantized BF16 lm_head. Reads headers and one tensor per kind."""
    from safetensors import safe_open

    names: dict[str, str] = {}
    lm_head_dtype = None
    for shard in sorted(checkpoint_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:  # type: ignore[no-untyped-call]
            for key in handle.keys():  # noqa: SIM118 (safe_open is not a dict)
                names[key] = shard.name
                if key == "lm_head.weight":
                    lm_head_dtype = str(handle.get_slice(key).get_dtype())
    q_proj = sorted(k for k in names if k.endswith("self_attn.q_proj.weight_packed"))
    zero_points = [k for k in names if k.endswith("weight_zero_point")]
    unpacked_linears = [
        k
        for k in names
        if k.endswith("proj.weight") and not k.endswith("weight_packed")
    ]
    failures = []
    if not q_proj:
        failures.append("no packed q_proj weights")
    if symmetric and zero_points:
        failures.append("symmetric scheme stored zero points")
    if not symmetric and not zero_points:
        failures.append("asymmetric scheme stored no zero points")
    if unpacked_linears:
        failures.append(f"unquantized linear weights: {unpacked_linears[:3]}")
    if lm_head_dtype not in {"BF16", "torch.bfloat16"}:
        failures.append(f"lm_head dtype {lm_head_dtype}")
    return {
        "tensor_count": len(names),
        "packed_q_proj_layers": len(q_proj),
        "zero_point_tensors": len(zero_points),
        "lm_head_dtype": lm_head_dtype,
        "failures": failures,
    }
