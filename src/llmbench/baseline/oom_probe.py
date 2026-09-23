"""Largest HF static batch that fits at the benchmark's lengths (SPEC §4).

    python -m llmbench.baseline.oom_probe --model PATH --prompts PROMPTS.npy \
        --output-tokens 256 --candidates 8,16,32,64 --out oom_probe.json

Each candidate batch runs the server's own `Baseline._generate` (same
greedy settings, exact output length) on real prompts, in increasing order,
stopping at the first CUDA out-of-memory error. The largest batch that
completed is the static batch size B. Peak memory and time are recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmbench.baseline.hf_server import Baseline, Job
from llmbench.prompts import load_pool


async def probe(
    baseline: Baseline, prompts: list[list[int]], output_tokens: int
) -> dict[str, Any]:
    """Generate one batch; raise torch.cuda.OutOfMemoryError if it does not fit."""
    jobs = [Job(prompt, output_tokens) for prompt in prompts]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    await asyncio.to_thread(baseline._generate, jobs, asyncio.get_running_loop())
    elapsed = time.perf_counter() - started
    counts = []
    for job in jobs:
        count = 0
        while (item := job.events.get_nowait()) is not None:
            if isinstance(item, BaseException):
                raise item
            count += 1
        counts.append(count)
    if set(counts) != {output_tokens}:
        raise RuntimeError(f"generated {sorted(set(counts))}, not {output_tokens}")
    return {
        "batch_size": len(jobs),
        "seconds": elapsed,
        "output_tokens_per_s": len(jobs) * output_tokens / elapsed,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model)  # type: ignore[no-untyped-call]
    model: Any = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32 if args.dtype == "float32" else torch.bfloat16,
        attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        model = model.cuda()
    baseline = Baseline(model, tokenizer, mode="static", batch_size=1)
    pool = load_pool(Path(args.prompts))
    candidates = sorted(int(c) for c in args.candidates.split(","))
    results: list[dict[str, Any]] = []
    stopped_by = None
    offset = 0
    for size in candidates:
        prompts = [[int(t) for t in pool[offset + i]] for i in range(size)]
        offset += size
        try:
            results.append(await probe(baseline, prompts, args.output_tokens))
        except torch.cuda.OutOfMemoryError as exc:
            stopped_by = {"batch_size": size, "error": str(exc)[:500]}
            torch.cuda.empty_cache()
            break
        print(json.dumps(results[-1]), flush=True)
    return {
        "model": args.model,
        "input_tokens": len(pool[0]),
        "output_tokens": args.output_tokens,
        "candidates": candidates,
        "fitted": results,
        "first_oom": stopped_by,
        "chosen_batch_size": results[-1]["batch_size"] if results else None,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m llmbench.baseline.oom_probe")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--candidates", default="8,16,32,48,64,96,128")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if result["chosen_batch_size"] is None:
        raise SystemExit("no candidate batch fitted")


if __name__ == "__main__":
    main()
