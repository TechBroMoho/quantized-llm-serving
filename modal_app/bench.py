"""Phase 6 performance benchmarks on Modal (BILLABLE: announce and ask first).

    uv run modal run --detach -m modal_app.bench::prepare      # CPU only
    uv run modal run --detach -m modal_app.bench::run --lifetime probe
    uv run modal run --detach -m modal_app.bench::run --lifetime awq   # etc.

`prepare` (CPU) checks every checkpoint file against the sha256 that Phase 4
recorded and builds the WikiText-103 prompt pool (ADR-021). Each GPU lifetime
refuses to start unless that check passed for its exact checkpoint path, so
the speed numbers come from the same bytes Phase 5 measured accuracy on.
Entrypoints `.spawn()` and exit, so a sleeping laptop cannot cancel a run
(ADR-018); poll with `modal app list` and sync with `make sync-bench`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import modal

from modal_app.common import (
    BENCH_AWQ_RESOURCES,
    BENCH_BF16_RESOURCES,
    BENCH_GPTQ_RESOURCES,
    BENCH_HF_RESOURCES,
    BENCH_MAXBATCH_RESOURCES,
    BENCH_PREPARE_RESOURCES,
    BENCH_PROBE_RESOURCES,
    EVAL_IMAGE,
    HF_IMAGE,
    HF_SECRET,
    OFFLINE_ENV,
    REPO,
    RESULTS,
    RESULTS_PATH,
    VLLM_CACHE,
    VLLM_CACHE_PATH,
    VLLM_IMAGE,
    WEIGHTS,
    WEIGHTS_PATH,
    Resources,
    load_config,
    model_dir,
    quantized_dir,
    run_stamp,
)

app = modal.App("llmbench-bench")

CONFIG = "phase6_bench.yaml"
PORT = 8000
VERIFIED_DIR = f"{WEIGHTS_PATH}/phase6/verified"
PROMPTS_DIR = f"{WEIGHTS_PATH}/phase6/prompts"


def checkpoint_path(run: dict[str, Any], variant: str) -> str:
    model = run["config"]["model"]
    if variant == "bf16":
        return model_dir(model["id"], model["revision"])
    return quantized_dir(run["quant_config"], variant)


def prompt_pool_path(config: dict[str, Any]) -> str:
    prompts = config["prompts"]
    name = (
        f"wikitext103-{prompts['revision'][:8]}-{prompts['count']}x"
        f"{prompts['input_tokens']}-seed{prompts['seed']}"
    )
    return f"{PROMPTS_DIR}/{name}.npy"


# --- CPU: checkpoint verification + prompt pool ---------------------------


@app.function(
    image=EVAL_IMAGE,
    secrets=[HF_SECRET],
    volumes={WEIGHTS_PATH: WEIGHTS, RESULTS_PATH: RESULTS},
    **BENCH_PREPARE_RESOURCES.function_kwargs(),
)
def prepare_fn(run: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from llmbench.bench import verify_checkpoint
    from llmbench.prompts import build_pool
    from llmbench.smoke import runtime_metadata, write_json

    config = run["config"]
    out = Path(RESULTS_PATH) / "phase6" / f"prepare-{run['stamp']}"
    summary: dict[str, Any] = {
        "run": {k: run[k] for k in ("stamp", "git", "config_sha256")},
        "resources": BENCH_PREPARE_RESOURCES.as_dict(),
        "runtime": runtime_metadata(),
        "checkpoints": {},
    }
    write_json(out / "prepare_summary.json", summary | {"state": "starting"})
    RESULTS.commit()

    Path(VERIFIED_DIR).mkdir(parents=True, exist_ok=True)
    for variant, expected in run["expected_checkpoints"].items():
        root = checkpoint_path(run, variant)
        result = verify_checkpoint(Path(root), expected["files"])
        result |= {"variant": variant, "evidence": expected["source"]}
        result["evidence_sha256"] = expected["sha256"]
        summary["checkpoints"][variant] = result
        write_json(Path(VERIFIED_DIR) / f"{variant}.json", result)
        print(f"{variant}: passed={result['passed']} in {result['seconds']:.0f} s")
    WEIGHTS.commit()
    write_json(out / "prepare_summary.json", summary | {"state": "verified"})
    RESULTS.commit()

    prompts = config["prompts"]
    lines: list[str] = []
    downloaded = []
    for filename in prompts["files"]:
        path = hf_hub_download(
            prompts["dataset"],
            filename,
            repo_type="dataset",
            revision=prompts["revision"],
        )
        downloaded.append({"file": filename, "bytes": Path(path).stat().st_size})
        lines.extend(pq.read_table(path, columns=["text"]).column("text").to_pylist())
    tokenizer_dir = checkpoint_path(run, "bf16")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    pool, info = build_pool(
        lines,
        lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
        special_ids=tokenizer.all_special_ids,
        count=int(prompts["count"]),
        length=int(prompts["input_tokens"]),
        seed=int(prompts["seed"]),
        oversample=float(prompts["oversample"]),
    )
    target = Path(prompt_pool_path(config))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, pool)
    manifest = info | {
        "path": str(target),
        "dataset": prompts["dataset"],
        "dataset_revision": prompts["revision"],
        "files": downloaded,
        "lines": len(lines),
        "tokenizer": tokenizer_dir,
        "license": "CC BY-SA 3.0 / GFDL (text is not redistributed; IDs only)",
        "sample_decoded": tokenizer.decode(pool[0][:64].tolist()),
    }
    write_json(target.with_suffix(".json"), manifest)
    WEIGHTS.commit()
    summary["prompts"] = manifest
    summary["passed"] = all(c["passed"] for c in summary["checkpoints"].values())
    write_json(out / "prepare_summary.json", summary | {"state": "finished"})
    RESULTS.commit()
    if not summary["passed"]:
        raise RuntimeError(f"checkpoint verification failed: {summary['checkpoints']}")
    return summary


# --- GPU lifetimes ---------------------------------------------------------


def _require_prepared(run: dict[str, Any], variant: str) -> dict[str, Any]:
    """Refuse to spend GPU time on an unverified checkpoint or prompt pool."""
    from llmbench.prompts import load_pool, pool_sha256

    record_path = Path(VERIFIED_DIR) / f"{variant}.json"
    if not record_path.exists():
        raise RuntimeError(f"run prepare first: no {record_path}")
    record = json.loads(record_path.read_text())
    expected = run["expected_checkpoints"][variant]
    if not (
        record["passed"]
        and record["root"] == checkpoint_path(run, variant)
        and record["evidence_sha256"] == expected["sha256"]
    ):
        raise RuntimeError(f"checkpoint verification does not cover this run: {record}")
    pool_path = Path(prompt_pool_path(run["config"]))
    manifest = json.loads(pool_path.with_suffix(".json").read_text())
    if pool_sha256(load_pool(pool_path)) != manifest["pool_sha256"]:
        raise RuntimeError("prompt pool does not match its manifest")
    return {"checkpoint_verification": record, "prompt_pool": manifest}


def _metadata(run: dict[str, Any], resources: Resources) -> dict[str, Any]:
    return {
        "config": run["config"],
        "config_yaml": run["config_yaml"],
        "config_sha256": run["config_sha256"],
        "git": run["git"],
        "resources": resources.as_dict(),
    }


def _cross_check(run: dict[str, Any], served: str, tokenizer: str) -> Any:
    """`vllm bench serve` at matching settings, after our own points."""
    from llmbench.bench import bench_serve_flag_failures, vllm_bench_serve_command

    workload = run["config"]["workload"]
    settings = run["config"]["lifetimes"][run["lifetime"]].get("cross_check", [])

    def after(out_dir: Path) -> dict[str, Any]:
        results: dict[str, Any] = {"settings": settings, "runs": [], "failures": []}
        result_dir = out_dir / "vllm_bench_serve"
        result_dir.mkdir(parents=True, exist_ok=True)
        for index, setting in enumerate(settings):
            command = vllm_bench_serve_command(
                base_url=f"http://127.0.0.1:{PORT}",
                model_name=served,
                tokenizer=tokenizer,
                concurrency=int(setting["concurrency"]),
                num_prompts=int(setting["num_prompts"]),
                input_tokens=int(workload["input_tokens"]),
                output_tokens=int(workload["output_tokens"]),
                result_dir=result_dir,
                seed=int(workload["seed"]),
            )
            if index == 0:
                flag_failures = bench_serve_flag_failures(command, result_dir)
                if flag_failures:
                    results["failures"] += flag_failures
                    return results
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=1200, check=False
            )
            log = result_dir / f"c{setting['concurrency']}.log"
            log.write_text(done.stdout + done.stderr)
            entry: dict[str, Any] = {"command": command, "returncode": done.returncode}
            saved = result_dir / f"vllm_bench_serve_c{setting['concurrency']}.json"
            if saved.exists():
                entry["result"] = json.loads(saved.read_text())
            else:
                results["failures"].append(f"no result for {setting}")
            results["runs"].append(entry)
            RESULTS.commit()
        return results

    return after


def _vllm_lifetimes(run: dict[str, Any], names: list[str], resources: Resources) -> Any:
    import asyncio

    from llmbench.bench import resolve_points, run_lifetime
    from llmbench.prompts import NpyPromptPayloads
    from llmbench.smoke import verify_vllm_flags

    config = run["config"]
    workload = config["workload"]
    summaries = {}
    for name in names:
        spec = config["lifetimes"][name]
        variant = spec["variant"]
        prepared = _require_prepared(run, variant)
        out = Path(RESULTS_PATH) / "phase6" / f"{name}-{run['stamp']}"
        engine_args = [
            *config["vllm"]["engine_args"],
            *spec.get("extra_engine_args", []),
        ]
        missing = verify_vllm_flags(engine_args, out)
        if missing:
            raise RuntimeError(f"vLLM flags not in the pinned --help: {missing}")
        path = checkpoint_path(run, variant)
        served = f"qwen3-8b-{variant}"
        command = [
            "vllm",
            "serve",
            path,
            "--served-model-name",
            served,
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            *engine_args,
        ]
        summaries[name] = asyncio.run(
            run_lifetime(
                label=name,
                command=command,
                base_url=f"http://127.0.0.1:{PORT}",
                kind="vllm",
                payloads=NpyPromptPayloads(
                    prompt_pool_path(config),
                    output_tokens=int(workload["output_tokens"]),
                    seed=int(workload["seed"]),
                    model=served,
                ),
                points=resolve_points(config["points"], spec["points"]),
                processes=int(workload["client_processes"]),
                out_dir=out,
                input_tokens=int(workload["input_tokens"]),
                output_tokens=int(workload["output_tokens"]),
                health_timeout_s=float(config["vllm"]["health_timeout_s"]),
                idle_timeout_s=float(workload["idle_timeout_s"]),
                validated_chunks_per_s=float(workload["validated_client_chunks_per_s"]),
                env=OFFLINE_ENV,
                checkpoint=RESULTS.commit,
                metadata=_metadata(run, resources)
                | prepared
                | {"lifetime": name, "variant": variant, "checkpoint": path},
                after_points=(
                    _cross_check(run | {"lifetime": name}, served, path)
                    if spec.get("cross_check")
                    else None
                ),
            )
        )
        VLLM_CACHE.commit()
    return summaries


_VLLM_VOLUMES = {
    WEIGHTS_PATH: WEIGHTS,
    RESULTS_PATH: RESULTS,
    VLLM_CACHE_PATH: VLLM_CACHE,
}


@app.function(
    image=VLLM_IMAGE,
    env=OFFLINE_ENV,
    volumes=_VLLM_VOLUMES,
    **BENCH_PROBE_RESOURCES.function_kwargs(),
)
def probe_fn(run: dict[str, Any]) -> Any:
    return _vllm_lifetimes(run, ["probe"], BENCH_PROBE_RESOURCES)


@app.function(
    image=VLLM_IMAGE,
    env=OFFLINE_ENV,
    volumes=_VLLM_VOLUMES,
    **BENCH_BF16_RESOURCES.function_kwargs(),
)
def bf16_fn(run: dict[str, Any]) -> Any:
    return _vllm_lifetimes(run, ["bf16"], BENCH_BF16_RESOURCES)


@app.function(
    image=VLLM_IMAGE,
    env=OFFLINE_ENV,
    volumes=_VLLM_VOLUMES,
    **BENCH_AWQ_RESOURCES.function_kwargs(),
)
def awq_fn(run: dict[str, Any]) -> Any:
    return _vllm_lifetimes(run, ["awq"], BENCH_AWQ_RESOURCES)


@app.function(
    image=VLLM_IMAGE,
    env=OFFLINE_ENV,
    volumes=_VLLM_VOLUMES,
    **BENCH_GPTQ_RESOURCES.function_kwargs(),
)
def gptq_fn(run: dict[str, Any]) -> Any:
    return _vllm_lifetimes(run, ["gptq"], BENCH_GPTQ_RESOURCES)


@app.function(
    image=VLLM_IMAGE,
    env=OFFLINE_ENV,
    volumes=_VLLM_VOLUMES,
    **BENCH_MAXBATCH_RESOURCES.function_kwargs(),
)
def maxbatch_fn(run: dict[str, Any]) -> Any:
    return _vllm_lifetimes(
        run, ["maxbatch-16", "maxbatch-64"], BENCH_MAXBATCH_RESOURCES
    )


@app.function(
    image=HF_IMAGE,
    env=OFFLINE_ENV,
    volumes={WEIGHTS_PATH: WEIGHTS, RESULTS_PATH: RESULTS},
    **BENCH_HF_RESOURCES.function_kwargs(),
)
def hf_fn(run: dict[str, Any]) -> Any:
    import asyncio
    import sys

    from llmbench.bench import resolve_points, run_lifetime
    from llmbench.prompts import NpyPromptPayloads

    config = run["config"]
    workload = config["workload"]
    hf = config["hf"]
    prepared = _require_prepared(run, "bf16")
    path = checkpoint_path(run, "bf16")
    summaries = {}
    for name in ("hf-naive", "hf-static"):
        spec = config["lifetimes"][name]
        out = Path(RESULTS_PATH) / "phase6" / f"{name}-{run['stamp']}"
        out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "llmbench.baseline.hf_server",
            "--model",
            path,
            "--mode",
            spec["mode"],
            "--dtype",
            hf["dtype"],
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ]
        batch_size = None
        if spec["mode"] == "static":
            probe_out = out / "oom_probe.json"
            done = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "llmbench.baseline.oom_probe",
                    "--model",
                    path,
                    "--prompts",
                    prompt_pool_path(config),
                    "--output-tokens",
                    str(workload["output_tokens"]),
                    "--candidates",
                    ",".join(str(c) for c in hf["oom_candidates"]),
                    "--dtype",
                    hf["dtype"],
                    "--out",
                    str(probe_out),
                ],
                capture_output=True,
                text=True,
                timeout=1200,
                check=False,
            )
            (out / "oom_probe.log").write_text(done.stdout + done.stderr)
            RESULTS.commit()
            if done.returncode != 0:
                raise RuntimeError(f"OOM probe failed: {done.stderr[-2000:]}")
            batch_size = int(json.loads(probe_out.read_text())["chosen_batch_size"])
            command += [
                "--batch-size",
                str(batch_size),
                "--batch-wait-ms",
                str(hf["batch_wait_ms"]),
            ]
        summaries[name] = asyncio.run(
            run_lifetime(
                label=name,
                command=command,
                base_url=f"http://127.0.0.1:{PORT}",
                kind="hf",
                payloads=NpyPromptPayloads(
                    prompt_pool_path(config),
                    output_tokens=int(workload["output_tokens"]),
                    seed=int(workload["seed"]),
                    model="qwen3-8b-bf16-hf",
                ),
                points=resolve_points(config["points"], spec["points"], batch_size),
                processes=int(workload["client_processes"]),
                out_dir=out,
                input_tokens=int(workload["input_tokens"]),
                output_tokens=int(workload["output_tokens"]),
                health_timeout_s=float(hf["health_timeout_s"]),
                idle_timeout_s=float(workload["idle_timeout_s"]),
                validated_chunks_per_s=float(workload["validated_client_chunks_per_s"]),
                env=OFFLINE_ENV,
                checkpoint=RESULTS.commit,
                metadata=_metadata(run, BENCH_HF_RESOURCES)
                | prepared
                | {
                    "lifetime": name,
                    "variant": "bf16",
                    "engine": "hf",
                    "static_batch_size": batch_size,
                    "checkpoint": path,
                },
            )
        )
    return summaries


# --- Local entrypoints ------------------------------------------------------


def _prepare_run() -> dict[str, Any]:
    """Laptop pre-flight: config, git state and Phase 4's expected hashes."""
    import hashlib

    from llmbench.bench import expected_checkpoint_files
    from llmbench.smoke import config_sha256

    config, text = load_config(CONFIG)
    quant_config, _ = load_config(config["quantization_config"])

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=REPO
        ).stdout.strip()

    expected = {}
    for variant, source in config["expected_checkpoints"].items():
        raw = (REPO / source).read_bytes()
        expected[variant] = {
            "source": source,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "files": expected_checkpoint_files(json.loads(raw)),
        }
    return {
        "config": config,
        "config_yaml": text,
        "config_sha256": config_sha256(config),
        "quant_config": quant_config,
        "expected_checkpoints": expected,
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "-s"))},
        "stamp": run_stamp(),
    }


_GPU_FUNCTIONS = {
    "probe": probe_fn,
    "bf16": bf16_fn,
    "awq": awq_fn,
    "gptq": gptq_fn,
    "maxbatch": maxbatch_fn,
    "hf": hf_fn,
}


@app.local_entrypoint()
def prepare() -> None:
    run = _prepare_run()
    call = prepare_fn.spawn(run)
    print(f"spawned prepare {run['stamp']} ({call.object_id}); poll modal app list")


@app.local_entrypoint()
def run(lifetime: str) -> None:
    if lifetime not in _GPU_FUNCTIONS:
        raise SystemExit(f"lifetime must be one of {sorted(_GPU_FUNCTIONS)}")
    plan = _prepare_run()
    if plan["git"]["dirty"]:
        print("warning: uncommitted changes; the recorded commit may not match")
    call = _GPU_FUNCTIONS[lifetime].spawn(plan)
    print(f"spawned {lifetime} {plan['stamp']} ({call.object_id}); poll modal app list")
