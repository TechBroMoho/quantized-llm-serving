"""L4 smoke tests: vLLM from our Dockerfile image, then the HF baseline image.

    uv run modal run --detach -m modal_app.smoke::vllm
    uv run modal run --detach -m modal_app.smoke::hf

Both use the same committed config, the same seeded 512-token-ID prompts and
localhost traffic inside one container. They are functional checks only.
"""

from __future__ import annotations

import json
from typing import Any

import modal

from modal_app.common import (
    HF_IMAGE,
    HF_SMOKE_RESOURCES,
    OFFLINE_ENV,
    RESULTS,
    RESULTS_PATH,
    VLLM_CACHE,
    VLLM_CACHE_PATH,
    VLLM_IMAGE,
    VLLM_SMOKE_RESOURCES,
    WEIGHTS,
    WEIGHTS_PATH,
    load_config,
    manifest_path,
    model_dir,
    run_stamp,
)

app = modal.App("llmbench-smoke")


def _metadata(run: dict[str, Any], resources: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": run["config"],
        "config_yaml": run["config_yaml"],
        "config_sha256": run["config_sha256"],
        "git": run["git"],
        "model_manifest": run["manifest"],
        "resources": resources,
    }


@app.function(
    image=VLLM_IMAGE,
    env=OFFLINE_ENV,
    volumes={
        WEIGHTS_PATH: WEIGHTS,
        RESULTS_PATH: RESULTS,
        VLLM_CACHE_PATH: VLLM_CACHE,
    },
    **VLLM_SMOKE_RESOURCES.function_kwargs(),
)
def vllm_smoke(run: dict[str, Any]) -> dict[str, Any]:
    import asyncio
    import os
    from pathlib import Path

    from llmbench.smoke import (
        VLLM_LOG_PATTERNS,
        SmokeWorkload,
        pool_for_model,
        run_server_smoke,
        vllm_metrics_collector,
    )

    config = run["config"]
    model_id = config["model"]["id"]
    path = model_dir(model_id, config["model"]["revision"])
    workload = SmokeWorkload.from_dict(config["workload"])
    out = Path(RESULTS_PATH) / "phase3" / f"smoke-{run['stamp']}" / "vllm"
    port = 8000
    command = [
        "vllm",
        "serve",
        path,
        "--served-model-name",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *config["vllm"]["engine_args"],
    ]
    metadata = _metadata(run, VLLM_SMOKE_RESOURCES.as_dict())
    metadata["image"] = os.environ.get("LLMBENCH_VLLM_IMAGE")
    summary = asyncio.run(
        run_server_smoke(
            label="vllm",
            command=command,
            base_url=f"http://127.0.0.1:{port}",
            model_name=model_id,
            pool=pool_for_model(path, workload),
            workload=workload,
            out_dir=out,
            health_timeout_s=float(config["vllm"]["health_timeout_s"]),
            collect=vllm_metrics_collector(out),
            log_patterns=VLLM_LOG_PATTERNS,
            checkpoint=RESULTS.commit,
            metadata=metadata,
        )
    )
    VLLM_CACHE.commit()
    if not summary["passed"]:
        raise RuntimeError(f"vLLM smoke failed: {summary['failures']}")
    return summary


@app.function(
    image=HF_IMAGE,
    env=OFFLINE_ENV,
    volumes={WEIGHTS_PATH: WEIGHTS, RESULTS_PATH: RESULTS},
    **HF_SMOKE_RESOURCES.function_kwargs(),
)
def hf_smoke(run: dict[str, Any]) -> dict[str, Any]:
    import asyncio
    from pathlib import Path

    from llmbench.smoke import (
        SmokeWorkload,
        hf_stats_collector,
        pool_for_model,
        run_server_smoke,
    )

    config = run["config"]
    model_id = config["model"]["id"]
    path = model_dir(model_id, config["model"]["revision"])
    workload = SmokeWorkload.from_dict(config["workload"])
    hf = config["hf"]
    results = {}
    for index, mode_config in enumerate(hf["modes"]):
        mode = mode_config["mode"]
        port = 8100 + index
        batch_size = int(mode_config.get("batch_size", 1))
        command = [
            "python",
            "-m",
            "llmbench.baseline.hf_server",
            "--model",
            path,
            "--mode",
            mode,
            "--dtype",
            hf["dtype"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        if mode == "static":
            command += [
                "--batch-size",
                str(batch_size),
                "--batch-wait-ms",
                str(mode_config["batch_wait_ms"]),
            ]
        out = Path(RESULTS_PATH) / "phase3" / f"smoke-{run['stamp']}" / f"hf_{mode}"
        results[mode] = asyncio.run(
            run_server_smoke(
                label=f"hf_{mode}",
                command=command,
                base_url=f"http://127.0.0.1:{port}",
                model_name=model_id,
                # Same seed per server lifetime: identical prompts across
                # systems, each prompt still used once within a run.
                pool=pool_for_model(path, workload),
                workload=workload,
                out_dir=out,
                health_timeout_s=float(hf["health_timeout_s"]),
                collect=hf_stats_collector(mode, batch_size, workload.total_requests),
                checkpoint=RESULTS.commit,
                metadata=_metadata(run, HF_SMOKE_RESOURCES.as_dict()),
            )
        )
    failed = {mode: r["failures"] for mode, r in results.items() if not r["passed"]}
    if failed:
        raise RuntimeError(f"HF smoke failed: {failed}")
    return results


def _prepare(config_name: str, expected_gpu: str | None) -> dict[str, Any]:
    """Local pre-flight: config, git state and a verified weights manifest."""
    import subprocess

    from llmbench.smoke import config_sha256

    config, text = load_config(config_name)
    if config["gpu"] != expected_gpu:
        raise SystemExit(f"config GPU {config['gpu']} != function GPU {expected_gpu}")
    model = config["model"]
    try:
        raw = b"".join(WEIGHTS.read_file(manifest_path(model["id"], model["revision"])))
    except (FileNotFoundError, modal.exception.NotFoundError) as exc:
        raise SystemExit(
            "no verified weights manifest; run modal_app.download"
        ) from exc
    manifest = json.loads(raw)
    if not manifest.get("verified"):
        raise SystemExit("weights manifest is not verified")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "config": config,
        "config_yaml": text,
        "config_sha256": config_sha256(config),
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "-s"))},
        "manifest": {k: v for k, v in manifest.items() if k != "failures"},
        "stamp": run_stamp(),
    }


def _report(summary: dict[str, Any]) -> None:
    keys = ("label", "passed", "failures", "health_wait_s", "server_lifetime_s")
    print(json.dumps({key: summary.get(key) for key in keys}, indent=2))
    print(json.dumps(summary.get("load"), indent=2))


@app.local_entrypoint()
def vllm(config: str = "phase3_smoke.yaml") -> None:
    run = _prepare(config, VLLM_SMOKE_RESOURCES.gpu)
    print(f"results: llmbench-results:/phase3/smoke-{run['stamp']}/vllm")
    _report(vllm_smoke.remote(run))


@app.local_entrypoint()
def hf(config: str = "phase3_smoke.yaml") -> None:
    run = _prepare(config, HF_SMOKE_RESOURCES.gpu)
    print(f"results: llmbench-results:/phase3/smoke-{run['stamp']}/hf_*")
    for summary in hf_smoke.remote(run).values():
        _report(summary)
