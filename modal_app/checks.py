"""CPU-only checks inside the serving images, run before any GPU is attached.

    uv run modal run --detach -m modal_app.checks::vllm_env
    uv run modal run --detach -m modal_app.checks::hf_env
    uv run modal run --detach -m modal_app.checks::loadtest

`vllm_env` builds the Dockerfile image, records versions, verifies every
configured engine flag against the pinned server's own `--help`, and reruns
the Phase 1 mock timing/capacity validation inside that container.

`loadtest` reruns the timing gates and the 2-process capacity gate inside the
vLLM image with the Phase 6 container's CPU count (ADR-013). It must pass
before any Phase 6 benchmark.
"""

from __future__ import annotations

import json
from typing import Any

import modal

from modal_app.common import (
    HF_CHECK_RESOURCES,
    HF_IMAGE,
    LOADTEST_CHECK_RESOURCES,
    RESULTS,
    RESULTS_PATH,
    VLLM_CHECK_RESOURCES,
    VLLM_IMAGE,
    load_config,
    run_stamp,
)

app = modal.App("llmbench-checks")


def _run(command: list[str], timeout_s: float) -> dict[str, Any]:
    import subprocess

    done = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_s, check=False
    )
    return {
        "command": command,
        "returncode": done.returncode,
        "stdout": done.stdout,
        "stderr": done.stderr,
    }


def _flags(args: list[str]) -> list[str]:
    return [arg for arg in args if arg.startswith("--")]


@app.function(
    image=VLLM_IMAGE,
    volumes={RESULTS_PATH: RESULTS},
    **VLLM_CHECK_RESOURCES.function_kwargs(),
)
def vllm_env_check(config: dict[str, Any], stamp: str) -> dict[str, Any]:
    import os
    from pathlib import Path

    from llmbench.smoke import missing_flags, runtime_metadata, write_json

    out = Path(RESULTS_PATH) / "phase3" / f"vllm-env-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "image": os.environ.get("LLMBENCH_VLLM_IMAGE"),
        "resources": VLLM_CHECK_RESOURCES.as_dict(),
        "runtime": runtime_metadata(),
    }
    freeze = _run(["python3", "-m", "pip", "freeze"], 60)
    (out / "pip_freeze.txt").write_text(freeze["stdout"])
    help_run = _run(["vllm", "serve", "--help"], 120)
    (out / "vllm_serve_help.txt").write_text(help_run["stdout"] + help_run["stderr"])
    required = sorted(
        set(_flags(config["vllm"]["engine_args"]))
        | {"--served-model-name", "--host", "--port"}
    )
    help_text = help_run["stdout"]
    summary["help_returncode"] = help_run["returncode"]
    summary["flags_checked"] = required
    summary["flags_missing"] = missing_flags(required, help_text)
    # vLLM 0.10.2 cannot build its parser without a GPU ("Failed to infer
    # device type"); the GPU smoke verifies flags before starting the server.
    summary["help_needs_gpu"] = "Failed to infer device type" in help_run["stderr"]
    write_json(out / "env_summary.json", summary)
    RESULTS.commit()

    validation = _run(
        ["python3", "-m", "llmbench.loadtest.validation", "--output-dir", str(out)],
        150,
    )
    (out / "mock_validation_stdout.txt").write_text(
        validation["stdout"] + validation["stderr"]
    )
    summary["mock_validation_returncode"] = validation["returncode"]
    capacity_file = out / "capacity_summary.json"
    if capacity_file.exists():
        capacity = json.loads(capacity_file.read_text())
        summary["capacity"] = {
            key: capacity.get(key)
            for key in (
                "measured_text_chunks_per_s",
                "capacity_threshold_multiple",
                "client_process_cpu_cores_average",
                "errored_or_cancelled_requests",
                "rejected_requests",
                "late_completed_requests",
            )
        }
    timing_file = out / "timing_accuracy_summary.json"
    if timing_file.exists():
        timing = json.loads(timing_file.read_text())
        summary["timing"] = {
            key: timing.get(key)
            for key in (
                "passed",
                "median_ttft_s",
                "median_server_ttft_s",
                "median_itl_s",
                "median_server_itl_s",
            )
        }
    help_ok = summary["help_needs_gpu"] or (
        help_run["returncode"] == 0 and not summary["flags_missing"]
    )
    summary["passed"] = help_ok and validation["returncode"] == 0
    write_json(out / "env_summary.json", summary)
    RESULTS.commit()
    if not summary["passed"]:
        raise RuntimeError(f"vLLM environment check failed: {summary}")
    return summary


@app.function(
    image=HF_IMAGE,
    volumes={RESULTS_PATH: RESULTS},
    **HF_CHECK_RESOURCES.function_kwargs(),
)
def hf_env_check(stamp: str) -> dict[str, Any]:
    from pathlib import Path

    from llmbench.smoke import runtime_metadata, write_json

    out = Path(RESULTS_PATH) / "phase3" / f"hf-env-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    freeze = _run(["python", "-m", "pip", "freeze"], 60)
    if freeze["returncode"] != 0:  # uv-built images may lack pip
        freeze = _run(["uv", "pip", "freeze"], 60)
    (out / "pip_freeze.txt").write_text(freeze["stdout"] + freeze["stderr"])
    server_help = _run(["python", "-m", "llmbench.baseline.hf_server", "--help"], 90)
    runtime = runtime_metadata()
    summary = {
        "resources": HF_CHECK_RESOURCES.as_dict(),
        "runtime": runtime,
        "server_help_returncode": server_help["returncode"],
        "server_help": server_help["stdout"] + server_help["stderr"],
    }
    cuda = (runtime["gpu"].get("torch_cuda") or {}).get("torch_cuda_version")
    summary["passed"] = server_help["returncode"] == 0 and cuda is not None
    write_json(out / "env_summary.json", summary)
    RESULTS.commit()
    if not summary["passed"]:
        raise RuntimeError(f"HF environment check failed: {summary}")
    return summary


@app.function(
    image=VLLM_IMAGE,
    volumes={RESULTS_PATH: RESULTS},
    **LOADTEST_CHECK_RESOURCES.function_kwargs(),
)
def loadtest_check(stamp: str) -> dict[str, Any]:
    from pathlib import Path

    from llmbench.smoke import runtime_metadata, write_json

    out = Path(RESULTS_PATH) / "phase6" / f"loadtest-check-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "resources": LOADTEST_CHECK_RESOURCES.as_dict(),
        "runtime": runtime_metadata(),
    }
    write_json(out / "check_summary.json", summary | {"state": "starting"})
    RESULTS.commit()
    validation = _run(
        [
            "python3",
            "-m",
            "llmbench.loadtest.validation",
            "--output-dir",
            str(out),
            "--processes",
            "2",
        ],
        540,
    )
    (out / "validation_stdout.txt").write_text(
        validation["stdout"] + validation["stderr"]
    )
    summary["validation_returncode"] = validation["returncode"]
    for name in ("timing_accuracy", "timing_multiprocess"):
        path = out / f"{name}_summary.json"
        if path.exists():
            timing = json.loads(path.read_text())
            summary[name] = {"passed": timing.get("passed"), "gate": timing.get("gate")}
    capacity_file = out / "capacity_summary.json"
    if capacity_file.exists():
        capacity = json.loads(capacity_file.read_text())
        summary["capacity"] = {
            key: capacity.get(key)
            for key in (
                "measured_text_chunks_per_s",
                "capacity_threshold_multiple",
                "client_processes",
                "client_process_cpu_cores_average",
                "shard_window_process_cpu_seconds",
                "mock_server_cpu_cores_average",
                "errored_or_cancelled_requests",
                "rejected_requests",
                "late_completed_requests",
                "process_clock_check",
            )
        }
    summary["passed"] = validation["returncode"] == 0
    summary["state"] = "finished"
    write_json(out / "check_summary.json", summary)
    RESULTS.commit()
    if not summary["passed"]:
        raise RuntimeError(f"load tester check failed: {summary}")
    return summary


@app.local_entrypoint()
def loadtest() -> None:
    result = loadtest_check.remote(run_stamp())
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def vllm_env(config: str = "phase3_smoke.yaml") -> None:
    settings, _ = load_config(config)
    result = vllm_env_check.remote(settings, run_stamp())
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def hf_env() -> None:
    result = hf_env_check.remote(run_stamp())
    print(json.dumps({k: v for k, v in result.items() if k != "server_help"}, indent=2))
