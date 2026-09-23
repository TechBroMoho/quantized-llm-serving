"""Server smoke tests shared by Modal functions and local CPU rehearsals.

One smoke run starts a server subprocess, waits for `/health`, runs a warmup
and a fixed number of closed-loop requests, then validates every record.
Raw records, the server log and a summary are written before any check can
fail, and the server is always terminated in `finally`.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import aiohttp

from llmbench.loadtest.io import write_results
from llmbench.loadtest.metrics import RequestRecord
from llmbench.loadtest.runner import run_load
from llmbench.loadtest.workloads import (
    TokenPromptPool,
    completions_payload,
    regular_token_ids,
)

RUNTIME_PACKAGES = (
    "llmbench",
    "vllm",
    "torch",
    "transformers",
    "tokenizers",
    "aiohttp",
    "fastapi",
    "uvicorn",
    "huggingface-hub",
    "modal",
)

# vLLM startup lines that document the resolved engine configuration.
VLLM_LOG_PATTERNS = (
    r"non-default args",
    r"Model loading took",
    r"GPU KV cache size",
    r"Maximum concurrency",
    r"prefix caching",
    r"[Ee]ager",
    r"Using .* backend",
)


@dataclass(frozen=True)
class SmokeWorkload:
    """A small fixed-length workload; values come from a committed config."""

    input_tokens: int
    output_tokens: int
    concurrency: int
    warmup_requests: int
    measured_requests: int
    timeout_s: float
    seed: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmokeWorkload:
        return cls(
            input_tokens=int(data["input_tokens"]),
            output_tokens=int(data["output_tokens"]),
            concurrency=int(data["concurrency"]),
            warmup_requests=int(data["warmup_requests"]),
            measured_requests=int(data["measured_requests"]),
            timeout_s=float(data["timeout_s"]),
            seed=int(data["seed"]),
        )

    @property
    def total_requests(self) -> int:
        return self.warmup_requests + self.measured_requests


def missing_flags(flags: Sequence[str], help_text: str) -> list[str]:
    """Flags absent from help as whole tokens (so --seed never matches --seeds)."""
    return [
        flag
        for flag in flags
        if not re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text)
    ]


def verify_vllm_flags(
    engine_args: Sequence[str], out_dir: Path, timeout_s: float = 180
) -> list[str]:
    """Check configured flags against the pinned server's own `--help`.

    Plain `--help` is the full argparse listing; in vLLM 0.10.2
    `--help=<word>` is a keyword filter (`--help=all` matches only names
    containing "all"), so it must not be used here.

    vLLM 0.10.2 builds its parser from config dataclasses that must infer a
    device, so this only works on a GPU host. Returns failure messages.
    """
    flags = sorted(
        {arg for arg in engine_args if arg.startswith("--")}
        | {"--served-model-name", "--host", "--port"}
    )
    done = subprocess.run(
        ["vllm", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vllm_serve_help.txt").write_text(done.stdout + done.stderr, "utf-8")
    if done.returncode != 0:
        return [f"vllm serve --help exited {done.returncode}"]
    return [
        f"flag not in pinned --help: {flag}"
        for flag in missing_flags(flags, done.stdout)
    ]


def config_sha256(config: dict[str, Any]) -> str:
    return sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pool_for_tokenizer(tokenizer: Any, workload: SmokeWorkload) -> TokenPromptPool:
    """Sample prompts from the tokenizer's base vocabulary, excluding specials."""
    allowed = regular_token_ids(
        int(tokenizer.vocab_size), [int(i) for i in tokenizer.all_special_ids]
    )
    return TokenPromptPool(
        allowed_ids=allowed,
        count=workload.total_requests,
        length=workload.input_tokens,
        seed=workload.seed,
    )


def pool_for_model(model_path: str, workload: SmokeWorkload) -> TokenPromptPool:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore[no-untyped-call]
    return pool_for_tokenizer(tokenizer, workload)


def package_versions(packages: Sequence[str] = RUNTIME_PACKAGES) -> dict[str, str]:
    found: dict[str, str] = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            continue
    return found


def gpu_metadata() -> dict[str, Any]:
    """Record the GPU from nvidia-smi and torch; both absent on CPU hosts."""
    result: dict[str, Any] = {"nvidia_smi": None, "torch_cuda": None}
    if shutil.which("nvidia-smi"):
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        result["nvidia_smi"] = {
            "returncode": query.returncode,
            "stdout": query.stdout.strip(),
            "stderr": query.stderr.strip(),
        }
    try:
        import torch

        result["torch_cuda"] = {
            "available": torch.cuda.is_available(),
            "torch_cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
    except ImportError:
        pass
    return result


def runtime_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "gpu": gpu_metadata(),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
    }


def start_server(
    command: Sequence[str], log_path: Path, env: dict[str, str] | None = None
) -> subprocess.Popen[bytes]:
    """Start a server in its own process group so children are also stopped."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("wb")
    try:
        return subprocess.Popen(
            list(command),
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})},
            start_new_session=True,
        )
    finally:
        log.close()


def stop_server(process: subprocess.Popen[bytes], grace_s: float = 30) -> int | None:
    """SIGTERM the process group, then SIGKILL it after a bounded grace period."""
    if process.poll() is None:
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=grace_s)
    # Engine workers can outlive the parent; clear the whole group either way.
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    return process.returncode


def log_tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return "\n".join(text.splitlines()[-lines:])


def matching_lines(path: Path, patterns: Sequence[str], limit: int = 50) -> list[str]:
    if not path.exists():
        return []
    compiled = [re.compile(pattern) for pattern in patterns]
    found = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(pattern.search(line) for pattern in compiled):
            found.append(line.strip()[:1000])
            if len(found) >= limit:
                break
    return found


async def wait_for_health(
    base_url: str, process: subprocess.Popen[bytes], timeout_s: float
) -> float:
    """Poll `/health`; fail fast if the server exits. Returns seconds waited."""
    started = time.perf_counter()
    deadline = started + timeout_s
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {process.returncode} before health"
                )
            try:
                async with session.get(f"{base_url}/health") as response:
                    if response.status == 200:
                        return time.perf_counter() - started
            except (aiohttp.ClientError, TimeoutError):
                pass
            await asyncio.sleep(1)
    raise TimeoutError(f"server not healthy after {timeout_s:.0f} s")


def fetch_text(url: str, timeout_s: float = 10) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return {"status": response.status, "text": response.read().decode()}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"status": None, "text": "", "error": str(exc)}


async def run_workload(
    url: str, model: str, pool: TokenPromptPool, workload: SmokeWorkload
) -> tuple[dict[str, Any], list[RequestRecord]]:
    return await run_load(
        url=url,
        concurrency=workload.concurrency,
        timeout_s=workload.timeout_s,
        drain_s=0,
        payload_factory=lambda index: completions_payload(
            prompt=pool.take(index),
            request_index=index,
            output_tokens=workload.output_tokens,
            seed=workload.seed,
            model=model,
            ignore_eos=True,
        ),
        mode="closed",
        request_count=workload.measured_requests,
        warmup_requests=workload.warmup_requests,
        seed=workload.seed,
    )


def validate_run(
    summary: dict[str, Any], records: list[RequestRecord], workload: SmokeWorkload
) -> list[str]:
    """Return every failed smoke invariant; an empty list means the run passed."""
    failures = []
    expected = {
        "errored_or_cancelled_requests": 0,
        "rejected_requests": 0,
        "late_completed_requests": 0,
        "on_time_completed_requests": workload.measured_requests,
        "warmup_requests_completed": workload.warmup_requests,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            failures.append(f"{key} = {summary.get(key)!r}, expected {value!r}")
    if len(records) != workload.measured_requests:
        failures.append(
            f"{len(records)} records, expected {workload.measured_requests}"
        )
    for row in records:
        problems = []
        if row.status != "ok":
            problems.append(f"status {row.status} ({row.error})")
        if row.prompt_tokens != workload.input_tokens:
            problems.append(f"prompt_tokens {row.prompt_tokens}")
        if row.completion_tokens != workload.output_tokens:
            problems.append(f"completion_tokens {row.completion_tokens}")
        if row.usage_events != 1:
            problems.append(f"usage_events {row.usage_events}")
        if row.text_chunks < 1:
            problems.append("no text chunks")
        if problems:
            failures.append(f"{row.request_id}: " + "; ".join(problems))
    return failures


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", "utf-8")


async def run_server_smoke(
    *,
    label: str,
    command: Sequence[str],
    base_url: str,
    model_name: str,
    pool: TokenPromptPool,
    workload: SmokeWorkload,
    out_dir: Path,
    health_timeout_s: float,
    env: dict[str, str] | None = None,
    collect: Callable[[str], tuple[dict[str, Any], list[str]]] | None = None,
    log_patterns: Sequence[str] = (),
    checkpoint: Callable[[], None] = lambda: None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one server lifetime; always saves `smoke_summary.json` in `out_dir`.

    `collect` runs against the live server after the load (for example vLLM's
    `/metrics` or the HF `/stats` endpoint) and returns extra evidence plus
    any additional failures. `checkpoint` persists partial results (a Modal
    Volume commit) after each step.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"
    summary: dict[str, Any] = {
        "label": label,
        "command": list(command),
        "base_url": base_url,
        "model_name": model_name,
        "workload": asdict(workload),
        "runtime": runtime_metadata(),
        **(metadata or {}),
    }
    failures: list[str] = []
    write_json(out_dir / "smoke_summary.json", summary | {"state": "starting"})
    started = time.perf_counter()
    process = start_server(command, log_path, env)
    try:
        summary["health_wait_s"] = await wait_for_health(
            base_url, process, health_timeout_s
        )
        write_json(out_dir / "smoke_summary.json", summary | {"state": "healthy"})
        await asyncio.to_thread(checkpoint)
        load, records = await run_workload(
            f"{base_url}/v1/completions", model_name, pool, workload
        )
        load["prompt_pool"] = pool.describe()
        write_results(out_dir / "load_summary.json", load, records)
        summary["load"] = {
            key: load[key]
            for key in (
                "on_time_completed_requests",
                "errored_or_cancelled_requests",
                "rejected_requests",
                "late_completed_requests",
                "warmup_requests_completed",
                "window_duration_s",
                "request_throughput_per_s",
                "output_token_throughput_per_s",
                "ttft_s",
                "tpot_s",
                "e2e_s",
                "window_process_cpu_cores_average",
            )
        }
        summary["prompt_pool"] = pool.describe()
        failures.extend(validate_run(load, records, workload))
        if pool.describe()["issued"] != workload.total_requests:
            failures.append("prompt pool issue count differs from total requests")
        if collect is not None:
            evidence, extra_failures = collect(base_url)
            summary["collected"] = evidence
            failures.extend(extra_failures)
        await asyncio.to_thread(checkpoint)
    except Exception as exc:  # recorded, then re-raised by the caller's check
        failures.append(f"{type(exc).__name__}: {exc}")
        summary["server_log_tail"] = log_tail(log_path)
    finally:
        summary["server_returncode_after_stop"] = stop_server(process)
        summary["server_lifetime_s"] = time.perf_counter() - started
        summary["log_excerpts"] = matching_lines(log_path, log_patterns)
        summary["failures"] = failures
        summary["passed"] = not failures
        summary["state"] = "finished"
        write_json(out_dir / "smoke_summary.json", summary)
        await asyncio.to_thread(checkpoint)
    return summary


def hf_stats_collector(
    mode: str, batch_size: int, total_requests: int
) -> Callable[[str], tuple[dict[str, Any], list[str]]]:
    """Check the HF server's real generate() batch sizes for the chosen mode."""

    def collect(base_url: str) -> tuple[dict[str, Any], list[str]]:
        response = fetch_text(f"{base_url}/stats")
        failures = []
        try:
            stats: dict[str, Any] = json.loads(response["text"])
        except json.JSONDecodeError:
            return response, ["HF /stats did not return JSON"]
        sizes = [int(size) for size in stats.get("generated_batch_sizes", [])]
        if sum(sizes) != total_requests:
            failures.append(f"batch sizes sum to {sum(sizes)}, not {total_requests}")
        if mode == "naive" and any(size != 1 for size in sizes):
            failures.append(f"naive mode generated batches {sizes}")
        if mode == "static":
            if any(size > batch_size for size in sizes):
                failures.append(f"static batch exceeded {batch_size}: {sizes}")
            if not sizes or max(sizes) < 2:
                failures.append(f"static mode never batched: {sizes}")
        return stats, failures

    return collect


def vllm_metrics_collector(
    out_dir: Path,
) -> Callable[[str], tuple[dict[str, Any], list[str]]]:
    """Save vLLM's Prometheus text and extract prefix-cache counters."""

    def collect(base_url: str) -> tuple[dict[str, Any], list[str]]:
        response = fetch_text(f"{base_url}/metrics")
        (out_dir / "metrics.prom").write_text(response["text"], encoding="utf-8")
        wanted = ("prefix_cache", "prompt_tokens_total", "generation_tokens_total")
        lines = [
            line
            for line in response["text"].splitlines()
            if not line.startswith("#") and any(key in line for key in wanted)
        ]
        failures = [] if response["status"] == 200 else ["vLLM /metrics failed"]
        return {"metrics_status": response["status"], "metric_lines": lines}, failures

    return collect
