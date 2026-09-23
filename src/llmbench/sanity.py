"""Phase 4 sanity: a checkpoint loads in vLLM and completes fixed prompts.

Records vLLM's own "Model loading took" and KV-cache lines and the on-disk
checkpoint size (the two weight-memory metrics in SPEC §5). Outputs are saved
verbatim for a human coherence review; this is not an accuracy measurement.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from llmbench.smoke import (
    VLLM_LOG_PATTERNS,
    log_tail,
    matching_lines,
    runtime_metadata,
    start_server,
    stop_server,
    wait_for_health,
    write_json,
)


def checkpoint_bytes(model_path: str) -> dict[str, int]:
    files = [p for p in Path(model_path).rglob("*") if p.is_file()]
    files = [p for p in files if ".cache" not in p.relative_to(model_path).parts]
    return {
        "total_bytes": sum(p.stat().st_size for p in files),
        "safetensors_bytes": sum(
            p.stat().st_size for p in files if p.suffix == ".safetensors"
        ),
    }


def complete(base_url: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "n": 1,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        payload: dict[str, Any] = json.loads(response.read())
    choice = payload["choices"][0]
    return {
        "prompt": prompt,
        "text": choice["text"],
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage"),
        "request": body,
        "elapsed_s": time.perf_counter() - started,
    }


def check_outputs(outputs: list[dict[str, Any]], expected: int) -> list[str]:
    failures = []
    if len(outputs) != expected:
        failures.append(f"{len(outputs)} outputs, expected {expected}")
    for index, output in enumerate(outputs):
        usage = output.get("usage") or {}
        if not output["text"].strip():
            failures.append(f"prompt {index}: empty completion")
        if not usage.get("completion_tokens"):
            failures.append(f"prompt {index}: no completion tokens in usage")
    return failures


def run_sanity(
    *,
    label: str,
    model_path: str,
    served_name: str,
    sanity: dict[str, Any],
    out_dir: Path,
    metadata: dict[str, Any] | None = None,
    port: int = 8000,
) -> dict[str, Any]:
    """One server lifetime; always saves `sanity.json` and `server.log`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"
    base_url = f"http://127.0.0.1:{port}"
    command = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *sanity["engine_args"],
    ]
    summary: dict[str, Any] = {
        "label": label,
        "model_path": model_path,
        "command": command,
        "runtime": runtime_metadata(),
        "checkpoint_bytes": checkpoint_bytes(model_path),
        **(metadata or {}),
    }
    failures: list[str] = []
    outputs: list[dict[str, Any]] = []
    started = time.perf_counter()
    process = start_server(command, log_path)
    try:
        summary["health_wait_s"] = asyncio.run(
            wait_for_health(
                base_url, process, float(sanity.get("health_timeout_s", 900))
            )
        )
        for prompt in sanity["prompts"]:
            outputs.append(
                complete(base_url, served_name, prompt, int(sanity["max_tokens"]))
            )
        failures += check_outputs(outputs, len(sanity["prompts"]))
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        summary["server_log_tail"] = log_tail(log_path)
    finally:
        summary["server_returncode_after_stop"] = stop_server(process)
        summary["server_lifetime_s"] = time.perf_counter() - started
        summary["log_excerpts"] = matching_lines(log_path, VLLM_LOG_PATTERNS)
        summary["outputs"] = outputs
        summary["failures"] = failures
        summary["passed"] = not failures
        write_json(out_dir / "sanity.json", summary)
    return summary
