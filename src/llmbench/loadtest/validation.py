"""Run and save the Phase 1 near-zero-latency capacity validation."""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import platform
import socket
import statistics
import time
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from llmbench.loadtest.client import stream_request
from llmbench.loadtest.io import write_results
from llmbench.loadtest.metrics import RequestRecord
from llmbench.loadtest.runner import run_load
from llmbench.loadtest.workloads import completions_payload
from llmbench.mock.server import MockConfig, make_app, serve

CAPACITY_THRESHOLD_CHUNKS_PER_S = 6_000
CAPACITY_WINDOW_S = 30.0
CAPACITY_CONCURRENCY = 256
CLIENT_CPU_CORE_LIMIT = 2.0
EXPECTED_TTFT_S = 0.2
EXPECTED_ITL_S = 0.02
TIMING_RELATIVE_TOLERANCE = 0.05


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("llmbench", "aiohttp", "modal"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


async def measure_timing_accuracy() -> tuple[dict[str, Any], list[RequestRecord]]:
    """Compare client timing with actual monotonic mock write timestamps."""
    port = _free_port()
    sent: dict[str, dict[str, Any]] = {}

    def capture(request_id: str, kind: str, timestamp: float) -> None:
        entry = sent.setdefault(request_id, {"text": []})
        if kind == "start":
            entry["start"] = timestamp
        else:
            entry["text"].append(timestamp)

    config = MockConfig(
        ttft_s=EXPECTED_TTFT_S,
        itl_s=EXPECTED_ITL_S,
        output_tokens=4,
        timestamp_sink=capture,
    )
    server = web.AppRunner(make_app(config))
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", port)
    await site.start()
    base_url = f"http://127.0.0.1:{port}"
    records = []
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=2)
        ) as session:
            for index in range(5):
                records.append(
                    await stream_request(
                        session,
                        f"{base_url}/v1/completions",
                        completions_payload(
                            prompt="timing validation",
                            request_index=index,
                            output_tokens=4,
                            seed=0,
                        ),
                        request_id=f"timing-{index}",
                    )
                )
    finally:
        await server.cleanup()
    ttfts = [row.ttft_s for row in records if row.ttft_s is not None]
    itls = [gap for row in records for gap in row.itl_s]
    server_ttfts = [
        float(sent[row.request_id]["text"][0]) - float(sent[row.request_id]["start"])
        for row in records
        if row.request_id in sent
        and sent[row.request_id].get("start") is not None
        and len(sent[row.request_id]["text"]) == 4
    ]
    server_itls = [
        right - left
        for row in records
        if row.request_id in sent and len(sent[row.request_id]["text"]) == 4
        for left, right in zip(
            sent[row.request_id]["text"],
            sent[row.request_id]["text"][1:],
            strict=False,
        )
    ]
    median_ttft = statistics.median(ttfts) if ttfts else None
    median_itl = statistics.median(itls) if itls else None
    median_server_ttft = statistics.median(server_ttfts) if server_ttfts else None
    median_server_itl = statistics.median(server_itls) if server_itls else None
    timing_ok = (
        len(records) == 5
        and all(row.status == "ok" and row.usage_events == 1 for row in records)
        and len(ttfts) == 5
        and len(itls) == 15
        and len(server_ttfts) == 5
        and len(server_itls) == 15
        and median_ttft is not None
        and median_itl is not None
        and median_server_ttft is not None
        and median_server_itl is not None
        and abs(median_ttft - median_server_ttft)
        <= median_server_ttft * TIMING_RELATIVE_TOLERANCE
        and abs(median_itl - median_server_itl)
        <= median_server_itl * TIMING_RELATIVE_TOLERANCE
    )
    summary = {
        "validation": "phase1_timing_accuracy",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": {
            "mock_ttft_s": EXPECTED_TTFT_S,
            "mock_itl_s": EXPECTED_ITL_S,
            "output_tokens": 4,
            "samples": 5,
            "empty_text_before_first_token": True,
            "final_usage_only_event": True,
            "reference": (
                "server monotonic timestamps after each text response.write completes"
            ),
        },
        "versions": _package_versions(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
        "gpu_name": None,
        "expected_ttft_s": EXPECTED_TTFT_S,
        "expected_itl_s": EXPECTED_ITL_S,
        "median_server_ttft_s": median_server_ttft,
        "median_server_itl_s": median_server_itl,
        "relative_tolerance": TIMING_RELATIVE_TOLERANCE,
        "median_ttft_s": median_ttft,
        "median_itl_s": median_itl,
        "ttft_relative_error": (
            abs(median_ttft - median_server_ttft) / median_server_ttft
            if median_ttft is not None and median_server_ttft is not None
            else None
        ),
        "itl_relative_error": (
            abs(median_itl - median_server_itl) / median_server_itl
            if median_itl is not None and median_server_itl is not None
            else None
        ),
        "empty_text_events_per_request": 1,
        "usage_events_per_request": 1,
        "passed": bool(timing_ok),
    }
    summary["config_sha256"] = sha256(
        json.dumps(summary["config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return summary, records


def _mock_server_process(port: int, config: MockConfig) -> None:
    asyncio.run(serve("127.0.0.1", port, config))


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


async def _wait_until_ready(process: BaseProcess, url: str) -> None:
    deadline = time.perf_counter() + 10
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=0.25)
    ) as session:
        while time.perf_counter() < deadline:
            if not process.is_alive():
                raise RuntimeError("capacity mock server exited before readiness")
            try:
                async with session.get(f"{url}/health") as response:
                    if response.status == 200:
                        return
            except aiohttp.ClientError:
                await asyncio.sleep(0.05)
    raise TimeoutError("capacity mock server did not become ready in 10 seconds")


async def _measure(output_dir: Path) -> dict[str, Any]:
    timing_summary, timing_records = await measure_timing_accuracy()
    write_results(
        output_dir / "timing_accuracy_summary.json",
        timing_summary,
        timing_records,
    )
    print(
        "timing: "
        f"TTFT {timing_summary['median_ttft_s']:.4f}s "
        f"(expected {EXPECTED_TTFT_S:.3f}s), "
        f"ITL {timing_summary['median_itl_s']:.4f}s "
        f"(expected {EXPECTED_ITL_S:.3f}s)"
    )
    if not timing_summary["passed"]:
        raise RuntimeError("timing accuracy validation failed")

    port = _free_port()
    process = multiprocessing.get_context("spawn").Process(
        target=_mock_server_process,
        args=(
            port,
            MockConfig(
                ttft_s=0,
                itl_s=0,
                output_tokens=3,
                send_empty_chunk=False,
            ),
        ),
        daemon=True,
    )
    process.start()
    base_url = f"http://127.0.0.1:{port}"
    url = f"{base_url}/v1/completions"
    try:
        await _wait_until_ready(process, base_url)
        summary, records = await run_load(
            url=url,
            concurrency=CAPACITY_CONCURRENCY,
            timeout_s=5,
            drain_s=0.5,
            payload_factory=lambda index: completions_payload(
                prompt="capacity validation",
                request_index=index,
                output_tokens=3,
                seed=0,
            ),
            mode="closed",
            duration_s=CAPACITY_WINDOW_S,
            warmup_requests=0,
            seed=0,
        )
        cpu_seconds = float(summary["window_process_cpu_seconds"])
        duration = float(summary["window_duration_s"])
        event_count = int(summary["measurement_text_chunks"])
        rate = event_count / duration
        cpu_cores = float(summary["window_process_cpu_cores_average"])
        summary.update(
            {
                "validation": "phase1_client_capacity",
                "capacity_threshold_chunks_per_s": CAPACITY_THRESHOLD_CHUNKS_PER_S,
                "capacity_window_s": CAPACITY_WINDOW_S,
                "capacity_concurrency": CAPACITY_CONCURRENCY,
                "capacity_headroom_multiple": rate / 2_000,
                "capacity_threshold_multiple": (rate / CAPACITY_THRESHOLD_CHUNKS_PER_S),
                "capacity_margin_chunks_per_s": (
                    rate - CAPACITY_THRESHOLD_CHUNKS_PER_S
                ),
                "measured_text_chunks_per_s": rate,
                "measurement_text_chunks": event_count,
                "client_process_cpu_seconds": cpu_seconds,
                "client_process_cpu_cores_average": cpu_cores,
                "client_cpu_core_limit": CLIENT_CPU_CORE_LIMIT,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "gpu_name": None,
                "versions": _package_versions(),
                "host": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "logical_cpu_count": os.cpu_count(),
                    "mock_server_separate_process": True,
                },
                "config": {
                    **summary["config"],
                    "mock_server": {
                        "ttft_s": 0,
                        "itl_s": 0,
                        "output_tokens": 3,
                    },
                },
            }
        )
        summary["config_sha256"] = sha256(
            json.dumps(
                summary["config"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        write_results(
            output_dir / "capacity_summary.json",
            summary,
            records,
            raw_path=output_dir / "capacity_requests.jsonl.gz",
        )
        print(
            f"capacity: {rate:.1f} chunks/s "
            f"({rate / CAPACITY_THRESHOLD_CHUNKS_PER_S:.2f}x threshold), "
            f"CPU: {cpu_cores:.2f} cores"
        )
        if rate < CAPACITY_THRESHOLD_CHUNKS_PER_S:
            raise RuntimeError(
                f"capacity {rate:.1f}/s is below the "
                f"{CAPACITY_THRESHOLD_CHUNKS_PER_S}/s threshold"
            )
        if cpu_cores > CLIENT_CPU_CORE_LIMIT:
            raise RuntimeError(
                f"client CPU {cpu_cores:.2f} cores exceeds "
                f"{CLIENT_CPU_CORE_LIMIT:.1f} core limit"
            )
        return summary
    finally:
        process.terminate()
        process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join(timeout=3)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m llmbench.loadtest.validation")
    parser.add_argument("--output-dir", type=Path, default=Path("results/validation"))
    args = parser.parse_args()
    asyncio.run(_measure(args.output_dir))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
