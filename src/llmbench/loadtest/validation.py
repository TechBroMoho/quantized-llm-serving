"""Run and save the load tester's timing and capacity validation (ADR-010/013/019)."""

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
from dataclasses import dataclass
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
from llmbench.loadtest.metrics import RequestRecord, percentile
from llmbench.loadtest.runner import run_load
from llmbench.loadtest.workloads import TextPayloads, completions_payload
from llmbench.mock.server import MockConfig, make_app, serve

CAPACITY_THRESHOLD_CHUNKS_PER_S = 6_000
CAPACITY_WINDOW_S = 30.0
CAPACITY_CONCURRENCY = 256
CLIENT_CPU_CORE_LIMIT = 2.0
# Two single-threaded client processes fit the two-core budget (ADR-013).
CAPACITY_CLIENT_PROCESSES = 2
EXPECTED_TTFT_S = 0.2
EXPECTED_ITL_S = 0.02
TIMING_RELATIVE_TOLERANCE = 0.05
# ADR-019. The full gate adds p99 only where p99 is not just the maximum.
MIN_P99_SAMPLES = 100
# Unit-gate tail bound on every paired observation. The largest host
# scheduling delay measured on the laptop was 33 ms (a TTFT with 6 of 10
# cores spinning); a corrupted request in the regression test is 100 ms off.
UNIT_MAX_ABS_ERROR_S = 0.050


@dataclass(frozen=True)
class TimingPlan:
    """Sample size and gate for one timing validation run (ADR-019)."""

    requests: int
    output_tokens: int
    concurrency: int
    full: bool  # True: p50 + p99 gate; False: p50 + absolute tail bound


# `make check`: 10 sequential requests x 21 tokens (10 TTFTs, 200 gaps).
TIMING_UNIT = TimingPlan(requests=10, output_tokens=21, concurrency=1, full=False)
# Evidence runs (mock-validate, in-Modal): 200 TTFT/E2E/TPOT and 4,000 ITL
# samples, one stream per client process. The mock shares this process's
# event loop, so concurrent streams delay each other's reads: 8 streams gave
# an ITL p99 of 6.21% with p50 0.16% (ADR-019).
TIMING_FULL = TimingPlan(requests=200, output_tokens=21, concurrency=1, full=True)


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("llmbench", "aiohttp", "modal"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def evaluate_timing(
    records: list[RequestRecord],
    traces: dict[str, dict[str, Any]],
    *,
    plan: TimingPlan = TIMING_UNIT,
    tolerance: float = TIMING_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Gate client timings against the mock's own monotonic write timestamps.

    Every request must be structurally complete, and for TTFT, E2E, TPOT and
    ITL the client's p50 must be within `tolerance` of the paired server
    references' p50. The full gate also requires p99 within `tolerance`, and
    at least MIN_P99_SAMPLES per metric. The unit gate instead bounds every
    paired observation by UNIT_MAX_ABS_ERROR_S: with 10 requests a p99 is
    just the maximum (ADR-019). TTFT and E2E are referenced to the client's
    request start so connection setup is included.
    """
    requests, output_tokens = plan.requests, plan.output_tokens
    client: dict[str, list[float]] = {
        "ttft_s": [],
        "e2e_s": [],
        "tpot_s": [],
        "itl_s": [],
    }
    reference: dict[str, list[float]] = {name: [] for name in client}
    observations = []
    structural_ok = len(records) == requests
    for row in records:
        trace = traces.get(row.request_id, {})
        writes: list[float] = trace.get("text", [])
        errors: dict[str, float] = {}
        complete = (
            row.status == "ok"
            and row.text_chunks == output_tokens
            and row.empty_text_chunks == 1
            and row.usage_events == 1
            and row.completion_tokens == output_tokens
            and trace.get("start") is not None
            and len(writes) == output_tokens
            and len(row.itl_s) == output_tokens - 1
            and row.ttft_s is not None
            and row.e2e_s is not None
            and row.tpot_s is not None
        )
        if complete:
            assert row.ttft_s is not None and row.e2e_s is not None
            assert row.tpot_s is not None
            paired = {
                "ttft_s": (row.ttft_s, writes[0] - row.started_at),
                "e2e_s": (row.e2e_s, writes[-1] - row.started_at),
                "tpot_s": (row.tpot_s, (writes[-1] - writes[0]) / (output_tokens - 1)),
            }
            gaps = [b - a for a, b in zip(writes, writes[1:], strict=False)]
            for name, (actual, expected) in paired.items():
                client[name].append(actual)
                reference[name].append(expected)
                errors[name] = abs(actual - expected) / expected
            for i, (actual, expected) in enumerate(zip(row.itl_s, gaps, strict=True)):
                client["itl_s"].append(actual)
                reference["itl_s"].append(expected)
                errors[f"itl_{i}"] = abs(actual - expected) / expected
        structural_ok = structural_ok and complete
        observations.append(
            {
                "request_id": row.request_id,
                "complete": complete,
                "client_start_monotonic_s": row.started_at,
                "server_start_monotonic_s": trace.get("start"),
                "server_text_write_monotonic_s": writes,
                "relative_errors": errors,
                "over_tolerance": sorted(k for k, v in errors.items() if v > tolerance),
            }
        )

    gate: dict[str, Any] = {}
    for name in client:
        entry: dict[str, Any] = {"samples": len(client[name])}
        passed = bool(client[name])
        if plan.full and len(client[name]) < MIN_P99_SAMPLES:
            passed = False
            entry["error"] = f"p99 needs at least {MIN_P99_SAMPLES} samples"
        for q in (50, 99) if plan.full else (50,):
            client_q = percentile(client[name], q)
            reference_q = percentile(reference[name], q)
            error = (
                abs(client_q - reference_q) / reference_q
                if client_q is not None and reference_q
                else float("inf")
            )
            entry[f"client_p{q}_s"] = client_q
            entry[f"reference_p{q}_s"] = reference_q
            entry[f"p{q}_relative_error"] = error
            passed = passed and error <= tolerance
        pairs = list(zip(client[name], reference[name], strict=True))
        pair_errors = [abs(a - b) / b for a, b in pairs]
        max_abs = max((abs(a - b) for a, b in pairs), default=None)
        entry["max_paired_relative_error"] = max(pair_errors, default=None)
        entry["max_paired_abs_error_s"] = max_abs
        entry["paired_over_tolerance"] = sum(e > tolerance for e in pair_errors)
        if not plan.full:
            entry["max_abs_error_bound_s"] = UNIT_MAX_ABS_ERROR_S
            passed = passed and max_abs is not None and max_abs <= UNIT_MAX_ABS_ERROR_S
        entry["passed"] = passed
        gate[name] = entry

    handler_ttfts = [
        float(traces[row.request_id]["text"][0])
        - float(traces[row.request_id]["start"])
        for row in records
        if traces.get(row.request_id, {}).get("start") is not None
        and traces[row.request_id].get("text")
    ]
    return {
        "structural_ok": structural_ok,
        "gate": gate,
        "paired_observations": observations,
        "median_ttft_s": gate["ttft_s"]["client_p50_s"],
        "median_itl_s": gate["itl_s"]["client_p50_s"],
        # Handler-relative, descriptive only (omits connection setup).
        "median_server_ttft_s": (
            statistics.median(handler_ttfts) if handler_ttfts else None
        ),
        "median_server_itl_s": gate["itl_s"]["reference_p50_s"],
        "ttft_relative_error": gate["ttft_s"]["p50_relative_error"],
        "itl_relative_error": gate["itl_s"]["p50_relative_error"],
        "passed": structural_ok and all(entry["passed"] for entry in gate.values()),
    }


async def measure_timing_accuracy(
    processes: int = 1, plan: TimingPlan = TIMING_UNIT
) -> tuple[dict[str, Any], list[RequestRecord]]:
    """Compare client timing with actual monotonic mock write timestamps.

    With one process and one stream, requests go one at a time from this
    process (the ADR-010 check). Otherwise they go through the load runner;
    with `processes=2` a worker's timestamps are checked against this
    process's server clock, the shared-clock assumption behind ADR-013.
    """
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
        output_tokens=plan.output_tokens,
        timestamp_sink=capture,
    )
    server = web.AppRunner(make_app(config))
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", port)
    await site.start()
    url = f"http://127.0.0.1:{port}/v1/completions"
    records: list[RequestRecord] = []
    try:
        if processes == 1 and plan.concurrency == 1:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                for index in range(plan.requests):
                    records.append(
                        await stream_request(
                            session,
                            url,
                            completions_payload(
                                prompt="timing validation",
                                request_index=index,
                                output_tokens=plan.output_tokens,
                                seed=0,
                            ),
                            request_id=f"timing-{index}",
                        )
                    )
        else:
            _, records = await run_load(
                url=url,
                concurrency=max(processes, plan.concurrency),
                processes=processes,
                timeout_s=5,
                drain_s=0,
                payload_factory=TextPayloads(
                    prompt="timing validation",
                    output_tokens=plan.output_tokens,
                    seed=0,
                ),
                mode="closed",
                request_count=plan.requests,
            )
    finally:
        await server.cleanup()
    result = evaluate_timing(records, sent, plan=plan)
    summary = {
        "validation": "phase1_timing_accuracy",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": {
            "mock_ttft_s": EXPECTED_TTFT_S,
            "mock_itl_s": EXPECTED_ITL_S,
            "output_tokens": plan.output_tokens,
            "samples": plan.requests,
            "concurrency": max(processes, plan.concurrency),
            "client_processes": processes,
            "empty_text_before_first_token": True,
            "final_usage_only_event": True,
            "reference": (
                "server monotonic timestamps after each text response.write completes"
            ),
            "gate": (
                "structural completeness; client p50 and p99 of TTFT, E2E, TPOT "
                "and ITL each within tolerance of the paired server reference"
                if plan.full
                else "structural completeness; client p50 of TTFT, E2E, TPOT and "
                "ITL within tolerance; every paired observation within "
                f"{UNIT_MAX_ABS_ERROR_S * 1000:g} ms"
            ),
            "full_gate": plan.full,
            "unit_max_abs_error_s": None if plan.full else UNIT_MAX_ABS_ERROR_S,
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
        "relative_tolerance": TIMING_RELATIVE_TOLERANCE,
        "empty_text_events_per_request": 1,
        "usage_events_per_request": 1,
        **result,
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


async def _mock_cpu_seconds(session: aiohttp.ClientSession, base_url: str) -> float:
    async with session.get(f"{base_url}/stats") as response:
        return float((await response.json())["process_cpu_seconds"])


async def _measure(
    output_dir: Path, processes: int = CAPACITY_CLIENT_PROCESSES
) -> dict[str, Any]:
    for timing_processes, name in ((1, "timing_accuracy"), (2, "timing_multiprocess")):
        timing_summary, timing_records = await measure_timing_accuracy(
            timing_processes, TIMING_FULL
        )
        write_results(
            output_dir / f"{name}_summary.json", timing_summary, timing_records
        )
        gate = timing_summary["gate"]
        print(
            f"{name} ({timing_processes} client process(es)): "
            + ", ".join(
                f"{metric} (n={entry['samples']}) "
                f"p50 {entry['p50_relative_error']:.2%} "
                f"p99 {entry['p99_relative_error']:.2%}"
                for metric, entry in gate.items()
            )
        )
        if not timing_summary["passed"]:
            raise RuntimeError(f"{name} validation failed")

    # One mock process per client process, so the mock is not the bottleneck.
    context = multiprocessing.get_context("spawn")
    mock_config = MockConfig(ttft_s=0, itl_s=0, output_tokens=3, send_empty_chunk=False)
    ports = [_free_port() for _ in range(processes)]
    mocks = [
        context.Process(
            target=_mock_server_process, args=(port, mock_config), daemon=True
        )
        for port in ports
    ]
    for mock in mocks:
        mock.start()
    base_urls = [f"http://127.0.0.1:{port}" for port in ports]
    try:
        for mock, base_url in zip(mocks, base_urls, strict=True):
            await _wait_until_ready(mock, base_url)
        async with aiohttp.ClientSession() as stats_session:
            mock_cpu_before = [
                await _mock_cpu_seconds(stats_session, url) for url in base_urls
            ]
            mock_wall_before = time.perf_counter()
            summary, records = await run_load(
                url=[f"{url}/v1/completions" for url in base_urls],
                processes=processes,
                concurrency=CAPACITY_CONCURRENCY,
                timeout_s=5,
                drain_s=0.5,
                payload_factory=TextPayloads(
                    prompt="capacity validation", output_tokens=3, seed=0
                ),
                mode="closed",
                duration_s=CAPACITY_WINDOW_S,
                warmup_requests=0,
                seed=0,
            )
            mock_wall = time.perf_counter() - mock_wall_before
            mock_cpu_after = [
                await _mock_cpu_seconds(stats_session, url) for url in base_urls
            ]
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
                # Summed over all client processes (ADR-013).
                "client_process_cpu_seconds": cpu_seconds,
                "client_process_cpu_cores_average": cpu_cores,
                "client_cpu_core_limit": CLIENT_CPU_CORE_LIMIT,
                # Descriptive: each mock's CPU over the whole run_load call
                # (process spawn, window and drain), not only the window.
                "mock_server_cpu_cores_average": [
                    (after - before) / mock_wall
                    for before, after in zip(
                        mock_cpu_before, mock_cpu_after, strict=True
                    )
                ],
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "gpu_name": None,
                "versions": _package_versions(),
                "host": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "logical_cpu_count": os.cpu_count(),
                    "mock_server_separate_process": True,
                    "mock_server_processes": processes,
                },
                "config": {
                    **summary["config"],
                    "mock_server": {
                        "ttft_s": 0,
                        "itl_s": 0,
                        "output_tokens": 3,
                        "processes": processes,
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
            f"client CPU: {cpu_cores:.2f} cores over {processes} processes, "
            "mock CPU: "
            + ", ".join(f"{c:.2f}" for c in summary["mock_server_cpu_cores_average"])
        )
        check_capacity(summary)
        return summary
    finally:
        for mock in mocks:
            mock.terminate()
            mock.join(timeout=3)
            if mock.is_alive():
                mock.kill()
                mock.join(timeout=3)


def check_capacity(summary: dict[str, Any]) -> None:
    """Capacity is useful only if the client also preserves valid outcomes."""
    if summary["errored_or_cancelled_requests"] or summary["rejected_requests"]:
        raise RuntimeError("capacity run has errors or rejections")
    rate = float(summary["measured_text_chunks_per_s"])
    cores = float(summary["client_process_cpu_cores_average"])
    if rate < CAPACITY_THRESHOLD_CHUNKS_PER_S:
        raise RuntimeError(f"capacity {rate:.1f}/s is below threshold")
    if cores > CLIENT_CPU_CORE_LIMIT:
        raise RuntimeError(f"client CPU {cores:.2f} cores exceeds limit")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m llmbench.loadtest.validation")
    parser.add_argument("--output-dir", type=Path, default=Path("results/validation"))
    parser.add_argument(
        "--processes",
        type=int,
        default=CAPACITY_CLIENT_PROCESSES,
        help="client processes for the capacity run (ADR-013)",
    )
    args = parser.parse_args()
    asyncio.run(_measure(args.output_dir, args.processes))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
