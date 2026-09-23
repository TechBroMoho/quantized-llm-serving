"""Command-line entry point for local mock and load-testing tools."""

import argparse
import asyncio
import json
import os
import platform
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from llmbench.loadtest.io import write_results
from llmbench.loadtest.runner import run_load
from llmbench.loadtest.workloads import completions_payload
from llmbench.mock.server import MockConfig, serve


def _versions() -> dict[str, str]:
    packages = ("llmbench", "aiohttp", "modal", "torch", "transformers", "fastapi")
    found: dict[str, str] = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            continue
    return found


def _mock_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ttft-ms", type=float, default=200)
    parser.add_argument("--itl-ms", type=float, default=20)
    parser.add_argument("--tokens", type=int, default=3)
    parser.add_argument("--no-empty-chunk", action="store_true")
    parser.add_argument("--no-usage", action="store_true")
    parser.add_argument("--hold-open-ms", type=float, default=0)


def _load_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--mode", choices=("closed", "open"), default="closed")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--drain-time", type=float, default=2)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="A reproducible benchmark prompt")
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--workload", choices=("synthetic",), default="synthetic")
    parser.add_argument("--out", type=Path, required=True)


async def _run_load(args: argparse.Namespace) -> dict[str, Any]:
    if args.duration is None and args.requests is None:
        args.duration = 10.0
    if args.duration is not None and args.requests is not None:
        raise ValueError("choose --duration or --requests, not both")
    if args.mode == "open" and args.rate is None:
        raise ValueError("open-loop mode requires --rate")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests cannot be negative")

    config: dict[str, Any] = {
        "url": args.url,
        "model": args.model,
        "workload": args.workload,
        "mode": args.mode,
        "concurrency": args.concurrency,
        "duration_s": args.duration,
        "request_count": args.requests,
        "rate_per_s": args.rate,
        "warmup_requests": args.warmup_requests,
        "timeout_s": args.timeout,
        "drain_s": args.drain_time,
        "output_tokens": args.output_tokens,
        "seed": args.seed,
        "generation": {
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": True,
            "include_usage": True,
        },
    }
    config_hash = sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary, records = await run_load(
        url=args.url,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
        drain_s=args.drain_time,
        payload_factory=lambda index: completions_payload(
            prompt=args.prompt,
            request_index=index,
            output_tokens=args.output_tokens,
            seed=args.seed,
            model=args.model,
        ),
        mode=args.mode,
        duration_s=args.duration,
        request_count=args.requests,
        rate_per_s=args.rate,
        warmup_requests=args.warmup_requests,
        seed=args.seed,
    )
    measured_cpu = float(summary["window_process_cpu_seconds"])
    duration = summary["window_duration_s"]
    summary.update(
        {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "config": config,
            "config_sha256": config_hash,
            "versions": _versions(),
            "gpu_name": None,
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "logical_cpu_count": os.cpu_count(),
                "process_cpu_seconds": measured_cpu,
                "process_cpu_cores_average": (
                    measured_cpu / duration if duration else 0
                ),
            },
        }
    )
    write_results(args.out, summary, records)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmbench")
    commands = parser.add_subparsers(dest="command", required=True)
    load_parser = commands.add_parser("load", help="run an async streaming load test")
    _load_arguments(load_parser)
    mock_parser = commands.add_parser("mock", help="serve a controllable mock")
    mock_commands = mock_parser.add_subparsers(dest="mock_command", required=True)
    mock_server_parser = mock_commands.add_parser(
        "serve", help="start the mock SSE API"
    )
    _mock_arguments(mock_server_parser)
    args = parser.parse_args()
    if args.command == "mock":
        asyncio.run(
            serve(
                args.host,
                args.port,
                MockConfig(
                    ttft_s=args.ttft_ms / 1000,
                    itl_s=args.itl_ms / 1000,
                    output_tokens=args.tokens,
                    send_empty_chunk=not args.no_empty_chunk,
                    include_usage=not args.no_usage,
                    hold_open_s=args.hold_open_ms / 1000,
                ),
            )
        )
        return
    summary = asyncio.run(_run_load(args))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["errored_or_cancelled_requests"] or summary["rejected_requests"]:
        raise SystemExit(1)
