"""Integration coverage for the open-loop and bounded-window runner paths."""

from __future__ import annotations

import asyncio

from aiohttp.test_utils import TestServer

from llmbench.loadtest.runner import run_load
from llmbench.loadtest.workloads import completions_payload
from llmbench.mock.server import MockConfig, make_app


def test_open_loop_poisson_mode_completes_requests() -> None:
    async def run() -> tuple[dict, list]:
        server = TestServer(
            make_app(MockConfig(ttft_s=0.002, itl_s=0.001, output_tokens=2))
        )
        await server.start_server()
        try:
            return await run_load(
                url=str(server.make_url("/v1/completions")),
                concurrency=4,
                timeout_s=1,
                drain_s=0.2,
                payload_factory=lambda index: completions_payload(
                    prompt="open loop",
                    request_index=index,
                    output_tokens=2,
                    seed=7,
                ),
                mode="open",
                duration_s=0.15,
                rate_per_s=60,
                warmup_requests=3,
                seed=7,
            )
        finally:
            await server.close()

    summary, rows = asyncio.run(run())
    assert summary["config"]["mode"] == "open"
    assert summary["offered_requests"] > 0
    assert summary["warmup_requests_completed"] == 3
    assert summary["on_time_completed_requests"] > 0
    assert summary["rejected_requests"] == 0
    assert all(row.status in {"ok", "late_completion"} for row in rows)


def test_deadline_cancels_work_after_bounded_drain() -> None:
    async def run() -> tuple[dict, list]:
        server = TestServer(
            make_app(
                MockConfig(
                    ttft_s=0.2,
                    itl_s=0,
                    output_tokens=2,
                )
            )
        )
        await server.start_server()
        try:
            return await run_load(
                url=str(server.make_url("/v1/completions")),
                concurrency=2,
                timeout_s=1,
                drain_s=0.01,
                payload_factory=lambda index: completions_payload(
                    prompt="bounded drain",
                    request_index=index,
                    output_tokens=2,
                    seed=3,
                ),
                mode="closed",
                duration_s=0.03,
                seed=3,
            )
        finally:
            await server.close()

    summary, rows = asyncio.run(run())
    assert summary["on_time_completed_requests"] == 0
    assert summary["in_flight_at_deadline"] == 2
    assert summary["errored_or_cancelled_requests"] == 2
    assert summary["request_throughput_per_s"] == 0
    assert all(row.status == "drain_timeout" for row in rows)
