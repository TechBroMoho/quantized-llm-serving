"""End-to-end timing checks against the controllable SSE mock."""

from __future__ import annotations

import asyncio

import aiohttp
from aiohttp.test_utils import TestServer

from llmbench.loadtest.client import stream_request
from llmbench.loadtest.validation import measure_timing_accuracy
from llmbench.loadtest.workloads import completions_payload
from llmbench.mock.server import MockConfig, make_app


def test_mock_timing_accuracy() -> None:
    summary, rows = asyncio.run(measure_timing_accuracy())
    assert summary["passed"], (
        f"median TTFT={summary['median_ttft_s']:.6f}s "
        f"(client/server error={summary['ttft_relative_error']:.1%}); "
        f"median ITL={summary['median_itl_s']:.6f}s "
        f"(error={summary['itl_relative_error']:.1%})"
    )
    assert len(rows) == 5
    assert all(row.status == "ok" for row in rows)
    assert all(row.empty_text_chunks == 1 for row in rows)
    assert all(row.usage_events == 1 for row in rows)
    assert all(row.completion_tokens == 4 for row in rows)


def test_single_token_has_no_tpot_or_inter_chunk_latency() -> None:
    async def run() -> None:
        server = TestServer(
            make_app(MockConfig(ttft_s=0.005, itl_s=0, output_tokens=1))
        )
        await server.start_server()
        try:
            async with aiohttp.ClientSession() as session:
                row = await stream_request(
                    session,
                    str(server.make_url("/v1/completions")),
                    completions_payload(
                        prompt="one token",
                        request_index=0,
                        output_tokens=1,
                        seed=0,
                    ),
                    request_id="single-token",
                )
            assert row.status == "ok"
            assert row.ttft_s is not None
            assert row.itl_s == []
            assert row.tpot_s is None
        finally:
            await server.close()

    asyncio.run(run())


def test_missing_usage_is_invalid() -> None:
    async def run() -> None:
        server = TestServer(
            make_app(
                MockConfig(
                    ttft_s=0,
                    itl_s=0,
                    output_tokens=1,
                    send_empty_chunk=False,
                    include_usage=False,
                )
            )
        )
        await server.start_server()
        try:
            async with aiohttp.ClientSession() as session:
                row = await stream_request(
                    session,
                    str(server.make_url("/v1/completions")),
                    completions_payload(
                        prompt="no usage",
                        request_index=0,
                        output_tokens=1,
                        seed=0,
                    ),
                    request_id="missing-usage",
                )
            assert row.status == "error"
            assert row.error == "stream ended without valid usage"
        finally:
            await server.close()

    asyncio.run(run())


def test_cancelled_request_is_kept_as_a_failed_record() -> None:
    async def run() -> None:
        server = TestServer(
            make_app(
                MockConfig(
                    ttft_s=0,
                    itl_s=0,
                    output_tokens=1,
                    hold_open_s=1,
                )
            )
        )
        await server.start_server()
        try:
            async with aiohttp.ClientSession() as session:
                task = asyncio.create_task(
                    stream_request(
                        session,
                        str(server.make_url("/v1/completions")),
                        completions_payload(
                            prompt="cancel me",
                            request_index=0,
                            output_tokens=1,
                            seed=0,
                        ),
                        request_id="cancelled",
                    )
                )
                await asyncio.sleep(0.05)
                task.cancel("user cancellation")
                row = await task
            assert row.status == "cancelled"
            assert row.error == "request task cancelled: user cancellation"
            assert row.completed_at is not None
        finally:
            await server.close()

    asyncio.run(run())
