"""End-to-end timing checks against the controllable SSE mock."""

from __future__ import annotations

import asyncio

import aiohttp
from aiohttp.test_utils import TestServer

from llmbench.loadtest import client
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
        f"(error={summary['itl_relative_error']:.1%}); "
        f"structural_ok={summary['structural_ok']}; failing gate entries: "
        + str({k: v for k, v in summary["gate"].items() if not v["passed"]})
        + "; over-tolerance observations: "
        + str(
            {
                o["request_id"]: o["over_tolerance"]
                for o in summary["paired_observations"]
                if o["over_tolerance"]
            }
        )
    )
    assert len(rows) == 10
    assert all(row.status == "ok" for row in rows)
    assert all(row.empty_text_chunks == 1 for row in rows)
    assert all(row.usage_events == 1 for row in rows)
    assert all(row.completion_tokens == 21 for row in rows)
    assert summary["gate"]["itl_s"]["samples"] == 200


def test_mock_timing_accuracy_through_worker_process() -> None:
    # Half the requests come from a spawned process; the server clock is here.
    summary, rows = asyncio.run(measure_timing_accuracy(processes=2))
    assert summary["passed"], summary["gate"]
    assert sorted(row.request_id for row in rows) == [
        f"request-{i:09d}" for i in range(10)
    ]


def test_timing_gate_catches_empty_chunk_timed_as_text(monkeypatch) -> None:
    # ADR-010's mutation: the leading empty chunk sets TTFT.
    original = client._consume_event

    def broken(record, data, received_at, window=None):
        if '"text":""' in data:
            record.add_text(received_at)
        return original(record, data, received_at, window)

    monkeypatch.setattr(client, "_consume_event", broken)
    summary, _ = asyncio.run(measure_timing_accuracy())
    assert not summary["passed"]
    assert summary["gate"]["ttft_s"]["p50_relative_error"] > 0.5


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
