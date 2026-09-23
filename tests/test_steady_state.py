"""Steady-state windows (ADR-020): per-chunk usage, token bins and summaries."""

from __future__ import annotations

import asyncio
import json

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from llmbench.loadtest.client import stream_request
from llmbench.loadtest.metrics import RequestRecord, TokenWindow, summarize_steady
from llmbench.loadtest.runner import run_load
from llmbench.loadtest.workloads import TextPayloads, completions_payload
from llmbench.mock.server import MockConfig, make_app


def test_token_window_bins() -> None:
    window = TokenWindow(10.0, 20.0, bins=10)
    assert window.bin_of(9.99) is None and window.bin_of(20.01) is None
    assert window.bin_of(10.0) == 0 and window.bin_of(14.5) == 4
    assert window.bin_of(20.0) == 9


def _stream(events: list[dict], window: TokenWindow | None = None) -> RequestRecord:
    async def run() -> RequestRecord:
        async def handler(_: web.Request) -> web.Response:
            body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
            return web.Response(text=body + "data: [DONE]\n\n")

        app = web.Application()
        app.router.add_post("/", handler)
        async with TestServer(app) as server, aiohttp.ClientSession() as session:
            return await stream_request(
                session,
                str(server.make_url("/")),
                completions_payload(
                    prompt=[1, 2], request_index=0, output_tokens=3, seed=0
                ),
                request_id="r",
                window=window,
            )

    return asyncio.run(run())


def _usage(done: int) -> dict[str, int]:
    return {"prompt_tokens": 2, "completion_tokens": done, "total_tokens": 2 + done}


def test_cumulative_usage_is_progress_and_final_must_match() -> None:
    chunks = [{"choices": [{"text": "a"}], "usage": _usage(i)} for i in (1, 2, 3)]
    wide = TokenWindow(0.0, 1e12, bins=2)
    ok = _stream([*chunks, {"choices": [], "usage": _usage(3)}], wide)
    assert ok.status == "ok", ok.error
    assert ok.usage_progress_events == 3 and ok.streamed_tokens == 3
    assert ok.usage_events == 1 and ok.window_tokens == 3
    assert ok.window_chunks == 3
    # Tokens merged into one chunk still count exactly.
    merged = _stream(
        [
            {"choices": [{"text": "ab"}], "usage": _usage(2)},
            {"choices": [{"text": ""}], "usage": _usage(3)},
            {"choices": [], "usage": _usage(3)},
        ],
        wide,
    )
    assert merged.status == "ok" and merged.window_tokens == 3
    assert merged.window_chunks == 2  # two chunks carried the three tokens
    wrong = _stream([*chunks, {"choices": [], "usage": _usage(2)}])
    assert wrong.status == "error" and "streamed usage" in (wrong.error or "")
    backwards = _stream([chunks[1], chunks[0], {"choices": [], "usage": _usage(3)}])
    assert backwards.status == "error" and "decreased" in (backwards.error or "")


def test_tokens_outside_the_window_are_not_counted() -> None:
    chunks = [{"choices": [{"text": "a"}], "usage": _usage(i)} for i in (1, 2, 3)]
    row = _stream([*chunks, {"choices": [], "usage": _usage(3)}], TokenWindow(0, 1))
    assert row.status == "ok" and row.window_tokens == 0


def _steady(**kwargs: object) -> tuple[dict, list[RequestRecord]]:
    async def run() -> tuple[dict, list[RequestRecord]]:
        server = TestServer(
            make_app(MockConfig(ttft_s=0.02, itl_s=0.005, output_tokens=20))
        )
        await server.start_server()
        try:
            return await run_load(
                url=str(server.make_url("/v1/completions")),
                timeout_s=5,
                drain_s=0,
                payload_factory=TextPayloads("p", 20, 0, continuous_usage=True),
                mode="steady",
                **kwargs,  # type: ignore[arg-type]
            )
        finally:
            await server.close()

    return asyncio.run(run())


def test_steady_window_counts_tokens_and_cuts_in_flight_requests() -> None:
    summary, rows = _steady(
        concurrency=8,
        processes=2,
        duration_s=1.5,
        warmup_s=0.4,
        ramp_s=0.12,
        index_offset=1000,
    )
    assert summary["window_rule"] == "steady_state_adr020"
    assert summary["errored_or_cancelled_requests"] == 0
    assert {row.status for row in rows} == {"ok", "window_end"}
    # At most one cut request per user; a user that finished just before the
    # end starts nothing new, since admission stops at the window end.
    assert 1 <= summary["cut_at_window_end_requests"] <= 8
    window_end = summary["window_end_monotonic_s"]
    assert all(r.started_at < window_end for r in rows if r.status == "window_end")
    assert summary["steady_state_ok"], summary["window_token_bins"]
    assert summary["request_rate_ok"]
    # Token arrivals and completed requests measure the same steady rate.
    assert (
        abs(
            summary["output_token_throughput_per_s"]
            / summary["completed_request_output_tokens_per_s"]
            - 1
        )
        < 0.05
    )
    indices = sorted(int(r.request_id.split("-")[1]) for r in rows)
    assert indices[0] == 1000 and len(set(indices)) == len(indices)
    assert summary["max_request_index"] == indices[-1]
    assert summary["process_clock_check"]["passed"]


def test_steady_state_check_flags_a_rate_change_within_the_window() -> None:
    def row(i: int, bins: list[int]) -> RequestRecord:
        r = RequestRecord(f"r{i}", "ok", 0.0, completed_at=5.0, completion_tokens=4)
        r.window_token_bins = bins
        return r

    flat = summarize_steady(
        [row(i, [1, 1, 1, 1]) for i in range(10)],
        window_start=0.0,
        window_end=10.0,
        offered_requests=10,
        config={},
    )
    assert flat["steady_state_ok"] and flat["output_token_throughput_per_s"] == 4.0
    ramping = summarize_steady(
        [row(i, [0, 1, 1, 2]) for i in range(10)],
        window_start=0.0,
        window_end=10.0,
        offered_requests=10,
        config={},
    )
    assert not ramping["steady_state_ok"]
    # All ten completed at one instant: moving an edge could drop them all.
    assert ramping["completion_bunch_size"] == 10
    assert not ramping["request_rate_ok"]
