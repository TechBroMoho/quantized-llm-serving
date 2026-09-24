"""Integration coverage for the open-loop and bounded-window runner paths."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from llmbench.loadtest.runner import run_load
from llmbench.loadtest.workloads import (
    PoolPayloads,
    TextPayloads,
    TokenPromptPool,
    completions_payload,
)
from llmbench.mock.server import CONFIG_KEY, MockConfig, completion, make_app


def _open_loop(
    concurrency: int, ttft_s: float, drain_s: float = 0.2
) -> tuple[dict, list]:
    """0.15 s at 60 arrivals/s, seed 7: 17 scheduled arrivals (6.5 ... 135.4 ms)."""

    async def run() -> tuple[dict, list]:
        server = TestServer(
            make_app(MockConfig(ttft_s=ttft_s, itl_s=0.001, output_tokens=2))
        )
        await server.start_server()
        try:
            return await run_load(
                url=str(server.make_url("/v1/completions")),
                concurrency=concurrency,
                timeout_s=2,
                drain_s=drain_s,
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

    return asyncio.run(run())


def test_open_loop_poisson_mode_completes_requests() -> None:
    # The cap (32) is above the 17 scheduled arrivals, so no arrival can be
    # rejected however slowly this machine serves the in-process mock. With a
    # cap of 4, a loaded laptop stretched requests from ~4 ms to 35-52 ms and
    # 4 were still in flight at the 61-74 ms arrival cluster: a correct
    # rejection, but a flaky test (the cap itself is tested below).
    summary, rows = _open_loop(concurrency=32, ttft_s=0.002)
    assert summary["config"]["mode"] == "open"
    assert 0 < summary["offered_requests"] <= 17
    assert summary["warmup_requests_completed"] == 3
    assert summary["on_time_completed_requests"] > 0
    assert summary["rejected_requests"] == 0
    assert len(rows) == summary["offered_requests"]
    assert all(row.status in {"ok", "late_completion"} for row in rows)
    assert summary["open_loop_max_dispatch_lag_s"] >= 0


def test_open_loop_rejects_arrivals_that_find_every_slot_busy() -> None:
    # Each request holds its slot for >= 0.5 s, longer than the whole window,
    # so the first 2 arrivals take both slots and every later one is rejected.
    summary, rows = _open_loop(concurrency=2, ttft_s=0.5, drain_s=2)
    assert summary["offered_requests"] >= 3
    assert summary["rejected_requests"] == summary["offered_requests"] - 2
    assert len(rows) == 2 and all(row.status == "late_completion" for row in rows)


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


def _record_prompts(prompts: list[object]) -> web.Application:
    """The mock's completion handler, recording every prompt it is sent."""

    async def recording(request: web.Request) -> web.StreamResponse:
        prompts.append(tuple((await request.json())["prompt"]))
        return await completion(request)

    app = web.Application()
    app[CONFIG_KEY] = MockConfig(ttft_s=0.002, itl_s=0.001, output_tokens=2)
    app.router.add_post("/v1/completions", recording)
    return app


def test_sharded_request_count_run_covers_every_index_once() -> None:
    prompts: list[object] = []
    pool = TokenPromptPool(allowed_ids=range(3, 500), count=4 + 10, length=8, seed=5)

    async def run() -> tuple[dict, list]:
        server = TestServer(_record_prompts(prompts))
        await server.start_server()
        try:
            return await run_load(
                url=str(server.make_url("/v1/completions")),
                concurrency=3,
                processes=2,
                timeout_s=5,
                drain_s=0,
                payload_factory=PoolPayloads(
                    pool=pool,
                    warmup_requests=4,
                    output_tokens=2,
                    seed=5,
                    model="mock-model",
                ),
                mode="closed",
                request_count=10,
                warmup_requests=4,
            )
        finally:
            await server.close()

    summary, rows = asyncio.run(run())
    assert sorted(row.request_id for row in rows) == [
        f"request-{i:09d}" for i in range(10)
    ]
    assert all(row.status == "ok" for row in rows)
    assert summary["client_processes"] == 2
    assert summary["shard_users"] == [2, 1]
    assert summary["warmup_requests_completed"] == 4
    assert summary["on_time_completed_requests"] == 10
    assert summary["process_clock_check"]["passed"]
    assert len(summary["shard_window_process_cpu_seconds"]) == 2
    # Every warmup and measured request got its own pool prompt, across processes.
    assert len(prompts) == 14
    assert set(prompts) == {tuple(prompt) for prompt in pool.prompts}


def test_sharded_duration_and_open_loop_runs_share_one_window() -> None:
    async def run(mode: str) -> dict:
        server = TestServer(
            make_app(MockConfig(ttft_s=0.002, itl_s=0.001, output_tokens=2))
        )
        await server.start_server()
        try:
            summary, _ = await run_load(
                url=str(server.make_url("/v1/completions")),
                concurrency=4,
                processes=2,
                timeout_s=2,
                drain_s=0.5,
                payload_factory=TextPayloads(prompt="p", output_tokens=2, seed=1),
                mode=mode,
                duration_s=0.5,
                rate_per_s=40 if mode == "open" else None,
                seed=1,
            )
            return summary
        finally:
            await server.close()

    for mode in ("closed", "open"):
        summary = asyncio.run(run(mode))
        assert summary["window_duration_s"] == pytest.approx(0.5)
        assert summary["on_time_completed_requests"] > 0
        assert summary["errored_or_cancelled_requests"] == 0
        assert summary["config"]["processes"] == 2


def test_sharded_run_needs_a_picklable_factory_and_enough_users() -> None:
    options = dict(timeout_s=1, drain_s=0, mode="closed", request_count=2)
    with pytest.raises(ValueError, match="picklable"):
        asyncio.run(
            run_load(
                url="http://127.0.0.1:1",
                concurrency=2,
                processes=2,
                payload_factory=lambda index: {},
                **options,
            )
        )
    with pytest.raises(ValueError, match="between 1 and concurrency"):
        asyncio.run(
            run_load(
                url="http://127.0.0.1:1",
                concurrency=1,
                processes=2,
                payload_factory=TextPayloads(prompt="p", output_tokens=1, seed=0),
                **options,
            )
        )


def test_worker_failure_is_raised_in_the_parent() -> None:
    # Nothing listens on port 1, so every shard's warmup fails.
    with pytest.raises(RuntimeError, match="warmup failed"):
        asyncio.run(
            run_load(
                url="http://127.0.0.1:1/v1/completions",
                concurrency=2,
                processes=2,
                timeout_s=1,
                drain_s=0,
                payload_factory=TextPayloads(prompt="p", output_tokens=1, seed=0),
                mode="closed",
                request_count=2,
                warmup_requests=2,
            )
        )


def test_pool_payloads_map_indices_to_distinct_prompts() -> None:
    pool = TokenPromptPool(allowed_ids=range(3, 99), count=5, length=4, seed=0)
    payloads = PoolPayloads(
        pool=pool, warmup_requests=2, output_tokens=1, seed=0, model="m"
    )
    assert [payloads.position(i) for i in (-1, -2, 0, 1, 2)] == [0, 1, 2, 3, 4]
    assert payloads(-1)["prompt"] == pool.prompts[0]
    assert payloads(0)["skip_special_tokens"] is False
    assert payloads(0)["ignore_eos"] is True
    with pytest.raises(IndexError, match="exhausted"):
        payloads(3)
    with pytest.raises(IndexError, match="no prompt"):
        payloads(-3)
