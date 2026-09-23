"""Bounded closed-loop and Poisson open-loop load generation."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from typing import Any

import aiohttp

from llmbench.loadtest.client import stream_request
from llmbench.loadtest.metrics import RequestRecord, summarize

PayloadFactory = Callable[[int], dict[str, Any]]


async def run_load(
    *,
    url: str,
    concurrency: int,
    timeout_s: float,
    drain_s: float,
    payload_factory: PayloadFactory,
    mode: str,
    duration_s: float | None = None,
    request_count: int | None = None,
    rate_per_s: float | None = None,
    warmup_requests: int = 0,
    seed: int = 0,
) -> tuple[dict[str, Any], list[RequestRecord]]:
    """Run a load window, stop admission, then drain only for a bounded time."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if (duration_s is None) == (request_count is None):
        raise ValueError("specify exactly one of duration_s or request_count")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if request_count is not None and request_count < 1:
        raise ValueError("request_count must be positive")
    if mode not in {"closed", "open"}:
        raise ValueError("mode must be 'closed' or 'open'")
    if mode == "open" and (rate_per_s is None or rate_per_s <= 0):
        raise ValueError("open-loop mode needs a positive rate_per_s")
    if mode == "open" and request_count is not None:
        raise ValueError("open-loop mode uses a duration window")

    records: list[RequestRecord] = []
    request_number = 0
    offered = 0
    rejected = 0
    rng = random.Random(seed)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout_s),
        connector=aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency),
    ) as session:

        async def execute(index: int) -> None:
            record = await stream_request(
                session,
                url,
                payload_factory(index),
                request_id=f"request-{index:09d}",
            )
            records.append(record)

        workers, remainder = divmod(warmup_requests, concurrency)

        async def warmup_worker(worker_index: int, count: int) -> int:
            for index in range(count):
                request_index = -(worker_index * warmup_requests + index + 1)
                warmup = await stream_request(
                    session,
                    url,
                    payload_factory(request_index),
                    request_id=f"warmup-{worker_index:06d}-{index:06d}",
                )
                if warmup.status != "ok":
                    raise RuntimeError(f"warmup failed: {warmup.error}")
            return count

        warmup_counts = [
            workers + (1 if worker_index < remainder else 0)
            for worker_index in range(min(concurrency, warmup_requests))
        ]
        warmup_done = sum(
            await asyncio.gather(
                *(
                    warmup_worker(worker_index, count)
                    for worker_index, count in enumerate(warmup_counts)
                )
            )
        )

        window_cpu_start = time.process_time()
        window_start = time.perf_counter()
        window_end = window_start
        window_cpu_seconds = 0.0
        pending: set[asyncio.Task[None]] = set()

        async def worker() -> None:
            nonlocal request_number, offered
            while True:
                if (
                    duration_s is not None
                    and time.perf_counter() >= window_start + duration_s
                ):
                    return
                if request_count is not None and request_number >= request_count:
                    return
                index = request_number
                request_number += 1
                offered += 1
                await execute(index)

        if mode == "closed":
            pending = {
                asyncio.create_task(worker(), name=f"closed-loop-{i}")
                for i in range(concurrency)
            }
            if duration_s is not None:
                await asyncio.sleep(duration_s)
                window_end = window_start + duration_s
                window_cpu_seconds = time.process_time() - window_cpu_start
                done, pending = await asyncio.wait(pending, timeout=drain_s)
                for task in pending:
                    task.cancel("bounded drain ended")
                await asyncio.gather(*done, *pending, return_exceptions=True)
            else:
                await asyncio.gather(*pending)
                window_end = time.perf_counter()
                window_cpu_seconds = time.process_time() - window_cpu_start
        else:
            assert duration_s is not None and rate_per_s is not None
            deadline = window_start + duration_s
            next_arrival = window_start + rng.expovariate(rate_per_s)
            while next_arrival < deadline:
                await asyncio.sleep(max(0.0, next_arrival - time.perf_counter()))
                if time.perf_counter() >= deadline:
                    break
                done_now = {task for task in pending if task.done()}
                if done_now:
                    await asyncio.gather(*done_now, return_exceptions=True)
                pending.difference_update(done_now)
                offered += 1
                if len(pending) < concurrency:
                    task = asyncio.create_task(execute(request_number))
                    request_number += 1
                    pending.add(task)
                else:
                    rejected += 1
                next_arrival += rng.expovariate(rate_per_s)
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
            window_end = deadline
            window_cpu_seconds = time.process_time() - window_cpu_start
            done = set()
            if pending:
                done, pending = await asyncio.wait(pending, timeout=drain_s)
            for task in pending:
                task.cancel("bounded drain ended")
            await asyncio.gather(*done, *pending, return_exceptions=True)

    for record in records:
        if (
            record.status == "ok"
            and record.completed_at is not None
            and record.completed_at > window_end
        ):
            record.status = "late_completion"

    summary = summarize(
        records,
        window_start=window_start,
        window_end=window_end,
        offered_requests=offered,
        rejected_requests=rejected,
        config={
            "mode": mode,
            "concurrency": concurrency,
            "duration_s": duration_s,
            "request_count": request_count,
            "rate_per_s": rate_per_s,
            "timeout_s": timeout_s,
            "drain_s": drain_s,
            "warmup_requests": warmup_requests,
            "seed": seed,
        },
    )
    summary["drain_duration_s"] = drain_s
    summary["warmup_requests_completed"] = warmup_done
    summary["window_process_cpu_seconds"] = window_cpu_seconds
    summary["window_process_cpu_cores_average"] = (
        window_cpu_seconds / (window_end - window_start)
        if window_end > window_start
        else 0.0
    )
    summary["measurement_text_chunks"] = sum(
        row.text_chunks
        for row in records
        if row.status == "ok"
        and row.started_at >= window_start
        and row.completed_at is not None
        and row.completed_at <= window_end
    )
    return summary, records
