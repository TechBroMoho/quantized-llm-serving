"""Bounded closed-loop and Poisson open-loop load generation.

With `processes > 1` the load is sharded (ADR-013): the parent runs shard 0
and spawns one worker process per remaining shard. Each shard runs the same
closed/open-loop code over a disjoint share of the virtual users and request
indices. All shards start their measurement window at one instant chosen by
the parent on `time.perf_counter`, which is system-wide on Linux and macOS,
so the merged records share one clock and go through the unchanged summary.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import pickle
import random
import time
import traceback
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

import aiohttp

from llmbench.loadtest.client import WINDOW_END_REASON, stream_request
from llmbench.loadtest.metrics import (
    MAX_HALF_WINDOW_DEVIATION,
    RequestRecord,
    TokenWindow,
    summarize,
    summarize_steady,
)

PayloadFactory = Callable[[int], dict[str, Any]]
StartWindow = Callable[[int], Awaitable[float]]

# Delay between the parent's start decision and the common window start, so
# every worker has received the start time before the window opens.
SHARD_START_MARGIN_S = 0.25


@dataclass(frozen=True)
class _Shard:
    """One process's share: users, request counts and a disjoint index set.

    Shard p of P uses measured indices p, p+P, p+2P, ... and warmup indices
    -(1+p), -(1+p+P), ... With even splits, the measured indices of a
    request-count run are exactly 0..N-1 and the warmup indices -1..-W.
    """

    index: int
    count: int
    url: str
    concurrency: int
    warmup_requests: int
    request_count: int | None
    rate_per_s: float | None
    seed: int | str
    total_users: int = 1
    index_offset: int = 0

    def warmup_index(self, k: int) -> int:
        return -(1 + self.index + k * self.count)

    def request_index(self, k: int) -> int:
        return self.index_offset + self.index + k * self.count

    def user_id(self, k: int) -> int:
        """Global virtual-user number of this shard's k-th user."""
        return self.index + k * self.count


@dataclass
class _ShardResult:
    records: list[RequestRecord]
    window_start: float
    window_end: float
    cpu_seconds: float
    offered: int
    rejected: int
    warmup_done: int
    start_received_at: float | None = None
    # Open loop: the latest a scheduled arrival was dispatched. A client that
    # falls behind its schedule sends bursts, which distorts the arrivals.
    max_dispatch_lag_s: float = 0.0


@dataclass
class _Options:
    """Settings shared by every shard (picklable for worker processes)."""

    timeout_s: float
    drain_s: float
    payload_factory: PayloadFactory
    mode: str
    duration_s: float | None
    warmup_s: float = 0.0
    ramp_s: float = 0.0


def _raise_worker_errors(results: list[Any]) -> None:
    """`gather(return_exceptions=True)` must not hide a failed virtual user
    (e.g. an exhausted prompt pool); cancellation at a deadline is expected."""
    for result in results:
        if isinstance(result, BaseException) and not isinstance(
            result, asyncio.CancelledError
        ):
            raise RuntimeError(f"a load worker failed: {result!r}") from result


def _split(total: int, parts: int) -> list[int]:
    share, remainder = divmod(total, parts)
    return [share + (1 if part < remainder else 0) for part in range(parts)]


async def _start_now(_: int) -> float:
    return time.perf_counter()


async def _run_shard(
    shard: _Shard, options: _Options, start_window: StartWindow
) -> _ShardResult:
    """Warm up, wait for the common window start, run the window, drain."""
    records: list[RequestRecord] = []
    request_number = 0
    offered = 0
    rejected = 0
    max_dispatch_lag_s = 0.0
    rng = random.Random(shard.seed)
    payload_factory = options.payload_factory
    duration_s = options.duration_s
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=options.timeout_s),
        connector=aiohttp.TCPConnector(
            limit=shard.concurrency, limit_per_host=shard.concurrency
        ),
    ) as session:

        async def execute(index: int) -> None:
            record = await stream_request(
                session,
                shard.url,
                payload_factory(index),
                request_id=f"request-{index:09d}",
            )
            records.append(record)

        warmup_taken = 0

        async def warmup_worker(count: int) -> int:
            nonlocal warmup_taken
            for _ in range(count):
                index = shard.warmup_index(warmup_taken)
                warmup_taken += 1
                warmup = await stream_request(
                    session,
                    shard.url,
                    payload_factory(index),
                    request_id=f"warmup-{-index:09d}",
                )
                if warmup.status != "ok":
                    raise RuntimeError(f"warmup failed: {warmup.error}")
            return count

        warmup_counts = _split(
            shard.warmup_requests, min(shard.concurrency, shard.warmup_requests) or 1
        )
        warmup_done = sum(
            await asyncio.gather(
                *(warmup_worker(count) for count in warmup_counts if count)
            )
        )

        window_start = await start_window(warmup_done)
        await asyncio.sleep(max(0.0, window_start - time.perf_counter()))
        window_cpu_start = time.process_time()
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
                if (
                    shard.request_count is not None
                    and request_number >= shard.request_count
                ):
                    return
                index = shard.request_index(request_number)
                request_number += 1
                offered += 1
                await execute(index)

        if options.mode == "steady":
            assert duration_s is not None
            measure_start = window_start + options.warmup_s
            window_end = measure_start + duration_s
            token_window = TokenWindow(measure_start, window_end)

            async def steady_user(k: int) -> None:
                nonlocal request_number, offered
                # Staggered start: users are spread evenly over the ramp.
                user_start = window_start + options.ramp_s * shard.user_id(k) / max(
                    1, shard.total_users
                )
                await asyncio.sleep(max(0.0, user_start - time.perf_counter()))
                while time.perf_counter() < window_end:
                    index = shard.request_index(request_number)
                    request_number += 1
                    offered += 1
                    record = await stream_request(
                        session,
                        shard.url,
                        payload_factory(index),
                        request_id=f"request-{index:09d}",
                        window=token_window,
                    )
                    records.append(record)

            pending = {
                asyncio.create_task(steady_user(k), name=f"steady-{k}")
                for k in range(shard.concurrency)
            }
            await asyncio.sleep(max(0.0, measure_start - time.perf_counter()))
            window_cpu_start = time.process_time()
            await asyncio.sleep(max(0.0, window_end - time.perf_counter()))
            window_cpu_seconds = time.process_time() - window_cpu_start
            for task in pending:
                task.cancel(WINDOW_END_REASON)
            _raise_worker_errors(await asyncio.gather(*pending, return_exceptions=True))
            # Report the measurement window, not the warmup start.
            window_start = measure_start
        elif options.mode == "closed":
            pending = {
                asyncio.create_task(worker(), name=f"closed-loop-{i}")
                for i in range(shard.concurrency)
            }
            if duration_s is not None:
                deadline = window_start + duration_s
                await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
                window_end = deadline
                window_cpu_seconds = time.process_time() - window_cpu_start
                done, pending = await asyncio.wait(pending, timeout=options.drain_s)
                for task in pending:
                    task.cancel("bounded drain ended")
                _raise_worker_errors(
                    await asyncio.gather(*done, *pending, return_exceptions=True)
                )
            else:
                await asyncio.gather(*pending)
                window_end = time.perf_counter()
                window_cpu_seconds = time.process_time() - window_cpu_start
        else:
            assert duration_s is not None and shard.rate_per_s is not None
            rate_per_s = shard.rate_per_s
            deadline = window_start + duration_s
            next_arrival = window_start + rng.expovariate(rate_per_s)
            while next_arrival < deadline:
                await asyncio.sleep(max(0.0, next_arrival - time.perf_counter()))
                now = time.perf_counter()
                if now >= deadline:
                    break
                max_dispatch_lag_s = max(max_dispatch_lag_s, now - next_arrival)
                done_now = {task for task in pending if task.done()}
                if done_now:
                    await asyncio.gather(*done_now, return_exceptions=True)
                pending.difference_update(done_now)
                offered += 1
                if len(pending) < shard.concurrency:
                    task = asyncio.create_task(
                        execute(shard.request_index(request_number))
                    )
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
                done, pending = await asyncio.wait(pending, timeout=options.drain_s)
            for task in pending:
                task.cancel("bounded drain ended")
            await asyncio.gather(*done, *pending, return_exceptions=True)

    return _ShardResult(
        records=records,
        window_start=window_start,
        window_end=window_end,
        cpu_seconds=window_cpu_seconds,
        offered=offered,
        rejected=rejected,
        warmup_done=warmup_done,
        max_dispatch_lag_s=max_dispatch_lag_s,
    )


def _shard_process(conn: Connection, shard: _Shard, options: _Options) -> None:
    """Worker entry point: report readiness, wait for the start, send results."""
    received: dict[str, float] = {}

    async def start_window(warmup_done: int) -> float:
        conn.send(("ready", warmup_done))
        kind, start_at = await asyncio.to_thread(conn.recv)
        received["at"] = time.perf_counter()
        if kind != "start":
            raise RuntimeError(f"expected a start message, got {kind!r}")
        return float(start_at)

    try:
        result = asyncio.run(_run_shard(shard, options, start_window))
        result.start_received_at = received.get("at")
        conn.send(("done", result))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


def _receive(conn: Connection, process: BaseProcess, expected: str) -> Any:
    """Blocking receive of one message from a worker (run in a thread)."""
    try:
        kind, value = conn.recv()
    except EOFError as exc:
        process.join(timeout=5)
        raise RuntimeError(
            f"load shard process exited with code {process.exitcode} "
            f"before sending {expected!r}"
        ) from exc
    if kind == "error":
        raise RuntimeError(f"load shard process failed:\n{value}")
    if kind != expected:
        raise RuntimeError(f"expected {expected!r} from load shard, got {kind!r}")
    return value


async def _run_sharded(
    shards: list[_Shard], options: _Options
) -> tuple[list[_ShardResult], dict[str, Any]]:
    """Run shard 0 here and the others in spawned processes."""
    context = multiprocessing.get_context("spawn")
    workers: list[tuple[Connection, BaseProcess]] = []
    clock: dict[str, Any] = {}
    try:
        for shard in shards[1:]:
            parent_end, child_end = context.Pipe()
            spawned = context.Process(
                target=_shard_process,
                args=(child_end, shard, options),
                name=f"llmbench-load-shard-{shard.index}",
                daemon=True,
            )
            spawned.start()
            child_end.close()  # so a dead worker shows up as EOF here
            workers.append((parent_end, spawned))

        async def start_window(_: int) -> float:
            await asyncio.gather(
                *(
                    asyncio.to_thread(_receive, conn, process, "ready")
                    for conn, process in workers
                )
            )
            sent_at = time.perf_counter()
            clock["start_sent_at"] = sent_at
            start_at = sent_at + SHARD_START_MARGIN_S
            for conn, _process in workers:
                conn.send(("start", start_at))
            return start_at

        own = await _run_shard(shards[0], options, start_window)
        remote: list[_ShardResult] = list(
            await asyncio.gather(
                *(
                    asyncio.to_thread(_receive, conn, process, "done")
                    for conn, process in workers
                )
            )
        )
        clock["results_received_at"] = time.perf_counter()
    finally:
        for conn, process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            conn.close()

    # A worker saw the start message after it was sent and before its results
    # came back. A per-process clock would fall outside this interval.
    received = [result.start_received_at for result in remote]
    clock["worker_start_received_at"] = received
    clock["passed"] = all(
        at is not None and clock["start_sent_at"] <= at <= clock["results_received_at"]
        for at in received
    )
    if not clock["passed"]:
        raise RuntimeError(f"worker clocks disagree with the parent: {clock}")
    return [own, *remote], clock


async def run_load(
    *,
    url: str | Sequence[str],
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
    processes: int = 1,
    warmup_s: float = 0.0,
    ramp_s: float = 0.0,
    index_offset: int = 0,
    max_half_deviation: float = MAX_HALF_WINDOW_DEVIATION,
) -> tuple[dict[str, Any], list[RequestRecord]]:
    """Run a load window, stop admission, then drain only for a bounded time.

    `processes` shards users, requests, warmups and the open-loop rate across
    client processes; `payload_factory` must then be picklable. `url` may list
    one URL per process (the capacity validation gives each its own mock).

    `mode="steady"` (ADR-020): users start staggered over `ramp_s`, run a
    closed loop, and are measured over `duration_s` after `warmup_s`;
    requests still running at the window end are cut. `index_offset` keeps
    request indices (and so prompts) unique across runs on one server.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if (duration_s is None) == (request_count is None):
        raise ValueError("specify exactly one of duration_s or request_count")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if request_count is not None and request_count < 1:
        raise ValueError("request_count must be positive")
    if mode not in {"closed", "open", "steady"}:
        raise ValueError("mode must be 'closed', 'open' or 'steady'")
    if mode == "steady" and (
        duration_s is None or warmup_requests or not 0 <= ramp_s <= warmup_s
    ):
        raise ValueError(
            "steady mode needs duration_s, no warmup requests, 0 <= ramp_s <= warmup_s"
        )
    if mode == "open" and (rate_per_s is None or rate_per_s <= 0):
        raise ValueError("open-loop mode needs a positive rate_per_s")
    if mode == "open" and request_count is not None:
        raise ValueError("open-loop mode uses a duration window")
    if not 1 <= processes <= concurrency:
        raise ValueError("processes must be between 1 and concurrency")
    urls = [url] * processes if isinstance(url, str) else list(url)
    if len(urls) != processes:
        raise ValueError("give one URL, or one URL per process")
    if processes > 1:
        try:
            pickle.dumps(payload_factory)
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            raise ValueError(
                "payload_factory must be picklable when processes > 1 "
                "(use a module-level class such as workloads.TextPayloads)"
            ) from exc

    users = _split(concurrency, processes)
    warmups = _split(warmup_requests, processes)
    counts = (
        _split(request_count, processes)
        if request_count is not None
        else [None] * processes
    )
    shards = [
        _Shard(
            index=p,
            count=processes,
            url=urls[p],
            concurrency=users[p],
            warmup_requests=warmups[p],
            request_count=counts[p],
            rate_per_s=None if rate_per_s is None else rate_per_s / processes,
            # One process keeps the historical seed; shards need independent
            # arrival streams (string seeds are deterministic across runs).
            seed=seed if processes == 1 else f"{seed}/{p}",
            total_users=concurrency,
            index_offset=index_offset,
        )
        for p in range(processes)
    ]
    options = _Options(
        timeout_s=timeout_s,
        drain_s=drain_s,
        payload_factory=payload_factory,
        mode=mode,
        duration_s=duration_s,
        warmup_s=warmup_s,
        ramp_s=ramp_s,
    )
    clock: dict[str, Any] | None = None
    if processes == 1:
        results = [await _run_shard(shards[0], options, _start_now)]
    else:
        results, clock = await _run_sharded(shards, options)

    window_start = results[0].window_start
    window_end = max(result.window_end for result in results)
    records = sorted(
        (record for result in results for record in result.records),
        key=lambda record: record.started_at,
    )
    if len({record.request_id for record in records}) != len(records):
        raise RuntimeError("duplicate request IDs across load shards")
    config = {
        "mode": mode,
        "concurrency": concurrency,
        "duration_s": duration_s,
        "request_count": request_count,
        "rate_per_s": rate_per_s,
        "timeout_s": timeout_s,
        "drain_s": drain_s,
        "warmup_requests": warmup_requests,
        "seed": seed,
        "processes": processes,
    }
    if mode == "steady":
        config |= {"warmup_s": warmup_s, "ramp_s": ramp_s, "index_offset": index_offset}
        summary = summarize_steady(
            records,
            window_start=window_start,
            window_end=window_end,
            offered_requests=sum(result.offered for result in results),
            config=config,
            max_half_deviation=max_half_deviation,
        )
    for record in records if mode != "steady" else []:
        if (
            record.status == "ok"
            and record.completed_at is not None
            and record.completed_at > window_end
        ):
            record.status = "late_completion"

    window_cpu_seconds = sum(result.cpu_seconds for result in results)
    if mode != "steady":
        summary = summarize(
            records,
            window_start=window_start,
            window_end=window_end,
            offered_requests=sum(result.offered for result in results),
            rejected_requests=sum(result.rejected for result in results),
            config=config,
        )
    summary["drain_duration_s"] = drain_s
    summary["warmup_requests_completed"] = sum(r.warmup_done for r in results)
    # Summed over every client process: the tester's total CPU (ADR-005).
    summary["window_process_cpu_seconds"] = window_cpu_seconds
    summary["window_process_cpu_cores_average"] = (
        window_cpu_seconds / (window_end - window_start)
        if window_end > window_start
        else 0.0
    )
    summary["client_processes"] = processes
    summary["shard_window_process_cpu_seconds"] = [r.cpu_seconds for r in results]
    summary["shard_users"] = users
    if mode == "open":
        summary["open_loop_max_dispatch_lag_s"] = max(
            r.max_dispatch_lag_s for r in results
        )
    if clock is not None:
        summary["process_clock_check"] = clock
    summary["max_request_index"] = max(
        (int(r.request_id.rsplit("-", 1)[1]) for r in records), default=None
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
