"""Phase 6 benchmark driver, shared by Modal functions and CPU rehearsals.

One lifetime = one server process for a whole list of sweep points. For each
point: wait until the server is idle, snapshot its counters, run one
steady-state window (ADR-020) with the multi-process load tester (ADR-013),
snapshot again, validate, and save the point before starting the next one.
Every point draws fresh prompts from one pool, so no prompt is sent twice
in a lifetime and every variant gets the same prompts in the same order.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

from llmbench.loadtest.io import write_results
from llmbench.loadtest.metrics import MAX_HALF_WINDOW_DEVIATION, RequestRecord
from llmbench.loadtest.runner import run_load
from llmbench.prompts import NpyPromptPayloads
from llmbench.smoke import (
    fetch_text,
    log_tail,
    matching_lines,
    missing_flags,
    runtime_metadata,
    start_server,
    stop_server,
    wait_for_health,
    write_json,
)

BENCH_LOG_PATTERNS = (
    r"non-default args",
    r"Model loading took",
    r"GPU KV cache size",
    r"Maximum concurrency",
    r"[Gg]raph capturing",
    r"torch\.compile",
    r"prefix caching",
    r"Using .* backend",
    r"KV cache",
)


@dataclass(frozen=True)
class BenchPoint:
    """One steady-state measurement at a fixed closed-loop concurrency."""

    label: str
    concurrency: int
    ramp_s: float
    warmup_s: float
    window_s: float
    timeout_s: float
    # Reported, never used for headline numbers or the lifetime's pass/fail
    # (e.g. the all-at-once c=256 run that probes the cross-check gap).
    diagnostic: bool = False
    # False keeps ramp_s as configured (an all-at-once start uses ramp 0).
    adapt_ramp: bool = True
    # False: timings already derived elsewhere (HF static, from batch times).
    adapt: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchPoint:
        return cls(
            label=str(data["label"]),
            concurrency=int(data["concurrency"]),
            ramp_s=float(data["ramp_s"]),
            warmup_s=float(data["warmup_s"]),
            window_s=float(data["window_s"]),
            timeout_s=float(data["timeout_s"]),
            diagnostic=bool(data.get("diagnostic", False)),
            adapt_ramp=bool(data.get("adapt_ramp", True)),
            adapt=bool(data.get("adapt", True)),
        )

    @property
    def planned_s(self) -> float:
        return self.warmup_s + self.window_s


# ADR-022: from this concurrency on, users must be spread over a whole
# request duration and the window must cover >= 10 of them.
HIGH_CONCURRENCY = 128
WINDOW_DURATIONS = 10


def _ceil5(seconds: float) -> float:
    return float(5 * -(-seconds // 5))


def expected_e2e(
    concurrency: int,
    history: Sequence[dict[str, Any]],
    seeds: dict[int, float],
    output_tokens: int,
) -> tuple[float, str] | None:
    """Expected request duration at `concurrency` (ADR-022), in order of trust:
    a point already measured at this concurrency in this lifetime, a value
    measured earlier for this variant (config seed), or an extrapolation from
    this lifetime's lower-concurrency points (the larger of linear-in-TPOT
    and proportional-to-concurrency, both of which underestimated AWQ)."""
    same = [h for h in history if h["concurrency"] == concurrency and h["e2e_p50"]]
    if same:
        return float(same[-1]["e2e_p50"]), f"measured here ({same[-1]['label']})"
    if concurrency in seeds:
        return float(seeds[concurrency]), "measured earlier (config seed)"
    lower = sorted(
        (h for h in history if h["concurrency"] < concurrency and h["e2e_p50"]),
        key=lambda h: h["concurrency"],
    )
    if not lower:
        return None
    last = lower[-1]
    scale = concurrency / last["concurrency"]
    estimate = float(last["e2e_p50"]) * scale
    source = f"x{scale:g} of {last['label']}"
    if len(lower) >= 2 and lower[-2]["concurrency"] < last["concurrency"]:
        prev = lower[-2]
        slope = (last["tpot_p50"] - prev["tpot_p50"]) / (
            last["concurrency"] - prev["concurrency"]
        )
        tpot = last["tpot_p50"] + slope * (concurrency - last["concurrency"])
        linear = last["ttft_p50"] * scale + (output_tokens - 1) * tpot
        if linear > estimate:
            estimate, source = (
                linear,
                f"TPOT extrapolated from {prev['label']}, {last['label']}",
            )
    return estimate, source


def adapt_point(
    point: BenchPoint,
    history: Sequence[dict[str, Any]],
    seeds: dict[int, float],
    output_tokens: int,
) -> tuple[BenchPoint, dict[str, Any] | None]:
    """Apply ADR-022 at c >= 128: ramp = one expected E2E, warmup >= ramp +
    E2E, window >= 10 E2E. Never shortens a configured timing."""
    if point.concurrency < HIGH_CONCURRENCY or not point.adapt:
        return point, None
    found = expected_e2e(point.concurrency, history, seeds, output_tokens)
    if found is None:
        return point, {"rule": "ADR-022", "expected_e2e_s": None, "note": "no data"}
    e2e, source = found
    ramp = float(-(-e2e // 1)) if point.adapt_ramp else point.ramp_s
    adapted = BenchPoint(
        label=point.label,
        concurrency=point.concurrency,
        ramp_s=ramp,
        warmup_s=max(point.warmup_s, _ceil5(ramp + e2e)),
        window_s=max(point.window_s, _ceil5(WINDOW_DURATIONS * e2e)),
        timeout_s=max(point.timeout_s, 4 * e2e),
        diagnostic=point.diagnostic,
        adapt_ramp=point.adapt_ramp,
        adapt=point.adapt,
    )
    return adapted, {"rule": "ADR-022", "expected_e2e_s": e2e, "source": source}


_METRIC_LINE = re.compile(r"^(vllm:[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")


def vllm_counters(text: str) -> dict[str, float]:
    """Sum each `vllm:*` Prometheus series over its labels."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _METRIC_LINE.match(line.strip())
        if match:
            totals[match[1]] = totals.get(match[1], 0.0) + float(match[2])
    return totals


def server_state(base_url: str, kind: str) -> dict[str, Any]:
    """Counters and in-flight work, from vLLM `/metrics` or HF `/stats`."""
    if kind == "vllm":
        response = fetch_text(f"{base_url}/metrics")
        counters = vllm_counters(response["text"])
        busy = counters.get("vllm:num_requests_running", 0) + counters.get(
            "vllm:num_requests_waiting", 0
        )
        return {"status": response["status"], "counters": counters, "busy": busy}
    if kind == "hf":
        response = fetch_text(f"{base_url}/stats")
        try:
            stats = json.loads(response["text"])
        except json.JSONDecodeError:
            return {"status": response["status"], "stats": None, "busy": None}
        busy = int(stats.get("pending", 0)) + int(bool(stats.get("busy")))
        return {"status": response["status"], "stats": stats, "busy": busy}
    return {"status": None, "busy": 0}


async def wait_idle(base_url: str, kind: str, timeout_s: float) -> dict[str, Any]:
    """Wait until the server has no running or queued work (bounded)."""
    started = time.perf_counter()
    while True:
        state = await asyncio.to_thread(server_state, base_url, kind)
        if state["busy"] == 0:
            return state | {"waited_s": time.perf_counter() - started}
        if time.perf_counter() - started > timeout_s:
            raise TimeoutError(f"server still busy after {timeout_s:.0f} s: {state}")
        await asyncio.sleep(0.5)


def read_process_table(root: Path = Path("/proc")) -> dict[str, Any]:
    """Cumulative CPU seconds of every process, and the host's busy seconds.

    Linux only (`/proc/<pid>/stat` utime + stime, `/proc/stat`); an empty
    table elsewhere. Used to show which process a benchmark is CPU-bound in.
    """
    import os

    if not (root / "stat").exists():
        return {"processes": {}, "busy_s": None}
    ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    processes: dict[str, list[Any]] = {}
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text()
        except OSError:
            continue  # the process exited while we were reading
        comm = text[text.index("(") + 1 : text.rindex(")")]
        fields = text[text.rindex(")") + 2 :].split()
        pgrp, utime, stime = int(fields[2]), int(fields[11]), int(fields[12])
        processes[entry.name] = [comm, pgrp, (utime + stime) / ticks]
    first = (root / "stat").read_text().splitlines()[0].split()[1:]
    values = [int(v) for v in first]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return {"processes": processes, "busy_s": (sum(values[:8]) - idle) / ticks}


def cores_between(
    samples: Sequence[tuple[float, dict[str, Any]]],
    start: float,
    end: float,
    server_pgid: int | None,
) -> dict[str, Any] | None:
    """Average CPU cores per process over [start, end], from ~1 s samples.

    A process present throughout is measured between the samples bracketing
    the window. One that starts or exits inside it (a client worker exits as
    the window closes) is measured over the samples in which it appears, from
    the sample before its first appearance, so it is not missed.
    """
    before = [s for s in samples if s[0] <= start]
    after = [s for s in samples if s[0] >= end]
    if not before or not after:
        return None
    t0, t1 = before[-1][0], after[0][0]
    span_samples = [s for s in samples if t0 <= s[0] <= t1]
    seen: dict[str, list[Any]] = {}  # pid -> [comm, pgrp, t_base, cpu_base, t, cpu]
    previous_t = t0
    for at, table in span_samples:
        for pid, (comm, pgrp, cpu) in table["processes"].items():
            if pid not in seen:
                started_here = at > t0
                seen[pid] = [
                    comm,
                    pgrp,
                    previous_t if started_here else at,
                    0.0 if started_here else cpu,
                    at,
                    cpu,
                ]
            seen[pid][4:] = [at, cpu]
        previous_t = at
    rows = []
    for pid, (comm, pgrp, t_base, cpu_base, t_last, cpu_last) in seen.items():
        duration = t_last - t_base
        if duration <= 0:
            continue
        cores = (cpu_last - cpu_base) / duration
        if cores >= 0.01:
            rows.append(
                {
                    "pid": int(pid),
                    "comm": comm,
                    "group": "server" if pgrp == server_pgid else "client_or_other",
                    "cores": cores,
                    "observed_s": duration,
                }
            )
    rows.sort(key=lambda row: -row["cores"])
    first, last = before[-1][1], after[0][1]
    busy = (
        (last["busy_s"] - first["busy_s"]) / (t1 - t0)
        if first["busy_s"] is not None and last["busy_s"] is not None
        else None
    )
    return {
        "sample_span_s": t1 - t0,
        "processes": rows,
        "server_cores": sum(r["cores"] for r in rows if r["group"] == "server"),
        "client_or_other_cores": sum(
            r["cores"] for r in rows if r["group"] != "server"
        ),
        "container_busy_cores": busy,
    }


class CpuSampler:
    """`read_process_table()` every `interval_s` in a thread, for a lifetime."""

    def __init__(self, interval_s: float = 1.0) -> None:
        import threading

        self.interval_s = interval_s
        self.samples: list[tuple[float, dict[str, Any]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.enabled = bool(read_process_table()["processes"])

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), read_process_table()))
            self._stop.wait(self.interval_s)

    def __enter__(self) -> CpuSampler:
        if self.enabled:
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self.enabled:
            self._thread.join(timeout=5)


class GpuSampler:
    """`nvidia-smi` samples every second into a CSV, for the whole lifetime."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> GpuSampler:
        if shutil.which("nvidia-smi"):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("wb") as out:
                self.process = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,memory.used,memory.total,"
                        "utilization.gpu,power.draw",
                        "--format=csv,nounits",
                        "--loop-ms=1000",
                    ],
                    stdout=out,
                    stderr=subprocess.STDOUT,
                )
        return self

    def __exit__(self, *_: object) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


def validate_point(
    summary: dict[str, Any],
    records: list[RequestRecord],
    *,
    input_tokens: int,
    output_tokens: int,
) -> tuple[list[str], list[str]]:
    """Failures invalidate a point; warnings qualify its requests/s only."""
    failures = []
    if summary["errored_or_cancelled_requests"]:
        failures.append(
            f"{summary['errored_or_cancelled_requests']} errored requests, e.g. "
            + "; ".join(
                f"{r.request_id}: {r.error}"
                for r in records
                if r.status not in {"ok", "window_end"}
            )[:500]
        )
    completed = [r for r in records if r.status == "ok"]
    if not summary["completed_in_window_requests"]:
        failures.append("no request completed inside the window")
    for row in completed:
        if row.prompt_tokens != input_tokens or row.completion_tokens != output_tokens:
            failures.append(
                f"{row.request_id}: usage {row.prompt_tokens}/{row.completion_tokens}"
            )
            break
        if not row.usage_progress_events:
            failures.append(f"{row.request_id}: no per-chunk usage (ADR-020)")
            break
    if summary["text_chunk_shortfall_requests"]:
        failures.extend(summary["text_chunk_shortfall_examples"][:3])
    if not summary["steady_state_ok"]:
        failures.append(
            "window halves differ by "
            f"{summary['half_window_token_deviation']}: not steady state"
        )
    warnings = []
    if not summary["request_rate_ok"]:
        warnings.append(
            "requests/s edge error bound "
            f"{summary['request_rate_edge_error_bound']:.3f} exceeds the limit"
        )
    return failures, warnings


async def run_points(
    *,
    base_url: str,
    kind: str,
    payloads: NpyPromptPayloads,
    points: Sequence[BenchPoint],
    processes: int,
    out_dir: Path,
    input_tokens: int,
    output_tokens: int,
    idle_timeout_s: float,
    checkpoint: Callable[[], None] = lambda: None,
    index_offset: int = 0,
    max_half_deviation: float = MAX_HALF_WINDOW_DEVIATION,
    cpu: CpuSampler | None = None,
    server_pgid: int | None = None,
    validated_chunks_per_s: float | None = None,
    e2e_seeds: dict[int, float] | None = None,
    repeat_best: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Run each point in order, saving it before the next; returns the offset.

    With `repeat_best`, the best passing non-diagnostic point (by output
    tokens/s) is then repeated that many times as `<label>-r2`, `-r3`, ...
    """
    results: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    queue = list(points)
    repeats_added = repeat_best == 0
    while queue or not repeats_added:
        if not queue:
            repeats_added = True
            candidates = [
                r
                for r in results
                if r["passed"]
                and not r["diagnostic"]
                and not re.search(r"-r\d+$", r["label"])
            ]
            if not candidates:
                break
            best = max(candidates, key=lambda r: r["output_token_throughput_per_s"])
            source = next(p for p in points if p.label == best["label"])
            queue = [
                BenchPoint(
                    f"{best['label']}-r{k}",
                    source.concurrency,
                    source.ramp_s,
                    source.warmup_s,
                    source.window_s,
                    source.timeout_s,
                    source.diagnostic,
                    source.adapt_ramp,
                    source.adapt,
                )
                for k in range(2, 2 + repeat_best)
            ]
            continue
        configured = queue.pop(0)
        point, timing = adapt_point(configured, history, e2e_seeds or {}, output_tokens)
        point_dir = out_dir / point.label
        idle = await wait_idle(base_url, kind, idle_timeout_s)
        before = await asyncio.to_thread(server_state, base_url, kind)
        summary, records = await run_load(
            url=f"{base_url}/v1/completions",
            concurrency=point.concurrency,
            processes=min(processes, point.concurrency),
            timeout_s=point.timeout_s,
            drain_s=0,
            payload_factory=payloads,
            mode="steady",
            duration_s=point.window_s,
            warmup_s=point.warmup_s,
            ramp_s=point.ramp_s,
            index_offset=index_offset,
            max_half_deviation=max_half_deviation,
        )
        after = await asyncio.to_thread(server_state, base_url, kind)
        if cpu is not None and cpu.enabled:
            await asyncio.sleep(1.5 * cpu.interval_s)  # a sample after the window
        cpu_in_window = (
            cores_between(
                list(cpu.samples),
                summary["window_start_monotonic_s"],
                summary["window_end_monotonic_s"],
                server_pgid,
            )
            if cpu is not None
            else None
        )
        first_index = index_offset
        if summary["max_request_index"] is not None:
            index_offset = int(summary["max_request_index"]) + 1
        failures, warnings = validate_point(
            summary, records, input_tokens=input_tokens, output_tokens=output_tokens
        )
        if validated_chunks_per_s is not None:
            # ADR-005: the client's validated capacity must be at least 3x the
            # chunk rate it actually received, or the point is not accepted.
            multiple = validated_chunks_per_s / max(
                summary["chunks_in_window_per_s"], 1e-9
            )
            summary["client_capacity_headroom"] = multiple
            if multiple < 3:
                failures.append(
                    f"client headroom {multiple:.2f}x < 3x: "
                    f"{summary['chunks_in_window_per_s']:.0f} chunks/s received, "
                    f"{validated_chunks_per_s:.0f}/s validated (ADR-005)"
                )
        if kind == "vllm":
            hits = after["counters"].get("vllm:prefix_cache_hits_total", 0) - before[
                "counters"
            ].get("vllm:prefix_cache_hits_total", 0)
            if hits:
                failures.append(f"prefix cache hits {hits} (caching must be off)")
        if point.diagnostic:
            summary["diagnostic_findings"] = failures + warnings
            failures, warnings = [], []
        summary |= {
            "point": asdict(point),
            "configured_point": asdict(configured),
            "timing_rule": timing,
            "diagnostic": point.diagnostic,
            "idle_before": idle,
            "server_before": before,
            "server_after": after,
            "prompt_index_range": [first_index, index_offset],
            # Direct evidence of where CPU goes: the client shards' own
            # measurement over the exact window, and /proc for every process.
            "client_shard_cpu_cores": [
                seconds / summary["window_duration_s"]
                for seconds in summary["shard_window_process_cpu_seconds"]
            ],
            "cpu_in_window": cpu_in_window,
            "failures": failures,
            "warnings": warnings,
            "passed": not failures,
        }
        write_results(
            point_dir / "summary.json",
            summary,
            records,
            raw_path=point_dir / "requests.jsonl.gz",
        )
        await asyncio.to_thread(checkpoint)
        results.append(
            {
                key: summary[key]
                for key in (
                    "passed",
                    "failures",
                    "warnings",
                    "output_token_throughput_per_s",
                    "request_throughput_per_s",
                    "completed_in_window_requests",
                    "half_window_token_deviation",
                    "request_rate_edge_error_bound",
                    "window_process_cpu_cores_average",
                    "client_shard_cpu_cores",
                    "cpu_in_window",
                    "chunks_in_window_per_s",
                    "ttft_s",
                    "tpot_s",
                    "itl_s",
                    "e2e_s",
                )
            }
            | {
                "label": point.label,
                "concurrency": point.concurrency,
                "diagnostic": point.diagnostic,
                "diagnostic_findings": summary.get("diagnostic_findings"),
                "timing_rule": timing,
                "point": asdict(point),
            }
        )
        history.append(
            {
                "label": point.label,
                "concurrency": point.concurrency,
                "e2e_p50": summary["e2e_s"]["p50"],
                "tpot_p50": summary["tpot_s"]["p50"],
                "ttft_p50": summary["ttft_s"]["p50"],
            }
        )
        print(
            f"{point.label}: {summary['output_token_throughput_per_s']:.1f} tok/s, "
            f"{summary['request_throughput_per_s']:.3f} req/s, "
            f"passed={not failures}",
            flush=True,
        )
    return results, index_offset


def vllm_bench_serve_command(
    *,
    base_url: str,
    model_name: str,
    tokenizer: str,
    concurrency: int,
    num_prompts: int,
    input_tokens: int,
    output_tokens: int,
    result_dir: Path,
    seed: int,
) -> list[str]:
    """`vllm bench serve` at a matching setting (flags checked in v0.10.2)."""
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        base_url,
        "--endpoint",
        "/v1/completions",
        "--model",
        model_name,
        "--tokenizer",
        tokenizer,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_tokens),
        "--random-output-len",
        str(output_tokens),
        "--random-range-ratio",
        "0.0",
        "--ignore-eos",
        "--max-concurrency",
        str(concurrency),
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        "inf",
        "--seed",
        str(seed),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        f"vllm_bench_serve_c{concurrency}.json",
        "--disable-tqdm",
    ]


def bench_serve_flag_failures(command: Sequence[str], out_dir: Path) -> list[str]:
    """Check every flag in `command` against the pinned `vllm bench serve --help`."""
    done = subprocess.run(
        ["vllm", "bench", "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (out_dir / "vllm_bench_serve_help.txt").write_text(done.stdout + done.stderr)
    if done.returncode != 0:
        return [f"vllm bench serve --help exited {done.returncode}"]
    flags = [arg for arg in command if arg.startswith("--")]
    return [f"flag not in --help: {f}" for f in missing_flags(flags, done.stdout)]


async def run_lifetime(
    *,
    label: str,
    command: Sequence[str],
    base_url: str,
    kind: str,
    payloads: NpyPromptPayloads,
    points: Sequence[BenchPoint],
    processes: int,
    out_dir: Path,
    input_tokens: int,
    output_tokens: int,
    health_timeout_s: float,
    idle_timeout_s: float,
    env: dict[str, str] | None = None,
    checkpoint: Callable[[], None] = lambda: None,
    metadata: dict[str, Any] | None = None,
    after_points: Callable[[Path], dict[str, Any]] | None = None,
    max_half_deviation: float = MAX_HALF_WINDOW_DEVIATION,
    validated_chunks_per_s: float | None = None,
    e2e_seeds: dict[int, float] | None = None,
    repeat_best: int = 0,
) -> dict[str, Any]:
    """One server lifetime; always writes `lifetime_summary.json`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"
    summary: dict[str, Any] = {
        "label": label,
        "command": list(command),
        "points_planned": [asdict(p) for p in points],
        "runtime": runtime_metadata(),
        **(metadata or {}),
    }
    failures: list[str] = []
    write_json(out_dir / "lifetime_summary.json", summary | {"state": "starting"})
    started = time.perf_counter()
    process = start_server(command, log_path, env)
    cpu = CpuSampler()
    try:
        with GpuSampler(out_dir / "nvidia_smi.csv"), cpu:
            summary["health_wait_s"] = await wait_for_health(
                base_url, process, health_timeout_s
            )
            summary["server_at_health"] = await asyncio.to_thread(
                server_state, base_url, kind
            )
            write_json(
                out_dir / "lifetime_summary.json", summary | {"state": "healthy"}
            )
            await asyncio.to_thread(checkpoint)
            summary["points"], summary["next_prompt_index"] = await run_points(
                base_url=base_url,
                kind=kind,
                payloads=payloads,
                points=points,
                processes=processes,
                out_dir=out_dir,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                idle_timeout_s=idle_timeout_s,
                checkpoint=checkpoint,
                max_half_deviation=max_half_deviation,
                cpu=cpu,
                server_pgid=process.pid,  # start_server makes a new session
                validated_chunks_per_s=validated_chunks_per_s,
                e2e_seeds=e2e_seeds,
                repeat_best=repeat_best,
            )
            failures.extend(
                f"{point['label']}: {failure}"
                for point in summary["points"]
                for failure in point["failures"]
            )
            if after_points is not None:
                summary["after_points"] = await asyncio.to_thread(after_points, out_dir)
                failures.extend(summary["after_points"].get("failures", []))
    except Exception as exc:  # recorded, then re-raised by the caller's check
        failures.append(f"{type(exc).__name__}: {exc}")
        summary["server_log_tail"] = log_tail(log_path)
    finally:
        summary["server_returncode_after_stop"] = stop_server(process)
        summary["server_lifetime_s"] = time.perf_counter() - started
        summary["log_excerpts"] = matching_lines(log_path, BENCH_LOG_PATTERNS)
        summary["cpu_sampler_enabled"] = cpu.enabled
        if cpu.samples:
            import gzip

            with gzip.open(out_dir / "cpu_samples.jsonl.gz", "wt") as stream:
                for at, table in cpu.samples:
                    stream.write(json.dumps({"at": at, **table}) + "\n")
        summary["failures"] = failures
        summary["passed"] = not failures
        summary["state"] = "finished"
        write_json(out_dir / "lifetime_summary.json", summary)
        await asyncio.to_thread(checkpoint)
    return summary


async def post_json(url: str, body: dict[str, Any], timeout_s: float) -> Any:
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as s,
        s.post(url, json=body) as response,
    ):
        return await response.json()


def resolve_points(
    definitions: dict[str, dict[str, Any]],
    names: Sequence[str],
    batch_size: int | None = None,
) -> list[BenchPoint]:
    """Config point names to points. `NAME-rK` repeats NAME under a new label;
    concurrency `B` / `2B` is the HF static batch size from the OOM probe."""
    points = []
    for name in names:
        base = re.sub(r"-r\d+$", "", name)
        if base not in definitions:
            raise KeyError(f"unknown point {name!r}")
        data = dict(definitions[base])
        concurrency = str(data["concurrency"])
        if concurrency in {"B", "2B"}:
            if batch_size is None:
                raise ValueError(f"{name} needs the static batch size")
            data["concurrency"] = batch_size * (2 if concurrency == "2B" else 1)
        points.append(BenchPoint.from_dict(data | {"label": name}))
    if len({p.label for p in points}) != len(points):
        raise ValueError("point labels must be unique within a lifetime")
    return points


def expected_checkpoint_files(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-file sha256 list from a Phase 4 download manifest or quantize summary."""
    files = evidence.get("files") or evidence.get("checkpoint", {}).get("files")
    if not files:
        raise ValueError("evidence has no file list")
    return [
        {"path": f["path"], "bytes": int(f["bytes"]), "sha256": f["sha256"]}
        for f in files
    ]


def verify_checkpoint(root: Path, expected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Hash every expected file under `root`; any missing or changed file fails."""
    from llmbench.quantization import sha256_file

    mismatches = []
    started = time.perf_counter()
    for item in expected:
        path = root / item["path"]
        if not path.is_file():
            mismatches.append({"path": item["path"], "problem": "missing"})
            continue
        size = path.stat().st_size
        if size != item["bytes"]:
            mismatches.append(
                {"path": item["path"], "problem": f"bytes {size} != {item['bytes']}"}
            )
            continue
        digest = sha256_file(path)
        if digest != item["sha256"]:
            mismatches.append({"path": item["path"], "problem": f"sha256 {digest}"})
    return {
        "root": str(root),
        "files_checked": len(expected),
        "bytes_checked": sum(int(item["bytes"]) for item in expected),
        "mismatches": mismatches,
        "seconds": time.perf_counter() - started,
        "passed": not mismatches,
    }


def static_points_from_probe(
    points: Sequence[BenchPoint], probe: dict[str, Any], margin: float = 1.25
) -> tuple[list[BenchPoint], list[dict[str, Any]]]:
    """HF static timings from the OOM probe's measured batch times (ADR-020).

    A batch of up to B requests takes T(n) seconds, taken from the smallest
    probed batch size >= n. With c users and batch size B, a request waits
    for ceil(c / B) batches, so E2E ~= ceil(c / B) * T(min(c, B)). Warmup is
    ramp + `margin` x E2E (rounded up to 5 s). The window covers at least
    three batch periods, so each half holds more than one complete batch.
    """
    import math

    fitted = sorted(probe["fitted"], key=lambda r: r["batch_size"])
    batch_size = int(probe["chosen_batch_size"])

    def batch_seconds(n: int) -> float:
        for row in fitted:
            if row["batch_size"] >= n:
                return float(row["seconds"])
        return float(fitted[-1]["seconds"])

    planned, plan = [], []
    for point in points:
        in_batch = min(point.concurrency, batch_size)
        period = batch_seconds(in_batch)
        e2e = math.ceil(point.concurrency / batch_size) * period
        warmup = max(point.warmup_s, 5 * math.ceil((point.ramp_s + margin * e2e) / 5))
        cycles = WINDOW_DURATIONS if point.concurrency >= HIGH_CONCURRENCY else 3
        # Static batches complete together, so a cycle is one batch (ADR-022).
        window = max(point.window_s, 5 * math.ceil(cycles * period / 5))
        timeout = max(point.timeout_s, 3 * e2e)
        planned.append(
            BenchPoint(
                point.label,
                point.concurrency,
                point.ramp_s,
                warmup,
                window,
                timeout,
                point.diagnostic,
                adapt_ramp=False,
                adapt=False,
            )
        )
        plan.append(
            {
                "label": point.label,
                "concurrency": point.concurrency,
                "batch_rows": in_batch,
                "batch_seconds": period,
                "expected_e2e_s": e2e,
                "warmup_s": warmup,
                "window_s": window,
            }
        )
    return planned, plan
