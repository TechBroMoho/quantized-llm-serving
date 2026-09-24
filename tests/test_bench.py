"""CPU rehearsals of the Phase 6 benchmark driver (mock and tiny HF model)."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
from pathlib import Path

import numpy as np

from llmbench.bench import BenchPoint, run_lifetime, vllm_counters
from llmbench.prompts import NpyPromptPayloads
from tests.test_smoke import _port, _save_tiny_model

POINTS = [
    BenchPoint("c2", 2, ramp_s=0.1, warmup_s=0.4, window_s=1.2, timeout_s=10),
    BenchPoint("c4", 4, ramp_s=0.1, warmup_s=0.4, window_s=1.2, timeout_s=10),
]


def _pool(tmp_path: Path, length: int) -> str:
    rng = np.random.default_rng(0)
    pool = rng.integers(3, 32, size=(20000, length), dtype=np.int32)
    path = tmp_path / "prompts.npy"
    np.save(path, pool)
    return str(path)


def test_vllm_counters_sum_labelled_series() -> None:
    text = "\n".join(
        [
            "# HELP vllm:num_requests_running x",
            'vllm:num_requests_running{model_name="a"} 2.0',
            'vllm:num_requests_running{model_name="b"} 1.0',
            'vllm:prefix_cache_hits_total{model_name="a"} 0.0',
            "process_cpu_seconds_total 5",
        ]
    )
    counters = vllm_counters(text)
    assert counters["vllm:num_requests_running"] == 3.0
    assert counters["vllm:prefix_cache_hits_total"] == 0.0
    assert "process_cpu_seconds_total" not in counters


def _lifetime(tmp_path: Path, command: list[str], port: int, kind: str) -> dict:
    return asyncio.run(
        run_lifetime(
            label=kind,
            command=command,
            base_url=f"http://127.0.0.1:{port}",
            kind=kind,
            payloads=NpyPromptPayloads(
                _pool(tmp_path, 16), output_tokens=4, seed=0, model="tiny"
            ),
            points=POINTS,
            processes=2,
            out_dir=tmp_path / "out",
            input_tokens=16,
            output_tokens=4,
            health_timeout_s=120,
            idle_timeout_s=30,
            env={"HF_HUB_OFFLINE": "1"},
            # These rehearsals check the mechanics on a busy laptop with ~1 s
            # windows; the 5% default applies to real runs and is unit-tested
            # in test_steady_state.py.
            max_half_deviation=0.5,
        )
    )


def _check_saved(tmp_path: Path, summary: dict) -> None:
    assert summary["passed"], summary["failures"]
    saved = json.loads((tmp_path / "out" / "lifetime_summary.json").read_text())
    assert saved["state"] == "finished" and saved["passed"]
    ranges = []
    for point in POINTS:
        point_summary = json.loads(
            (tmp_path / "out" / point.label / "summary.json").read_text()
        )
        assert (tmp_path / "out" / point.label / "requests.jsonl.gz").exists()
        assert point_summary["passed"] and point_summary["output_tokens_in_window"] > 0
        ranges.append(point_summary["prompt_index_range"])
    # Each point uses fresh prompts: index ranges follow on without overlap.
    assert ranges[0][0] == 0 and ranges[0][1] == ranges[1][0] < ranges[1][1]


def test_mock_lifetime_runs_every_point_and_saves_it(tmp_path: Path) -> None:
    port = _port()
    command = [
        sys.executable,
        "-m",
        "llmbench.mock.server",
        "--port",
        str(port),
        "--ttft-ms",
        "5",
        "--itl-ms",
        "2",
        "--tokens",
        "4",
    ]
    _check_saved(tmp_path, _lifetime(tmp_path, command, port, "none"))


def test_hf_lifetime_on_cpu_cuts_windows_and_waits_for_idle(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _save_tiny_model(model)
    port = _port()
    command = [
        sys.executable,
        "-m",
        "llmbench.baseline.hf_server",
        "--model",
        str(model),
        "--mode",
        "static",
        "--batch-size",
        "2",
        "--batch-wait-ms",
        "20",
        "--dtype",
        "float32",
        "--port",
        str(port),
    ]
    summary = _lifetime(tmp_path, command, port, "hf")
    _check_saved(tmp_path, summary)
    c2, c4 = (
        json.loads((tmp_path / "out" / label / "summary.json").read_text())
        for label in ("c2", "c4")
    )
    # Both windows ended by cutting requests mid-stream (closed-loop users
    # always have one in flight); after c2's cuts, c4 began only once the
    # server's own /stats showed nothing queued or generating. The waiting
    # logic itself is tested with a scripted server below.
    assert c2["cut_at_window_end_requests"] > 0
    assert c4["cut_at_window_end_requests"] > 0
    stats = c4["server_before"]["stats"]
    assert stats["pending"] == 0 and not stats["busy"]
    assert stats["generated_batch_sizes"] and max(stats["generated_batch_sizes"]) <= 2


def test_wait_idle_waits_through_busy_and_unreachable_states(monkeypatch) -> None:
    import pytest

    from llmbench import bench

    # busy, unreachable (busy None: unknown, never idle), busy, then idle.
    states = iter([{"busy": 2}, {"busy": None}, {"busy": 1}, {"busy": 0}])
    calls = []

    def scripted(base_url: str, kind: str) -> dict:
        calls.append(kind)
        return next(states)

    monkeypatch.setattr(bench, "server_state", scripted)  # polls every 0.5 s
    state = asyncio.run(bench.wait_idle("http://x", "hf", timeout_s=60))
    assert state["busy"] == 0 and len(calls) == 4
    monkeypatch.setattr(bench, "server_state", lambda *_: {"busy": None})
    with pytest.raises(TimeoutError, match="still busy"):
        asyncio.run(bench.wait_idle("http://x", "vllm", timeout_s=0.01))


def test_unreachable_vllm_metrics_are_unknown_not_idle(monkeypatch) -> None:
    from llmbench import bench

    monkeypatch.setattr(
        bench, "fetch_text", lambda url: {"status": None, "text": "", "error": "x"}
    )
    state = bench.server_state("http://x", "vllm")
    assert state["busy"] is None and state["counters"] is None


def test_prefix_cache_check_needs_both_snapshots() -> None:
    from llmbench.bench import PREFIX_HITS, prefix_cache_failures

    def snap(hits: float | None) -> dict:
        counters = None if hits is None else {PREFIX_HITS: hits}
        return {"status": 200 if counters else None, "counters": counters}

    assert prefix_cache_failures(snap(5), snap(5)) == []
    assert "hits 3" in prefix_cache_failures(snap(5), snap(8))[0]
    # Formerly an unreachable /metrics made both counts 0 and the check passed.
    assert "unavailable" in prefix_cache_failures(snap(None), snap(8))[0]
    assert "unavailable" in prefix_cache_failures(snap(None), snap(None))[0]


def test_fetch_text_returns_resets_and_early_closes_as_failures() -> None:
    import threading

    from llmbench.smoke import fetch_text

    # A server that accepts, reads the request and closes without replying:
    # urlopen raises RemoteDisconnected, which is not a URLError.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve() -> None:
        connection, _ = listener.accept()
        connection.recv(1024)
        connection.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    port = listener.getsockname()[1]
    try:
        result = fetch_text(f"http://127.0.0.1:{port}/metrics", timeout_s=5)
    finally:
        thread.join(timeout=5)
        listener.close()
    assert result["status"] is None and "RemoteDisconnected" in result["error"]


def test_exhausted_prompt_pool_fails_the_point_loudly(tmp_path: Path) -> None:
    # Regression: a failed virtual user was once swallowed by gather().
    port = _port()
    command = [sys.executable, "-m", "llmbench.mock.server", "--port", str(port)]
    tiny = tmp_path / "tiny.npy"
    np.save(tiny, np.full((5, 16), 7, dtype=np.int32))
    summary = asyncio.run(
        run_lifetime(
            label="exhausted",
            command=command + ["--ttft-ms", "1", "--itl-ms", "1", "--tokens", "4"],
            base_url=f"http://127.0.0.1:{port}",
            kind="none",
            payloads=NpyPromptPayloads(str(tiny), output_tokens=4, seed=0, model="m"),
            points=POINTS[:1],
            processes=1,
            out_dir=tmp_path / "out",
            input_tokens=16,
            output_tokens=4,
            health_timeout_s=60,
            idle_timeout_s=10,
        )
    )
    assert not summary["passed"]
    assert any("prompt pool has 5 prompts" in f for f in summary["failures"])


def test_oom_probe_runs_the_server_generate_path_on_cpu(tmp_path: Path) -> None:
    import argparse

    from llmbench.baseline.oom_probe import run

    model = tmp_path / "model"
    _save_tiny_model(model)
    args = argparse.Namespace(
        model=str(model),
        prompts=_pool(tmp_path, 16),
        output_tokens=4,
        candidates="4,2",
        dtype="float32",
    )
    result = asyncio.run(run(args))
    assert [r["batch_size"] for r in result["fitted"]] == [2, 4]
    assert result["chosen_batch_size"] == 4 and result["first_oom"] is None
    assert result["input_tokens"] == 16


def test_points_resolve_repeats_and_the_static_batch_size() -> None:
    from llmbench.bench import resolve_points

    definitions = {
        "c1": {
            "concurrency": 1,
            "ramp_s": 0,
            "warmup_s": 5,
            "window_s": 9,
            "timeout_s": 60,
        },
        "hf-c2B": {
            "concurrency": "2B",
            "ramp_s": 1,
            "warmup_s": 5,
            "window_s": 9,
            "timeout_s": 60,
        },
    }
    points = resolve_points(definitions, ["c1", "hf-c2B", "c1-r2"], batch_size=24)
    assert [(p.label, p.concurrency) for p in points] == [
        ("c1", 1),
        ("hf-c2B", 48),
        ("c1-r2", 1),
    ]
    import pytest

    with pytest.raises(ValueError, match="static batch size"):
        resolve_points(definitions, ["hf-c2B"])
    with pytest.raises(ValueError, match="unique"):
        resolve_points(definitions, ["c1", "c1"])


def test_checkpoint_verification_catches_changed_and_missing_files(
    tmp_path: Path,
) -> None:
    from llmbench.bench import expected_checkpoint_files, verify_checkpoint
    from llmbench.quantization import sha256_file

    (tmp_path / "a.safetensors").write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    evidence = {
        "checkpoint": {
            "files": [
                {
                    "path": p,
                    "bytes": (tmp_path / p).stat().st_size,
                    "sha256": sha256_file(tmp_path / p),
                }
                for p in ("a.safetensors", "config.json")
            ]
        }
    }
    expected = expected_checkpoint_files(evidence)
    assert verify_checkpoint(tmp_path, expected)["passed"]
    (tmp_path / "a.safetensors").write_bytes(b"weightz")  # same size, new bytes
    (tmp_path / "config.json").unlink()
    result = verify_checkpoint(tmp_path, expected)
    assert not result["passed"]
    assert [m["path"] for m in result["mismatches"]] == ["a.safetensors", "config.json"]


def test_process_cpu_is_attributed_to_server_and_clients(tmp_path: Path) -> None:
    import os
    import shutil

    from llmbench.bench import cores_between, read_process_table

    ticks = os.sysconf("SC_CLK_TCK")

    def write(busy_ticks: int, procs: dict[int, tuple[str, int, int]]) -> dict:
        (tmp_path / "stat").write_text(f"cpu  {busy_ticks} 0 0 1000 0 0 0 0 0 0\n")
        for pid, (comm, pgrp, used) in procs.items():
            (tmp_path / str(pid)).mkdir(exist_ok=True)
            fields = ["S", "1", str(pgrp)] + ["0"] * 8 + [str(used), "0"]
            (tmp_path / str(pid) / "stat").write_text(
                f"{pid} ({comm}) {' '.join(fields)} 0 0\n"
            )
        return read_process_table(tmp_path)

    server, parent = ("EngineCore", 10), ("python3", 20)
    first = write(0, {11: (*server, 0), 20: (*parent, 0)})
    # A client worker (pid 30) appears mid-window and exits before the end.
    middle = write(
        2 * ticks,
        {
            11: (*server, ticks),
            20: (*parent, ticks // 2),
            30: ("python3", 20, ticks // 2),
        },
    )
    shutil.rmtree(tmp_path / "30")
    last = write(4 * ticks, {11: (*server, 2 * ticks), 20: (*parent, ticks)})
    assert first["processes"]["11"][:2] == ["EngineCore", 10]
    samples = [(0.0, first), (1.0, middle), (2.0, last)]
    result = cores_between(samples, 0.5, 1.5, server_pgid=10)
    assert result is not None
    assert result["server_cores"] == 1.0
    worker = next(r for r in result["processes"] if r["pid"] == 30)
    assert worker["cores"] == 0.5 and worker["observed_s"] == 1.0
    assert result["client_or_other_cores"] == 1.0  # parent 0.5 + worker 0.5
    assert result["container_busy_cores"] == 2.0
    assert result["processes"][0]["comm"] == "EngineCore"
    assert cores_between([(0.0, first)], 0.5, 1.5, 10) is None


def test_points_fail_when_client_headroom_is_below_3x(tmp_path: Path) -> None:
    port = _port()
    command = [
        sys.executable,
        "-m",
        "llmbench.mock.server",
        "--port",
        str(port),
        "--ttft-ms",
        "5",
        "--itl-ms",
        "2",
        "--tokens",
        "4",
    ]
    summary = asyncio.run(
        run_lifetime(
            label="headroom",
            command=command,
            base_url=f"http://127.0.0.1:{port}",
            kind="none",
            payloads=NpyPromptPayloads(
                _pool(tmp_path, 16), output_tokens=4, seed=0, model="m"
            ),
            points=POINTS[:1],
            processes=2,
            out_dir=tmp_path / "out",
            input_tokens=16,
            output_tokens=4,
            health_timeout_s=60,
            idle_timeout_s=10,
            max_half_deviation=0.5,
            validated_chunks_per_s=100.0,  # far below what the mock delivers
        )
    )
    assert not summary["passed"]
    assert any("headroom" in f and "< 3x" in f for f in summary["failures"])
    saved = json.loads((tmp_path / "out" / "c2" / "summary.json").read_text())
    assert saved["client_capacity_headroom"] < 3  # data kept, point not accepted


def test_static_points_follow_the_measured_batch_time() -> None:
    from llmbench.bench import static_points_from_probe

    probe = {
        "chosen_batch_size": 64,
        "fitted": [
            {"batch_size": 8, "seconds": 10.0},
            {"batch_size": 16, "seconds": 12.0},
            {"batch_size": 64, "seconds": 40.0},
        ],
    }
    base = [
        BenchPoint("c4", 4, ramp_s=5, warmup_s=20, window_s=120, timeout_s=600),
        BenchPoint("hf-c2B", 128, ramp_s=20, warmup_s=90, window_s=180, timeout_s=900),
    ]
    points, plan = static_points_from_probe(base, probe)
    c4, c2b = points
    assert plan[0]["batch_seconds"] == 10.0 and plan[0]["expected_e2e_s"] == 10.0
    assert c4.warmup_s == 20 and c4.window_s == 120  # config already long enough
    # 2B users wait for two 40 s batches: E2E 80 s -> warmup 20 + 1.25 * 80 = 120.
    assert plan[1]["expected_e2e_s"] == 80.0 and c2b.warmup_s == 120
    # c = 128 >= 128: the window covers 10 batch cycles of 40 s (ADR-022).
    assert c2b.window_s == 400 and c2b.timeout_s == 900
    assert not c2b.adapt and not c4.adapt  # timings are final; no re-adaptation
    slow = probe | {"fitted": [{"batch_size": 64, "seconds": 90.0}]}
    (slow_c4, slow_c2b), _ = static_points_from_probe(base, slow)
    assert slow_c2b.window_s == 900 and slow_c2b.warmup_s == 245
    assert slow_c4.window_s == 270  # below 128: three batch cycles


def test_perf_report_takes_medians_of_passed_repeats(tmp_path: Path) -> None:
    from llmbench.perf_report import headline, load_points

    def point(run: str, label: str, c: int, tps: float, tpot: float, ok=True):
        d = tmp_path / run / label
        d.mkdir(parents=True)
        pct = {"p50": tpot, "p90": tpot, "p95": tpot, "p99": tpot}
        d.joinpath("summary.json").write_text(
            json.dumps(
                {
                    "config": {"concurrency": c},
                    "passed": ok,
                    "failures": [],
                    "output_token_throughput_per_s": tps,
                    "request_throughput_per_s": 1.0,
                    "completed_in_window_requests": 10,
                    "ttft_s": pct,
                    "tpot_s": pct,
                    "itl_s": pct,
                    "e2e_s": pct,
                    "half_window_token_deviation": 0.01,
                }
            )
        )

    point("bf16-20260101T000000Z", "c1", 1, 50, 0.020)
    point("bf16-20260101T000000Z", "c1-r2", 1, 50, 0.022)
    point("bf16-20260101T000000Z", "c1-r3", 1, 50, 0.021)
    point("bf16-20260101T000000Z", "c256", 256, 3000, 0.1)
    point("awq-20260101T000000Z", "c1", 1, 100, 0.010)
    point("awq-20260101T000000Z", "c256", 256, 2000, 0.1)
    point("awq-20260101T000000Z", "c256-r2", 256, 9999, 0.1, ok=False)  # excluded
    point("hf-naive-20260101T000000Z", "c1", 1, 30, 0.03)
    point("prepare-20260101T000000Z", "x", 1, 1, 1)  # not a lifetime
    rows = load_points(tmp_path)
    assert len(rows) == 8
    head = headline(rows)
    assert head["single_user_decode"]["bf16"]["tpot"]["median"] == 0.021
    assert round(head["decode_speedup_vs_bf16"]["awq"], 2) == 2.1
    assert head["peak_output_tokens_per_s"]["awq"]["median"] == 2000
    assert head["engine_gain_vllm_bf16_vs_hf"]["naive"] == 100.0


def test_high_concurrency_timing_follows_the_expected_request_duration() -> None:
    from llmbench.bench import adapt_point, expected_e2e

    history = [
        {
            "label": "c64",
            "concurrency": 64,
            "e2e_p50": 8.19,
            "tpot_p50": 0.0317,
            "ttft_p50": 0.0828,
        },
        {
            "label": "c128",
            "concurrency": 128,
            "e2e_p50": 14.87,
            "tpot_p50": 0.0578,
            "ttft_p50": 0.162,
        },
    ]
    base = BenchPoint("c256", 256, ramp_s=30, warmup_s=80, window_s=180, timeout_s=600)
    # Extrapolated from c64/c128 (AWQ's real E2E at c256 was 35.3 s).
    e2e, source = expected_e2e(256, history, {}, 256)
    assert round(e2e, 1) == 29.7 and "x2" in source
    point, timing = adapt_point(base, history, {}, 256)
    assert point.ramp_s == 30 and point.warmup_s == 80 and point.window_s == 300
    # A measured seed wins over extrapolation.
    seeded, timing = adapt_point(base, history, {256: 35.3}, 256)
    assert seeded.ramp_s == 36 and seeded.warmup_s == 80  # max(80, 75)
    assert seeded.window_s == 355 and timing["source"].startswith("measured earlier")
    # A point already measured in this lifetime wins over the seed.
    again, timing = adapt_point(
        base,
        history + [dict(history[1], label="c256", concurrency=256, e2e_p50=40.0)],
        {256: 35.3},
        256,
    )
    assert again.window_s == 400 and timing["source"].startswith("measured here")
    # Below 128, diagnostics' ramp, and pre-derived timings are left alone.
    low = BenchPoint("c64", 64, 15, 40, 180, 600)
    assert adapt_point(low, history, {}, 256) == (low, None)
    sync = BenchPoint(
        "c256-sync", 256, 0, 80, 180, 600, diagnostic=True, adapt_ramp=False
    )
    adapted_sync, _ = adapt_point(sync, history, {256: 35.3}, 256)
    assert adapted_sync.ramp_s == 0 and adapted_sync.window_s == 355


def test_best_passing_point_is_repeated_after_the_sweep(tmp_path: Path) -> None:
    port = _port()
    command = [
        sys.executable,
        "-m",
        "llmbench.mock.server",
        "--port",
        str(port),
        "--ttft-ms",
        "5",
        "--itl-ms",
        "2",
        "--tokens",
        "4",
    ]
    summary = asyncio.run(
        run_lifetime(
            label="repeats",
            command=command,
            base_url=f"http://127.0.0.1:{port}",
            kind="none",
            payloads=NpyPromptPayloads(
                _pool(tmp_path, 16), output_tokens=4, seed=0, model="m"
            ),
            points=POINTS,
            processes=2,
            out_dir=tmp_path / "out",
            input_tokens=16,
            output_tokens=4,
            health_timeout_s=60,
            idle_timeout_s=10,
            max_half_deviation=0.5,
            repeat_best=2,
        )
    )
    labels = [p["label"] for p in summary["points"]]
    best = max(summary["points"][:2], key=lambda p: p["output_token_throughput_per_s"])
    assert labels == ["c2", "c4", f"{best['label']}-r2", f"{best['label']}-r3"]
    assert summary["passed"], summary["failures"]


def test_perf_report_merges_follow_ups_and_skips_diagnostics(tmp_path: Path) -> None:
    from llmbench.perf_report import headline, load_points

    def point(run: str, label: str, tps: float, ok=True, diagnostic=False):
        d = tmp_path / run / label
        d.mkdir(parents=True)
        pct = {"p50": 0.01, "p90": 0.01, "p95": 0.01, "p99": 0.01}
        d.joinpath("summary.json").write_text(
            json.dumps(
                {
                    "config": {"concurrency": 128},
                    "passed": ok,
                    "failures": [],
                    "diagnostic": diagnostic,
                    "output_token_throughput_per_s": tps,
                    "request_throughput_per_s": 1.0,
                    "completed_in_window_requests": 10,
                    "ttft_s": pct,
                    "tpot_s": pct,
                    "itl_s": pct,
                    "e2e_s": pct,
                    "half_window_token_deviation": 0.01,
                }
            )
        )

    point("awq-20260101T000000Z", "c128", 2200)
    point("awq-followup-20260102T000000Z", "c128-r2", 2100)
    point("awq-followup-20260102T000000Z", "c128-r3", 2300)
    point("awq-followup-20260102T000000Z", "c256-sync", 9000, diagnostic=True)
    head = headline(load_points(tmp_path))
    peak = head["peak_output_tokens_per_s"]["awq"]
    assert peak == {
        "median": 2200,
        "runs": 3,
        "min": 2100,
        "max": 2300,
        "point": "c128",
    }


def test_perf_report_peak_requests_are_chosen_by_requests_per_s(
    tmp_path: Path,
) -> None:
    from llmbench.perf_report import headline, load_points

    def point(run: str, label: str, c: int, tps: float, rps: float) -> None:
        d = tmp_path / run / label
        d.mkdir(parents=True)
        pct = {"p50": 0.01, "p90": 0.01, "p95": 0.01, "p99": 0.01}
        summary = {
            "config": {"concurrency": c},
            "passed": True,
            "failures": [],
            "output_token_throughput_per_s": tps,
            "request_throughput_per_s": rps,
            "completed_in_window_requests": 10,
            "ttft_s": pct,
            "tpot_s": pct,
            "itl_s": pct,
            "e2e_s": pct,
            "half_window_token_deviation": 0.01,
        }
        d.joinpath("summary.json").write_text(json.dumps(summary))

    # Tokens/s and requests/s peak at different points (edge effects).
    point("awq-20260101T000000Z", "c64", 64, 2000, 8.0)
    point("awq-20260101T000000Z", "c128", 128, 2100, 7.5)
    point("hf-naive-20260101T000000Z", "c4", 4, 40, 0.16)
    head = headline(load_points(tmp_path))
    assert head["peak_output_tokens_per_s"]["awq"]["point"] == "c128"
    assert head["peak_requests_per_s"]["awq"]["point"] == "c64"
    assert head["requests_per_s_ratios"]["awq_vs_hf-naive"] == 50.0
    assert head["requests_per_s_ratios"]["awq_vs_bf16"] is None  # no BF16 rows
