"""CPU rehearsals of the Phase 6 benchmark driver (mock and tiny HF model)."""

from __future__ import annotations

import asyncio
import json
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
    c4 = json.loads((tmp_path / "out" / "c4" / "summary.json").read_text())
    # c2 ended with requests cut mid-stream; c4 still started from an idle
    # server, and the HF server reported the jobs whose clients had gone.
    assert c4["idle_before"]["busy"] == 0
    assert c4["cut_at_window_end_requests"] > 0


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
    assert c2b.window_s == 180 and c2b.timeout_s == 900
    slow = probe | {"fitted": [{"batch_size": 64, "seconds": 90.0}]}
    (_, slow_c2b), _ = static_points_from_probe(base, slow)
    assert slow_c2b.window_s == 270 and slow_c2b.warmup_s == 245
