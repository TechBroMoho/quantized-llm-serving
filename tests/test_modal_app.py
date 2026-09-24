"""Static checks on Modal settings; nothing here contacts Modal."""

from __future__ import annotations

import json

from llmbench.smoke import missing_flags
from modal_app.common import (
    BENCH_AWQ_FOLLOWUP_RESOURCES,
    BENCH_AWQ_RESOURCES,
    BENCH_BF16_RESOURCES,
    BENCH_GPTQ_RESOURCES,
    BENCH_GPTQ_TRIMMED_RESOURCES,
    BENCH_HF_NAIVE_RESOURCES,
    BENCH_HF_STATIC_RESOURCES,
    BENCH_MAXBATCH16_RESOURCES,
    BENCH_MAXBATCH64_RESOURCES,
    BENCH_PREPARE_RESOURCES,
    BENCH_PROBE_RESOURCES,
    DOWNLOAD_LARGE_RESOURCES,
    DOWNLOAD_RESOURCES,
    EVAL_FULL_RESOURCES,
    EVAL_ONE_RESOURCES,
    EVAL_PREFETCH_RESOURCES,
    EVAL_PROBE_RESOURCES,
    HF_CHECK_RESOURCES,
    HF_SMOKE_RESOURCES,
    LOADTEST_CHECK_RESOURCES,
    QUANT_PREP_RESOURCES,
    QUANTIZE_AWQ_RESOURCES,
    QUANTIZE_GPTQ_RESOURCES,
    SANITY_RESOURCES,
    VLLM_CHECK_RESOURCES,
    VLLM_SMOKE_RESOURCES,
    load_config,
)


def test_every_function_is_bounded_and_gpus_are_only_on_smokes() -> None:
    # Explicit envelopes: changing any of them must be a visible test change.
    expected = {
        DOWNLOAD_RESOURCES: (None, 600),
        DOWNLOAD_LARGE_RESOURCES: (None, 1500),
        QUANT_PREP_RESOURCES: (None, 1500),
        HF_CHECK_RESOURCES: (None, 120),
        VLLM_CHECK_RESOURCES: (None, 180),
        LOADTEST_CHECK_RESOURCES: (None, 600),
        HF_SMOKE_RESOURCES: ("L4", 600),
        VLLM_SMOKE_RESOURCES: ("L4", 900),
        QUANTIZE_AWQ_RESOURCES: ("L40S", 3600),
        QUANTIZE_GPTQ_RESOURCES: ("L40S", 4500),
        SANITY_RESOURCES: ("L40S", 1800),
        EVAL_PREFETCH_RESOURCES: (None, 1800),
        EVAL_PROBE_RESOURCES: ("L40S", 1200),
        EVAL_FULL_RESOURCES: ("L40S", 15000),
        EVAL_ONE_RESOURCES: ("L40S", 5400),
        BENCH_PREPARE_RESOURCES: (None, 1200),
        BENCH_PROBE_RESOURCES: ("L40S", 720),
        BENCH_AWQ_RESOURCES: ("L40S", 3600),
        BENCH_AWQ_FOLLOWUP_RESOURCES: ("L40S", 2200),
        BENCH_BF16_RESOURCES: ("L40S", 3600),
        BENCH_HF_NAIVE_RESOURCES: ("L40S", 1820),
        BENCH_HF_STATIC_RESOURCES: ("L40S", 3600),
        BENCH_GPTQ_RESOURCES: ("L40S", 3250),
        BENCH_GPTQ_TRIMMED_RESOURCES: ("L40S", 2120),
        BENCH_MAXBATCH16_RESOURCES: ("L40S", 1170),
        BENCH_MAXBATCH64_RESOURCES: ("L40S", 780),
    }
    for resources, (gpu, timeout) in expected.items():
        kwargs = resources.function_kwargs()
        assert resources.gpu == gpu
        assert kwargs["timeout"] == timeout
        assert 0 < kwargs["startup_timeout"] <= 300
        assert kwargs["retries"] == 0 and kwargs["max_containers"] == 1
        # Request == limit, so billing can never exceed the listed envelope.
        assert kwargs["cpu"][0] == kwargs["cpu"][1]
        assert kwargs["memory"][0] == kwargs["memory"][1]


def test_smoke_config_matches_function_gpu_and_fair_settings() -> None:
    config, _ = load_config("phase3_smoke.yaml")
    assert config["gpu"] == VLLM_SMOKE_RESOURCES.gpu
    assert len(config["model"]["revision"]) == 40
    args = config["vllm"]["engine_args"]
    assert "--no-enable-prefix-caching" in args
    assert args[args.index("--generation-config") + 1] == "vllm"
    assert "@sha256:" in config["vllm"]["image"]
    dockerfile = (load_config.__globals__["REPO"] / "docker" / "Dockerfile").read_text()
    assert f"FROM {config['vllm']['image']}" in dockerfile


def test_flag_check_matches_whole_flags_only() -> None:
    help_text = "  --seeds SEEDS\n  --enable-prefix-caching, --no-enable-prefix-caching"
    assert missing_flags(
        ["--seed", "--no-enable-prefix-caching", "--enable-prefix-caching"], help_text
    ) == ["--seed"]


def test_phase4_config_matches_official_examples() -> None:
    config, _ = load_config("phase4_quantize.yaml")
    assert config["gpu"] == QUANTIZE_AWQ_RESOURCES.gpu == SANITY_RESOURCES.gpu
    awq, gptq = config["variants"]["awq"], config["variants"]["gptq"]
    assert (awq["scheme"], awq["expected_symmetric"]) == ("W4A16_ASYM", False)
    assert (gptq["scheme"], gptq["expected_symmetric"]) == ("W4A16", True)
    assert awq["ignore"] == gptq["ignore"] == ["lm_head"]
    assert (
        awq["calibration"]["num_samples"],
        awq["calibration"]["max_seq_length"],
    ) == (
        256,
        512,
    )
    assert (
        gptq["calibration"]["num_samples"],
        gptq["calibration"]["max_seq_length"],
    ) == (512, 2048)
    for variant in (awq, gptq):
        assert len(variant["calibration"]["revision"]) == 40
    assert len(config["sanity"]["prompts"]) == 5
    assert "--no-enable-prefix-caching" in config["sanity"]["engine_args"]


def test_effective_packages_follow_import_precedence() -> None:
    from modal_app.evaluate import effective_packages

    entries = [
        ["aiohttp", "3.12.15", "/usr/local/lib/python3.12/dist-packages"],
        ["six", "1.17.0", "/usr/local/lib/python3.12/dist-packages"],
        ["aiohttp", "3.12.7", "/pkg"],
        ["Six", "1.16.0", "/usr/lib/python3/dist-packages"],
    ]
    effective, shadowed = effective_packages(entries)
    assert effective == {"aiohttp": "3.12.15", "six": "1.17.0"}
    assert [(item["name"], item["version"]) for item in shadowed] == [
        ("aiohttp", "3.12.7"),
        ("Six", "1.16.0"),
    ]


def test_bench_config_resolves_and_every_lifetime_has_a_function() -> None:
    from llmbench.bench import resolve_points
    from modal_app import bench

    config, _ = load_config("phase6_bench.yaml")
    assert config["gpu"] == "L40S"
    for name, spec in config["lifetimes"].items():
        points = resolve_points(config["points"], spec["points"], batch_size=32)
        assert points, name
        assert all(p.warmup_s >= p.ramp_s for p in points), name
    args = config["vllm"]["engine_args"]
    assert "--no-enable-prefix-caching" in args and "--enforce-eager" not in args
    # Every configured lifetime can be launched, and nothing else.
    assert set(bench._GPU_FUNCTIONS) == set(config["lifetimes"])
    # The expected hashes come from the committed Phase 4 records.
    plan = bench._prepare_run()
    assert set(plan["expected_checkpoints"]) == {"bf16", "awq", "gptq"}
    safetensors = [
        f
        for f in plan["expected_checkpoints"]["awq"]["files"]
        if f["path"].endswith(".safetensors")
    ]
    assert sum(f["bytes"] for f in safetensors) == 6_098_617_040


def _drive_hf_lifetime(monkeypatch, tmp_path, name: str) -> dict:
    """Run modal_app.bench._hf_lifetime with every Modal and GPU effect
    replaced: the OOM probe subprocess writes a fixed result (B = 32) and
    run_lifetime only records what it was given."""
    import subprocess
    from types import SimpleNamespace

    import llmbench.bench
    from modal_app import bench
    from modal_app.bench import _prepare_run

    probe = {"chosen_batch_size": 32, "fitted": [{"batch_size": 32, "seconds": 20.0}]}
    seen: dict = {"probe_calls": 0}

    def fake_subprocess_run(command, **kwargs):
        assert command[2] == "llmbench.baseline.oom_probe"
        seen["probe_calls"] += 1
        out = command[command.index("--out") + 1]
        with open(out, "w") as stream:
            json.dump(probe, stream)
        return subprocess.CompletedProcess(command, 0, "", "")

    async def fake_run_lifetime(**kwargs):
        seen["lifetime"] = kwargs
        return {"passed": True}

    run = _prepare_run() | {"drop": []}  # reads git, before subprocess is faked
    assert run["git"]["dirty"] == bool(run["git"]["dirty_files"])
    monkeypatch.setattr(bench, "RESULTS_PATH", str(tmp_path))
    monkeypatch.setattr(bench, "RESULTS", SimpleNamespace(commit=lambda: None))
    monkeypatch.setattr(bench, "_require_prepared", lambda run, variant: {})
    monkeypatch.setattr(bench.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(llmbench.bench, "run_lifetime", fake_run_lifetime)
    resources = {
        "hf-static": BENCH_HF_STATIC_RESOURCES,
        "hf-naive": BENCH_HF_NAIVE_RESOURCES,
    }[name]
    bench._hf_lifetime(run, name, resources)
    return seen


def test_hf_static_lifetime_resolves_points_from_the_oom_probe(
    monkeypatch, tmp_path
) -> None:
    """Regression: hf-static once resolved hf-cB before the probe gave B and
    crashed at startup ("hf-cB needs the static batch size")."""
    seen = _drive_hf_lifetime(monkeypatch, tmp_path, "hf-static")
    assert seen["probe_calls"] == 1
    lifetime = seen["lifetime"]
    assert [p.concurrency for p in lifetime["points"]] == [4, 16, 32, 64]
    command = lifetime["command"]
    assert command[command.index("--batch-size") + 1] == "32"
    assert lifetime["metadata"]["static_batch_size"] == 32
    (plan_path,) = (tmp_path / "phase6").glob("hf-static-*/static_plan.json")
    assert json.loads(plan_path.read_text())["batch_size"] == 32


def test_hf_naive_lifetime_runs_without_an_oom_probe(monkeypatch, tmp_path) -> None:
    seen = _drive_hf_lifetime(monkeypatch, tmp_path, "hf-naive")
    assert seen["probe_calls"] == 0
    assert [p.label for p in seen["lifetime"]["points"]] == ["c1", "hf-c4", "hf-c16"]
    assert "--batch-size" not in seen["lifetime"]["command"]
