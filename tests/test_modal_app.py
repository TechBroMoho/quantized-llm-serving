"""Static checks on Modal settings; nothing here contacts Modal."""

from __future__ import annotations

from llmbench.smoke import missing_flags
from modal_app.common import (
    DOWNLOAD_LARGE_RESOURCES,
    DOWNLOAD_RESOURCES,
    EVAL_FULL_RESOURCES,
    EVAL_PREFETCH_RESOURCES,
    EVAL_PROBE_RESOURCES,
    HF_CHECK_RESOURCES,
    HF_SMOKE_RESOURCES,
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
        HF_SMOKE_RESOURCES: ("L4", 600),
        VLLM_SMOKE_RESOURCES: ("L4", 900),
        QUANTIZE_AWQ_RESOURCES: ("L40S", 3600),
        QUANTIZE_GPTQ_RESOURCES: ("L40S", 4500),
        SANITY_RESOURCES: ("L40S", 1800),
        EVAL_PREFETCH_RESOURCES: (None, 1800),
        EVAL_PROBE_RESOURCES: ("L40S", 1200),
        EVAL_FULL_RESOURCES: ("L40S", 15000),
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
