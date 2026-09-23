"""Static checks on Modal settings; nothing here contacts Modal."""

from __future__ import annotations

from modal_app.checks import missing_flags
from modal_app.common import (
    DOWNLOAD_RESOURCES,
    HF_CHECK_RESOURCES,
    HF_SMOKE_RESOURCES,
    VLLM_CHECK_RESOURCES,
    VLLM_SMOKE_RESOURCES,
    load_config,
)


def test_every_function_is_bounded_and_gpus_are_only_on_smokes() -> None:
    for resources in (
        DOWNLOAD_RESOURCES,
        HF_CHECK_RESOURCES,
        VLLM_CHECK_RESOURCES,
        HF_SMOKE_RESOURCES,
        VLLM_SMOKE_RESOURCES,
    ):
        kwargs = resources.function_kwargs()
        assert 0 < kwargs["timeout"] <= 900
        assert 0 < kwargs["startup_timeout"] <= 300
        assert kwargs["retries"] == 0 and kwargs["max_containers"] == 1
        # Request == limit, so billing can never exceed the listed envelope.
        assert kwargs["cpu"][0] == kwargs["cpu"][1]
        assert kwargs["memory"][0] == kwargs["memory"][1]
    for cpu_only in (DOWNLOAD_RESOURCES, HF_CHECK_RESOURCES, VLLM_CHECK_RESOURCES):
        assert cpu_only.gpu is None
    assert HF_SMOKE_RESOURCES.gpu == VLLM_SMOKE_RESOURCES.gpu == "L4"


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
