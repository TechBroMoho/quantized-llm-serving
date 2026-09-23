"""Phase 4: Qwen3-8B download, calibration prep, AWQ/GPTQ, vLLM sanity.

    uv run modal run --detach -m modal_app.quantize::prepare   # CPU only
    uv run modal run --detach -m modal_app.quantize::awq       # L40S
    uv run modal run --detach -m modal_app.quantize::gptq      # L40S
    uv run modal run --detach -m modal_app.quantize::sanity    # L40S, vLLM image

Each GPU entrypoint first checks, from the laptop, that its inputs exist on the
weights Volume, so a GPU is never attached to a job that cannot start.
"""

from __future__ import annotations

import json
from typing import Any

import modal

from modal_app.common import (
    DOWNLOAD_IMAGE,
    DOWNLOAD_LARGE_RESOURCES,
    HF_SECRET,
    OFFLINE_ENV,
    QUANT_IMAGE,
    QUANT_PREP_RESOURCES,
    QUANTIZE_AWQ_RESOURCES,
    QUANTIZE_GPTQ_RESOURCES,
    RESULTS,
    RESULTS_PATH,
    SANITY_RESOURCES,
    VLLM_CACHE,
    VLLM_CACHE_PATH,
    VLLM_IMAGE,
    WEIGHTS,
    WEIGHTS_PATH,
    Resources,
    load_config,
    manifest_path,
    model_dir,
    quantized_dir,
    run_stamp,
)

app = modal.App("llmbench-quantize")
CONFIG = "phase4_quantize.yaml"


def calibration_dir(config: dict[str, Any], variant: str) -> str:
    calibration = config["variants"][variant]["calibration"]
    return (
        f"{WEIGHTS_PATH}/calibration/{variant}-{config['model']['revision'][:8]}"
        f"-{calibration['revision'][:8]}"
    )


def _packages() -> dict[str, str]:
    from importlib.metadata import distributions

    return {
        dist.metadata["Name"]: dist.version
        for dist in distributions()
        if dist.metadata["Name"]
    }


@app.function(
    image=DOWNLOAD_IMAGE,
    volumes={WEIGHTS_PATH: WEIGHTS, RESULTS_PATH: RESULTS},
    secrets=[HF_SECRET],
    **DOWNLOAD_LARGE_RESOURCES.function_kwargs(),
)
def download_8b(model_id: str, revision: str, stamp: str) -> dict[str, Any]:
    """CPU only: the same verified download as Phase 3, with a longer timeout."""
    from modal_app.download import download_and_verify

    return download_and_verify(model_id, revision, stamp, phase="phase4")


@app.function(
    image=QUANT_IMAGE,
    volumes={WEIGHTS_PATH: WEIGHTS, RESULTS_PATH: RESULTS},
    secrets=[HF_SECRET],
    **QUANT_PREP_RESOURCES.function_kwargs(),
)
def prepare_calibration_data(config: dict[str, Any], stamp: str) -> dict[str, Any]:
    """CPU: build each variant's tokenized calibration set from the 8B tokenizer."""
    from pathlib import Path

    from transformers import AutoTokenizer

    from llmbench.quantization import prepare_calibration

    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir(model["id"], model["revision"]))
    out = Path(RESULTS_PATH) / "phase4" / f"calibration-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, variant in config["variants"].items():
        target = Path(calibration_dir(config, name))
        summary = prepare_calibration(tokenizer, variant["calibration"], target)
        summary["path"] = str(target)
        results[name] = summary
        (out / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
        WEIGHTS.commit()
        RESULTS.commit()
    (out / "packages.json").write_text(json.dumps(_packages(), indent=2) + "\n")
    RESULTS.commit()
    return results


def _quantize(run: dict[str, Any], variant_name: str, resources: Resources) -> Any:
    import shutil
    import time
    from pathlib import Path

    from loguru import logger

    from llmbench.quantization import (
        check_checkpoint,
        inspect_tensors,
        run_quantization,
    )
    from llmbench.smoke import gpu_metadata, write_json

    config = run["config"]
    model = config["model"]
    variant = config["variants"][variant_name]
    out = Path(RESULTS_PATH) / "phase4" / f"quantize-{variant_name}-{run['stamp']}"
    final = Path(quantized_dir(config, variant_name))
    manifest_file = Path(WEIGHTS_PATH) / "manifests" / f"{final.name}.json"
    # The verified manifest, written last, is the only completion marker.
    if manifest_file.exists():
        raise RuntimeError(f"{final} is already verified; refusing to overwrite it")
    shutil.rmtree(final, ignore_errors=True)  # leftovers of a failed attempt
    out.mkdir(parents=True, exist_ok=True)
    logger.add(str(out / "llmcompressor.log"), level="INFO")
    record: dict[str, Any] = {
        "variant_name": variant_name,
        "config": config,
        "config_sha256": run["config_sha256"],
        "git": run["git"],
        "model_manifest": run["manifest"],
        "resources": resources.as_dict(),
        "gpu": gpu_metadata(),
        "packages": _packages(),
        "state": "started",
    }

    def progress(stage: str, data: dict[str, Any]) -> None:
        record.update(data)
        record["state"] = stage
        write_json(out / "summary.json", record)
        RESULTS.commit()

    progress("started", {})
    started = time.perf_counter()
    failures: list[str] = []
    try:
        summary = run_quantization(
            model_path=model_dir(model["id"], model["revision"]),
            calibration_dir=Path(calibration_dir(config, variant_name)),
            out_dir=final,
            variant=variant,
            calibration=variant["calibration"],
            progress=progress,
        )
        record.update(summary)
        failures += check_checkpoint(summary, variant)
        record["tensors"] = inspect_tensors(final, bool(variant["expected_symmetric"]))
        failures += record["tensors"]["failures"]
        record["checkpoint_path"] = str(final)
        if not failures:
            manifest = {
                **summary["checkpoint"],
                "verified": True,
                "source_run": f"quantize-{variant_name}-{run['stamp']}",
            }
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        WEIGHTS.commit()
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["function_elapsed_s"] = time.perf_counter() - started
        record["failures"] = failures
        record["passed"] = not failures
        progress("finished", {})
    if failures:
        raise RuntimeError(f"{variant_name} checkpoint failed checks: {failures}")
    return {k: v for k, v in record.items() if k not in {"config", "packages"}}


@app.function(
    image=QUANT_IMAGE,
    env=OFFLINE_ENV,
    volumes={WEIGHTS_PATH: WEIGHTS, RESULTS_PATH: RESULTS},
    **QUANTIZE_AWQ_RESOURCES.function_kwargs(),
)
def quantize_awq(run: dict[str, Any]) -> Any:
    return _quantize(run, "awq", QUANTIZE_AWQ_RESOURCES)


@app.function(
    image=QUANT_IMAGE,
    env=OFFLINE_ENV,
    volumes={WEIGHTS_PATH: WEIGHTS, RESULTS_PATH: RESULTS},
    **QUANTIZE_GPTQ_RESOURCES.function_kwargs(),
)
def quantize_gptq(run: dict[str, Any]) -> Any:
    return _quantize(run, "gptq", QUANTIZE_GPTQ_RESOURCES)


@app.function(
    image=VLLM_IMAGE,
    env=OFFLINE_ENV,
    volumes={
        WEIGHTS_PATH: WEIGHTS,
        RESULTS_PATH: RESULTS,
        VLLM_CACHE_PATH: VLLM_CACHE,
    },
    **SANITY_RESOURCES.function_kwargs(),
)
def vllm_sanity(run: dict[str, Any], variants: list[str]) -> dict[str, Any]:
    """Load each variant in vLLM and complete the five fixed prompts greedily."""
    from pathlib import Path

    from llmbench.sanity import run_sanity
    from llmbench.smoke import verify_vllm_flags, write_json

    config = run["config"]
    model = config["model"]
    sanity = config["sanity"]
    out = Path(RESULTS_PATH) / "phase4" / f"sanity-{run['stamp']}"
    flag_failures = verify_vllm_flags(sanity["engine_args"], out)
    if flag_failures:
        write_json(out / "flag_failures.json", flag_failures)
        RESULTS.commit()
        raise RuntimeError(f"engine flags failed verification: {flag_failures}")
    paths = {"bf16": model_dir(model["id"], model["revision"])}
    paths |= {name: quantized_dir(config, name) for name in config["variants"]}
    results = {}
    for name in variants:
        results[name] = run_sanity(
            label=name,
            model_path=paths[name],
            served_name=f"{model['id']}-{name}",
            sanity=sanity,
            out_dir=out / name,
            metadata={"git": run["git"], "config_sha256": run["config_sha256"]},
        )
        RESULTS.commit()
    VLLM_CACHE.commit()
    failed = {k: v["failures"] for k, v in results.items() if not v["passed"]}
    if failed:
        raise RuntimeError(f"sanity failed: {failed}")
    return {
        k: {key: v[key] for key in ("passed", "log_excerpts", "checkpoint_bytes")}
        for k, v in results.items()
    }


def _prepare_run(require: list[str]) -> dict[str, Any]:
    """Laptop pre-flight: config, git state, and required Volume inputs."""
    import subprocess

    from llmbench.smoke import config_sha256

    config, text = load_config(CONFIG)
    model = config["model"]
    raw = _read(manifest_path(model["id"], model["revision"]))
    if raw is None or not json.loads(raw).get("verified"):
        raise SystemExit(
            "no verified Qwen3-8B manifest; run modal_app.quantize::prepare"
        )
    for path in require:
        if _read(path) is None:
            raise SystemExit(f"missing required Volume file {path}")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    manifest = json.loads(raw)
    return {
        "config": config,
        "config_yaml": text,
        "config_sha256": config_sha256(config),
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "dirty_tracked": bool(git("status", "-s", "--untracked-files=no")),
        },
        "manifest": {
            k: v for k, v in manifest.items() if k not in {"files", "failures"}
        },
        "stamp": run_stamp(),
    }


def _read(path: str) -> bytes | None:
    try:
        return b"".join(WEIGHTS.read_file(path))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None


def _volume_path(absolute: str) -> str:
    return absolute.removeprefix(WEIGHTS_PATH + "/")


@app.local_entrypoint()
def prepare() -> None:
    """CPU only: download + verify Qwen3-8B, then build calibration sets."""
    config, _ = load_config(CONFIG)
    model = config["model"]
    stamp = run_stamp()
    if _read(manifest_path(model["id"], model["revision"])) is None:
        manifest = download_8b.remote(model["id"], model["revision"], stamp)
        print(
            f"downloaded {manifest['total_bytes']} bytes, "
            f"verified={manifest['verified']}"
        )
    for name, summary in prepare_calibration_data.remote(config, stamp).items():
        keys = ("samples", "tokens_total", "tokens_min", "tokens_max", "path")
        print(name, json.dumps({k: summary[k] for k in keys}))


def _calibration_file(variant: str) -> str:
    config, _ = load_config(CONFIG)
    return _volume_path(calibration_dir(config, variant)) + "/calibration.json"


@app.local_entrypoint()
def awq() -> None:
    run = _prepare_run([_calibration_file("awq")])
    print(json.dumps(quantize_awq.remote(run), indent=2, default=str)[:4000])


@app.local_entrypoint()
def gptq() -> None:
    run = _prepare_run([_calibration_file("gptq")])
    print(json.dumps(quantize_gptq.remote(run), indent=2, default=str)[:4000])


@app.local_entrypoint()
def sanity(variants: str = "bf16,awq,gptq") -> None:
    config, _ = load_config(CONFIG)
    names = [name.strip() for name in variants.split(",") if name.strip()]
    required = [
        f"manifests/{quantized_dir(config, name).rsplit('/', 1)[-1]}.json"
        for name in names
        if name != "bf16"
    ]
    run = _prepare_run(required)
    print(json.dumps(vllm_sanity.remote(run, names), indent=2))
