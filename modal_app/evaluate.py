"""Phase 5: lm-eval accuracy (MMLU 5-shot, WikiText-2) for BF16, AWQ and GPTQ.

    uv run modal run --detach -m modal_app.evaluate::prefetch  # CPU only
    uv run modal run --detach -m modal_app.evaluate::probe     # L40S, --limit
    uv run modal run --detach -m modal_app.evaluate::full      # L40S, 3 variants

`prefetch` checks that installing lm-eval left the vLLM image's packages
untouched, downloads the two datasets once into the weights Volume, reloads
them offline and measures every request's token length with lm-eval's own
request construction. GPU functions run fully offline from that copy, so every
variant is evaluated on byte-identical data. Each local entrypoint checks its
inputs on the Volume first, so a GPU is never attached to a job that cannot
start.
"""

from __future__ import annotations

import json
from typing import Any

import modal

from modal_app.common import (
    EVAL_FULL_RESOURCES,
    EVAL_IMAGE,
    EVAL_PREFETCH_RESOURCES,
    EVAL_PROBE_RESOURCES,
    HF_SECRET,
    OFFLINE_ENV,
    RESULTS,
    RESULTS_PATH,
    WEIGHTS,
    WEIGHTS_PATH,
    Resources,
    load_config,
    manifest_path,
    model_dir,
    quantized_dir,
    run_stamp,
)

app = modal.App("llmbench-evaluate")
CONFIG = "phase5_accuracy.yaml"
EVAL_CACHE = f"{WEIGHTS_PATH}/eval-cache/hf-home"
EVAL_CACHE_MANIFEST = "eval-cache/manifest.json"  # relative to the weights Volume
LOCAL_HF_HOME = "/tmp/hf-eval"
TASKS = ("mmlu", "wikitext")
# Per lm-eval process; the function timeout bounds the whole run.
PROBE_TASK_TIMEOUT_S = 600
FULL_TASK_TIMEOUT_S = 5400
IMAGE_CONSTRAINTS = "/opt/llmbench/requirements/vllm-image-constraints.txt"
RECORDED_PACKAGES = (
    "lm_eval",
    "vllm",
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "evaluate",
    "compressed-tensors",
    "huggingface-hub",
)


def checkpoint_paths(quant_config: dict[str, Any]) -> dict[str, str]:
    model = quant_config["model"]
    paths = {"bf16": model_dir(model["id"], model["revision"])}
    paths |= {
        name: quantized_dir(quant_config, name) for name in quant_config["variants"]
    }
    return paths


def offline_env(hf_home: str) -> dict[str, str]:
    return {
        **OFFLINE_ENV,
        "HF_HOME": hf_home,
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


# Lists every distribution in sys.path order, as the lm-eval subprocess sees
# it. Modal's function process also carries Modal's own client packages, and
# the image has Ubuntu's system dist-packages; the first prefetch mistook
# those shadowed copies for changes to the image.
_LIST_DISTRIBUTIONS = """
import json
from importlib.metadata import distributions
print(json.dumps([[d.metadata["Name"], d.version, str(d._path.parent)]
                  for d in distributions() if d.metadata["Name"]]))
"""


def effective_packages(
    entries: list[list[str]],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """The version Python imports for each package, plus shadowed copies.

    `entries` are (name, version, location) in sys.path order; the first
    occurrence of a name is the one `import` resolves.
    """
    import re

    effective: dict[str, str] = {}
    first: dict[str, tuple[str, str]] = {}
    shadowed = []
    for name, version, location in entries:
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in first:
            shadowed.append(
                {
                    "name": name,
                    "version": version,
                    "location": location,
                    "shadowed_by": f"{first[key][0]} in {first[key][1]}",
                }
            )
            continue
        first[key] = (version, location)
        effective[name] = version
    return effective, shadowed


def _installed(env: dict[str, str]) -> list[list[str]]:
    """Distributions visible to a subprocess with lm-eval's environment."""
    import os
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-c", _LIST_DISTRIBUTIONS],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=True,
        timeout=120,
    )
    entries: list[list[str]] = json.loads(done.stdout)
    return entries


def _run(command: list[str], log: Any, env: dict[str, str], timeout_s: float) -> int:
    import os
    import subprocess

    done = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ, **env},
        timeout=timeout_s,
        check=False,
    )
    return done.returncode


def _copy_tree(source: str, target: str) -> int:
    """Replace `target` with a copy of `source`; returns the bytes copied."""
    import shutil
    from pathlib import Path

    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target)
    return sum(p.stat().st_size for p in Path(target).rglob("*") if p.is_file())


@app.function(
    image=EVAL_IMAGE,
    volumes={WEIGHTS_PATH: WEIGHTS, RESULTS_PATH: RESULTS},
    secrets=[HF_SECRET],
    **EVAL_PREFETCH_RESOURCES.function_kwargs(),
)
def eval_prefetch(run: dict[str, Any]) -> dict[str, Any]:
    """CPU: image check, dataset download, offline reload and length audit."""
    import subprocess
    import sys
    import time
    from pathlib import Path

    from llmbench.accuracy import package_drift
    from llmbench.smoke import package_versions, write_json

    out = Path(RESULTS_PATH) / "phase5" / f"prefetch-{run['stamp']}"
    out.mkdir(parents=True, exist_ok=True)
    entries = _installed(offline_env(LOCAL_HF_HOME))
    installed, shadowed = effective_packages(entries)
    constraints = Path(IMAGE_CONSTRAINTS).read_text(encoding="utf-8")
    record: dict[str, Any] = {
        "run": {k: v for k, v in run.items() if k != "config_yaml"},
        "resources": EVAL_PREFETCH_RESOURCES.as_dict(),
        "packages": package_versions(RECORDED_PACKAGES),
        "image_package_drift": package_drift(installed, constraints),
        "shadowed_distributions": shadowed,
        "failures": [],
        "steps": {},
    }
    record["failures"] += [
        f"image package changed: {d}" for d in record["image_package_drift"]
    ]
    write_json(
        out / "installed_packages.json", {"effective": installed, "all": entries}
    )
    help_text = subprocess.run(
        ["lm-eval", "run", "--help"], capture_output=True, text=True, check=False
    )
    (out / "lm_eval_run_help.txt").write_text(help_text.stdout + help_text.stderr)

    def step(
        name: str, command: list[str], env: dict[str, str], timeout_s: float
    ) -> int:
        started = time.time()
        with (out / f"{name}.log").open("w") as log:
            code = _run(command, log, env, timeout_s)
        record["steps"][name] = {
            "command": command,
            "returncode": code,
            "elapsed_s": round(time.time() - started, 1),
        }
        if code != 0:
            record["failures"].append(f"{name} exited {code}")
        write_json(out / "summary.json", record)
        RESULTS.commit()
        return code

    # 1. Online: lm-eval's own task construction downloads exactly what it loads.
    download_home = "/tmp/hf-download"
    download = (
        "from lm_eval.tasks import TaskManager, get_task_dict; "
        f"get_task_dict({list(TASKS)!r}, TaskManager())"
    )
    online_env = {
        "HF_HOME": download_home,
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if step("download", [sys.executable, "-c", download], online_env, 1200) == 0:
        for secret_file in ("token", "stored_tokens"):
            Path(download_home, secret_file).unlink(missing_ok=True)
        cache_bytes = _copy_tree(download_home, EVAL_CACHE)
        datasets_dir = Path(EVAL_CACHE) / "datasets"
        record["eval_cache"] = {
            "path": EVAL_CACHE,
            "bytes": cache_bytes,
            # Cache folders are named <dataset>/<config>/<version>/<hub commit>.
            "dataset_dirs": sorted(
                str(p.relative_to(datasets_dir))
                for p in datasets_dir.glob("*/*/*/*")
                if p.is_dir()
            ),
        }
        WEIGHTS.commit()

        # 2. Offline, from a copy of the Volume cache, exactly as GPU runs do:
        #    the full-length audit, then the probe's token volume.
        _copy_tree(EVAL_CACHE, LOCAL_HF_HOME)
        config = run["config"]
        tokenizer = checkpoint_paths(run["quant_config"])["bf16"]
        max_length = str(config["lm_eval"]["model_args"]["max_model_len"])
        audit = [sys.executable, "-m", "llmbench.eval_audit", "--tokenizer", tokenizer]
        audit += ["--max-length", max_length]
        step(
            "audit_full",
            [*audit, "--out", str(out / "prompt_lengths_full.json")],
            offline_env(LOCAL_HF_HOME),
            1200,
        )
        probe_limits = [f"{t}={n}" for t, n in config["probe"]["limit"].items()]
        step(
            "audit_probe",
            [*audit, "--out", str(out / "prompt_lengths_probe.json")]
            + [arg for item in probe_limits for arg in ("--limit", item)],
            offline_env(LOCAL_HF_HOME),
            600,
        )
        for name in ("full", "probe"):
            path = out / f"prompt_lengths_{name}.json"
            if path.exists():
                report = json.loads(path.read_text())
                record[f"prompt_lengths_{name}"] = report
                if not report["fits"]:
                    record["failures"].append(
                        f"{name}: prompts exceed max_model_len - 1"
                    )
    record["passed"] = not record["failures"]
    write_json(out / "summary.json", record)
    RESULTS.commit()
    if record["passed"]:
        manifest = {
            "verified": True,
            "source_run": f"prefetch-{run['stamp']}",
            **record["eval_cache"],
            "packages": record["packages"],
            "prompt_lengths_full": record["prompt_lengths_full"]["tasks"],
        }
        target = Path(WEIGHTS_PATH) / EVAL_CACHE_MANIFEST
        target.write_text(json.dumps(manifest, indent=2) + "\n")
        WEIGHTS.commit()
    return {k: record[k] for k in ("passed", "failures", "packages", "steps")}


def _evaluate(
    run: dict[str, Any],
    variants: list[str],
    limits: dict[str, int | None],
    label: str,
    resources: Resources,
    task_timeout_s: float,
) -> dict[str, Any]:
    from pathlib import Path

    from llmbench.accuracy import run_variant, variant_model_args
    from llmbench.quantization import GpuMemorySampler
    from llmbench.smoke import gpu_metadata, package_versions, write_json

    config = run["config"]
    settings = config["lm_eval"]
    paths = checkpoint_paths(run["quant_config"])
    out = Path(RESULTS_PATH) / "phase5" / f"{label}-{run['stamp']}"
    cache_bytes = _copy_tree(EVAL_CACHE, LOCAL_HF_HOME)
    meta = {
        "run": {k: v for k, v in run.items() if k != "config_yaml"},
        "label": label,
        "variants": variants,
        "limits": limits,
        "resources": resources.as_dict(),
        "gpu": gpu_metadata(),
        "packages": package_versions(RECORDED_PACKAGES),
        "eval_cache_bytes": cache_bytes,
        "checkpoints": paths,
    }
    write_json(out / "run.json", meta)
    RESULTS.commit()
    results = {}
    for name in variants:
        with GpuMemorySampler() as sampler:
            record = run_variant(
                settings,
                label=name,
                # Every variant uses the BF16 tokenizer files (config: tokenizer).
                model_args=variant_model_args(
                    settings, paths[name], paths[settings["tokenizer"]]
                ),
                out_dir=out / name,
                limits=limits,
                timeout_s=task_timeout_s,
                env=offline_env(LOCAL_HF_HOME),
                on_progress=RESULTS.commit,
            )
        record["gpu_peak_mib_nvidia_smi"] = sampler.peak_mib
        record["gpu_memory_samples"] = sampler.samples
        write_json(out / name / "summary.json", record)
        RESULTS.commit()
        results[name] = {
            key: record.get(key)
            for key in ("passed", "failures", "gpu_peak_mib_nvidia_smi")
        } | {
            task: {
                "elapsed_s": record["tasks"].get(task, {}).get("elapsed_s"),
                "loglikelihood_progress": record["tasks"]
                .get(task, {})
                .get("loglikelihood_progress"),
            }
            for task in TASKS
        }
    failed = {k: v["failures"] for k, v in results.items() if not v["passed"]}
    if failed:
        raise RuntimeError(f"evaluation failed: {failed}")
    return results


@app.function(
    image=EVAL_IMAGE,
    env=OFFLINE_ENV,
    volumes={WEIGHTS_PATH: WEIGHTS, RESULTS_PATH: RESULTS},
    **EVAL_PROBE_RESOURCES.function_kwargs(),
)
def eval_probe(run: dict[str, Any]) -> dict[str, Any]:
    """One variant with `--limit`, to time the full run before approving it."""
    probe = run["config"]["probe"]
    limits = {task: probe["limit"].get(task) for task in TASKS}
    return _evaluate(
        run,
        [probe["variant"]],
        limits,
        "probe",
        EVAL_PROBE_RESOURCES,
        PROBE_TASK_TIMEOUT_S,
    )


@app.function(
    image=EVAL_IMAGE,
    env=OFFLINE_ENV,
    volumes={WEIGHTS_PATH: WEIGHTS, RESULTS_PATH: RESULTS},
    **EVAL_FULL_RESOURCES.function_kwargs(),
)
def eval_full(run: dict[str, Any], variants: list[str]) -> dict[str, Any]:
    """Full MMLU 5-shot and WikiText-2 for each variant, saved after each task."""
    limits: dict[str, int | None] = {task: None for task in TASKS}
    return _evaluate(
        run, variants, limits, "full", EVAL_FULL_RESOURCES, FULL_TASK_TIMEOUT_S
    )


def _read(path: str) -> bytes | None:
    try:
        return b"".join(WEIGHTS.read_file(path))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None


def _prepare_run(require_eval_cache: bool) -> dict[str, Any]:
    """Laptop pre-flight: configs, git state and required Volume inputs."""
    import subprocess

    from llmbench.smoke import config_sha256

    config, text = load_config(CONFIG)
    quant_config, _ = load_config(config["checkpoints"])
    model = quant_config["model"]
    required = [manifest_path(model["id"], model["revision"])]
    required += [
        f"manifests/{quantized_dir(quant_config, name).rsplit('/', 1)[-1]}.json"
        for name in quant_config["variants"]
    ]
    if require_eval_cache:
        required.append(EVAL_CACHE_MANIFEST)
    for path in required:
        raw = _read(path)
        if raw is None or not json.loads(raw).get("verified"):
            raise SystemExit(f"missing or unverified Volume file {path}")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "config": config,
        "config_yaml": text,
        "config_sha256": config_sha256(config),
        "quant_config": quant_config,
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "dirty_tracked": bool(git("status", "-s", "--untracked-files=no")),
        },
        "stamp": run_stamp(),
    }


@app.local_entrypoint()
def prefetch() -> None:
    run = _prepare_run(require_eval_cache=False)
    print(json.dumps(eval_prefetch.remote(run), indent=2))


@app.local_entrypoint()
def probe() -> None:
    run = _prepare_run(require_eval_cache=True)
    print(json.dumps(eval_probe.remote(run), indent=2))


@app.local_entrypoint()
def full(variants: str = "bf16,awq,gptq") -> None:
    run = _prepare_run(require_eval_cache=True)
    names = [name.strip() for name in variants.split(",") if name.strip()]
    unknown = set(names) - set(run["config"]["variants"])
    if unknown:
        raise SystemExit(f"unknown variants {sorted(unknown)}")
    print(json.dumps(eval_full.remote(run, names), indent=2))
