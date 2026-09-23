"""Shared Modal images, Volumes and resource limits for every llmbench app.

Resource limits are (request, limit) pairs so a function can never be billed
for more CPU or memory than listed here. Every function has an execution
timeout, a startup timeout, no retries and at most one container.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import modal

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO / "configs"

WEIGHTS_PATH = "/weights"
RESULTS_PATH = "/results"
VLLM_CACHE_PATH = "/root/.cache/vllm"

WEIGHTS = modal.Volume.from_name("llmbench-weights", create_if_missing=True)
RESULTS = modal.Volume.from_name("llmbench-results", create_if_missing=True)
VLLM_CACHE = modal.Volume.from_name("llmbench-vllm-cache", create_if_missing=True)

HF_SECRET = modal.Secret.from_name("huggingface")

# GPU containers must never reach the Hub: weights come from the Volume.
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HOME": "/tmp/hf-home",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "PYTHONUNBUFFERED": "1",
}

# Built from our Dockerfile; the inherited api_server ENTRYPOINT must be
# cleared so Modal can start its own container entrypoint.
_VLLM_BASE = modal.Image.from_dockerfile(
    REPO / "docker" / "Dockerfile", context_dir=REPO
).entrypoint([])
VLLM_IMAGE = _VLLM_BASE.add_local_python_source("modal_app")

# Phase 5: the same serving image plus lm-eval. The resolved set only adds
# packages; the image's own freeze is a pip constraint, so vLLM, torch and
# transformers cannot move (checked again inside the image by eval_env).
_EVAL_REQ = "/opt/llmbench/requirements"
EVAL_IMAGE = (
    _VLLM_BASE.add_local_file(
        REPO / "requirements" / "eval-linux.txt",
        f"{_EVAL_REQ}/eval-linux.txt",
        copy=True,
    )
    .add_local_file(
        REPO / "requirements" / "vllm-image-constraints.txt",
        f"{_EVAL_REQ}/vllm-image-constraints.txt",
        copy=True,
    )
    .run_commands(
        "python3 -m pip install --no-cache-dir"
        f" -r {_EVAL_REQ}/eval-linux.txt -c {_EVAL_REQ}/vllm-image-constraints.txt"
    )
    # llmbench itself is already on PYTHONPATH from the Dockerfile's COPY.
    .add_local_python_source("modal_app")
)

# The HF baseline image pins the same torch/transformers as the local lock.
# torch 2.8.0's default Linux wheel is a CUDA 12.8 build.
HF_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.8.0",
        "transformers==4.56.2",
        "tokenizers==0.22.2",
        "safetensors==0.6.2",
        "huggingface-hub==0.36.2",
        "fastapi==0.117.1",
        "uvicorn==0.36.0",
        "aiohttp==3.14.3",
        "pyyaml==6.0.3",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("llmbench", "modal_app")
)

# Phase 4 quantization stack (llm-compressor 0.7.1 needs transformers 4.55.2).
QUANT_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(requirements=[str(REPO / "requirements" / "quantize.in")])
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("llmbench", "modal_app")
)

DOWNLOAD_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub==0.36.2", "hf-xet==1.6.0", "pyyaml==6.0.3")
    .add_local_python_source("modal_app")
)


@dataclass(frozen=True)
class Resources:
    """Billing envelope for one function; recorded in its results."""

    gpu: str | None
    cpu_cores: float
    memory_mib: int
    timeout_s: int
    startup_timeout_s: int

    def function_kwargs(self) -> dict[str, Any]:
        return {
            "gpu": self.gpu,
            "cpu": (self.cpu_cores, self.cpu_cores),
            "memory": (self.memory_mib, self.memory_mib),
            "timeout": self.timeout_s,
            "startup_timeout": self.startup_timeout_s,
            "retries": 0,
            "max_containers": 1,
            "scaledown_window": 2,
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DOWNLOAD_RESOURCES = Resources(None, 2, 4096, 600, 300)
VLLM_CHECK_RESOURCES = Resources(None, 4, 4096, 180, 300)
HF_CHECK_RESOURCES = Resources(None, 1, 2048, 120, 300)
VLLM_SMOKE_RESOURCES = Resources("L4", 4, 16384, 900, 300)
HF_SMOKE_RESOURCES = Resources("L4", 4, 8192, 600, 300)
# Phase 4. Host memory: AWQ holds the 16.4 GB BF16 model plus ~1 GiB of cached
# 256x512 activations; GPTQ's 512x2048 calibration caches ~8 GiB more.
DOWNLOAD_LARGE_RESOURCES = Resources(None, 2, 4096, 1500, 300)
QUANT_PREP_RESOURCES = Resources(None, 2, 8192, 1500, 300)
QUANTIZE_AWQ_RESOURCES = Resources("L40S", 4, 49152, 3600, 300)
QUANTIZE_GPTQ_RESOURCES = Resources("L40S", 4, 65536, 4500, 300)
SANITY_RESOURCES = Resources("L40S", 4, 32768, 1800, 300)
# Phase 5. The prefetch downloads the two datasets and runs the length audit.
EVAL_PREFETCH_RESOURCES = Resources(None, 2, 8192, 1800, 300)
# Probe: one variant with --limit; bounded well below the planned envelope.
EVAL_PROBE_RESOURCES = Resources("L40S", 4, 32768, 1200, 300)
# Full run, three variants in one container. Probe-20260923T140749Z measured
# 10,555 MMLU tokens/s on BF16, i.e. ~66 min per variant, ~3.3 h in total;
# the timeout allows ~25% more. Host RSS peaked at 13.0 GiB in the probe.
EVAL_FULL_RESOURCES = Resources("L40S", 4, 32768, 15000, 300)
# One variant (e.g. rerunning GPTQ). full-20260923T142440Z: BF16 took 4,256 s,
# AWQ 4,342 s (MMLU + WikiText), so 5,400 s leaves ~24%.
EVAL_ONE_RESOURCES = Resources("L40S", 4, 32768, 5400, 300)


def load_config(name: str) -> tuple[dict[str, Any], str]:
    """Read a committed YAML config locally; return it with its raw text."""
    import yaml

    text = (CONFIG_DIR / name).read_text(encoding="utf-8")
    return yaml.safe_load(text), text


def model_dir(model_id: str, revision: str) -> str:
    return f"{WEIGHTS_PATH}/{model_id}/{revision}"


def quantized_dir(config: dict[str, Any], variant: str) -> str:
    """Where Phase 4 wrote a verified quantized checkpoint on the weights Volume."""
    name = config["model"]["id"].split("/")[-1]
    return (
        f"{WEIGHTS_PATH}/quantized/{name}-{variant}-{config['model']['revision'][:8]}"
    )


def manifest_path(model_id: str, revision: str) -> str:
    return f"manifests/{model_id.replace('/', '--')}--{revision}.json"


def run_stamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
