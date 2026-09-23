# Quantized LLM Serving & Benchmarking

This project will quantify the accuracy, memory, and serving performance of BF16,
AWQ, and GPTQ variants of `Qwen/Qwen3-8B`. Results will come from reproducible
configs and saved raw measurements. The target numbers in the specification are
aspirations, not measured findings.

See [the specification](docs/SPEC.md), [progress](PROGRESS.md), and
[decisions](docs/DECISIONS.md). Phase 0 creates the local project foundation.

## Local setup

Install `uv` and run:

```sh
make setup
make check
make mock-validate
```

GPU experiments are outside the local workflow and require a separate cost
estimate and explicit approval before each run.

## CPU Hugging Face baseline

The baseline has `naive` (one generation at a time) and `static` (short
collection window, left-padded batches) modes. It streams each generated token
through `/v1/completions` and reports token counts in the final `usage` event.
For a local smoke run, start one mode in one terminal:

```sh
uv run python -m llmbench.baseline.hf_server --model sshleifer/tiny-gpt2 --mode naive --dtype float32 --port 8765
# or: --mode static --batch-size 2 --batch-wait-ms 50
```

Then run the client in another terminal:

```sh
uv run llmbench load --url http://127.0.0.1:8765/v1/completions --model sshleifer/tiny-gpt2 --concurrency 2 --requests 4 --output-tokens 3 --timeout 20 --out results/validation/hf_naive_cpu.json
```

Use a matching output path for static mode. The CPU checks use a deterministic
tiny Transformers model and also verify that disabling the minimum output
length produces an error when `usage.completion_tokens` is short. GPU baseline
runs will use BF16 with SDPA and a batch size established by an OOM probe.

## Reproduce the pre-Phase-3 audit

All of these run locally on CPU; no Modal function is invoked:

```sh
LLMBENCH_AUDIT_EVIDENCE_DIR=results/validation/audit make check
UV_CACHE_DIR=.cache/uv uv run python -m llmbench.loadtest.validation --output-dir results/validation/audit
```

The full check includes hostile model defaults, actual batched generation and
per-row token routing, short-output mutations for both HF modes, protocol
validity, and a single-request timing corruption that pooled medians miss.
`audit/actual_batch.json` saves inputs, masks, returned and streamed token IDs.
The timing summary saves paired server writes; its sibling JSONL saves client
records. The mock uses a shared local clock/event loop: this does not replace
an independent vLLM cross-check on the eventual GPU host. Historical evidence
limits are recorded in ADR-011. The CLI returns nonzero on failed/rejected
requests after saving the raw run.

The current CLI's `synthetic` workload uses variable-length text. It is for
functional checks, not the future 512/256-token controlled performance claims.

## Phase 3: Docker and Modal smoke tests

`docker/Dockerfile` extends the digest-pinned `vllm/vllm-openai:v0.10.2`
image with this repo's load tester. Modal builds that file directly; Compose
runs the same image on a local NVIDIA GPU:

```sh
make docker-check                          # hadolint + docker compose config ($0)
MODEL=Qwen/Qwen3-0.6B docker compose up --build   # needs a local NVIDIA GPU
```

Reproducing the Modal smoke is **billable** (Phase 3 actual total: $0.0975;
see `PROGRESS.md`). Every run is detached and bounded by timeouts and
CPU/memory limits (`modal_app/common.py`):

```sh
make modal-download   # CPU: verified Qwen3-0.6B snapshot into the weights Volume
make modal-checks     # CPU: build images, versions, in-container mock validation
make smoke            # L4: vLLM smoke, then HF naive + static smoke
make sync-results     # copy phase3/ results into results/validation/
uv run modal app list # confirm nothing is still running
```

The config is `configs/phase3_smoke.yaml`. Raw results and the billing report
are in `results/validation/phase3/`. These runs are functional checks (L4,
0.6B model, 16 requests), not performance results. See ADR-012 to ADR-014 for
the design and the two open findings, and `docs/PHASE4_6_ESTIMATE.md` for the
next phases' costs.
