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
See [the pending Phase 3 plan and itemized budget](docs/PHASE3_PLAN.md).
