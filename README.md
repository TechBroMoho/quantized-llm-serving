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
