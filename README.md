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
```

GPU experiments are outside the local workflow and require a separate cost
estimate and explicit approval before each run.
