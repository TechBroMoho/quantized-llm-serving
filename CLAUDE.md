# CLAUDE.md: Quantized LLM Serving & Benchmarking (`llmbench`)

**Instruction sync:** `CLAUDE.md` is the source of truth. Keep `AGENTS.md` byte-for-byte identical whenever these instructions change.

We quantize an open 8B LLM (default `Qwen/Qwen3-8B`) to 4-bit with AWQ and GPTQ (via `llm-compressor`), measure the accuracy cost (lm-eval: MMLU + perplexity), serve BF16/AWQ/GPTQ with vLLM in Docker on Modal GPUs, and benchmark them against a plain Hugging Face `transformers` baseline using **our own** validated async load tester (up to 256 concurrent users).

**Full spec: `docs/SPEC.md`. Status log: `PROGRESS.md`. Read both at the start of every session**, then summarize the current state in 3–5 lines before doing anything else.

## Non-negotiables (details and reasons in SPEC §3)

1. **Every number is real.** Metrics come from saved raw results produced by committed scripts/configs, with reproduce commands. Headline numbers are the median of 3 runs. Never fabricate, extrapolate, or cherry-pick to match the resume targets in SPEC §0.
2. **Fair comparisons.**
   - Same GPU type, greedy decoding, warmup excluded.
   - Token-ID prompts of exactly 512 tokens, **unique per request**, with vLLM prefix caching off for perf runs.
   - Output lengths identical, checked via `usage`.
   - Report the engine gains (vLLM vs. HF) separately from the quantization gains. Don't chase past the physical ceilings in SPEC §0.
3. **Hard budget: $30 out of pocket; target ≤ $25 total GPU spend** (Modal Starter includes $30/month free compute; per-phase caps are in SPEC §3).
   - **Before every GPU run:** print the GPU type, the expected minutes, the estimated cost (GPU + CPU + memory), and the running total, and **wait for an explicit "yes"**.
   - Every Modal function has a `timeout`.
   - Use `uv run modal run` only; **never leave a deployed app running.**
   - No GPU until the code has been validated on CPU. Downloads happen in CPU-only functions.
   - **Stop all GPU work at a $25 cumulative estimate.**
4. **Never commit secrets** (the HF token lives in a Modal Secret) or model weights.
5. **Never weaken, skip, or delete a test to make it pass.**
6. **Verify by running commands** and report the actual output.
7. **Phase gates:** at the end of each phase, run the acceptance checks, update `PROGRESS.md` (including the Spend log) and `docs/DECISIONS.md`, commit, report, and **STOP**. Exception: if Mohammed says "continue through Phase N", keep going through $0 phases, but still stop before any GPU spend and whenever an acceptance check fails.
8. **Long jobs run detached.** Shell commands time out after ~10 minutes, and a killed foreground `modal run` stops the GPU job and loses results. Use `uv run modal run --detach`, save results to the Volume after each step, and poll with `modal app logs` / `modal app list`. Nothing is finished until `modal app list` shows no running apps.
9. **Pin and verify versions.** vLLM / llm-compressor / transformers / torch / lm-eval move fast. Check current docs or `--help` before relying on memory. Don't use AutoAWQ/AutoGPTQ (deprecated).
10. **Ask when blocked** (no Modal token, model gating, GPU unavailable). Don't fake or quietly shrink scope.

## Environment facts

- Mohammed's machine: macOS, no NVIDIA GPU. All GPU work runs on Modal via `modal run` from the laptop.
- Default GPU: **L40S** for everything compared; **L4** for smoke tests (not T4). Never mix GPU types within a comparison.
- Benchmarks run the server and the load tester **in the same Modal container over localhost**, with one server lifetime per variant for the whole sweep.
- The Dockerfile is `FROM vllm/vllm-openai:<pinned>`; Modal builds from it (`Image.from_dockerfile`; check whether the entrypoint needs clearing). Don't build it in CI; lint it with hadolint.
- HF baseline: custom per-token streamer (not `TextIteratorStreamer`). HF-naive is swept only at concurrency {1, 4, 16}. Token counts always come from `usage`.

## Commands (keep this list accurate as the Makefile evolves)

```
make setup / check / test                  # local, $0
make mock-validate                         # load tester validation vs mock server, $0
make docker-check                          # hadolint + docker compose config, $0
make modal-download / modal-checks         # Modal CPU-only download + image checks (BILLABLE, tiny)
make smoke (smoke-vllm / smoke-hf)         # Modal L4 smoke tests, detached (BILLABLE, small)
make sync-results                          # copy phase3 results from the Modal Volume, $0
make quantize-rehearsal                    # CPU rehearsal of the Phase 4 quantization code, $0
uv run modal run --detach -m modal_app.quantize::{prepare,awq,gptq,sanity}  # Phase 4 (BILLABLE, ask first)
make eval-rehearsal / eval-audit           # CPU rehearsal + prompt-length audit of the eval code, $0
make eval-prefetch                         # Modal CPU: eval image check + datasets (BILLABLE, tiny)
make eval-probe / eval-full                # Modal L40S accuracy runs, detached (BILLABLE, ask first)
make sync-accuracy / accuracy-table RUN_DIR=...  # copy Phase 5 results, build the table, $0
make modal-loadtest-check                  # Modal CPU: ADR-013 load tester gates in the vLLM image (BILLABLE, tiny)
make bench-prompts-rehearsal               # $0: Phase 6 prompt build on WikiText-103 validation
make bench-prepare                         # Modal CPU: checkpoint sha256 check + prompt pool (BILLABLE, tiny)
make bench LIFETIME=<name in phase6_bench.yaml> [DROP=c256]  # Phase 6 L40S lifetimes, spawned (BILLABLE, ask first)
make sync-loadtest-check / sync-bench      # copy Phase 6 check / benchmark results, $0
make plots / report                        # regenerate charts + RESULTS.md from results/, $0 (Phase 7)
```

## Style

- Python 3.12, `uv`, `src/llmbench`, `mypy --strict`, `ruff`.
- Config-driven experiments (YAML in `configs/`). Result files embed the config, versions, and the GPU name.
- Clarity over cleverness. Conventional commits.
