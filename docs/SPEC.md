# SPEC: Quantized LLM Serving & Benchmarking

> This is the full project specification. Claude Code: read this file, `CLAUDE.md`, and `PROGRESS.md` at the start of every session. This file is the source of truth. If you believe something here is wrong or outdated (this ecosystem moves fast), say so and record the decision in `docs/DECISIONS.md`. Do not silently deviate.

---

## 0. Context: who this is for and why it exists

- **Owner:** Mohammed, a UC Berkeley CS + Data Science student applying for SWE/MLE internships (summer 2027).
- **Purpose:** a portfolio project for his resume. It will be public on GitHub and discussed in interviews.
- **How it's built:** Claude Code is the primary implementer. Mohammed reviews each phase and studies the finished code afterward. So:
  - **Clarity beats cleverness.**
  - **Every non-obvious design decision is written down** in `docs/DECISIONS.md`.
  - **Every number must be real, measured, and reproducible.**

### Target resume bullets (placeholders, not facts)

```
Quantized LLM Serving & Benchmarking | Python, PyTorch, vLLM, Hugging Face, Docker
• Compressed an open-source 8B LLM to 4-bit precision (AWQ, GPTQ) and deployed it with vLLM in
  Docker, using 68% less memory and generating text 3.1x faster while keeping accuracy within 1.5%
  of the original
• Developed a load-testing tool simulating 256 simultaneous users to measure response speed under
  heavy traffic, showing the optimized setup handles 14x more requests per second than a standard
  Hugging Face setup
```

The numbers (68%, 3.1x, 1.5%, 256 users, 14x) are **ambition targets**. Pursue them with legitimate engineering and fair methodology. If reality differs, report the real numbers and propose reworded bullets at the end. **Never fabricate, extrapolate, or cherry-pick a number to match a bullet.** Section 8 maps every claim to the evidence that must back it.

**Physical ceilings. Compute these in Phase 0 and don't spend GPU money chasing past them.**
- Qwen3-8B has ~8.2B parameters, of which ~6.95B are non-embedding. The input embedding and a separate `lm_head` (~0.62B each) stay BF16 under our scheme.
- **Weight-memory ceiling:** BF16 ≈ 16.4 GB vs. W4A16 ≈ 6.1–6.3 GB, so a ~62–63% reduction is the most we can expect. The 68% placeholder is above the ceiling.
- **Single-user decode-speed ceiling:** batch-1 decode is memory-bandwidth-bound. Bytes read per token (the embedding lookup reads only one row; `lm_head` is read in full) give a ceiling of roughly 3x. Real kernels usually land around 2–2.5x.
- Verify these numbers yourself from the model config and record the derivation in DECISIONS.md.
- A quantized-`lm_head` variant may be added only as a **separately labelled** extra. It must never be swapped in to make a headline number.

---

## 1. What we're building (plain English)

An LLM is a big file of numbers. An **inference engine** is the software that runs it to produce text: the thing behind a chatbot's "send" button. This project:

1. **Compresses** an 8B-parameter open model from 16-bit to 4-bit weights using two methods (AWQ and GPTQ). This is like saving a RAW photo as a JPEG: much smaller, slightly lossy.
2. **Checks the quality cost** with a standard benchmark (MMLU) plus perplexity.
3. **Serves** the original and compressed models with vLLM (a production engine with continuous batching and paged KV-cache) inside Docker.
4. **Builds a load tester** that simulates up to 256 simultaneous users and measures response speed: time to first token, per-token speed, and tail latency.
5. **Compares** everything against a "standard Hugging Face setup" (plain `transformers.generate()` behind a simple web server) on the same GPU. It explains *where* each speedup comes from: engine vs. quantization.

---

## 2. Goals and non-goals

### Goals

1. AWQ and GPTQ W4A16 checkpoints produced **by us** with `llm-compressor`, loadable by vLLM.
2. An accuracy comparison (BF16 vs. AWQ vs. GPTQ) with `lm-evaluation-harness`.
3. **Our own** async load tester (streaming-aware), validated against a mock server and cross-checked against `vllm bench serve`.
4. A Hugging Face baseline server (naive + static-batching modes) with the same API shape.
5. Throughput/latency sweeps across concurrency levels and max batch sizes, with charts.
6. A Dockerized serving setup (the same Dockerfile runs on Modal and on any NVIDIA machine via Compose).
7. Everything reproducible from committed configs, within a **tight GPU budget**.

### Non-goals

- Training or fine-tuning.
- Writing custom CUDA kernels.
- Building our own inference engine. Custom KV-cache/continuous batching is a stretch idea only.
- A chat UI.
- A persistent public endpoint.
- Comparing against TGI/SGLang/TensorRT-LLM (optional stretch; the headline baseline is plain `transformers`).

---

## 3. Non-negotiable rules

1. **Honest numbers.** Every metric in README/RESULTS comes from a saved raw result file produced by a committed script/config, with the exact reproduce command. If a run fails or looks wrong, investigate. Never discard it silently or rerun until it looks good. Headline numbers are the **median of 3 runs**, with the spread reported.
2. **Fair comparisons.**
   - Same GPU type for everything that's compared.
   - Same prompts and exact token lengths; same sampling (greedy); same max output tokens with `ignore_eos` (or `min_new_tokens` for HF) so every system generates identical lengths.
   - Warmup excluded.
   - Configs recorded.
   - Report vLLM-BF16 vs. HF separately from quantized vs. BF16, so engine gains and quantization gains aren't conflated.
   - Also report against HF *static batching* so the baseline isn't a strawman.
3. **Budget (hard).** Mohammed's out-of-pocket GPU budget is **$30, hard cap**. Modal's Starter plan includes **$30/month of free compute** (verify on modal.com/pricing).
   - **Target total GPU spend: ≤ $25**, so ideally it all fits inside the free credits.
   - **Before any GPU run:** print the GPU type, the expected duration, the estimated cost (GPU + CPU + memory), and the running total. Wait for an explicit "yes".
   - Every Modal function sets a `timeout`.
   - **Never `modal deploy` a persistent endpoint.** Use `modal run` (ephemeral). If a demo endpoint is ever needed, it must scale to zero and be stopped (`modal app stop`) right after.
   - **Long GPU jobs run detached.** Claude Code's shell commands time out after ~10 minutes. An ephemeral Modal app stops when its client exits, so a killed foreground `modal run` wastes GPU money and loses results. For anything that might exceed ~8 minutes:
     - use `uv run modal run --detach ...` with a strict function `timeout`;
     - write results to the Volume **incrementally** (after each sweep point);
     - poll with short `modal app logs` / `modal app list` commands.
     A job is only finished when `modal app list` shows nothing still running.
   - **Per-phase hard caps** (estimates, re-check before each run), summing to ≤ $25: Phase 3 $1, Phase 4 $6, Phase 5 $6, Phase 6 $10, reserve $2. If a phase would exceed its cap, stop and ask.
   - GPU is not the only cost: CPU cores and memory are billed on top. Request only what's needed, and include them in estimates.
   - **No GPU until the code is validated on CPU.** All client, baseline, and analysis code is tested locally first.
   - **Do downloads and file shuffling in CPU-only functions**, never while holding a GPU.
   - Keep a running **Spend log** in PROGRESS.md: GPU-seconds × rate, reconciled with the Modal dashboard.
   - **Stop all GPU work if the cumulative estimate reaches $25** and ask how to proceed.
4. **Secrets.** Hugging Face token via a Modal Secret; never committed. No `.env` in git.
5. **Tests are the spec.** Never weaken, skip, or delete a test to make it pass.
6. **No fake verification.** Run it and report the actual output.
7. **Phase gates.** At the end of each phase: acceptance checks → update `PROGRESS.md` + `docs/DECISIONS.md` → commit → **stop and report** (§10).
8. **Ask when blocked.** No Modal token, model license not accepted, a GPU type unavailable, etc.: stop and ask. Don't work around it by faking or silently changing scope.
9. **Pin and verify versions.** vLLM, llm-compressor, transformers, torch, and lm-eval change quickly and have tight compatibility constraints. Check current docs/`--help`/release notes before relying on memory. Pin exact versions and record them in every result file.

---

## 4. Key decisions (defaults; deviations need a DECISIONS.md entry)

| Area | Default | Why / notes |
|---|---|---|
| GPU provider | **Modal** | $30/month free credits on Starter; per-second billing; scale-to-zero; driven from the laptop CLI (`modal run`), so Claude Code can run GPU jobs without SSH; Volumes cache weights and results. Fallback: RunPod. Mohammed should set a **workspace budget** on Modal's Usage & Billing page. |
| GPU type | **L40S (48 GB)**, ≈ $1.95/hr on Modal | Enough memory for BF16 8B plus KV cache; supports the fast 4-bit (Marlin) kernels. Alternative: A100-80GB (≈ $2.50/hr). **Never mix GPU types within a comparison.** Use **L4** (same Ada architecture, ≈ $0.80/hr) for smoke tests. Don't use T4: no BF16, no Marlin, different code paths. Record the exact GPU name from `nvidia-smi` in every result. |
| Model | **`Qwen/Qwen3-8B`** (Apache-2.0, ungated) | Alternative: `meta-llama/Llama-3.1-8B-Instruct` (gated; needs license approval). Before committing, verify llm-compressor AWQ/GPTQ support for the architecture (AWQ layer mappings). For perf benchmarks use raw `/v1/completions` prompts (no chat template), which sidesteps Qwen3's thinking-mode template. |
| Quantization tool | **`llm-compressor`** (vLLM project): `AWQModifier` and `GPTQModifier`, 4-bit weights / 16-bit activations, group size 128, `lm_head` ignored | Outputs the compressed-tensors format that vLLM loads natively. **Do not use AutoAWQ or AutoGPTQ** (deprecated/unmaintained). Start from the official llm-compressor examples for each method and use their recommended calibration data/sample counts; record the dataset, number of samples, seq len, and seed. The official AWQ example uses an asymmetric scheme (`W4A16_ASYM`), while GPTQ examples typically use symmetric `W4A16`. Decide deliberately (following each method's official example is fine), record the choice in DECISIONS.md, and label it in the results. Official pre-quantized checkpoints may be used only as a *sanity reference*, never as the headline result. |
| Serving engine | **vLLM** OpenAI-compatible server, pinned version | Record the engine args (`--max-num-seqs`, `--max-model-len`, `--gpu-memory-utilization`, etc.). |
| Container | `docker/Dockerfile` `FROM vllm/vllm-openai:<pinned tag>` + our serving script/config | Modal builds from this Dockerfile (`modal.Image.from_dockerfile`). `docker-compose.yml` runs the same image on any local NVIDIA GPU. This is what backs "deployed with vLLM in Docker". The base image defines an ENTRYPOINT, and Modal's own vLLM example clears the entrypoint. Check whether `.entrypoint([])` is needed. Don't build the vLLM image in GitHub Actions (too large for runners); lint it with `hadolint` instead. `modal` is a **project dependency**; run it as `uv run modal ...` so app files can import project code. |
| HF baseline | FastAPI + `transformers` `generate()` in BF16 with SDPA attention, the same `/v1/completions` streaming shape | Two modes: **naive** (one request at a time, i.e. a lock) and **static batching** (collect up to B requests within W ms, pad left, generate together). **Don't use `TextIteratorStreamer`:** it only supports batch size 1 and buffers text until word boundaries, which skews TTFT/ITL. Write a small custom `BaseStreamer` that emits one event per generated token per batch row. Choose B for static batching with a quick out-of-memory probe (the largest B that fits at 512+256 tokens), and record it. |
| Accuracy eval | `lm-evaluation-harness` with the vLLM backend: **MMLU 5-shot** (report acc and the delta in *percentage points*) + **WikiText-2 word perplexity** (fixed max length, recorded) | No `--apply_chat_template` (standard base-style evaluation, the same for all variants). Set `--max-model-len` high enough for the longest 5-shot prompts (e.g., 4096; verify). Optional if the budget allows: GSM8K (generative; quantization often hurts reasoning more, which is an interesting finding). |
| Load tester | **Our own:** Python `asyncio` + `httpx` (or `aiohttp`), SSE streaming | Backs the second resume bullet. Must be validated (Phase 1) and cross-checked vs `vllm bench serve` (Phase 6). |
| Benchmark topology | **Server and load tester run in the same Modal container**, talking over `localhost` | Removes internet latency from TTFT, needs no public endpoint, and the container exits when done (no idle billing). Give the container enough CPU (e.g., 4–8 cores) so the client isn't the bottleneck. CPU/memory are billed on top of the GPU, so request only what's needed. |

---

## 5. Metrics: precise definitions (use these names everywhere)

- **TTFT:** time from sending the request to receiving the first streamed chunk containing generated text.
- **ITL:** inter-chunk latency; the gaps between consecutive streamed chunks. SSE chunks don't always map one-to-one to tokens, so note this in RESULTS.
- **Token counts come from the server's `usage` field** (request `stream_options: {"include_usage": true}`; the HF server must return the same), never from counting chunks.
- **TPOT:** (E2E − TTFT) / (output_tokens − 1).
- **E2E latency:** request start → last token.
- **Request throughput:** completed requests / wall time over the measurement window (req/s).
- **Peak request throughput:** the maximum request throughput a system reaches across the concurrency sweep. The "14x" headline compares **peaks** (vLLM variant peak vs. HF-naive peak, and vs. HF-static peak).
- **Output token throughput:** generated tokens / wall time (tok/s).
- **Percentiles:** p50, p90, p95, p99 for TTFT, TPOT, ITL, and E2E.
- **Single-user decode speed:** 1 / median TPOT at concurrency 1 (tok/s). This backs "generates text Nx faster".
- **Model weight memory:** vLLM's reported model-loading memory (log line) **and** the on-disk checkpoint size. Report both. Explain in RESULTS why `nvidia-smi` total memory is *not* the right metric (vLLM preallocates KV cache according to `--gpu-memory-utilization`).
- **KV-cache capacity:** vLLM's logged KV-cache size (tokens) and the max concurrency at the given sequence length. 4-bit weights free memory for more KV cache; that's a key interview point.
- **Accuracy delta:** BF16 MMLU acc − quantized MMLU acc, in **absolute percentage points** (say so explicitly).

---

## 6. Workloads and sweeps

- **W1 (controlled, headline):** 512 input tokens / 256 output tokens.
  - Prompts are real text (e.g., a public-domain corpus) tokenized with the model's tokenizer and truncated to exactly 512 tokens.
  - Send prompts **as token-ID lists** to both vLLM and HF, so no re-tokenization drift. Assert `usage.prompt_tokens == 512` and `usage.completion_tokens == 256` on every response, and fail the run otherwise.
  - Greedy decoding; `ignore_eos=true` on vLLM, `min_new_tokens=max_new_tokens` on HF.
  - **Every request uses a unique prompt** (the pool size is at least the total requests in a run, seeded), **and** vLLM perf runs set `--no-enable-prefix-caching` (verify the flag name for the pinned version). vLLM caches repeated prefixes by default, which would inflate its numbers relative to HF. Log the prefix-cache hit rate if it's available.
- **W2 (optional, realistic):** variable input/output lengths from a ShareGPT-style distribution, as `vllm bench serve` supports. Use it for a single "real-world" data point, not the headline.
- **Concurrency sweep** (closed loop: N virtual users, each sends its next request when the previous one finishes):
  - vLLM-BF16, vLLM-AWQ, vLLM-GPTQ: `{1, 4, 16, 64, 128, 256}`. Use **one server lifetime per variant** for the whole sweep, to avoid paying for repeated cold starts.
  - **HF-naive: `{1, 4, 16}` only.** Its throughput is flat by construction (one request at a time), and higher concurrency just creates huge queues and timeouts. State this explicitly in RESULTS.
  - **HF-static:** concurrency up to a small multiple of its batch size B (from the OOM probe).
  - All HF runs are **time-boxed** (e.g., 3 min) with a **time-based warmup**.
  - Report only what was measured in the window; **no extrapolation**.
- **Max-batch sweep:** vLLM-AWQ at concurrency 256 with `--max-num-seqs ∈ {16, 64, 256}`. This shows the latency ↔ throughput trade-off (the "batch size" story).
- **Repeats:** headline configs × 3 runs; the others × 1 unless variance looks high.
- **Measurement window:** a warmup of ≥ 20 requests (or 30 s) is excluded. Record the exact window.

---

## 7. Phased plan with acceptance criteria

Estimated GPU cost per phase is on L40S at ~$1.95/hr plus CPU/mem. Re-estimate before each run.

### Phase 0: Bootstrap ($0)
- Create the repo layout (below), `pyproject.toml` (`uv`, Python 3.12), `Makefile` (`setup`, `fmt`, `lint`, `typecheck`, `test`, `check`), `.gitignore`, `PROGRESS.md`, `docs/DECISIONS.md`.
- Check that Modal is set up (`modal token` / `modal profile current`) and that the HF token Modal Secret exists. If not, give Mohammed exact steps and stop.
- Confirm the model choice: its license/gating, and llm-compressor example support for its architecture (cite the docs/examples you checked).
- Compute the **physical ceilings** from §0 (weight-memory reduction and batch-1 decode speedup) from the model's config. Write the derivation in DECISIONS.md.
- Add `modal` as a project dependency (`uv add modal`), so every Modal command runs as `uv run modal ...`.
- **Acceptance:** `make check` green. **Report:** your plan for Phases 1–3, the ceilings, and any questions. **STOP.**

```
.
├── CLAUDE.md  PROGRESS.md  README.md  Makefile  pyproject.toml  uv.lock
├── docker/Dockerfile  docker-compose.yml
├── configs/                  # YAML experiment configs (model, variant, engine args, workload, sweep)
├── src/llmbench/
│   ├── loadtest/             # client.py, workloads.py, metrics.py, cli.py
│   ├── baseline/             # hf_server.py (naive + static batching)
│   ├── mock/                 # mock SSE server with controllable TTFT/ITL
│   └── analysis/             # aggregate.py, plots.py, report.py
├── modal_app/                # common.py (images, volumes, secrets), download.py, quantize.py, evaluate.py, bench.py
├── tests/
├── results/{quantization,accuracy,perf,validation}/   # raw JSON/JSONL + charts (committed; keep them small)
└── docs/SPEC.md  DECISIONS.md  RESULTS.md
```

### Phase 1: Load tester + mock server ($0)
- **Mock server:** an OpenAI-style `/v1/completions` SSE endpoint with a configurable TTFT, ITL, and token count.
- **Load tester:**
  - closed-loop concurrency mode and open-loop Poisson-rate mode;
  - warmup, duration/request-count limits, and timeouts;
  - SSE parsing that timestamps each received text-bearing chunk; token counts come from the server's `usage` field because chunks do not necessarily correspond to tokens;
  - per-request JSONL (TTFT, ITL list, TPOT, E2E, tokens, status/error) and a summary JSON (throughputs, percentiles, error rate, config, versions, host info);
  - a CLI: `llmbench load --url ... --concurrency ... --workload ... --out ...`.
- **Validation (commit the results to `results/validation/`):**
  1. **Accuracy:** against a mock with TTFT=200ms and ITL=20ms, the measured medians are within ±5% (tests).
  2. **Client capacity:** against a near-zero-latency mock at 256 concurrent streams, the tester sustains well above the event rate the real benchmarks need. Show its CPU usage, and prove the client won't be the bottleneck.
- **Acceptance:** `make check` green; the validation report is committed. **STOP.**

### Phase 2: HF baseline server ($0)
- Implement the naive and static-batching modes with the same streaming API.
- Test on CPU with a tiny model (e.g., `Qwen/Qwen3-0.6B` or a tiny test model) using the load tester at low concurrency.
- Confirm that the output lengths equal `max_new_tokens` exactly.
- **Acceptance:** tests green; a small local run of the load tester against the HF server works. **STOP.**

### Phase 3: Modal plumbing + Docker smoke test (cap $1)
- Modal: one Volume for model weights/checkpoints, one for results, and one for vLLM's compile/CUDA-graph cache (saves time on every later cold start). Images: our Dockerfile (vLLM) + a baseline image (transformers).
- Add a CPU-only `download` function that pulls the model weights into the Volume once (no GPU cost).
- Practice the **detached run + polling** workflow (§3.3) here on a cheap job.
- **Smoke test on L4 with the tiny model:**
  - start the vLLM server from our image;
  - wait for `/health`;
  - run the load tester against `localhost` at concurrency 4;
  - save the results to the Volume and sync them to `results/`;
  - exit.
- Repeat for the HF baseline image.
- **Acceptance:** both smoke tests pass; actual cost recorded; `docker compose config` validates. **Report the cost estimate for Phases 4–6 and STOP for approval.**

### Phase 4: Quantization (cap $6)
- Download `Qwen3-8B` BF16 to the Volume (CPU function).
- Run **AWQ** and **GPTQ** 4-bit (W4A16-family, per the §4 decision) with llm-compressor on L40S, detached. Record wall time, peak GPU memory, calibration settings, and versions.
- Save the checkpoints to the Volume. Optionally push them to a *private* HF repo as a backup, so a lost Volume doesn't mean re-paying. That needs a **write**-scoped HF token; ask Mohammed first.
- **Sanity checks** for each variant:
  - it loads in vLLM;
  - it generates coherent text for 5 fixed prompts (save the outputs);
  - its on-disk size is recorded.
- **Acceptance:** 3 servable variants (BF16, AWQ, GPTQ), with sizes, logs, and sample outputs committed. **STOP.**

### Phase 5: Accuracy evaluation (cap $6)
- First run a short **timed probe** (`lm_eval ... --limit <small>`) on one variant. Use it to calibrate the full-run time and cost estimate, then show that estimate before the full runs.
- lm-eval (vLLM backend): MMLU 5-shot and WikiText-2 word perplexity for all 3 variants, with identical settings (§4), detached. Save the raw lm-eval JSON.
- Produce a table: accuracy, delta in percentage points, perplexity, and per-subject MMLU extremes (the biggest drops) as an interview talking point.
- **Acceptance:** results + table committed. **STOP.**

### Phase 6: Performance benchmarks (cap $10)
- A self-contained Modal benchmark function (run detached): start the server (variant + engine args from a config) → health check → warmup → run the sweep, **saving results after each sweep point** → collect the vLLM logs (weight memory, KV-cache size) and `nvidia-smi` samples → save raw → exit.
- Run the full §6 sweep plan:
  - vLLM-BF16/AWQ/GPTQ concurrency sweeps;
  - HF-naive and HF-static (time-boxed);
  - the max-batch sweep;
  - headline repeats ×3.
- **Cross-check:** run `vllm bench serve` (verify the current flags via `--help`) at 2–3 matching settings. Our tester and vLLM's should agree within ~10%. Investigate and document any gap.
- **Acceptance:** all raw results in `results/perf/`; the cross-check documented; spend logged. **STOP.**

### Phase 7: Analysis and charts ($0)
- Charts (matplotlib, clean and labeled, with GPU and model in the subtitle):
  1. request throughput vs. concurrency (all systems);
  2. p95 TTFT vs. concurrency;
  3. median TPOT vs. concurrency;
  4. a latency–throughput Pareto plot, including the max-batch sweep;
  5. weight memory + KV-cache capacity per variant;
  6. an accuracy table/bar chart.
- Sanity checks:
  - throughput rises then plateaus at saturation;
  - error rates ≈ 0 (report any non-zero);
  - the variance across repeats is reported.
- `docs/RESULTS.md`: the methodology (hardware, versions, configs, workloads, windows), all results, and a "where the speedup comes from" section: engine (continuous batching + paged KV cache) vs. quantization (memory-bandwidth-bound decode reads fewer bytes; more room for KV cache).
- **Acceptance:** charts + RESULTS.md committed. **STOP.**

### Phase 8: Docker/repro polish, CI, and hand-off ($0)
- `docker-compose.yml` to serve any variant on a local NVIDIA GPU; a README section with exact commands.
- GitHub Actions: ruff, mypy, pytest (CPU-only; mock server + tiny-model tests), hadolint. No GPU in CI.
- **README:**
  - what/why in plain English;
  - an architecture diagram (Mermaid);
  - headline results with charts;
  - how to reproduce on Modal (≤ 5 commands) and the **actual total spend**;
  - limitations (single GPU type, synthetic workload, one model);
  - future work.
- **Final resume bullets:** propose revised bullets using only real numbers, each linked to its evidence (§8). If a target wasn't hit, say so plainly and suggest honest wording.
- **Acceptance:** fresh clone → `make setup && make check` passes; the README renders correctly. **STOP.**

---

## 8. Claims → evidence (fill this in `docs/RESULTS.md`)

| Resume claim | Evidence required |
|---|---|
| "Compressed an open-source 8B LLM to 4-bit (AWQ, GPTQ)" | Our llm-compressor scripts + configs + logs; checkpoint sizes |
| "deployed it with vLLM in Docker" | The Dockerfile; the Modal image built from it; a benchmark log showing the server started from that image |
| "68% less memory" | Weight memory (vLLM log) and on-disk size, BF16 vs. the 4-bit variant; say which metric the % uses |
| "3.1x faster" | Single-user decode speed (1/median TPOT, concurrency 1), 4-bit vs. BF16 on vLLM, median of 3 runs |
| "accuracy within 1.5%" | MMLU 5-shot delta in percentage points, from the lm-eval JSON |
| "load-testing tool simulating 256 simultaneous users" | Our tester's code; the validation report; a concurrency-256 run summary |
| "14x more requests per second than a standard Hugging Face setup" | **Peak** request throughput: which vLLM variant vs. HF-naive peak, same GPU/workload, unique prompts, prefix caching off. Also report vs. the HF-static peak and vLLM-BF16 vs. HF |

Each row gets: the value, its definition, the environment (GPU, versions), the config file, the reproduce command, the raw file path, and the date.

---

## 9. Coding standards

- Type hints everywhere; `mypy --strict` on `src/`; `ruff`.
- Config-driven experiments: each run is defined by a committed YAML config. Result files embed the config, a config hash, the package versions, the GPU name, the driver/CUDA version, and a timestamp.
- Small modules; docstrings explain *why*.
- Tests:
  - unit tests (metrics math, SSE parsing, workload construction with exact token counts);
  - integration tests (tester ↔ mock server; tester ↔ HF server with a tiny model on CPU).
- Keep committed results small (JSON/JSONL, PNG). No model weights in git.
- Commits use conventional messages.

---

## 10. How Claude Code should work on this project

- **Session start:** read `CLAUDE.md`, this SPEC, and `PROGRESS.md`. Summarize the current state in 3–5 lines before doing anything.
- **Each phase:** a short plan (bullets) → implement → run acceptance checks → fix → update docs → commit → report → **stop**.
- **`PROGRESS.md` sections:**
  - `## Status`;
  - `## Phase log` (what was built, evidence: commands + key output lines, open issues);
  - `## Spend log` (per GPU run: GPU type, seconds, est. cost, running total);
  - `## Things that went wrong` (real bugs/surprises and fixes; these are the best interview stories).
- **`docs/DECISIONS.md`:** ADR-style entries (context → options → decision → consequences). Examples:
  - Modal vs. RunPod;
  - L40S vs. A100;
  - Qwen3 vs. Llama;
  - llm-compressor settings;
  - W4A16 group size;
  - why W4A16 rather than W8A8/FP8;
  - closed- vs. open-loop load testing;
  - why the HF baseline has two modes;
  - why the server and client share a container;
  - the metric definitions.
- **End-of-phase report (chat):** what was built (3–6 bullets), the evidence, the decisions, the spend so far, anything Mohammed should look at, and what the next phase needs from him.
- If Mohammed says "continue through Phase N", you may run $0 phases back-to-back, but **always stop before any GPU spend** for approval with an estimate, and stop if an acceptance check fails.
- **Long-running commands:** shell commands time out after ~10 minutes. GPU jobs run detached and are polled (§3.3). A job that was killed partway is a failed run, not a partial success.
- Keep the context lean: read targeted files/sections, and summarize long logs instead of dumping them.

---

## 11. Stretch goals (only after Phase 8, and only if Mohammed asks and the budget allows)

- FP8 KV cache or W8A8/FP8 weight quantization as extra variants.
- Speculative decoding on the best variant.
- A GSM8K accuracy comparison (a reasoning-sensitivity finding).
- A comparison with another engine (SGLang).
- A minimal educational KV-cache + continuous-batching loop in pure PyTorch, benchmarked against HF-naive (teaches the "why", not a replacement for vLLM).
