# Progress

## Status

Phase 5 in progress (2026-09-23). BF16 (MMLU 74.88%, word perplexity 12.742)
and AWQ (73.93%, 13.282) are complete. GPTQ was cancelled mid-run when the
Mac slept; the entrypoint now spawns the job so it runs independently of the
laptop. The GPTQ-only rerun awaits approval. Cumulative spend **$9.0154**
(dashboard). Before Phase 6 (approved, not yet built): a multi-process load
tester revalidated in Modal (ADR-013), and `skip_special_tokens=false` with a
chunk-count check (ADR-014).

## Phase log

- **Phase 0 (2026-09-23):** Copied the instruction file to `AGENTS.md` and
  made sync an explicit rule. Verified GitHub login as `TechBroMoho` and the
  presence of Modal secret `huggingface` by read-only checks. Created the
  Python 3.12 project layout and documented the model estimate and measurement
  rules. `make setup` resolved 44 packages and checked 43. `make check` passed:
  Ruff reported "All checks passed!" and 3 files formatted; mypy reported
  "Success: no issues found in 2 source files"; pytest reported "2 passed";
  `cmp CLAUDE.md AGENTS.md` succeeded. `uv run llmbench` displayed its help,
  and `uv run modal --version` reported 1.3.3. No GPU stack or model checkpoint
  has been exercised.
- **Phase 1 (2026-09-23):** Implemented the controllable OpenAI completions
  SSE mock; closed-loop and Poisson open-loop load generation; chunk-based
  timing, usage counts, bounded drains, and raw result writers. The initial
  spin-based timing/capacity figures are withdrawn as evidence: their matching
  raw run was replaced, not retained. The later timing audit below has retained
  files. Historical 11-test and CLI-smoke claims lack saved command transcripts.
- **Phase 1 timing audit (2026-09-23):** Replaced the mock's final-millisecond
  busy spin (which compensated for coarse asyncio timer granularity) with plain
  `asyncio.sleep`. Accuracy now compares client timings to same-host monotonic
  timestamps recorded at handler start and immediately after each text write
  returns; requested 200/20 ms delays are not the reference. Five-sample client
  medians were TTFT 0.202324 s vs server 0.201397 s (0.46% difference) and ITL
  0.021327 s vs server 0.021258 s (0.33%), within ±5%. Treating the initial
  empty event as text failed with 99.7% TTFT difference; production code was
  restored and the corrected acceptance passed. Capacity recheck sustained
  40,405.8 chunks/s (6.73× threshold, 34,405.8/s margin) at 0.98 client CPU
  cores, zero errors/rejections; 256 requests remained at the measurement
  deadline and completed during the drain, so they are separately reported as
  late and excluded from in-window throughput. Full `make check` passed (11
  tests, Ruff, mypy, instruction-file comparison). No Modal job ran.
- **Phase 2 (2026-09-23):** Added the Hugging Face `/v1/completions` streaming
  baseline with naive serialization and left-padded static batches. A custom
  `BaseStreamer` emits token IDs per batch row; final `usage` counts generated
  IDs. The load client now fails when output usage differs from `max_tokens`
  and checks token-ID prompt length. CPU integration tests used a deterministic
  tiny GPT-2 that prefers EOS: naive generation batches were `[1, 1]`, static
  mode batched two unequal-length inputs, and both modes returned three tokens
  per request. The `min_new_tokens=0` mutation yielded one token and client
  errors `completion_tokens 1 != max_tokens 3`; normal settings passed. A
  separate `sshleifer/tiny-gpt2` CPU load run completed 4/4 requests per mode,
  each with three output tokens and zero errors. Raw records and summaries are
  in `results/validation/hf_*_cpu.*`; reproduce with the commands in
  `README.md` (static server uses `--mode static --batch-size 2 --batch-wait-ms
  50`). `make check` passed: Ruff, strict mypy on 11 source files, 14 tests,
  and instruction-file comparison. These are functional checks, not performance
  comparisons.

- **Pre-Phase-3 skeptical audit (2026-09-23):** Reproduced bugs before fixes
  (8 initial regression failures, 2 acceptance-guard failures, 3 ignored-setting
  failures; saved logs in `results/validation/audit*failures.txt` and
  `audit_failing_tests.txt`). Disabled Transformers model-default fallback;
  reject ignored API settings; strengthened SSE/usage validity and CLI failure
  status; fixed empty/short open-loop windows; enforce zero-error capacity gate.
  Added actual generate-call, padding/mask and distinct-row streamer evidence,
  and tested EOS suppression in both modes. Paired timing validates TTFT/E2E/
  TPOT and every gap; server timestamps are now saved. Local capacity measured
  36,538.1 chunks/s (6.09× gate), 0.97 CPU cores; raw files in
  `results/validation/audit/`. `make check` passed: 30 tests, Ruff, strict
  mypy (11 source files), and instruction-file comparison; saved in
  `results/validation/audit_check.txt`.
  ADR-011 distinguishes verified results, historical unsupported claims and
  analytical estimates. The Phase 3 plan is in `docs/PHASE3_PLAN.md`.
  No Modal commands, image builds, downloads or functions ran during this audit.

- **Phase 3 (2026-09-23):** Modal plumbing, the Docker image and L4 smokes
  (ADR-012). The Dockerfile pins `vllm/vllm-openai:v0.10.2@sha256:607442e4…`;
  `make docker-check` passes (hadolint 2.15.1 reported no findings; `docker
  compose config` is valid). Local `make check`: Ruff, strict mypy (12 source
  files), 47 tests, instruction-file comparison. CPU rehearsals drive the exact
  smoke code path against the mock and a real HF server subprocess.
  Remote steps, in order:
  - The CPU download of `Qwen/Qwen3-0.6B@c1899de2…` (1,519,209,243 bytes, 10
    files) matched the Hub's sha256 for every LFS file.
  - The in-image CPU check recorded vLLM 0.10.2, torch 2.8.0+cu128 and
    transformers 4.56.1. Timing validation passed; the **capacity gate failed
    at 3,786.1 chunks/s (0.63×)** (ADR-013).
  - The HF image check recorded torch 2.8.0 (CUDA 12.8) and transformers
    4.56.2.
  - **The vLLM L4 smoke passed** (run `smoke-20260923T095423Z`): NVIDIA L4
    (driver 580.95.05); all configured flags were found in the pinned `vllm
    serve --help`; `/health` after 59.1 s; 4/4 warmups and 16/16 requests
    with `usage` 512/32 on every record, zero errors/rejections/late
    completions; server counters 10,240 prompt / 640 generated tokens (= 20
    requests); prefix-cache queries and hits both 0. The log shows "Model
    loading took 1.1201 GiB" and "GPU KV cache size: 159,840 tokens". Two
    records have misleading E2E/TPOT because of empty chunks (ADR-014).
  - **The HF L4 smoke passed** (run `smoke-20260923T100540Z`): bf16, SDPA,
    cuda:0; the same prompt pool SHA-256 as vLLM. Naive `generate()` batches
    were `[1]×20`, static (B=4, 50 ms) `[4]×5`; every record was 512/32 with
    32 text chunks; zero errors.
  Raw files are in `results/validation/phase3/`; billing is in
  `modal_billing_2026-09-23.json`. The smokes' `git.dirty=true` came only
  from untracked synced result files; the recorded commits (1439224 for vLLM,
  7e16306 for HF) contain all code. Reproduce with `make modal-download
  modal-checks smoke sync-results` (billable; see README). These are
  functional checks on L4 with a 0.6B model, not performance numbers.

- **Phase 4, preparation (2026-09-23):**
  - The pinned llm-compressor 0.7.1 / compressed-tensors 0.11.0 stack was
    chosen to match vLLM 0.10.2 (ADR-016).
  - `make quantize-rehearsal` ran both official recipes on a tiny Qwen3 with
    the real tokenizer: 4-bit g128 `pack-quantized`, 14 zero-point tensors for
    AWQ and none for GPTQ, BF16 `lm_head`; a wrong-symmetry mutation was
    rejected and the GPTQ reload forward was finite.
  - `make check`: Ruff, strict mypy (14 files), 58 tests.
  - The CPU prepare run downloaded `Qwen/Qwen3-8B@b968826d…` (16,397,461,266
    bytes; every LFS sha256 matched) and built both calibration sets. AWQ: 256
    samples, 90,735 tokens (9–512 per sample). GPTQ: 512 samples, 572,273
    tokens (277–2048).
  - Evidence: `results/quantization/phase4/{download,calibration}-20260923T102838Z/`.

- **Phase 4, AWQ (2026-09-23):** passed on an NVIDIA L40S (driver 580.95.05),
  run `quantize-awq-20260923T103930Z`, llmcompressor 0.7.1 / compressed-tensors
  0.11.0 / transformers 4.55.2 / torch 2.8.0.
  - Time: model load 3.0 s, `oneshot` 459.4 s, save 24.5 s; function 538.3 s.
  - Memory: peak GPU 8,791 MiB by `nvidia-smi` (torch peak allocated
    7,785,041,408 bytes); host peak RSS 28,312,344 KiB.
  - Checkpoint `/weights/quantized/Qwen3-8B-awq-b968826d`: `pack-quantized`,
    4-bit group-128 asymmetric (minmax observer), `lm_head` ignored and stored
    in BF16. Tensors: 36 packed q_proj layers and 252 zero-point tensors.
    Safetensors total 6,098,617,040 bytes (6,114,604,278 with tokenizer and
    config files).
  - Actual cost $0.3855 against $1.47 expected. `llmcompressor.log` is empty
    (llmcompressor's loguru setup bypassed the added sink), so the run log is
    the filtered client capture `client_log_filtered.txt`. The one warning,
    "Optimized model is not saved", is expected: we save with
    `save_pretrained(save_compressed=True)` afterwards.
  - Not yet loaded in vLLM; that is the sanity run.

- **Phase 4, GPTQ (2026-09-23):** passed on L40S, run
  `quantize-gptq-20260923T121835Z`.
  - Time: `oneshot` 893.6 s (37 layers × calibrate/propagate over 512
    samples), save 25.8 s; function 973.2 s.
  - Memory: peak GPU 4,683 MiB by `nvidia-smi` (torch peak allocated
    3,761,910,272 bytes); host peak RSS 32,342,920 KiB.
  - Checkpoint `/weights/quantized/Qwen3-8B-gptq-b968826d`: `pack-quantized`,
    4-bit group-128 symmetric, `lm_head` ignored and stored in BF16; no zero
    points; safetensors 6,071,454,192 bytes.
  - Actual cost $0.7261 (re-estimate $0.88; the billing report first showed
    $0.3240 while still settling).
- **Phase 4, vLLM sanity (2026-09-23):** one L40S lifetime, run
  `sanity-20260923T124020Z`, vLLM 0.10.2 image, eager mode, max-model-len
  4096, prefix caching off, `--generation-config vllm`, all flags checked
  against the pinned `--help`. For all three variants, all 5 prompts completed
  64 greedy tokens (`finish_reason=length`); outputs are saved verbatim in
  `*/sanity.json`.
  - My coherence read: all three correctly answer Paris, write a recursive
    Fibonacci, give Rayleigh scattering, translate to "Bonjour, comment
    allez-vous aujourd'hui ?", and list primes 2–11. Continuations then diverge
    (for example, BF16 repeats "The capital of France is Paris."; AWQ and GPTQ
    list other capitals). This is a sanity check, not an accuracy measure;
    accuracy is Phase 5.
  - Both quantized variants run on `MarlinLinearKernel` for
    `CompressedTensorsWNA16`.
  - Measured sizes (single run; weight memory is deterministic for a given
    stack):

    | Variant | Safetensors on disk (bytes) | vLLM "Model loading took" | GPU KV cache (tokens) |
    | --- | ---: | ---: | ---: |
    | BF16 | 16,381,516,776 | 15.2683 GiB | 175,728 |
    | AWQ | 6,098,617,040 (−62.77%) | 5.7088 GiB (−62.61%) | 245,344 (×1.396) |
    | GPTQ | 6,071,454,192 (−62.94%) | 5.6835 GiB (−62.78%) | 245,520 (×1.397) |

    These match ADR-001's analytical estimate (6.07–6.10 GB; 62.8–62.9%
    reduction) and fall short of the 68% placeholder. KV-cache capacity is at
    `--gpu-memory-utilization 0.90` and max-model-len 4096, with eager mode.
    Phase 6's CUDA-graph settings will change the absolute token counts.
  - Reproduce: `uv run modal run --detach -m modal_app.quantize::{prepare,awq,gptq,sanity}`
    (billable; `configs/phase4_quantize.yaml`). Raw files are in
    `results/quantization/phase4/`.

- **Phase 5, preparation (2026-09-23, $0):** ADR-017.
  - Stack: lm-eval **0.4.11** (0.4.12+ require vLLM ≥0.18), datasets 4.1.1,
    evaluate 0.4.6 on top of the Dockerfile image. Resolved against the
    image's own Phase 3 freeze as constraints: 33 packages added, 0 changed.
    Every vLLM call in 0.4.11's backend exists in v0.10.2 (checked in source).
    The CLI is `lm-eval run` (read from the pinned `--help`); MMLU (5-shot)
    and WikiText (0-shot) run as two processes with identical `--model_args`.
  - Prompt-length audit (`make eval-audit`, lm-eval's own request
    construction, pinned Qwen3-8B tokenizer): MMLU 5-shot 56,168 requests,
    39,160,912 tokens, longest **3,097** (high_school_european_history);
    WikiText-2 104 windows, 347,162 tokens, longest 4,095. The truncation
    limit at max-model-len 4096 is 4,095, so nothing is truncated
    (`results/accuracy/phase5/prompt_lengths_local.json`). The probe
    (`--limit` 10 / 5) is 2,280 requests / 1,382,712 tokens plus 8 windows /
    27,069 tokens.
  - vLLM 0.10.2 skips the prefix cache for prompt-logprob requests, so each
    variant prefills all ~39.2M MMLU tokens; this drives the cost.
  - CPU rehearsal (`make eval-rehearsal`): the exact `run_variant` →
    checks → `compare` path with lm-eval's hf backend on two tiny random
    Qwen3s, `--limit 2`/`1`. Both passed, 114 questions each, 0 truncation
    warnings, identical prompt fingerprints, and net flipped questions (33
    gained − 29 lost) reproduce the −3.51 pp delta
    (`results/validation/phase5/rehearsal/`). A first attempt was SIGKILLed:
    the hf backend with the vLLM chunk size 1024 ran the laptop out of
    memory; the rehearsal now uses batch 8.
  - `make check`: Ruff, strict mypy (16 files), 78 tests, instruction-file
    comparison. One `make check` run during a concurrent rehearsal had one
    failing test that I could not identify; 8 reruns passed and later passing
    runs cleared pytest's record. Watch for it.

- **Phase 5, prefetch 1 (2026-09-23, run `prefetch-20260923T134506Z`,
  $0.0225 settled; first read $0.0176): gate failed, probe not started.** Inside the eval image:
  lm-eval 0.4.11, vLLM 0.10.2, torch 2.8.0+cu128, transformers 4.56.1,
  datasets 4.1.1. Datasets downloaded through lm-eval's task loading (121 s,
  58,394,847 bytes; 57 MMLU subjects at Hub commit `c30699e8`, WikiText at
  `64723477`) and reloaded offline ("Using the latest cached version …
  offline mode"). The in-image audit matched the laptop exactly (full
  56,168 / 39,160,912 / max 3,097; probe 2,280 / 1,382,712). The failure was
  my package check: it listed 13 packages as older than the image freeze
  (e.g. aiohttp 3.12.7 vs 3.12.15, setuptools 59.6.0 vs 79.0.1). These are
  Modal's client packages and Ubuntu's system packages, shadowed on
  `sys.path`; the check kept the *last* copy of each name instead of the one
  Python imports. Fixed: packages are listed in a subprocess with lm-eval's
  environment, the first copy wins, and shadowed copies are recorded.
  Evidence: `results/accuracy/phase5/prefetch-20260923T134506Z/`.

- **Phase 5, prefetch 2 (run `prefetch-20260923T135927Z`, $0.0203):
  passed.** No package drift; 17 shadowed duplicate distributions recorded
  (Modal client and Ubuntu system copies, never the imported ones). Same
  58,394,847-byte dataset cache; the length audits matched again. The
  eval-cache manifest was written.
- **Phase 5, timed probe (run `probe-20260923T140749Z`, $0.2108): passed.**
  NVIDIA L40S (driver 580.95.05), BF16, `gpu_memory_utilization` 0.80.
  - MMLU `--limit 10` (570 questions, 2,280 requests, 1,382,712 tokens):
    scoring 131 s (lm-eval bar `2280/2280 [02:11]`) = **10,555 tokens/s**;
    the process took 205.5 s (74.5 s engine start, 11.1 s model load, context
    building). WikiText `--limit 5` (8 windows, 27,069 tokens): 3 s scoring,
    46.4 s process. 0 truncation warnings; all checks passed.
  - vLLM: `max_num_batched_tokens=8192`, eager, Flash Attention, "Model
    loading took 15.2683 GiB", KV cache 141,328 tokens. Peak GPU memory
    **45,233 of 46,068 MiB** by nvidia-smi: prompt-logprob logits sit
    outside vLLM's 36 GiB budget. Full runs therefore use 0.75 (ADR-017).
    Host peak RSS 13.0 GiB.
  - The probe's accuracies (MMLU 77.89% on 570 questions, word perplexity
    11.743 on 5 documents) are timing by-products on a subset, not results.
  - Full-run estimate from this rate: ~66 min per variant, ~3.3 h and ~$7.94
    for all three (timeout 15,000 s, envelope $9.98).

- **Phase 5, full run 1 (run `full-20260923T142440Z`, $7.3972): BF16 and
  AWQ complete; GPTQ cancelled.** L40S, `gpu_memory_utilization` 0.75.
  - **BF16:** MMLU 5-shot **74.88% ± 0.35** (all 14,042 questions; macro
    76.83%), WikiText-2 word perplexity **12.742** (62 documents). MMLU
    scoring 64.0 min (10,200 tokens/s), process 4,170 s; WikiText 86 s. GPU
    peak 42,929 MiB (~3.1 GiB headroom); host RSS 13.0 GiB. 0 truncations.
  - **AWQ:** MMLU **73.93% ± 0.35** (macro 75.92%), WikiText-2 **13.282**.
    MMLU scoring 66.2 min (3.5% slower than BF16), process 4,252 s.
    Prompt fingerprints identical to BF16's. 617 questions lost and 483
    gained (net 134 = 0.95 pp); McNemar χ² 16.08, p ≈ 6e-5.
  - **GPTQ:** the input was cancelled at 10:29:45, ~42 min into MMLU, with no
    error in the job. The Mac entered clamshell sleep at 10:28:05 (`pmset`
    log); the local entrypoint was blocked on `.remote()`, and Modal stopped
    the app about 100 s after its client went silent. lm-eval writes results
    only at the end, so GPTQ's partial MMLU is lost (the partial `summary.json`
    stays as evidence). Fix: `full` now `.spawn()`s the job and returns, so it
    no longer depends on the laptop; a single variant runs as `eval_one`
    with a 5,400 s timeout.

## Spend log

| Date | Phase | Activity | GPU | Seconds | Cost (Phase 3+: actual) | Running total |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 2026-09-23 | 0 | Local setup and read-only account checks | None | 0 | $0.00 | $0.00 |
| 2026-09-23 | 1 | Mock, tests, timing and capacity validation, CLI smoke | None | 0 | $0.00 | $0.00 |
| 2026-09-23 | 2 | HF baseline, CPU integration and local load runs | None | 0 | $0.00 | $0.00 |
| 2026-09-23 | Audit | Local regression and mock revalidation | None | 0 | $0.00 | $0.00 |
| 2026-09-23 | 3 | Local plumbing, CPU rehearsals, lint | None | 0 | $0.00 | $0.00 |
| 2026-09-23 | 3 | Qwen3-0.6B download (CPU, `ap-xHIXkmtbBrUnz0xwBBkRs9`) | None | ≈20 | $0.000705 | $0.000705 |
| 2026-09-23 | 3 | vLLM env check, failed Python detection; includes image builds (`ap-5JIms1ZiZW6cas7KanUpFl`) | None | n/a | $0.005496 | $0.006200 |
| 2026-09-23 | 3 | vLLM env check + in-container mock validation (`ap-Dsss0oKkfcbF1FyX3ZhXnY`) | None | ≈112 | $0.006870 | $0.013071 |
| 2026-09-23 | 3 | HF env check (`ap-Gz7DnyqOIrQW8FZmA2mkky`) | None | ≈75 | $0.001320 | $0.014390 |
| 2026-09-23 | 3 | vLLM smoke, aborted at the flag pre-flight (`ap-eDYO92BJ50vQOAKBpmJ6oK`) | L4 | ≈50 | $0.015520 | $0.029910 |
| 2026-09-23 | 3 | vLLM smoke, passed (`ap-zTOJ6QYPv3WGEcInqkulAL`) | L4 | ≈129 | $0.040044 | $0.069953 |
| 2026-09-23 | 3 | HF naive + static smoke, passed (`ap-e1i6lvPAhpupn9WHoeTtW3`) | L4 | ≈94 | $0.027521 | $0.097475 |
| 2026-09-23 | 4 | Qwen3-8B download + calibration prep, CPU, includes quant image build (`ap-EyzLyIROLh9zFAktXmwkMu`) | None | n/a | $0.024180 | $0.121655 |
| 2026-09-23 | 4 | AWQ W4A16_ASYM quantization, passed (`ap-bi7KgCITqb40g0uoaypiVe`) | L40S | 538 function | $0.385489 | $0.507144 |
| 2026-09-23 | 4 | GPTQ W4A16 quantization, passed (`ap-PA7aqlQxiqweqcBW0wJNKj`) | L40S | 973 function | $0.726136 | $1.233280 |
| 2026-09-23 | 4 | vLLM sanity, BF16 + AWQ + GPTQ, passed (`ap-rbKztZzbSRbnw9hvYsp33H`) | L40S | n/a | $0.131285 | $1.364565 |
| 2026-09-23 | 5 | Eval prefetch, CPU, includes eval image build; gate failed on a false package-drift check (`ap-88OReY3mpGKjX70ev99mWR`) | None | n/a | $0.022514 | $1.387079 |
| 2026-09-23 | 5 | Eval prefetch rerun with the fixed check, passed (`ap-BH19KuquiKpyE9PMWKPgEt`) | None | n/a | $0.020285 | $1.407364 |
| 2026-09-23 | 5 | Timed probe, BF16, MMLU `--limit 10` + WikiText `--limit 5`, passed (`ap-BDLHfeHEYkcp93cPSBkH2Y`) | L40S | ≈317 | $0.210801 | $1.618165 |
| 2026-09-23 | 5 | Full run 1: BF16 + AWQ complete; GPTQ cancelled ~42 min into MMLU when the Mac slept (`ap-upiufFw2kJEquSk8ZS6hmT`); cost may still settle | L40S | ≈11,380 | $7.397234 | $9.015399 |

Phase 3 amounts are **actual** per-app costs from `modal billing report --for
today --json` (saved in `results/validation/phase3/modal_billing_2026-09-23.json`),
not estimates. The first read, a few minutes after the HF run, showed
$0.022840 for that app; the report lags, so the later value is used. Seconds
are derived as cost ÷ the function's envelope rate. The dashboard reports cost,
not seconds, and derived values also absorb any build charges. Phase 3 planned
$0.35 (worst case $0.81) against the $1 cap; actual: **$0.0975**. Remaining
Starter credit is not verified. Volume storage (~1.5 GiB of weights) is within
the 1 TiB/month included.

## Things that went wrong

- The local sandbox could not reach GitHub, Modal, or PyPI. Read-only account
  checks and the local dependency install passed with network access enabled.
  No remote compute was started.
- The sandbox denied localhost socket binding. The local integration and
  acceptance checks ran with localhost access; a first capacity attempt also
  exposed a wrong health-probe path. The probe was fixed and the full
  validation then passed.
- Phase 2's first dependency download and localhost bind were blocked by the
  sandbox. Approved network/local access let the CPU checks run. The cached
  tiny model still attempted a metadata lookup; `HF_HUB_OFFLINE=1` made the
  reproducible server run fully offline.

- The audit exposed Transformers model defaults overriding explicit-looking
  config values; a fresh GenerationConfig alone was insufficient. Local socket
  tests needed sandbox network access; no remote compute was involved.

- Phase 3: Modal rejected the vLLM image ("unable to determine the version of
  Python"): the base has only `python3`. A `python` symlink fixed it.
- Phase 3: `vllm serve --help` crashes without a GPU in 0.10.2 ("Failed to
  infer device type"), so flag verification moved to the GPU host, before the
  server starts. There, `--help=all` turned out to be a keyword filter and
  every flag looked missing. The pre-flight aborted the run for $0.0155
  instead of starting a misconfigured server. Fixed with plain `--help`.
- Phase 3: inside Modal, the single-process client managed 3,786 chunks/s,
  about a tenth of the laptop rate: gVisor plus one saturated core (ADR-013).
- Phase 3: two vLLM requests streamed 30–31 empty chunks (probably skipped
  special tokens after EOS was ignored), which truncates their last-text E2E
  (ADR-014). The HF server keeps special tokens, so the engines chunk
  differently.
- Phase 3: a local wait loop grepped the wrong file and ran for 10 minutes
  after the download had finished. No remote cost; the job itself took ~25 s.
- Phase 4: compressed-tensors 0.11.0 cannot decompress packed zero points
  inside transformers, so the rehearsal's HF reload failed for asymmetric AWQ.
  The rehearsal now checks stored tensors instead, and vLLM (which supports
  asymmetric WNA16) loaded the real AWQ checkpoint.
- Phase 4: llm-compressor writes `sparse_logs/` into the working directory.
  Rehearsal logs slipped into a local commit and were removed before pushing;
  the directory is now gitignored. Its loguru setup also bypassed our file
  sink, so the run logs come from the filtered client capture.
- Phase 4: the billing report lags. GPTQ first read $0.3240, then settled at
  $0.7261, which matches duration × rate. Always re-read the report before
  recording a cost.
- Phase 4: progress-bar output flooded two log monitors, which had to be
  stopped. The fix is to wait on process exit and read a filtered log.
- Phase 5: the first prefetch failed its "image unchanged" gate on 13
  phantom downgrades. Modal's function process has its own client packages
  (aiohttp, protobuf, yarl, …) on `sys.path`, and the image has Ubuntu's
  system dist-packages (six, setuptools, wheel, distro). The check kept the
  last copy of each name, not the one Python imports. Cost $0.0176; the probe
  was not started because the gate had failed.
- Phase 5: the full run lost GPTQ ~42 min into MMLU (about $1.65 of GPU
  time). The Mac went to clamshell sleep while `modal run --detach` waited on
  `.remote()`, and Modal cancelled the input ~100 s later. `--detach` did not
  protect a blocked client that went silent. The entrypoint now spawns the
  job and exits. BF16 and AWQ, already saved, were unaffected.
- Phase 5: `tests/test_timing_accuracy.py::test_mock_timing_accuracy` fails
  intermittently (2 of 25 full-suite runs, 0 of 6 alone): a few per-request
  gaps exceed ±5% (~1 ms at 20 ms ITL) under suite load, while the medians are
  within 0.5%. The test and gate are unchanged; a separate investigation is
  proposed.
