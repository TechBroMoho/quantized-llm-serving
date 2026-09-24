# Progress

## Status

**Phase 8 complete (2026-09-24), stopped at the final phase gate.** All
SPEC phases are done. CPU-only CI runs on GitHub Actions. Compose serves any
checkpoint from a local directory. The README is rewritten for a public
audience, and the final resume bullets are generated into
`docs/RESULTS.md#resume-bullets` with each number traced to its evidence
(ADR-025). Spend unchanged: **$20.3659** total (Phase 8 $0); nothing running.
- **Engine (vLLM BF16 vs HF, same weights):** peak output tokens/s 48.79×
  HF naive, 2.00× HF static (B = 128, largest *tested*; upper bound).
- **Quantization (AWQ vs vLLM BF16):** single-user decode 2.35×; weight
  memory −62.6%; KV cache 1.44×; peak throughput only 1.12×; MMLU −0.95 pp.
- **Both (AWQ vs HF):** peak requests/s 54.14× HF naive, 2.27× HF static.
- Resume bullets: 63% smaller model memory, 2.3× single-user speed, within
  1 percentage point on MMLU; served with vLLM, 256 simulated users, 54× a
  basic (one-at-a-time) HF server.
- Next: nothing scheduled. SPEC §11 stretch goals only if Mohammed asks and
  the budget allows ($4.63 left under the $25 target).

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

- **Phase 5, GPTQ rerun (run `full-20260923T175000Z`, $3.0496): passed.**
  Launched with `--detach` and `.spawn()`; `modal app list` still showed the
  app running 3 minutes after the entrypoint exited. Same config sha256
  (`364e97d9d36a`) and package versions as run 1 (commit c300da5 vs 69a80c0:
  only the launch and report code differ). MMLU scoring 65.0 min, process
  4,224 s; WikiText 74 s; GPU peak 42,942 MiB; 0 truncations.
- **Phase 5 results (acceptance: results + table committed).** Table:
  `results/accuracy/phase5/full-20260923T142440Z/accuracy_table.md`
  (+ `comparison.json`, `awq_vs_gptq.json`). All 14,042 MMLU questions and
  all 62 WikiText documents for every variant; identical prompt fingerprints
  across variants; every check passed.

  | Variant | MMLU 5-shot | Δ vs BF16 (pp) | Lost / gained | McNemar p | WikiText-2 word ppl |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | BF16 | 74.88% ± 0.35 | — | — | — | 12.742 |
  | AWQ W4A16-asym (pile-val) | 73.93% ± 0.35 | +0.95 | 617 / 483 | 6.1e-5 | 13.282 (+4.2%) |
  | GPTQ W4A16-sym (UltraChat) | 73.24% ± 0.36 | +1.65 | 678 / 447 | 7e-12 | 13.662 (+7.2%) |

  - Both drops are statistically real (paired test); only AWQ's is under
    1.5 pp. GPTQ is 0.69 pp below AWQ (754 lost / 657 gained, p = 0.011), but
    the two used different calibration data, so this does not isolate the
    method.
  - Category deltas: AWQ 0.75–1.26 pp, evenly spread (STEM +0.82); GPTQ
    1.42–2.19 pp, largest in STEM (+2.19).
  - Largest subject drops: AWQ high_school_chemistry +5.42 (203 q),
    college_mathematics +5.00 (100 q), college_physics +4.90 (102 q); GPTQ
    college_mathematics +8.00 (100 q), global_facts +7.00 (100 q),
    college_physics +6.86 (102 q). Single subjects are noisy (1 question =
    1 pp in a 100-question subject).
  - Relative drops, secondary: AWQ 1.27%, GPTQ 2.20% of BF16's accuracy.
  - Reproduce (billable; ADR-017/018): `make eval-prefetch eval-probe
    eval-full sync-accuracy`, then `make accuracy-table RUN_DIR=...` (for
    this split run: `uv run python -m llmbench.accuracy --run-dir
    results/accuracy/phase5/full-20260923T142440Z --variant-dir
    gptq=results/accuracy/phase5/full-20260923T175000Z`).

- **Phase 6 prerequisites (2026-09-23, $0, local only).**
  - *ADR-014.* The smoke driver and the new `PoolPayloads` send
    `skip_special_tokens: false` (verified in vLLM v0.10.2's
    `CompletionRequest`). The HF server accepts only `false`. Every completed
    request must have ≥ 0.5 text-bearing chunks per usage token
    (`MIN_TEXT_CHUNKS_PER_TOKEN`); the ratio is 0.5 rather than ~0.9 because
    vLLM's `RequestOutputCollector` merges deltas under load. A mock that
    streams the last 3 of 4 tokens as empty text now fails the smoke, and a
    threshold-0 mutation fails both new tests. Records gain a diagnostic
    `stream_end_s`.
  - *ADR-013.* `run_load(processes=P)` runs shard 0 in the parent and P−1
    spawned workers. The shards have disjoint users and request indices,
    share one window start on `perf_counter`, and pass a clock check (the
    worker's receipt of the start lies between the parent's send and its
    receipt of the results). Records are merged before the unchanged summary.
    `make mock-validate MOCK_VALIDATE_DIR=results/validation/multiprocess-local`
    (3 min 57 s): full timing gate passed with 1 and 2 client processes (ITL
    p99 0.42% / 0.39% over 4,000 gaps; TTFT/E2E/TPOT ≤ 0.14%); capacity
    **88,246.5 chunks/s (14.71×)** with 2 client processes at 1.94 cores,
    mocks 0.74 cores each, 0 errors or rejections, 256 late completions
    reported separately. The raw capacity file (46.1 MB, sha256
    `4751c849…917912`) is gitignored; the summaries are committed. New `make
    modal-loadtest-check` (CPU, 8 cores, 600 s timeout) runs the same gates
    inside the vLLM image. **Not run yet.**
  - *ADR-019, the flaky timing test.* Cause: host CPU contention. The
    process is descheduled for 2.5–6.4 ms (worst seen 33 ms) between the
    mock's write and the client's read, which stretches one gap and shrinks
    the next; medians stay within 0.6%. GC is not the cause (no gen-2 pause
    during any timing test, with a 6.3M-object heap or in the suite).
    - A first fix (p50 + p99 on 10 × 21) was still the every-request rule for
      the 10-sample metrics. It failed 3 of 25 suite runs.
    - Final: the unit test gates strict ±5% medians plus a 50 ms bound on
      every paired observation.
    - The evidence runs (200 × 21, sequential) gate strict medians plus
      p99 ±5%. A first evidence run with 8 in-process streams failed ITL p99
      at 6.21%, because the mock shares the client's event loop; it is kept.
    - Probes and logs: `results/validation/timing-investigation/`. The final
      unit gate was not loop-tested (the laptop was overheating).
  - Payload factories must now be picklable (`TextPayloads`, `PoolPayloads`);
    `llmbench load --processes N` is available.
  - `make check` (once, final code): Ruff, strict mypy (16 files), 103
    tests passed, instruction-file comparison.

- **Phase 6, in-Modal load tester check (2026-09-23, app
  `ap-alL2wtq71x0HiFGB2iF5IT`, CPU 8 cores / 4 GiB, $0.028004): passed.**
  Full timing gate with 1 client process (ITL p99 0.56% over 4,000 gaps) and
  2 (1.54%). Capacity **9,124.1 chunks/s (1.52×)** at 1.86 client cores, 0
  errors or rejections, clock check passed. One mock was at 0.95 cores, so
  this is a lower bound. If the probe's c=256 output rate exceeds ~3,040
  tokens/s, the headroom falls below ADR-005's 3× and needs a decision.
  Evidence: `results/validation/phase6/loadtest-check-20260923T205147Z/`.
- **Phase 6, benchmark build (2026-09-23, $0).**
  - Steady-state mode (ADR-020): staggered ramp, warmup, a window, then
    cut. Tokens are counted by arrival from cumulative per-chunk usage (vLLM
    `continuous_usage_stats`; the HF server emits the same). Requests/s
    counts completions inside the window. Recorded checks: half-window
    stationarity (≤ 5%, else the point is invalid) and a request-count edge
    bound (a warning).
  - HF server: skips jobs whose client left and stops a batch whose clients
    have all gone (so cut windows cannot leak into the next point); `/stats`
    reports pending/busy for the idle wait.
  - Prompts (ADR-021): WikiText-103 articles → exact 512-token windows →
    50,000 distinct, seeded, as a memory-mapped int32 pool.
    `make bench-prompts-rehearsal` on the validation file: 60 articles,
    exact windows, the prompt decodes and re-tokenizes to 512.
  - `llmbench.bench`: lifetimes, points, idle waits, vLLM counters (prefix
    hits must be 0), `nvidia-smi` sampling, per-point saving, the `vllm
    bench serve` cross-check (flags checked in v0.10.2 source and again
    against the image's `--help`), and the HF OOM probe
    (`llmbench.baseline.oom_probe`, the server's own `_generate`).
  - `modal_app.bench`: `prepare` (CPU) verifies all three checkpoints against
    Phase 4's sha256 and builds the pool. GPU lifetimes refuse unverified
    checkpoints and spawn so they don't depend on the laptop.
    `configs/phase6_bench.yaml` holds planning point timings until the probe.
  - CPU rehearsals of the exact driver against the mock and a tiny HF server
    (static batches, cuts, idle waits) pass.

- **Phase 6, prepare (run `prepare-20260923T212502Z`, CPU, $0.010784):
  passed.**
  - Every file matched Phase 4's sha256: BF16 15 files / 16,397,461,266
    bytes, AWQ 13 / 6,114,604,278, GPTQ 13 / 6,087,417,885 (23.4 s, 10.9 s,
    10.4 s).
  - Prompt pool: 1,801,350 WikiText-103 train lines → 9,900 articles read →
    75,000 windows → **223 duplicate windows dropped** → 50,000 prompts of
    exactly 512 tokens, pool sha256 `155f08a3…`.
- **Phase 6, AWQ probe (run `probe-20260923T213438Z`, L40S, $0.281566):
  passed.** NVIDIA L40S, vLLM 0.10.2, CUDA graphs.
  - Startup: healthy after 101.2 s (torch.compile 44.2 s, graph capture
    4 s). "Model loading took 5.7088 GiB"; GPU KV cache 240,544 tokens
    (234.9× for 1,024-token requests).
  - **c=1** (60 s window): 105.8 output tokens/s; TPOT p50 9.34 ms (107.1
    tokens/s single-user decode); TTFT p50 37.4 ms; E2E p50 2.42 s; 24
    requests completed. The requests/s edge bound (8.3%) is a warning;
    tokens/s is unaffected.
  - **c=256** (90 s window): **1,944.0 output tokens/s**, 7.278 req/s;
    TTFT p50 308 ms; TPOT p50 133.7 ms; E2E p50 34.4 s; 655 completed, 256
    cut at the end, 0 errors. Halves agree within 0.43%; edge bound 1.5%. 1
    chunk per token (1,944 chunks/s). Prefix-cache hits 0, preemptions 0.
  - **Where the time goes (c=256).** GPU utilization p50 96% while busy,
    power up to 352.6 W (both from `nvidia_smi.csv`). vLLM EngineCore 2.16 cores, API
    server 0.11; **client processes 0.037 + 0.037 cores** (the internal
    window CPU and `/proc` agree once `/proc` also counts the worker that
    exits as the window closes). Container 2.35 of 8 cores busy. The client
    is not the limit.
  - **Headroom:** 9,124.1 / 1,944.0 = **4.69×** ≥ 3×. Every sweep point now
    fails automatically if its received chunk rate leaves less than 3×.
  - Point timings revised: c=256 warmup 70 → 80 s, a separate
    `--max-num-seqs 16` point (warmup 120 s), and HF naive c4/c16 warmups
    40/125 s, per ADR-020's rule (warmup ≥ ramp + one expected E2E).

- **Phase 6 run plan (approved by Mohammed, 2026-09-23).** Back to back:
  AWQ → BF16 → GPTQ → max-batch → HF, under a $10 Phase 6 cap. Before each
  launch: Phase 6 actual spend + that run's timeout envelope ≤ $10. Stop and
  ask if a run fails, a point fails its checks (including the 3× headroom
  rule), or the next run could exceed the cap.
  - **HF static timings** come from the OOM probe inside the HF run, by the
    ADR-020 rule (`llmbench.bench.static_points_from_probe`).
    - T(n) is the measured `generate()` time for the smallest probed batch
      size ≥ n; B is the largest batch that fit.
    - At concurrency c, E2E ≈ ⌈c/B⌉ · T(min(c, B)).
    - warmup = ramp + 1.25 · E2E (rounded up to 5 s, never below the
      config).
    - window = max(config window, k · T) with k = 3 below c=128 and k = 10
      from c=128 (ADR-022), so each half holds more than one whole batch.
  - The HF function writes `static_plan.json` and stops before starting the
    static server if the plan cannot finish in the time left in the function.

- **Phase 6, AWQ sweep (run `awq-20260923T220125Z`, L40S, $1.868826, 2,557 s
  server lifetime): 6 of 10 points passed; stopped before BF16 as the rule
  requires.** Table: `results/perf/phase6/results_table.md`.
  - Passed: c4 352.7, c16 1,094.8, c64 1,997.5, c128 **2,196.8** output
    tokens/s; c1-r2 100.2 and c1-r3 102.2 (TPOT p50 9.7 ms). Headroom
    4.2–91× on passing points; client ≤ 0.10 cores per process (≤ 0.19 for
    both together); 0 errors, prefix-cache hits 0, preemptions 0.
  - **c1 failed** (halves 7.53%): a slowdown in the first half of the
    window. Requests starting 12.9 and 17.2 s in ran at TPOT 17.0 / 14.1 ms
    (stalls up to 74 ms); the eight starting 20.9–46.9 s in ran at 9.98–10.65
    ms; the other 37 at 9.52–9.80 ms. The client was at 0.03 cores. Cause not
    identified.
  - **c256, c256-r2, c256-r3 failed** (halves 7.17 / 7.24 / 6.07%). Their
    18 s token bins alternate high/low (~29k / ~37k tokens): the users form
    two waves with a period ≈ E2E (35.3–35.8 s). The 180 s window holds 5
    periods, so each half holds 2.5 and the halves differ by about one bin.
    This is periodic, not a trend. The 30 s ramp was shorter than one E2E.
    Throughput 1,831.7–1,860.5 tokens/s, below c128.
  - **Cross-check (`vllm bench serve`, `random` dataset: 512-token target
    inputs, 509.0–510.8 on average after its tokenizer round trip; exactly
    256 outputs):** c1 101.1 vs ours 99.0 tokens/s (+2.1%), TPOT +0.9%; c64
    1,996.4 vs 1,997.5 (−0.1%), TPOT −3.3%; **c256 2,277.1 vs 1,850.8
    (+23.0%)**, TPOT 111.3 vs 137.2 ms. (Our c1 and c256 here are the
    points that failed the halves check; RESULTS.md compares with the
    passing medians.) Our TTFT is lower at c64 (82.8 vs 336.0 ms): bench serve starts
    all users at once. At c256 its duration includes the synchronized start
    and the ramp-down tail, while our window is a staggered steady state
    with prefill mixed into decode steps. Not yet confirmed.

- **Phase 6, AWQ follow-up (run `awq-followup-20260924T000015Z`, L40S,
  $1.204175 first read; 1,628 s lifetime): all 5 points passed.** ADR-022
  timings.
  - c1-r4 100.0 tokens/s (TPOT p50 9.8 ms).
  - c128-r2 2,225.9 and c128-r3 2,196.1 tokens/s (with the sweep's c128:
    median **2,196.8**).
  - c256 re-timed (ramp 36 s, warmup 80 s, window 360 s = 10 × the measured
    35.6 s E2E): **halves 0.00%**, 1,844.5 tokens/s. That matches the
    three failed runs' 1,832–1,861, so their rate was right and only the
    check tripped.
  - **Diagnostic c256-sync** (all users at once, window 360 s): 2,223.8
    tokens/s, TPOT p50 112.0 ms, within 2.4% / 0.6% of `vllm bench serve`
    (2,277.1, 111.3 ms). This confirms the 23% cross-check gap at c=256 is the
    arrival pattern: an all-at-once start separates prefill from decode, while
    staggered users mix prefill into every decode step. Diagnostic only.
  - BF16 launched at 17:30 PDT: Phase 6 actual $3.393 + $2.584 envelope =
    $5.98 ≤ $11. Its summary records `git.dirty=true` at `abdbd1a` but not
    which files; I believed they were untracked synced results, which the
    results cannot confirm. Runs now also record `git.dirty_files`.
- Billing note: `modal billing report --for today` is a UTC day. After
  midnight UTC, the Phase 6 total must merge `--start 2026-09-23` (whole days
  only) with `--for today`.

- **Phase 6, parallel launch (2026-09-24 00:46 UTC, approved by Mohammed).**
  HF naive (`hf-naive-20260924T004647Z`) and HF static
  (`hf-static-20260924T004648Z`) launched alongside BF16, each in its own
  L40S container. Sum of worst cases: $3.393 + $2.584 + $1.306 + $2.584 =
  $9.87 ≤ $11 (project ≤ $21.93). With GPTQ too it would have been $12.20,
  so GPTQ waits for budget to free up (`gptq-trimmed` without c=256 if full
  GPTQ still doesn't fit). **Max-batch 16/64 skipped** (ADR-022 addendum):
  no resume claim depends on them.

- **Phase 6, BF16 (run `bf16-20260924T003034Z`, L40S, $1.619558 settled;
  2,219 s lifetime): all 10 points passed.**
  - "Model loading took 15.2683 GiB"; KV cache 170,944 tokens (166.9× at
    1,024 tokens).
  - c1 43.6 / 43.6 / 43.5 tokens/s (TPOT p50 22.8–22.9 ms); c4 157.6; c16
    561.9; c64 1,476.9; c128 1,943.7 then repeats 1,958.7 / 1,957.2
    (median **1,957.2**); c256 1,784.3 (ADR-022 window 340 s from an
    extrapolated 33.7 s E2E; measured 36.7 s; halves 1.44%).
  - The best point was c128, repeated as `repeat_best`.
- **Phase 6, HF naive (run `hf-naive-20260924T004647Z`, L40S, $0.700659
  first read; 957 s lifetime): all 5 points passed.** 40.1 / 40.1 / 40.0
  tokens/s at c1/c4/c16 (one request at a time, as designed), repeats of the
  best (c4) 40.1 / 40.1.
- **Phase 6, HF static, first launch (`hf-static-20260924T004648Z`,
  $0.027280): failed at startup.** `_hf_lifetime` resolved the static points
  before the OOM probe had provided B ("hf-cB needs the static batch
  size"); no GPU work ran. It wrote no result files: the error text is in
  the `7b5ae56` commit message, and the behavioural regression test
  reproduces it when the bug is reintroduced. Only the cost is billed
  evidence.
  Relaunch pending Mohammed's OK.
- **Phase 6, GPTQ.** `gptq-trimmed` (no c=256) launched at 01:09 UTC,
  because full GPTQ plus a later HF static relaunch did not fit with BF16's
  worst-case remainder counted. BF16's last `nvidia-smi` sample is 01:08:01
  UTC, so it was already finishing; with its actual cost, full GPTQ would
  have fit.

- **Phase 6 standing rule (Mohammed, 2026-09-24).** If a run crashes at
  startup before any GPU measurement, and the fix comes with a test and the
  budget still fits, fix and relaunch without asking, and log it here. Stop
  only for failures during measurement, failed checks that cannot be
  diagnosed, or budget limits.
- HF static relaunched with the fix (`hf-static-20260924T013433Z`, 01:34
  UTC). Budget: Phase 6 $5.729 (the Spend-log first reads then) +
  gptq-trimmed $1.522 + HF static $2.584 = $9.84 ≤ $11.

- **Phase 6, GPTQ trimmed (run `gptq-trimmed-20260924T010937Z`, L40S; 1,492 s
  lifetime): all 7 points passed.** "Model loading took 5.6835 GiB"; KV
  cache 240,736 tokens. c1 101.9 tokens/s (TPOT p50 9.7 ms); c4 365.2; c16
  1,107.8; c64 2,001.0; c128 2,207.2 / 2,195.2 / 2,209.9 (median
  **2,207.2**). No c=256 (budget cut, ADR-022 addendum).

- **Phase 6, HF static relaunch (run `hf-static-20260924T013433Z`, L40S,
  $1.438650 first read; 1,922 s lifetime): all 6 points passed.**
  - OOM probe (`generate()` of 512 + 256 tokens): every candidate fit up to
    128 (8: 10.0 s, 205 tokens/s … 96: 24.4 s, 1,007 tokens/s; 128: 30.2 s,
    1,084 tokens/s). **No candidate hit OOM, so B = 128 is the largest
    *tested* size, not the proven maximum** (SPEC §4 asks for the largest that
    fits). A larger B might raise HF static's peak somewhat; its probe
    throughput was still rising slowly.
  - Plan (ADR-022, T = 30.2 s): cB warmup 60 s / window 305 s; c2B warmup
    100 s / window 305 s (≈ 10 batch cycles).
  - Results: c4 125.7, c16 393.6, cB (c=128) 906.5, c2B (c=256) 974.5 /
    993.8 / 977.0 tokens/s (median **977.0**).
  - Every point warns on requests/s (edge bound 13–23%), because whole
    batches of up to 128 complete together. Tokens/s, the primary metric, is
    unaffected.
- **Phase 6 acceptance (SPEC §7): met.** All raw results are in
  `results/perf/phase6/` (per-request `requests.jsonl.gz` per point,
  summaries, server logs, `nvidia-smi` and `/proc` samples). The cross-check
  is documented (below). Spend is logged.
  - Every accepted point passed the 5% steady-state check, the 3× client
    headroom (worst 4.1×), exact 512/256 usage, the chunk check, zero
    prefix-cache hits and zero errors.
  - Failed points stay in the table, marked (AWQ c1 and the first three
    c256).
  - **Headline table (medians of passing runs):**

    | | vLLM AWQ | vLLM GPTQ | vLLM BF16 | HF static | HF naive |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | Single-user tokens/s (1 / median TPOT, c=1) | 102.6 (3 runs) | 102.8 (1) | 43.7 (3) | — | — |
    | Peak output tokens/s | 2,196.8 @ c128 | 2,207.2 @ c128 | 1,957.2 @ c128 | 977.0 @ c256 | 40.1 @ c4 |
    | Peak requests/s | 8.572 | 8.606 | 7.556 | 3.777 | 0.158 |
    | Weight memory (vLLM log) | 5.7088 GiB | 5.6835 GiB | 15.2683 GiB | — | — |
    | KV cache (tokens, 1,024 max len) | 245,312 | 240,736 | 170,944 | — | — |

  - AWQ's KV cache is from the sweep and follow-up lifetimes (both
    245,312). The probe lifetime, same flags, logged 240,544 (−1.9%), so
    lifetimes vary by ~2% and AWQ vs GPTQ (240,736) is within that.
  - **Ratios:**
    - AWQ decode 2.35× BF16.
    - Peak tokens/s: AWQ vs HF naive 54.8×, vs HF static 2.25×; BF16 vs HF
      naive 48.8×, vs static 2.00×.
    - Requests/s (now computed by `perf_report`): AWQ vs HF naive 54.1×,
      vs static 2.27× (HF requests/s carries the edge-bound warning).
    - Quantization at peak: 1.12× (AWQ), 1.13× (GPTQ).
  - **Cross-check** (`vllm bench serve`, AWQ): c1 +2.1% tokens/s, c64 −0.1%,
    c256 +23.0%. The c256 gap is explained by arrival pattern: our
    all-at-once diagnostic gave 2,223.8 tokens/s, within 2.4% of its 2,277.1.

- **Phase 7 audit of Phase 6 (2026-09-24, $0; ADR-023).**
  - Pushed the 12 unpushed commits. Five `make check` runs with visible exit
    codes: runs 1, 2, 4, 5 exit 0; run 3 exit 2 with
    `test_open_loop_poisson_mode_completes_requests` (`rejected_requests
    1 != 0`). Reproduced: 0 of 100 runs idle, 2 of 100 under CPU contention
    (12 busy loops on 10 cores). Requests to the in-process mock took 35–52
    ms instead of ~4, and four were in flight at a seeded arrival cluster
    (61–74 ms). The rejection was correct and the test wrong. Fixed with a
    cap above the scheduled arrivals, a new deterministic rejection test,
    and a recorded `open_loop_max_dispatch_lag_s`. Stressed: 0 of 40.
  - The earlier HF-lifetime failure did not recur (0 of 23 runs, 8 under
    contention). Found while hunting it: `fetch_text` let
    `RemoteDisconnected`/resets escape (a candidate cause); an unreachable
    vLLM `/metrics` read as idle and passed the prefix-cache check
    vacuously; the HF server looked idle while collecting a batch. All
    fixed with regression tests that fail on the old code. The 45 recorded
    points all had good snapshots, so no result changes.
  - Tests that didn't test their names: the HF lifetime's idle assertion
    was a tautology, and the hf-static regression test grepped the source.
    Both now test behaviour; reintroducing the original bug fails the new
    one.
  - Numbers: a subagent traced every Phase 6 number in this file to its raw
    file. Corrected: AWQ KV cache 245,312 (not the probe's 240,544); BF16
    cost $1.619558 settled (the whole old $0.0115 gap); the c1 slowdown
    description; "per process" cores; the HF static window rule; four
    claims without a raw file. Peak requests/s is now computed by
    `perf_report`.
  - `make check` after the fixes: exit 0 three times (142 tests).
- **Phase 7 (2026-09-24, $0; ADR-024).** `src/llmbench/analysis/`
  (`aggregate`, `plots`, `report`); `make aggregate / plots / report`.
  - Charts in `results/analysis/`: request throughput, p95 TTFT and median
    TPOT vs concurrency; latency–throughput; weight memory + KV cache;
    accuracy. Palette validated (`validate_palette.js`: all hard checks
    pass; contrast relief via markers and tables). Each chart was rendered
    and inspected; label collisions were fixed.
  - `docs/RESULTS.md` (generated): summary with engine / quantization / both
    kept separate, the HF static caveat, SPEC §8 claims → evidence with
    targets vs measured, methodology (hardware, versions, commands, windows
    per point, lifetimes with commit and image), full sweep table,
    failed/diagnostic points, memory (with `nvidia-smi` peaks showing why it
    is not the metric), accuracy, cross-check against passing medians,
    sanity checks, where the speedup comes from, limitations.
  - Sanity checks (from the aggregate): 43 points, 38 accepted, 4 failed,
    1 diagnostic; 0 errored requests; vLLM prefix-cache queries/hits and
    preemptions all 0 over 34 points; every curve rises to its peak; AWQ
    and BF16 fall 16.0% / 8.8% from c=128 to c=256; repeated points spread
    ≤ 2.6%. Measurement config sections are identical across lifetimes, with
    one prompt pool.
  - The cross-check against passing medians: c1 +0.9%, c64 −0.1%, c256
    +23.5% (the all-at-once diagnostic is within +2.4%).
  - `make check` (final code): exit 0 twice. Ruff, strict mypy on 24 source
    files, 151 tests passed, instruction-file comparison.

- **Phase 8 (2026-09-24, $0; ADR-025).**
  - Pushed Phase 6 audit `1b5b818` and Phase 7 `cc7efdc` first.
  - CI (`.github/workflows/ci.yml`): `make setup` + `make check` on
    ubuntu-24.04, plus hadolint and `docker compose config`. Actions pinned
    to SHAs (checkout v7.0.1, setup-uv v10.2.0, hadolint-action v3.5.0). The
    lock now takes torch from the PyTorch CPU index on Linux only (`uv lock`:
    torch `2.8.0+cpu` there; 15 nvidia-* wheels and triton removed; macOS
    unchanged). Modal images pin their own CUDA torch, so no measured
    environment changes.
  - Linux rehearsal before pushing (`python:3.12-bookworm`, linux/amd64,
    uv 0.11.16): the first run installed `torch 2.8.0+cpu` and no nvidia
    package, then failed strict mypy (`Library stubs not installed for
    "yaml"`) on the new command. Fixed with `types-pyyaml` in the dev group.
    Local `make check` showed the same error; the stubs fixed both.
  - Compose: serves `MODEL_DIR` (required) read-only at `/model`, offline,
    with the Phase 6 engine flags as defaults, plus a Python healthcheck (no
    curl in the base image). `make docker-check` exits 0; with `MODEL_DIR`
    unset Compose refuses ("required variable MODEL_DIR is missing a
    value"). Not run on a GPU (none locally); vLLM 0.10.2's loader skips the
    Hub for local directories (checked in the v0.10.2 source).
  - `make fetch-checkpoint VARIANT=…` + `llmbench verify-checkpoint`:
    `modal volume get` keeps only the last path component (checked on a
    small results directory), so the target renames each download to
    `checkpoints/qwen3-8b-<variant>`. Against the real AWQ evidence with no
    local files, the verifier checked 13 files / 6,114,604,278 bytes and
    exited 1 with 13 missing. New tests cover a matching file, a same-size
    changed one, a longer one and a missing one.
  - Resume bullets are rendered by `report.py` from the aggregate, with an
    exact-value/definition/evidence table and the caveats (ADR-025 §4).
    `make perf-table` joined `make plots report`; the regenerated
    `results_table.md`, aggregate and charts are byte-identical.
  - README: plain-English purpose, results table (engine / quantization /
    accuracy / both), two charts, the engine-vs-quantization explanation,
    how the benchmark works (Mermaid diagram), a $0 quickstart (mock server
    + load tester, run here: 64 requests, 0 errors), local Docker serving,
    five Modal steps with actual per-phase cost, limitations, future work.
    `tests/test_readme.py` guards links/anchors, headline numbers and the
    bullets. A mutation (a changed anchor, number and bullet) failed all
    three tests. GitHub's GFM renderer produced both tables and the Mermaid
    block.
  - `make check` (local, final tree): exit 0; ruff clean, strict mypy on 24
    files, 158 tests passed, instruction files identical.

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
| 2026-09-23 | 5 | Full run 1: BF16 + AWQ complete; GPTQ cancelled ~42 min into MMLU when the Mac slept (`ap-upiufFw2kJEquSk8ZS6hmT`) | L40S | ≈11,380 | $7.397234 | $9.015399 |
| 2026-09-23 | 5 | GPTQ-only rerun, spawned, passed (`ap-ZlJlWJBg1CYBkMPungNRD3`) | L40S | ≈4,580 | $3.049610 | $12.065009 |
| 2026-09-23 | 6 | In-Modal load tester check, passed, includes vLLM image rebuild (`ap-alL2wtq71x0HiFGB2iF5IT`) | None | ≈240 | $0.028004 | $12.093013 |
| 2026-09-23 | 6 | Prepare: checkpoint sha256 check + WikiText-103 prompt pool, passed (`ap-3es0914tudLlXFvC62Qz7f`) | None | n/a | $0.010784 | $12.103797 |
| 2026-09-23 | 6 | AWQ probe, c=1 and c=256, passed (`ap-RyihFdSLFZprP7tolBMi7H`) | L40S | ≈392 | $0.281566 | $12.385363 |
| 2026-09-23 | 6 | AWQ sweep + repeats + cross-check; 4 points failed the steady-state check (`ap-wlDNh5vqN7p0fgg6ncYgEb`) | L40S | ≈2,600 | $1.868826 | $14.254189 |
| 2026-09-23/24 | 6 | AWQ follow-up (ADR-022), all 5 points passed (`ap-xh6csySKIIZ5HRZ4cLD0ri`) | L40S | ≈1,680 | $1.204175 | $15.458364 |
| 2026-09-24 | 6 | BF16 sweep + c1 and c128 repeats, all passed (`ap-M0EH6Yb26rksfXcKxtsv9v`) | L40S | ≈2,240 | $1.619558 | $17.077922 |
| 2026-09-24 | 6 | HF naive, all passed (`ap-rZcEliswiLDYYgL0RuUx2L`) | L40S | ≈975 | $0.700659 | $17.778581 |
| 2026-09-24 | 6 | HF static, failed at startup (point resolution bug) (`ap-YfAYWf7dsuIWowYM1twSeK`) | L40S | ≈38 | $0.027280 | $17.805861 |
| 2026-09-24 | 6 | GPTQ trimmed (no c256), all passed (`ap-FDnqJCa4CQ7vKLxYKIVg75`) | L40S | ≈1,560 | $1.121412 | $18.927273 |
| 2026-09-24 | 6 | HF static relaunch, all passed (`ap-FSNm8dgmEFBGmIAphj1loj`) | L40S | ≈2,000 | $1.438650 | $20.365923 |
| 2026-09-24 | 7 | Audit, charts, RESULTS.md (local only) | None | 0 | $0.00 | $20.365923 |
| 2026-09-24 | 8 | CI, Compose, README, resume bullets (local; Modal Volume listings/reads only, no compute) | None | 0 | $0.00 | $20.365923 |

**Phase 6 actual: $8.3009** against its $11 cap (raised from $10 by
Mohammed, ADR-022). Every Phase 6 row matches the saved
`results/perf/phase6/modal_billing_2026-09-24.json` (the AWQ follow-up is its
two UTC-day rows summed). The BF16 row first showed the pre-settlement
$1.608071; it was corrected in the Phase 7 audit, which removed the $0.0115
gap previously noted here.

**Phase 5 actual: $10.7004** against its $11.63 cap ($6 plus Phase 4's
unused $4.73 and Phase 3's unused $0.90, both reallocated by Mohammed;
ADR-018). It includes about $1.65 of GPU time lost to the cancelled GPTQ
attempt. Billing re-read at 12:13 PDT; the report is saved as
`results/accuracy/phase5/modal_billing_2026-09-23.json`.

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

- Phase 6 prerequisites: my first fix for the timing flake (a p99 over 10
  samples) was still the old every-request rule in disguise, and it failed 3
  of 25 suite runs. My first 25× loop also kept only each run's last line, so
  one failure could not be attributed. Then the first full-gate evidence run
  failed ITL p99 at 6.21% with 8 concurrent streams sharing the mock's event
  loop; the evidence runs now use one stream per process.
- Phase 6 build: a failed virtual user (the prompt pool ran out in a CPU
  rehearsal) was silently swallowed by `asyncio.gather(return_exceptions=True)`,
  and the run looked merely non-stationary. The steady-state check caught the
  symptom; worker exceptions now fail the run, and a regression test covers it.
  The same latent problem existed in the older duration mode.
- Phase 6 probe: the `/proc` CPU evidence first missed the spawned client
  worker, which exits as the window closes, before the next 1 s sample. The
  attribution now uses the samples in which each process appears; re-derived
  from the saved samples, the worker used 0.037 cores, matching its own
  measurement. `modal app logs` also streams forever; poll `modal app list
  --json` from a bounded script instead (macOS has no `timeout`).
- Phase 6: commit `099f4c9` was made after a `make check` that had one
  failure (`test_hf_lifetime_on_cpu_cuts_windows_and_waits_for_idle`). I had
  piped make into `tail`, which hid its exit status. The failure has not
  reproduced in 15 reruns (14 alone, 1 full suite), and its message was lost
  to the `tail`. It stays an open flake; the test prints its failure list if
  it recurs. Commits now check make's real exit status.
- Phase 6: the HF static OOM probe's candidate list stopped at 128, and
  all candidates fit, so the probe established "fits at 128", not the
  maximum. A larger B might raise HF static's peak (probe throughput was
  still rising: 1,007 → 1,084 tokens/s from 96 to 128). The comparison vs
  HF static is reported with that caveat.
- Phase 7 audit (ADR-023): five `make check` runs with visible exit codes
  exposed a *second* flaky test. `test_open_loop_poisson_mode_completes_requests`
  failed once on 1 rejected request: under CPU contention the in-process
  mock slowed to 35–52 ms per request, and a cluster of five seeded arrivals
  found all four slots busy. The runner was right; the test assumed an idle
  machine. The Phase 6 HF-lifetime flake still has not recurred (0 of 23,
  8 of them under contention). Two real bugs turned up while hunting it: a
  vLLM `/metrics` failure read as "idle" and let the prefix-cache check pass
  vacuously; and the HF server reported idle while it was collecting a
  batch. Neither affected a recorded point.
- Phase 8: Compose read a relative `MODEL_DIR` without `./`
  (`checkpoints/example`) as a *named volume* and rejected the project. The
  long bind syntax fixed it for relative and absolute paths. Also, the
  astral `uv:…-python3.12-bookworm` image tag used for the first Linux
  rehearsal does not exist; the rehearsal moved to `python:3.12-bookworm`
  with uv installed by pip.
