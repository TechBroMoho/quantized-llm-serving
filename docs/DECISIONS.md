# Decisions

Decisions are recorded before experiments so that later measurements cannot
quietly change the method. `CLAUDE.md` is the instruction source of truth; its
identical `AGENTS.md` copy is checked by `make check`.

## ADR-001 — Default model and analytical ceilings (2026-09-23)

**Context.** The portfolio claims need one open 8B model with a clear license,
distinct BF16 input and output embeddings, and documented AWQ/GPTQ support.

**Options.** Qwen3-8B is public and Apache-2.0. Llama-3.1-8B-Instruct is gated.

**Decision.** Use `Qwen/Qwen3-8B` by default. Its [published config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) gives 36 layers,
hidden size 4096, intermediate size 12288, 32 query heads, 8 KV heads,
head dimension 128, vocabulary size 151936, and untied embeddings. The
[model card](https://huggingface.co/Qwen/Qwen3-8B) states Apache-2.0. Current
[llm-compressor AWQ mappings](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/transform/awq/mappings.py)
include `Qwen3ForCausalLM`, and its
[GPTQ example](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a16/)
uses W4A16. This establishes documented architecture support, not a claim that
our checkpoints have been produced or served.

**Derivation.** Per layer, attention projection weights are
`2×4096² + 2×4096×(8×128) = 41,943,040`; MLP projection weights are
`3×4096×12288 = 150,994,944`. Across 36 layers, the target linear weights
number `6,945,767,424`. The input embedding and separate output head contain
`2×151936×4096 = 1,244,659,712` weights. The remaining normalization weights
are negligible for this estimate. BF16 weights therefore occupy approximately
`2×8.191B = 16.38 GB` (decimal). At 4 bits, linear weights occupy 3.473 GB;
two-byte scales per 128-weight group add about 0.109 GB. Keeping embeddings
and head in BF16 yields about 6.07 GB with symmetric groups. An illustrative
four-bit zero point per group adds about 0.027 GB for asymmetric groups, for
about 6.10 GB. That suggests a **weight-only reduction of about 62.8–62.9%**,
before exact checkpoint layout, alignment, or runtime overhead. The original
68% target exceeds this estimate.

For batch-one decoding, the input embedding is a lookup, while the full output
head and transformer weights are read. Ignoring cache traffic and compute,
BF16 weight reads are approximately `2×(6.946+0.622) = 15.14 GB/token`;
symmetric W4 reads are about `3.473+0.109+1.245 = 4.83 GB/token`. The ratio is
**about 3.14× as a weight-bandwidth-only estimate**, or roughly 3× after
allowing for simplified accounting. It is not a measured speedup or a strict
physical limit: kernels, KV traffic, and scheduling may change the result.

**Consequences.** The actual memory and speed claims await saved Phase 4–6
measurements. Quantizing `lm_head` would be a separately labelled variant.

## ADR-002 — Quantization methods and version boundaries (2026-09-23)

**Context.** AWQ and GPTQ must remain distinct and load through vLLM's
compressed-tensors path. The upstream recipes evolve.

**Options.** The current [AWQ recipe](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/awq/)
uses `AWQModifier` followed by `QuantizationModifier` with `W4A16_ASYM` and
an ignored `lm_head`. The [GPTQ recipe](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a16/)
uses `GPTQModifier` with `W4A16` and group size 128.

**Decision.** Keep AWQ and GPTQ as separate experiments. Before Phase 4, pin a
compatible vLLM/llm-compressor/transformers/torch set and verify the exact
recipes against those versions. Record calibration data, sample count, length,
seed, group size, symmetry, and excluded layers in each result. Prequantized
models can only be sanity references.

**Consequences.** Phase 0 pins only its local dependencies. Nothing in this
decision claims the GPU stack has already been tested.

## ADR-003 — Streaming measurement and result validity (2026-09-23)

**Context.** SSE chunks can contain zero, one, or several generated tokens.
The old Phase 1 wording incorrectly requested a timestamp for each token.

**Options.** Count chunks as tokens, retokenize the output, or use the server's
usage field. The first two can disagree with the serving engine.

**Decision.** Timestamp every received **text-bearing** chunk. Empty text,
role/metadata, and final usage-only events do not set TTFT, ITL, or the
last-text E2E time. They are retained as protocol evidence. Token counts come
only from the final `usage` field; missing or mismatched usage makes a request
invalid. TTFT is request start to first text-bearing chunk. ITL is the list of
gaps between text-bearing chunks. E2E ends at the last text-bearing chunk.
TPOT is `(E2E − TTFT)/(output_tokens − 1)` only when there are at least two
output tokens; a one-token output has an empty ITL list and null TPOT. Zero
text-bearing chunks are a failed request, even if a usage event appears.
Cancellation and timeout are recorded with their reason and timing, count
toward the error rate, and are excluded from completed-request latency and
throughput. Failed runs and raw records remain available for investigation.

**Consequences.** ITL measures inter-chunk latency, not exact per-token
latency. Server-generated text and usage must be consistent. `docs/SPEC.md`
Phase 1 wording has been corrected to match §5. See ADR-010 for the timing
validation correction and its mutation evidence.

## ADR-010 — Timing validation reference (2026-09-23)

**Context.** The initial Phase 1 mock waited until one millisecond before the
configured deadline, then busy-spun for the final millisecond to compensate
for coarse asyncio timer granularity. This changed the mock schedule to make
observed intervals closer to configured 200 ms / 20 ms values. Comparing the
client to those configured values could not distinguish a client timing bug
from host scheduling behavior and risked fitting the validation to its target.

**Decision.** Do not compensate for timer granularity: the mock now uses plain
`asyncio.sleep`. During timing validation, a same-process server records its
monotonic handler-start time and a timestamp immediately after each
text-bearing `response.write()` returns (the chunk has been handed to the
server transport). The client and server use the same monotonic clock domain.
Compare client median TTFT with median server handler-start-to-first-write
interval, and client ITL observations with the server's actual consecutive
text-write intervals, within ±5%. Configured 200 ms / 20 ms values identify
the requested delays only; they are not the accuracy reference.

**Consequences.** The check measures client timing fidelity against the
mock's realized scheduling and transport-write timing on that host, not wire
arrival at a remote peer. It still detects wrong-event timestamping: treating
the initial empty chunk as text causes client TTFT to disagree with the
server's first text write. A mutation check must fail and the production code
must be restored before acceptance. The former spin-based Phase 1 numbers are
not considered evidence under this corrected method.

**Revalidation (2026-09-23).** Client median TTFT was 0.202324 s against
server-observed 0.201397 s (0.46% difference); client median ITL was
0.021327 s against server-observed 0.021258 s (0.33% difference). Both passed
the ±5% gate. The empty-chunk-as-text mutation failed at 99.7% TTFT difference
and was restored. See `results/validation/timing_accuracy_summary.json` and
`results/validation/timing_mutation_failure.txt`. The 30-second capacity
recheck produced 40,405.8 text chunks/s (6.73× the 6,000/s threshold), 0.98
average client cores, no errors or rejections, and 256 requests completed only
during the bounded drain (reported late; excluded from throughput). The
measurement window alone supplies the rate numerator.

## ADR-004 — Explicit generation and fair comparisons (2026-09-23)

**Context.** A model's generation config can alter server defaults. Prefix
caching and reused prompts could exaggerate vLLM gains.

**Options.** Rely on model defaults or set and record parameters explicitly.

**Decision.** Set every relevant generation parameter in each request or
server config for both engines: greedy selection, output-token limit, EOS
handling, penalties, stop conditions, seed where relevant, and usage streaming.
Use the same seeded, unique 512-token ID prompts for compared runs. Assert
`usage.prompt_tokens == 512` and `usage.completion_tokens == 256`. Disable
vLLM prefix caching in performance runs; its current CLI exposes
[`--no-enable-prefix-caching`](https://docs.vllm.ai/en/latest/cli/serve/).
Record the full engine and request settings. Pin the model and tokenizer
revisions when final configurations are prepared.

**Consequences.** A comparison fails if output lengths differ. The HF and
vLLM endpoints may use different setting names, so tests must verify the
behavior rather than assume equivalent names mean equivalent behavior.

## ADR-005 — Load-generator capacity (2026-09-23)

**Context.** A localhost client can become the bottleneck at high concurrency.
Before GPU measurement, Phase 1 needs a quantitative local capacity gate.

**Options.** Assert that 256 connections open, or require sustained event
processing. Connection count alone says little about throughput.

**Decision.** Set a **provisional** planning peak of 2,000 text-bearing
chunks/second from the real server at concurrency 256. This is an assumption,
not a measured fact. Against a near-zero-latency mock, require the client to
handle at least **6,000 text-bearing chunks/second for 30 seconds at 256
concurrent streams** (3× headroom), with no client errors, and report client
CPU use. The CPU target is at most two full cores so a later 4–8 core
container has room for the server. Phase 1 must record the achieved rate,
CPU measurement method, host, and limits. Revalidate within the actual Modal
container in Phase 3 or 6; revise the capacity target if observed GPU event
rates exceed 2,000/s, before accepting performance comparisons.

**Consequences.** Passing on the laptop alone does not establish capacity in
Modal. The original 41,960.7 chunks/s claim has no retained matching raw run;
it is withdrawn as evidence. The retained pre-audit replacement is
`results/validation/capacity_summary.json` with its compressed request records.
The current audit run is in `results/validation/audit/`; see ADR-011.

## ADR-006 — Time-boxed windows and draining (2026-09-23)

**Context.** HF runs can have large queues. Counting work finished after a
deadline would overstate their measured throughput.

**Options.** Count every eventual completion, immediately cancel at the
deadline, or separate the measurement window from a bounded drain.

**Decision.** Keep warmup requests entirely outside the measurement sample.
At the measurement deadline, stop admitting new requests. Throughput uses
only successful requests **started and completed inside** the fixed window,
divided by the actual window duration. Report requests still in flight at the
deadline separately. Allow a bounded drain to preserve their outcomes and
latencies, labelled `late_completion`; exclude them from window throughput
and headline latency percentiles. At the drain deadline, cancel and record
remaining work as `drain_timeout`. Record warmup, window, drain duration,
admitted, on-time completed, late completed, cancelled, and errored counts.

**Consequences.** A timed-out or overloaded baseline cannot look faster by
dropping queued work. Success, error, and unfinished rates remain visible.

## ADR-007 — Budget and local phase boundaries (2026-09-23)

**Context.** Modal charges for CPU and memory as well as GPUs. The user
authorized only local, $0 work through Phase 2.

**Options.** Treat CPU-only Modal runs as free or defer all remote execution.

**Decision.** Phases 0–2 use only the local machine. We may inspect account
metadata, but run no Modal function, image build, GPU, or persistent app in
this session. Before each later GPU run, estimate GPU, CPU, memory, and any
other relevant costs; record the running total and await an explicit yes.
Keep the spec's $25 cumulative stop, per-phase caps, and $30 out-of-pocket
limit. Current [Modal pricing](https://modal.com/pricing) lists $30/month
Starter credit and per-second resource rates, but remaining credit has not
been checked.

**Consequences.** Phase 3 download and smoke jobs need cost authorization,
even though downloading uses CPU-only functions.

## ADR-008 — Evaluation commands to verify in Phase 5 (2026-09-23)

**Context.** The evaluation harness CLI evolves, and MMLU prompts may exceed
a guessed context limit.

**Options.** Freeze the old spec command or verify at the later pinned version.

**Decision.** In Phase 5 use the pinned package's `--help` and task list.
The [current interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)
documents `lm-eval run`; its [WikiText task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/wikitext.yaml)
reports `word_perplexity`. Measure actual five-shot MMLU prompt lengths and
choose a context limit that fits them before full evaluation.

**Consequences.** No evaluation package or GPU work is added in Phase 0.

## ADR-009 — Hugging Face baseline scheduling and exact output length (2026-09-23)

**Context.** Phase 2 needs a local reference server with fair token counts and
streaming behavior that can later be compared with vLLM. `TextIteratorStreamer`
buffers text and does not preserve one event per generated token for a batch.

**Options.** Serialize all requests, or collect a fixed group briefly and run
one padded `generate()` call. Use token events or decoded word events.

**Decision.** One worker owns model generation. Naive mode dispatches a batch
of one; static mode gathers up to `batch_size` jobs for `batch_wait_ms`, groups
only equal output limits, left pads inputs, and passes an attention mask.
A custom `BaseStreamer` sends one generated token ID per batch row to the
event loop; it skips the initial prompt callback. Decoding happens per token,
so the final `usage` count is generated token IDs, not text chunks or
retokenized text. The load client requires `usage.completion_tokens` to equal
requested `max_tokens`; token-ID prompts also require exact prompt usage.

Every generation call uses a fresh explicit `GenerationConfig`: greedy
(`do_sample=False`, one beam and return), `max_new_tokens` and
`min_new_tokens` both equal the requested limit, one cache policy, neutral
penalties, explicit BOS/EOS/pad IDs, and no forced tokens or stop sequence.
The API accepts only explicit greedy streaming request settings and rejects
unsupported stop sequences. The CPU CLI selects float32; the future GPU
baseline selects BF16 and SDPA. Pinning model and tokenizer revisions and
choosing the static batch size by an OOM probe remain Phase 6 work.

**Verification.** The deterministic tiny GPT-2 test model is configured to
prefer EOS immediately. With `min_new_tokens=3`, naive and static mode each
return three generated tokens; the static test proves a batch of two with
unequal prompt lengths. Temporarily setting `min_new_tokens=0` gives one
generated token and the load client marks both requests as errors with
`completion_tokens 1 != max_tokens 3`. The normal setting is restored by the
default path. A separate `sshleifer/tiny-gpt2` CPU run saved four requests per
mode in `results/validation/hf_{naive,static}_cpu.{json,jsonl}`; all eight
completed with exactly three output tokens and no errors. These runs are
functional checks, not performance comparisons.

## ADR-011 — Adversarial audit before Phase 3 (2026-09-23)

**Context.** Prior acceptance proved only a narrow happy path. Pooled timing
medians hid a corrupt request, the batching test observed the scheduler's own
counter, and identical token outputs could hide row-routing errors. The pinned
Transformers 4.56.2 `_prepare_generation_config` replaces global-default-valued
fields with model defaults unless `use_model_defaults=False` is passed.

**Decision.** Explicitly disable that fallback. Test with hostile sampling,
beam, repetition, and forced-EOS model defaults. Reject unsupported request
settings instead of silently ignoring them. Preserve the EOS-preferring tiny
model tests and run the short-output mutation in both scheduling modes.
Observe the actual `generate()` invocation and returned sequences; test
unequal input lengths, padding/masks, distinct per-row tokens, and delivery
before `end()`. This retains real Transformers generation with a deterministic
forward hook for row-specific logits, not a replacement fake generate method.

Replace the median-only timing gate with matched per-request/per-gap checks
for TTFT, E2E, TPOT and ITL, all at the existing 5% tolerance. Use client
request start for the server TTFT/E2E reference; retain handler-based medians
only for historical context. Save all server write timestamps in the summary.
A deliberately corrupted single request must fail even when medians pass.
This supersedes ADR-010's median-only acceptance rule. The server and client
share one loop/clock: this is localhost fidelity evidence, not remote arrival
truth or a substitute for the Phase 6 independent vLLM cross-check.

Require a complete SSE terminator and one usage object with nonnegative integer
counts and a consistent total (booleans are not counts). Exact prompt/output
lengths still come from usage, not chunk counting. Save failed requests, then
make the CLI exit nonzero on errors/rejections. Capacity acceptance must also
reject errors/rejections, regardless of rate. Open-loop windows now wait until
the deadline before draining and safely handle zero arrivals.

**Evidence.** `audit_failing_tests.txt` records 8 failures before fixes;
`audit_additional_failures.txt` records two missing acceptance guards;
`audit_settings_failures.txt` records three silently ignored settings. All are
under `results/validation/`. `audit_check.txt` records the final full check.
`audit/actual_batch.json` contains the real generate inputs, outputs and config;
`audit/timing_accuracy_summary.{json,jsonl}` and
`audit/capacity_{summary.json,requests.jsonl.gz}` retain the new measurements.
Reproduce commands are in README. Prior results remain unmodified.

**Evidence limits.** The Phase 0 account/setup claims and historical test counts
lack saved command transcripts and are not independently reverified here.
Analytical memory/bandwidth estimates in ADR-001 are calculations from model
config, not benchmark results. The old HF CLI summaries prove 4/4 completions
per mode but do not contain server batching/config traces; use the new actual
batch evidence for batching assertions. Initial spin-based timing/capacity
numbers lack the matching retained raw run and are not accepted evidence.
No GPU performance, accuracy, or exact 512/256-token workload has been validated.
The present CLI still constructs variable-length text prompts; pinned,
unique token-ID workload construction must be implemented before GPU comparisons.

## ADR-012 — Modal and Docker plumbing for Phase 3 (2026-09-23)

**Context.** Phase 3 must prove that the vLLM image built from our Dockerfile
and the HF baseline image both serve on a Modal GPU, with bounded cost, before
any L40S spend.

**Decision.**
- *Image.* `docker/Dockerfile` is `FROM vllm/vllm-openai:v0.10.2@sha256:607442e4…`
  (the multi-arch index digest resolved on 2026-09-23). Our `src/llmbench` is
  copied and put on `PYTHONPATH` rather than pip-installed, so the image's CUDA
  torch (2.8.0+cu128) and vLLM's resolved dependencies are untouched. The image
  therefore runs its own versions (transformers 4.56.1, aiohttp 3.12.15), which
  differ from the local lock (4.56.2, 3.14.3); every result records the
  versions it actually ran. The base ships only `python3`. Modal failed with
  "unable to determine the version of Python", so the Dockerfile adds a
  `python` symlink, the same step Modal's `add_python` images perform. Modal
  clears the inherited `api_server` ENTRYPOINT with `.entrypoint([])`; Compose
  keeps it. The HF image is `debian_slim` Python 3.12 with the local pins
  (torch 2.8.0's default Linux wheel is CUDA 12.8).
- *Volumes.* `llmbench-weights` (revision-pinned snapshots plus verified
  manifests), `llmbench-results`, and `llmbench-vllm-cache` mounted at
  `/root/.cache/vllm`. The eager smoke claims no cache reuse.
- *Bounded functions.* Every function sets CPU and memory as request == limit
  (billing uses max(request, actual), so the limit caps it), an execution
  timeout, a 300 s startup timeout, `retries=0`, `max_containers=1` and no
  deploy. Runs use `modal run --detach`. Downloads run CPU-only with the HF
  secret; GPU functions get no secret and run with `HF_HUB_OFFLINE=1`. The
  local entrypoint refuses to request a GPU unless a verified weights manifest
  exists (every file's sha256 is checked against the Hub's LFS metadata).
- *Smoke driver.* `llmbench.smoke.run_server_smoke` is the one code path for
  Modal and the CPU rehearsals (mock server; tiny on-disk GPT-2 through the
  real HF server subprocess). It starts the server in its own process group,
  waits for `/health`, runs warmup plus fixed closed-loop requests, saves raw
  records and the log, validates, and always kills the group in `finally`.
- *Workload.* A seeded pool of distinct 512-token prompts sampled uniformly
  from the tokenizer's base vocabulary minus special IDs. Each prompt is used
  once per server lifetime, and each system gets the same pool (the SHA-256 is
  recorded). This is a functional workload; the real-text W1 corpus remains
  Phase 6 work. The same payload goes to both engines: `ignore_eos: true` is
  vLLM's extension; the HF server accepts it (it always enforces
  `min_new_tokens == max_new_tokens`) and rejects `false`, which it could not
  honor.
- *Engine flags.* `--no-enable-prefix-caching`, `--generation-config vllm`
  (model sampling defaults ignored), and `--enforce-eager` (smoke only).
  vLLM 0.10.2 cannot build its CLI parser on a CPU host (`DeviceConfig` raises
  "Failed to infer device type"), so flags are verified on the GPU host against
  the pinned `vllm serve --help` before the server starts. `--help=<word>` is a
  keyword filter in 0.10.2 (`--help=all` matched only names containing "all");
  the unfiltered `--help` is used. HF static batches are read from the server's
  own `/stats` counter of real `generate()` calls.

**Consequences.** Both smokes passed on L4 (see PROGRESS). The smoke numbers
are functional evidence only: L4, a 0.6B model, eager vLLM, 16 requests, and
no repeats. They must not be compared or quoted as performance.

## ADR-013 — In-container client capacity is below the gate (2026-09-23, open)

**Context.** ADR-005 requires revalidating the 6,000 text-chunks/s client
capacity inside the Modal container before performance comparisons.

**Observation.** The unchanged Phase 1 validation ran CPU-only inside our vLLM
image (gVisor, 4 cores requested, `os.cpu_count()` = 4; the mock server ran as
a separate process in the same container). Timing passed: TTFT 0.20253 s vs
server 0.20120 s, ITL 0.021522 s vs 0.021497 s. **Capacity failed: 3,786.1
chunks/s (0.63× the 6,000/s gate)**, with the client process at 0.995 cores,
zero errors/rejections, and 256 late completions reported separately. The laptop
reached 36,538/s. The single-process asyncio client is CPU-bound on this host.
Raw files: `results/validation/phase3/vllm-env-20260923T093629Z/`.

**Decision.** The gate is not relaxed. Phase 3's own smokes run at
concurrency 4 with only a few hundred chunks/s, so they are unaffected. **This
blocks the Phase 6 concurrency-128/256 comparisons** until resolved. Options
for Mohammed:
(a) shard the load generator across worker processes (merging records on one
monotonic clock) and rerun the in-container gate, estimated at under $0.02 CPU;
(b) measure the real peak chunk rate at c=256 on L40S first, then require 3×
that rate. Recommended: (a), because it keeps the existing 6,000/s bar.

**Resolution (approved by Mohammed, 2026-09-23).** Option (a). The load
generator becomes multi-process: worker processes each run the existing
closed/open-loop client over a disjoint share of the virtual users, record
timestamps on the same host monotonic clock (`time.perf_counter` is
system-wide on Linux), and the parent merges records before the unchanged
summary/window logic. It is revalidated inside the Modal container against
the **unchanged** 6,000 chunks/s, 30 s, 256-stream, zero-error gate, plus the
timing gate, for an estimated ~$0.02 of CPU. This must pass before any Phase 6
run; Phases 4–5 do not use the load tester.

**Implementation (2026-09-23, $0; in-container run pending approval).**
- `run_load(processes=P)`: the parent runs shard 0 and spawns P−1 worker
  processes (`spawn` context). Shard p gets an even share of users, warmups,
  request count and open-loop rate (independent seeds `"{seed}/{p}"`).
  Measured indices are p, p+P, …; warmup indices −(1+p), −(1+p+P), …, so the
  indices never collide and a request-count run covers exactly 0..N−1. In
  open-loop mode the admission cap (rejections) applies per shard, to its
  share of `concurrency`; independent Poisson streams sum to the total rate.
- After every shard's warmup, the parent picks one window start 0.25 s ahead
  and sends it to the workers. Workers return their records; the parent
  merges them, relabels late completions against the common window end and
  calls the unchanged `summarize`. Duplicate request IDs are an error.
- Clock check: each worker records `perf_counter()` when the start message
  arrives. It must lie between the parent's send time and its receipt of the
  results, or the run fails. A per-process clock would fall outside.
- The timing gate now also runs through two processes with the mock in the
  parent, so a worker's timestamps are compared with the parent's server
  write timestamps.
- CPU is summed over all client processes. The capacity gate uses **2 client
  processes**: each is single-threaded, so the 2-core limit holds by
  construction, and the question is only the chunk rate. Each client process
  gets its own mock process so the mock is not the bottleneck; mock CPU is
  recorded (descriptive).
- Payload factories must be picklable (`TextPayloads`, `PoolPayloads`).
  `TokenPromptPool.take` issues prompts in call order, so copies in several
  processes would reuse prompts; `PoolPayloads` maps each request index to
  its own prompt instead.

**Local revalidation** (`make mock-validate
MOCK_VALIDATE_DIR=results/validation/multiprocess-local`, final code):
**88,246.5 chunks/s (14.71×)** with 2 client processes at 1.94 cores in
total, mock processes at 0.74 cores each, 0 errors or rejections, 256 late
completions reported separately. The full timing gate (ADR-019) passed with
1 and 2 client processes (ITL p99 0.42% / 0.39% over 4,000 gaps). The 46.1 MB
raw request file stays local (gitignored, sha256 `4751c849…917912`); the
summaries are committed. An earlier run of this commit's code, before the
final timing plan, gave 88,943.0 chunks/s.

**In-container revalidation: passed** (2026-09-23, `make modal-loadtest-check`,
app `ap-alL2wtq71x0HiFGB2iF5IT`, CPU only, 8 cores / 4 GiB, **$0.028004**).
Inside the vLLM image under gVisor (`os.cpu_count()` = 8):
- Timing, 1 client process: ITL p99 0.56% (4,000 gaps); TTFT/E2E/TPOT
  ≤ 0.12% at p50 and p99. Through a worker process: ITL p99 1.54%.
- Capacity: **9,124.1 chunks/s (1.52× the 6,000/s gate)** at 1.86 client
  cores (25.9 s and 29.9 s of CPU over 30 s), 0 errors or rejections, clock
  check passed. The mock serving shard 0 was at 0.95 cores, so 9,124/s is a
  lower bound on the client's capacity, not its limit. Phase 3's
  single-process client managed 3,786/s here.
- Evidence: `results/validation/phase6/loadtest-check-20260923T205147Z/`.

**Consequence for ADR-005's headroom.** The 6,000/s gate was 3× a
provisional 2,000/s peak. With `skip_special_tokens=false`, vLLM sends about
one chunk per token, so the peak chunk rate is roughly the peak output
tokens/s. If the probe measures more than ~3,040 tokens/s at c=256, the
validated 9,124/s is less than 3× the real peak. ADR-005 then requires
revising the target before comparisons are accepted, and the probe decides.
Every Phase 6 point also records the client's CPU.

## ADR-014 — vLLM empty text chunks distort per-request E2E/TPOT (2026-09-23, open)

**Observation.** In the vLLM smoke, 14 of 16 requests had 29–32 text-bearing
chunks for 32 usage tokens. Two had only 2 and 1 text chunks, followed by 30–31
empty-text chunks. ADR-003 ends E2E at the last text-bearing chunk, so those
two records show E2E 0.066/0.059 s and TPOT 0.0012/0.0 s, while the other
requests took about 0.79–0.83 s. The HF server returned 32 text chunks for
every request. Usage counts were exact in both engines, so the functional smoke
is valid, but these per-request latencies are wrong for those rows.

**Likely cause (not yet verified).** Greedy decoding with `ignore_eos` on
random-token prompts keeps generating special tokens, which vLLM's default
`skip_special_tokens=true` turns into empty text. The HF server decodes with
`skip_special_tokens=False`.

**Proposal before Phase 6 (needs approval).** Send `skip_special_tokens:
false` to vLLM and accept only `false` in the HF server, so both engines stream
one text-bearing chunk per token. Add a run-level check that fails when
`text_chunks` is far below `completion_tokens`. Keep ADR-003 unchanged, and
record the end-of-stream (usage/`[DONE]`) time as a second E2E field for
diagnosis. The real-text W1 prompts should also reduce this effect.

**Resolution (approved by Mohammed, 2026-09-23).** Requests to vLLM set
`skip_special_tokens: false`, so both servers decode and stream special tokens
the same way. The HF server accepts that field only with the value `false`
(it already decodes with `skip_special_tokens=False`). A run fails if any
completed request's text-bearing chunk count falls well short of its
`usage.completion_tokens`; the exact threshold is fixed in code and tests
before Phase 6 and recorded here. ADR-003's metric definitions are unchanged.
Implementation and a CPU test land before Phase 6.

**Implementation (2026-09-23).**
- Checked in the pinned source (v0.10.2): `CompletionRequest` has
  `skip_special_tokens: bool = True` (`entrypoints/openai/protocol.py`,
  passed to `SamplingParams`). The smoke driver and `PoolPayloads` send
  `false`. The HF server rejects any value other than `false` (including
  `true`, `null` and `0`); omitting it stays valid.
- **Threshold: at least 0.5 text-bearing chunks per usage completion token**
  for every completed request (`ok` or `late_completion`), in
  `metrics.MIN_TEXT_CHUNKS_PER_TOKEN`. Reason: healthy Phase 3 rows had ≥
  29/32 (0.91) and the broken rows 1–2/32 (≤ 0.06). vLLM's
  `RequestOutputCollector` (`v1/engine/output_processor.py`) merges deltas
  when the API server falls behind the engine, so under load one chunk can
  carry several tokens without distorting E2E. 0.5 tolerates an average of
  two tokens per chunk. A failure message gives the empty-chunk count, so
  merging (few empty chunks) can be told apart from text-less tokens (many).
- Checked by `smoke.validate_run`, reported in every summary
  (`text_chunk_shortfall_requests`), and it makes `llmbench load` exit
  non-zero. Each record also stores `stream_end_s` (start to `[DONE]`) for
  diagnosis only.
- Tests: the unit boundary (exactly 0.5 passes); a mock that streams the last
  3 of 4 tokens as empty text fails the smoke, although every record is `ok`
  by usage; and a threshold-0 mutation makes both tests fail.
- **Risk for Phase 6:** if vLLM merges more than two tokens per chunk at
  c=256, this check fails that sweep point. The results are still saved, and
  the empty-chunk counts will show which case it was.

## ADR-015 — Budget hard stop from free credits (2026-09-23)

**Context.** Mohammed reports that no payment card is on file with Modal and
that the account showed $30 of Starter credit before Phase 3.

**Decision.** No Modal workspace budget is configured: without a card, the
free credit is a hard stop. The project rules are unchanged: $25 cumulative
target, per-phase caps, and an explicit estimate and yes before each GPU run.
The balance before Phase 3 comes from Mohammed's report; Claude has not read
it from the dashboard.

**Consequences.** Remaining credit is about $30 − $0.0975 = **$29.90** by our
ledger. A job that hits the credit limit could be stopped mid-run, so long jobs
keep saving results incrementally to the Volume.

## ADR-016 — Phase 4 quantization stack and procedure (2026-09-23)

**Context.** ADR-002 left the llm-compressor version and exact recipes to be
fixed before Phase 4. The checkpoints must load in the pinned
`vllm/vllm-openai:v0.10.2`, whose `compressed-tensors` is 0.11.0.

**Options.** llm-compressor 0.7.1 pins `compressed-tensors==0.11.0`
(transformers ≤4.55.2, torch ≤2.8.0). 0.8.x writes with compressed-tensors
0.12.x, a format version vLLM 0.10.2 was not built against. 0.9+ also moves
torch/transformers. The 0.7.1.x post-releases (2026-07) carry the same pins.

**Decision.**
- Pin `llmcompressor==0.7.1`, `compressed-tensors==0.11.0`, torch 2.8.0,
  transformers 4.55.2, datasets 4.0.0, accelerate 1.10.0
  (`requirements/quantize.in`; resolved Linux set in
  `requirements/quantize-linux.txt`). This runs in its own Modal image and a
  local `.cache/quant-venv`, because the serving stack needs transformers 4.56.
- Model `Qwen/Qwen3-8B@b968826d…` (16,397,461,266 bytes), downloaded CPU-only
  and verified file-by-file against the Hub's LFS sha256.
- Recipes follow the official **0.7.1** examples exactly
  (`configs/phase4_quantize.yaml`):
  - **AWQ:** `AWQModifier(targets=[Linear], scheme=W4A16_ASYM,
    ignore=[lm_head])`, calibrated on `mit-han-lab/pile-val-backup@2f5e46ae…`
    `validation[:256]` at max 512 tokens, each text wrapped as one user turn.
  - **GPTQ:** `GPTQModifier(targets=Linear, scheme=W4A16, ignore=[lm_head])`
    (symmetric, group 128), calibrated on
    `HuggingFaceH4/ultrachat_200k@80496310…` `train_sft[:512]` at max 2048
    tokens.
  - **Both:** `shuffle(seed=42)` after slicing, as in the examples, and
    `add_special_tokens=False`. The two methods keep their own official
    calibration sets rather than a shared one, so the comparison is "each
    method as recommended". Any accuracy difference therefore mixes method and
    calibration data; RESULTS must say so.
- Qwen3's chat template renders assistant turns with an empty
  `<think>\n\n</think>` block (visible in the saved decoded prefix). That is
  the template's behavior, left as is. `pile-val-backup` has no license on its
  card; it is used only for calibration and never redistributed.
- **Deviation from the AWQ example:** preprocessing (template + tokenization)
  runs ahead of time on CPU, and `oneshot` receives token IDs, as in the GPTQ
  example. It is equivalent because the Qwen tokenizer adds no special tokens.
  The saved calibration records sample counts, token totals and an input-ID
  sha256.
- **Procedure:** AWQ runs first (L40S, 4 cores / 48 GiB, 60 min timeout). Its
  actual cost is reconciled, then GPTQ is re-estimated (64 GiB for its larger
  activation cache). The checkpoint is written in place; a verified manifest,
  written last, is the only completion marker. A verified checkpoint is never
  overwritten.
- **Checks after saving:** compressed-tensors `pack-quantized`, 4 bits, group
  128, the expected symmetry, `lm_head` ignored and stored in BF16, packed
  q_proj tensors, and zero points if and only if asymmetric.
- **Recorded measurements:** wall time per stage, torch peak allocated/reserved
  memory, peak `nvidia-smi` memory (2 s sampling), host peak RSS, the full
  package set, and every file's size/sha256.

**CPU rehearsal.** `make quantize-rehearsal` runs the same functions and both
recipes on a tiny random Qwen3 with the real Qwen3-8B tokenizer. Both passed
(`results/validation/phase4/rehearsal/rehearsal.json`). compressed-tensors
0.11.0 cannot decompress packed zero points inside transformers
("Decompression of packed zero points is currently not supported"), so only
the symmetric checkpoint gets an HF reload/forward check. The vLLM 0.10.2
source registers `weight_zero_point` for asymmetric WNA16 schemes, and the
Phase 4 vLLM sanity run is the real load test for AWQ.

**Consequences.** Accuracy and serving numbers apply to this stack only.
Newer llm-compressor releases are not evaluated.

**Outcome (2026-09-23).** Both recipes passed every structural check on
L40S: AWQ in 538 s ($0.3855) and GPTQ in 973 s ($0.7261). vLLM 0.10.2 loaded
both with `MarlinLinearKernel` for `CompressedTensorsWNA16`, as well as the
BF16 original. On-disk safetensors shrank 62.77% (AWQ) and 62.94% (GPTQ);
vLLM's model-loading memory shrank 62.61% and 62.78%. These are consistent
with the ADR-001 analytical estimate and below the 68% placeholder. Details
are in PROGRESS; the final claims wording is Phase 7/8 work.

## ADR-017 — Phase 5 accuracy stack and protocol (2026-09-23)

**Context.** SPEC Phase 5 asks for MMLU 5-shot and WikiText-2 word perplexity
for BF16, AWQ and GPTQ with lm-eval's vLLM backend, identical settings, no
chat template, and a context limit verified against the longest 5-shot prompt
(ADR-008). The serving image pins vLLM 0.10.2.

**Options.** lm-eval 0.4.13 (newest, 2026-08-31) and 0.4.12 declare
`vllm>=0.18` for their vLLM extra. 0.4.11 (2026-02-13) is the newest release
that still accepts older vLLM. The alternatives would be upgrading vLLM for
evaluation only (a different engine from the one we serve) or lm-eval's HF
backend (not the SPEC's backend, and transformers cannot decompress the AWQ
zero points; see ADR-016).

**Decision.**
- *Stack.* `lm-eval==0.4.11`, `datasets==4.1.1` (the newest release whose
  fsspec ≤2025.9.0 and dill <0.4.1 bounds admit the image's packages) and
  `evaluate==0.4.6`, installed on top of the Dockerfile image
  (`EVAL_IMAGE`). The resolved set (`requirements/eval-linux.txt`) was
  compiled with the image's own Phase 3 `pip freeze` as constraints
  (`requirements/vllm-image-constraints.txt`): it **adds 33 packages and
  changes none**. The prefetch step re-checks this inside the built image.
  Every vLLM call in 0.4.11's backend was checked against the v0.10.2 source:
  `resolve_hf_chat_template(..., model_config=)`, the fallback imports
  `vllm.transformers_utils.tokenizer.get_tokenizer` and `vllm.utils.get_open_port`,
  `TokensPrompt`, `LLM(swap_space=...)`; `ray` (imported at module load) ships
  with the CUDA image.
- *CLI.* 0.4.11 uses the `lm-eval run` subcommand (bare `lm-eval --model ...`
  still works); flags were read from its `--help`. `--num_fewshot` applies
  to every task in one call, so each variant runs two processes with
  identical `--model_args`: `--tasks mmlu --num_fewshot 5` and
  `--tasks wikitext` (zero-shot, task default).
- *Settings* (`configs/phase5_accuracy.yaml`, identical for all variants,
  only `pretrained` differs): `dtype=bfloat16`, `max_model_len=4096`,
  `gpu_memory_utilization=0.75` (0.80 in the probe; see below),
  `enable_prefix_caching=false`,
  `enforce_eager=true`, `seed=1234`, `--batch_size 1024`,
  `--seed 0,1234,1234,1234`, no `--apply_chat_template`, `--log_samples`.
  Every variant uses the BF16 checkpoint's tokenizer files.
- *Context limit.* A stub model fed through lm-eval's own `simple_evaluate`
  (`llmbench.eval_audit`) recorded the token length of every request exactly
  as the vLLM backend tokenizes it. MMLU 5-shot: 56,168 requests (14,042 ×
  4 choices), 39,160,912 tokens, mean 697, **longest 3,097**
  (`high_school_european_history`); WikiText-2: 104 rolling windows,
  347,162 tokens, longest 4,095. The backend left-truncates above
  `max_length − 1 = 4,095`, so **no request is truncated at 4096**. Any
  "Truncating context" warning in a run log fails that run.
- *No prefix-cache shortcut.* vLLM 0.10.2 skips the prefix cache for any
  request with `prompt_logprobs` (`vllm/v1/core/kv_cache_manager.py`), and
  lm-eval sends each of the four choices as a separate request, so all ~39.2M
  MMLU tokens are prefilled per variant. Caching is still set off explicitly.
- *Memory.* `--batch_size auto` would pass all 56,168 requests to one
  `generate()` call and keep every prompt position's logprob dict (~39M
  Python dicts, ~20 GB) alive at once, so requests go in fixed chunks of
  1,024. On vLLM this only groups requests into `generate()` calls; the engine
  still schedules its own token batches. Prompt-logprob logits (up to ~3.7 GB
  for a 4,095-token window) are allocated outside vLLM's memory profile, hence
  0.80 GPU memory utilization. Log-likelihood requests are prefill-only
  (`max_tokens=1`), so CUDA graphs (decode-only) are skipped with
  `enforce_eager`.
- *Data.* A CPU-only prefetch downloads both datasets through lm-eval's own
  task loading into the weights Volume (`eval-cache/hf-home`), recording the
  Hub commit in each cache folder name, and reloads them offline. GPU runs
  copy that cache to local disk and run with `HF_HUB_OFFLINE`/
  `HF_DATASETS_OFFLINE`, so all variants read byte-identical data. The
  MMLU few-shot examples are the first five `dev` questions of each subject
  (lm-eval's `first_n` sampler), so the prompts are deterministic. Each run
  stores a SHA-256 fingerprint of lm-eval's per-question `prompt_hash` values;
  the comparison requires equal fingerprints across variants.
- *Metrics.* The MMLU headline is lm-eval's `mmlu` group accuracy, weighted by
  subject size (all 14,042 questions); the macro mean over 57 subjects is
  secondary. **Accuracy delta = BF16 − quantized, in absolute percentage
  points**; a relative drop is shown only as a labelled secondary number. The
  per-subject extremes list each subject's question count (small subjects are
  noisy: in a 100-question subject, 1 question = 1 pp), and paired
  per-question flips (lost/gained vs BF16) are reported. WikiText-2 reports
  `word_perplexity` over the 62 test documents, with lm-eval's disjoint
  rolling windows of `max_length − 2 = 4,094` tokens; perplexity depends on
  that window, so it is fixed for every variant.
- *Calibration caveat.* AWQ was calibrated on pile-val (256 × 512 tokens) and
  GPTQ on UltraChat (512 × 2,048), each following its official example
  (ADR-016). An AWQ-vs-GPTQ difference therefore mixes method and calibration
  data and must be reported that way.
- *Base-style prompts.* Qwen3-8B is the post-trained (hybrid thinking) model.
  Scoring it with plain 5-shot prompts and no chat template is standard
  log-likelihood MMLU and is the same for every variant. The absolute numbers
  are not directly comparable to Qwen's published figures, which use their
  own harness.

**Procedure.** CPU rehearsal (`make eval-rehearsal`: the same `run_variant`
code with lm-eval's hf backend on tiny random Qwen3s) → CPU prefetch
(BILLABLE, tiny) → timed probe on BF16 with `--limit` (MMLU first 10 questions
per subject = 2,280 requests / 1,382,712 tokens; WikiText first 5 documents =
8 windows / 27,069 tokens) → a measured full-run estimate → full run, each
step with its own estimate and approval.

**Probe outcome and one change (2026-09-23).** The BF16 probe at 0.80 scored
10,555 MMLU tokens/s and passed every check, but `nvidia-smi` peaked at
45,233 of 46,068 MiB: the prompt-logprob logits for a 4,095-token window
exceed my ~3.7 GB estimate once allocator caching is included. Full runs use
**0.75** for all three variants (~3 GiB headroom). This bounds memory only:
the KV cache (~125k tokens) still far exceeds the 8,192 tokens scheduled per
step, and no score depends on it. The probe is a timing measurement, not a
result.

**Outcome (2026-09-23).** All three variants were evaluated on every MMLU
question (14,042) and WikiText-2 document (62) with identical settings and
byte-identical prompts (equal fingerprints), with 0 truncations:

| Variant | MMLU 5-shot | Δ vs BF16 (pp) | Paired McNemar p | WikiText-2 word ppl |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 74.88% ± 0.35 | — | — | 12.742 |
| AWQ W4A16-asym (pile-val) | 73.93% ± 0.35 | 0.95 | 6.1e-5 | 13.282 |
| GPTQ W4A16-sym (UltraChat) | 73.24% ± 0.36 | 1.65 | 7e-12 | 13.662 |

The per-variant lm-eval standard errors ignore that all variants answer the
same questions; the paired McNemar test on per-question outcomes is the
measure of the difference, and both drops are significant. AWQ vs GPTQ:
0.69 pp, p = 0.011, confounded by calibration data. The "within 1.5%" resume
wording holds for AWQ in absolute points (0.95 pp; 1.27% relative) but not
for GPTQ (1.65 pp; 2.20% relative); final wording is Phase 8 work.

## ADR-018 — Phase 5 budget reallocation and launch-independent runs (2026-09-23)

**Context.** The probe-based full-run estimate (~$7.94) exceeded Phase 5's
$6 cap, and the first full run lost GPTQ when the laptop went to clamshell
sleep: the local entrypoint was blocked on `.remote()`, and Modal cancelled
the input ~100 s after the client went silent, despite `--detach`.

**Decision.**
- *Budget.* Mohammed moved Phase 4's unused $4.73 and then Phase 3's unused
  $0.90 to Phase 5, making its cap **$11.63**, on condition that the
  projected project total (including the Phase 6 estimate) stays under $25.
  Full 5-shot MMLU was kept; no subsetting or 0-shot shortcut. Phase 5
  actual: **$10.7004**.
- *Launch.* Full runs `.spawn()` the Modal function and the entrypoint exits,
  so the job does not depend on the laptop's client. A single variant runs as
  `eval_one` (5,400 s timeout) rather than inside the three-variant envelope.
  The report can combine variants from runs with the same config sha256 and
  refuses unequal prompt fingerprints or per-question results that do not
  reproduce the delta.
- *Verification.* After launching, `modal app list` must still show the app
  running a few minutes after the entrypoint exits; the GPTQ rerun did.

**Consequences.** BF16 and AWQ come from `full-20260923T142440Z`, GPTQ from
`full-20260923T175000Z` (same config and package versions). Phase 6's long
runs should use the same spawn pattern. Remaining against the $25 target:
$12.93, with Phase 6 estimated at $6.72 (envelope $9.48).

## ADR-019 — Timing gates: strict medians, a tail bound sized to the sample (2026-09-23)

**Context.** `test_mock_timing_accuracy` failed in 2 of 25 full-suite runs
(0 of 6 alone). The gate required every paired TTFT, E2E, TPOT and ITL to be
within ±5% of the mock's own write timestamps, on 5 requests × 4 tokens;
medians were computed but not gated. With a 20 ms mock ITL, ±5% is 1 ms.

**Cause: host CPU contention, not the client.** Scratch probes, laptop with
10 cores (`results/validation/timing-investigation/`):
- A large heap is not the cause: with 6.3M live objects (torch and
  transformers imported), 30/30 runs passed, with no gen-2 GC during
  measurement. Four instrumented full-suite runs also logged no GC pause over
  0.5 ms during any timing test.
- Contention is. Old gate: 0/30 failures quiet, 3/30 with 2 cores spinning,
  7/30 with 6, and 30/30 with all 10. Failing gaps were off by 5–92% (up to
  18.5 ms); the worst TTFT miss was 16.7% (33 ms, 6 cores spinning). Medians
  stayed within 0.6% with up to 6 cores busy.
- The per-chunk delivery delay (client receipt − server write, same clock)
  was 0.1–0.2 ms at p50 and 0.5–0.9 ms at p99, never negative, with
  occasional 2.5–6.4 ms spikes. A spike stretches one gap and shrinks the
  next by the same amount. That is the process being descheduled between the
  mock's write and the client's read: attribution and the clock are right.

**First attempt, rejected.** p50 and p99 at ±5% on 10 × 21. With 10
requests, a p99 of TTFT/E2E/TPOT is just the maximum, so this was still the
every-request rule for those metrics. It failed 3 of 25 full-suite runs (plus
1 of 6 in an earlier loop) while the laptop was busy (load average ~7.7).

**Decision: two gates, each sized to what its sample supports.**
- **Unit gate** (`make check`; `TIMING_UNIT`, 10 sequential requests × 21
  tokens): structural completeness; the client's **p50** of TTFT, E2E, TPOT
  and ITL within **±5%** of the paired server references (unchanged
  tolerance, now actually gated); and **every** paired observation within
  **50 ms** absolute. The bound is absolute because scheduling delays are
  absolute time, not proportional to the interval: the same spikes hit 20 ms
  gaps and 200 ms TTFTs. 50 ms is 1.5× the worst delay measured under
  artificial heavy contention (33 ms), and half the 100 ms corruption that the
  audit regression test injects.
- **Full gate** (`make mock-validate` and the in-Modal check; `TIMING_FULL`,
  200 sequential requests × 21 tokens, i.e. 200 TTFT/E2E/TPOT and 4,000 ITL
  samples per run): structural completeness; client **p50 and p99** of all
  four metrics within **±5%**. It refuses to gate a p99 on fewer than 100
  samples. It runs twice: 1 client process, and 2 (one stream each).
- The full plan uses one stream per client process. A first run with 8
  concurrent streams failed ITL p99 at 6.21% (p50 0.16%): the mock shares the
  client's event loop, so its writes for other streams delay the client's
  reads. A real server is a separate process. Kept in
  `results/validation/multiprocess-local/failed-8-streams-in-process/`.

**What this gives up, and what still fails.** In the unit gate a single
paired observation may now be off by up to 50 ms instead of 5%; systematic
errors are caught by the strict medians, and tail distortion by the full
gate's p99. The unit gate still fails on:
- a 1.2 ms bias on every gap (median);
- one gap 60 ms off, or one request 100 ms late (tail bound);
- the ADR-010 mutation, an empty chunk timed as text (TTFT median > 50% off);
- the unchanged audit regression, one corrupted request.

The full gate fails on 4 ms delays on 10% of gaps (p99, median passing) and
passes one isolated 6.4 ms spike (`tests/test_timing_gate.py`).

**Evidence** (`make mock-validate
MOCK_VALIDATE_DIR=results/validation/multiprocess-local`, final code): 1
process, ITL p99 0.42% (4,000 gaps), TTFT/E2E/TPOT ≤ 0.12% at p50 and p99; 2
processes, ITL p99 0.39%. The final unit gate was not repeated in a loop (the
repeats were stopped because the laptop was overheating); its flake rate is
not measured beyond single `make check` runs.

**Consequences.** SPEC Phase 1 wording is updated. The earlier timing
summaries (ADR-010, audit, Phase 3) used the old sample and rule and stay as
they were. The in-Modal check runs the full gate.

## ADR-020 — Steady-state measurement windows for Phase 6 (2026-09-23)

**Context.** ADR-006 counts only requests that start *and* finish inside the
window, with every user starting at the window's start. At high concurrency
that undercounts by up to one request duration per window (for example,
~20 s of E2E in a 180 s window at c=256, about 10%). Because E2E differs
between systems, the bias would differ too, and it would distort the peak
throughput ratio.

**Decision (Mohammed, 2026-09-23).** Phase 6 uses a steady-state window,
applied identically to vLLM and HF (`run_load(mode="steady")`,
`summarize_steady`). ADR-006 stays in force for the smokes and validations.
- *Warmup.* Users start staggered evenly over `ramp_s`, then run closed
  loops. The window opens at `warmup_s` ≥ `ramp_s` (planned ≥ ramp + one
  expected E2E) and lasts `window_s`. At its end, admission stops and
  requests still in flight are cut (`window_end`, not an error). The next
  point waits until the server reports no running or queued work.
- *Primary metric: output tokens/s.* The server's tokens whose chunk arrived
  inside the window, divided by `window_s`. The counts come from cumulative
  per-chunk usage (`stream_options.continuous_usage_stats`, verified in vLLM
  v0.10.2's `serving_completion.py`; the HF server emits the same field), so
  merged or empty-text chunks are still counted exactly. The final usage
  must equal the last cumulative count, or the request is invalid.
- *Secondary: requests/s.* Requests that complete inside the window, divided
  by `window_s`; latency percentiles come from the same requests. Output
  tokens of completed requests per second is also reported.
- *Validity, recorded per point:*
  - the token rates in the window's two halves must agree within **5%**
    (steady state; a point that fails is invalid);
  - a request-count edge bound, 2B / n, where B is the largest group of
    completions within one median ITL of each other: moving an edge can
    change the count by at most B. Above 5% it is a warning that qualifies
    requests/s only; tokens/s is unaffected.
- *HF baseline.* A disconnected client's queued job is skipped, and a
  `generate()` call stops once every client in its batch has gone
  (`AllCancelled`), so cut requests never run into the next point.
- *Other checks per point:* zero errors, exact 512/256 usage on every
  completed request, per-chunk usage present, the ADR-014 chunk check, and
  zero vLLM prefix-cache hits (counter delta).
- *Prompts.* Each point takes fresh indices from the prompt pool, so no
  prompt repeats within a lifetime and every variant sees the same prompts
  in the same order.

**Consequences.** Throughput no longer depends on how a window aligns with
request starts. A CPU rehearsal against the mock gave 1,262.5 tokens/s from
arrivals vs 1,270.0 from completed requests (0.6% apart) with flat bins.
Found while building it: a failed virtual user (an exhausted prompt pool)
was silently swallowed by `gather(return_exceptions=True)` in both steady and
duration modes. Worker errors now fail the run, with a regression test. The
rehearsal tests use a 50% half-window limit, because ~1 s windows on a busy
laptop are not steady to 5%; the 5% default applies to real runs and has its
own unit test.

## ADR-021 — W1 prompts from WikiText-103 (2026-09-23)

**Decision (Mohammed, 2026-09-23).** The 512-token W1 prompts come from
WikiText-103 raw (`Salesforce/wikitext`, `wikitext-103-raw-v1`, train split,
revision `b08601e0…`).
- Articles are rebuilt at top-level ` = Title = ` headings and tokenized once
  with the pinned Qwen3-8B tokenizer, without special tokens. Each article is
  cut into consecutive non-overlapping 512-token windows; the remainder is
  dropped, so no window spans two articles.
- The first 75,000 windows (1.5× oversampling) are reduced to distinct
  windows without special tokens. 50,000 are then sampled with seed 20260923
  and stored as an int32 array on the weights Volume, with a manifest (pool
  sha256, dataset files and sizes, tokenizer).
- Prompts are sent as token IDs, so there is no re-tokenization drift.
- The pool is larger than any lifetime's planned requests (~25k).

**License.** WikiText is CC BY-SA 3.0 / GFDL, not public domain as SPEC §6's
example suggested. We store and send token IDs only. The manifest carries the
license and a 64-token decoded sample; no text is committed otherwise.

**Rehearsal ($0).** `make bench-prompts-rehearsal` ran the same code on the
WikiText-103 validation file: 60 articles, exact 512-token windows, no
duplicates, and the first prompt decodes to article text and re-tokenizes to
512 tokens (`results/validation/phase6/prompts_rehearsal.json`).

**Checkpoint identity.** The CPU `prepare` step checks every file of all
three checkpoints against the sha256 values Phase 4 recorded (the BF16
download manifest; the AWQ/GPTQ quantize summaries), writing a record per
variant on the Volume. Each GPU lifetime refuses to start unless that record
passed for its exact path and for the same evidence file hash. So AWQ's
speed and accuracy numbers come from the same bytes.
