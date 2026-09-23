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
