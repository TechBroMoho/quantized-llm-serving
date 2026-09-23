# Phase 3 plan — executed 2026-09-23

> **Status:** approved and executed. Actual spend was $0.0975 of the $1 cap;
> see `PROGRESS.md` (Phase log, Spend log) and ADR-012 to ADR-014. The text
> below is the plan as approved, kept unchanged. The pre-run wording ("not a
> claim…", "cumulative spend remains $0") describes the state before execution.
> Deviations: vLLM flags were verified on the GPU host, not on CPU (ADR-012);
> one extra CPU HF-image check ran; one vLLM smoke retry after a flag-check bug.

The audit is local and complete; this is a proposed execution plan, not a
claim that Docker or Modal have been validated. Phase 3's total cap is $1.
Cumulative project spend remains $0; remaining account credits are unverified.

## Implementation and acceptance order

1. Prepare local Dockerfile, Compose, Modal modules, smoke config and CPU tests.
   Use separate weight, result and compilation-cache Volumes. All functions
   have timeouts, no automatic retries, one container, and zero idle retention.
   Lint Dockerfile and validate `docker compose config` before any remote run.
2. Pin a compatible serving stack and immutable model/tokenizer revision.
   Candidate Docker base: `vllm/vllm-openai:v0.10.2`, with torch 2.8.0 and
   transformers 4.56.2. The tagged upstream
   [CUDA requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/requirements/cuda.txt)
   pin torch 2.8.0 and the
   [common requirements](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/requirements/common.txt)
   require transformers >=4.55.2. This is a compatibility candidate, not a
   tested image. Resolve the image digest and test dependency resolution
   before spending; keep the GPU image's CUDA torch build intact. Record
   installed versions, image digest and commands in each result.
3. After approval, build the vLLM image from our Dockerfile plus a separate
   HF image. Clear the inherited entrypoint for Modal as in the
   [official example](https://modal.com/docs/examples/vllm_inference).
   Do not attach a GPU to builders. Package downloads are included in the
   build allowance below; a small CPU download image is also included there.
4. CPU-only download of [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
   at the resolved revision, once to the weight Volume. Do not download the
   8B checkpoint in Phase 3; that belongs to Phase 4. Store manifests and
   verify local paths. GPU functions must use offline/local loading.
5. Run mock timing/capacity acceptance inside the serving environment in a
   CPU-only function (including the 30-second capacity window). Save evidence.
6. L4 vLLM smoke: detached ephemeral run, localhost health check, warmup,
   concurrency 4, fixed unique token-ID prompts (512 input / 32 output tokens
   for this small functional smoke), explicit greedy settings, ignore EOS,
   model generation defaults disabled, prefix caching disabled. Verify flags
   with the pinned server's `--help`. Use eager mode to bound smoke startup;
   record this difference from future performance runs. Mount compile cache
   now; cache reuse is not claimed by an eager smoke. Save raw results and logs
   incrementally; always terminate the server in `finally`.
7. Separate L4 HF smoke: same workload, BF16/SDPA, naive and static modes in
   the same function lifetime, static batch size 4. Record actual generate
   batch sizes. This is smoke coverage, not an OOM-derived performance batch
   size or a speed comparison.
8. Sync results locally; check every usage count, zero failures, required raw
   files and runtime metadata. Poll using `uv run modal app logs` / `app list`
   until no apps are running. Reconcile actual usage with the dashboard,
   update spend/decisions, commit and stop. Then estimate Phases 4–6 separately.

GPU invocations use `uv run modal run --detach`; there is no deploy. Before
**each** GPU run, restate GPU, expected minutes, resource cost and cumulative
ledger and await an explicit yes, as required by CLAUDE.md. Approval of this
plan does not authorize an unbounded retry or a GPU upgrade.

## Cost basis and itemized estimate

Verified on 2026-09-23 against [Modal pricing](https://modal.com/pricing):
L4 $0.000222/GPU-second, physical CPU $0.0000131/core-second, memory
$0.00000222/GiB-second. Functions use physical cores, not vCPUs. Formula:
`seconds × (GPU_count × GPU_rate + CPU_cores × CPU_rate + GiB × memory_rate)`.
[Resource billing](https://modal.com/docs/guide/resources) uses the greater of
requested and actual CPU/memory usage; requests are not billing ceilings.

These durations/resources are planning assumptions, not measured startup times.
Build steps are conservatively budgeted as billable CPU/memory; registry,
package downloads and the CPU download-image build are included in that line.

- **Image preparation/builds:** 20 total CPU minutes (10 per main image),
  2 cores / 8 GiB budget envelope: GPU $0; CPU $0.031440; memory $0.021312;
  **$0.052752**. Allow up to 40 total minutes: **$0.105504**.
- **Tiny-model download:** 5 minutes, 2 cores / 4 GiB: GPU $0;
  CPU $0.007860; memory $0.002664; **$0.010524**.
  Function timeout 10 minutes: **$0.021048** at this envelope.
- **CPU mock validation:** 2 minutes, 4 cores / 4 GiB: GPU $0;
  CPU $0.006288; memory $0.001066; **$0.007354**.
  Function timeout 3 minutes: **$0.011030**.
- **vLLM smoke, including cold start:** L4 for 10 minutes,
  4 cores / 16 GiB: GPU $0.133200; CPU $0.031440; memory $0.021312;
  **$0.185952**. Function timeout 15 minutes: **$0.278928**.
- **HF smoke, including both modes and cold start:** L4 for 5 minutes,
  4 cores / 8 GiB: GPU $0.066600; CPU $0.015720; memory $0.005328;
  **$0.087648**. Function timeout 10 minutes: **$0.175296**.

Expected total: **$0.344230**, rounded to **$0.35** (GPU $0.199800,
CPU $0.092748, memory $0.051682). The longer-duration envelopes sum to
**$0.591806**, rounded to **$0.60**. Reserve the remaining **$0.40** under the
$1 phase cap for build variability, startup/teardown overhead and unexpected
CPU/memory usage. These are estimates, not an enforceable billing guarantee.
Re-estimate if image builder resources differ; stop before another step if
its projected total exceeds $1. Do not automatically use the reserve for retries.

Volume storage is $0.09/GiB-month with 1 TiB/month included on the cited pricing
page. Assuming this workspace is within that allowance, incremental storage
is $0. If the allowance is exhausted, 10 GiB retained for one day is about
$0.03 using a 30-day month, covered by contingency; verify account usage before
execution. No region premium, snapshot feature or non-preemptible pricing is
requested. Free compute credits may cover the charges, but the estimate does
not subtract unverified credits. Local lint, tests and result sync preparation
cost $0. Record all billed resource usage, including builds, in the spend log.
