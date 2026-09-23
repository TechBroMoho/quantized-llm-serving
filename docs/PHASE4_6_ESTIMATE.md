# Phases 4–6 cost estimate

> **Update after Phase 5 (2026-09-23).** Phase 5 actual: **$10.7004**
> (prefetches $0.0428, probe $0.2108, full run 1 $7.3972 with ~$1.65 lost to
> the cancelled GPTQ attempt, GPTQ rerun $3.0496) against a cap raised to
> $11.63 (ADR-018). The probe predicted $7.94 for a complete three-variant
> run; the measured per-variant cost was ~$2.8–3.0 (BF16 MMLU 10,200 tokens/s,
> AWQ −3.5%, GPTQ −1.6% vs BF16). Cumulative actual: **$12.0650**; $12.93
> remains against the $25 target and about $17.93 of the $30 credit.
> Phase 6 below is unchanged ($6.72 expected, $9.48 envelope), for a
> projected total of about $18.8 (envelope $21.5). Phase 6 must re-estimate
> each run from its own probe, and runs longer than a few minutes should use
> the spawn pattern.

> **Update after Phase 4 (2026-09-23).** Phase 4 actual: **$1.2671** (prep
> $0.0242, AWQ $0.3855, GPTQ $0.7261, sanity $0.1313) against $4.16 expected.
> Measured L40S facts for later phases: vLLM healthy in 46–52 s for 8B
> variants in eager mode (CUDA-graph capture will add some); model load from
> the Volume took 7–11 s. Phases 5–6 below are unchanged pending the Phase 5
> timed probe, and are likely conservative on startup. Cumulative actual
> after Phase 4: **$1.3646**; about $28.64 of the reported $30 credit remains.

Prepared 2026-09-23 at the end of Phase 3. Rates were re-read from
[Modal pricing](https://modal.com/pricing) on 2026-09-23: L40S $0.000542/s,
physical CPU core $0.0000131/s, memory $0.00000222/GiB-s. Formula:
`seconds × (GPU + cores × CPU + GiB × memory)`, with CPU and memory set as
request == limit (ADR-012). Durations are **planning assumptions, not
measurements**. None of this has been timed on L40S. The only calibration
point is Phase 3: both L4 smokes cost less than planned ($0.040 vs $0.186
expected for vLLM, $0.028 vs $0.088 for HF), mainly because startup was
short (vLLM healthy in 59 s, HF in 10–11 s).

"Envelope" is the cost if every step ran to its planned timeout. Each GPU run
is still announced with its own estimate and needs an explicit yes. A phase
that projects past its cap stops for a decision.

## Phase 4: quantization (cap $6)

| Step | Resources | Expected | Envelope |
| --- | --- | ---: | ---: |
| Qwen3-8B BF16 + calibration data download (CPU only) | 2 cores / 8 GiB, 10 min (timeout 20) | $0.026 | $0.053 |
| AWQ `W4A16_ASYM`, llm-compressor official recipe | L40S, 4 cores / 64 GiB, 35 min (timeout 60) | $1.547 | $2.651 |
| GPTQ `W4A16`, group 128 | L40S, 4 cores / 64 GiB, 45 min (timeout 75) | $1.988 | $3.314 |
| vLLM load + 5 fixed prompts × 3 variants, on-disk sizes | L40S, 4 cores / 32 GiB, 15 min total (timeout 30) | $0.599 | $1.198 |
| **Phase 4 total** | | **$4.16** | **$7.22** |

The envelope exceeds the $6 cap. Mitigation: run AWQ first, reconcile its
actual cost, and re-estimate GPTQ before asking to run it. The 64 GiB host
memory is an assumption for loading and calibrating a 16 GB BF16 model; it
costs $0.51/h at this rate and will be checked against the pinned
llm-compressor's documented needs.

## Phase 5: accuracy (cap $6)

| Step | Resources | Expected | Envelope |
| --- | --- | ---: | ---: |
| lm-eval timed probe (`--limit`, one variant) | L40S, 4 cores / 32 GiB, 5 min (timeout 10) | $0.200 | $0.399 |
| MMLU 5-shot + WikiText-2 word perplexity × 3 | L40S, 4 cores / 32 GiB, 60 min total (timeout 90) | $2.396 | $3.593 |
| **Phase 5 total** | | **$2.60** | **$3.99** |

The probe replaces the 60-minute assumption with a measured rate before the
full runs are approved (SPEC Phase 5).

## Phase 6: performance (cap $10)

8 cores / 32 GiB is assumed so the load generator does not compete with the
server (see ADR-013).

| Step | Resources | Expected | Envelope |
| --- | --- | ---: | ---: |
| vLLM BF16/AWQ/GPTQ, c ∈ {1,4,16,64,128,256}, one lifetime each | L40S, 54 min (timeout 75) | $2.326 | $3.230 |
| Headline repeats (2 more runs at c=1 and peak, × 3 variants) | L40S, 30 min (timeout 45) | $1.292 | $1.938 |
| Max-batch sweep: AWQ, c=256, `--max-num-seqs` ∈ {16,64,256} | L40S, 17 min (timeout 25) | $0.732 | $1.077 |
| HF-naive {1,4,16} + HF-static OOM probe and sweep (time-boxed) | L40S, 45 min (timeout 60) | $1.938 | $2.584 |
| `vllm bench serve` cross-check, 3 settings | L40S, 10 min (timeout 15) | $0.431 | $0.646 |
| **Phase 6 total** | | **$6.72** | **$9.48** |

## Summary

| | Expected | Envelope |
| --- | ---: | ---: |
| Phase 3 actual (dashboard) | $0.097 | — |
| Phase 4 actual (dashboard) | $1.267 | — |
| Phases 5–6 (unchanged estimates) | $9.32 | $13.47 |
| Cumulative | **≈ $10.69** | ≈ $14.83 |

Both are under the $25 target and inside the $30 credit Mohammed reported
before Phase 3 (no card on file, so the credit is a hard stop; ADR-015). Items to resolve before Phase 6 (no cost to
decide): ADR-013 (client capacity inside Modal) and ADR-014 (empty vLLM
chunks), plus the real-text W1 prompt corpus.
