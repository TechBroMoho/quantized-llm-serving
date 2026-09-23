# Phases 4–6 cost estimate (for approval, nothing launched)

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
| Phases 4–6 | $13.47 | $20.68 |
| Cumulative | **≈ $13.57** | ≈ $20.78 |

Both are under the $25 target and inside the $30/month Starter credit
(remaining credit not verified). Items to resolve before Phase 6 (no cost to
decide): ADR-013 (client capacity inside Modal) and ADR-014 (empty vLLM
chunks), plus the real-text W1 prompt corpus.
