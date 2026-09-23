| Variant | MMLU 5-shot acc (± stderr) | Δ vs BF16 (pp) | WikiText-2 word perplexity | Δ perplexity |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 74.88% ± 0.35 | — | 12.7418 | — |
| AWQ W4A16-asym (pile-val calib.) | 73.93% ± 0.35 | +0.95 | 13.2822 | +4.24% |
| GPTQ W4A16-sym (ultrachat calib.) | 73.24% ± 0.36 | +1.65 | 13.6619 | +7.22% |

Δ MMLU is BF16 accuracy minus the variant's, in absolute percentage points (positive = lower accuracy). Accuracy is lm-eval's size-weighted mean over all 14,042 test questions.

**Paired per-question comparison vs BF16** (same 14,042 prompts; McNemar test with continuity correction)

| Variant | Lost (BF16 right, variant wrong) | Gained | Net lost | χ² | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| AWQ W4A16-asym (pile-val calib.) | 617 | 483 | 134 | 16.08 | 6.1e-05 |
| GPTQ W4A16-sym (ultrachat calib.) | 678 | 447 | 231 | 47.02 | 7e-12 |

**MMLU category deltas vs BF16** (pp)

| Category | AWQ W4A16-asym (pile-val calib.) Δ (pp) | GPTQ W4A16-sym (ultrachat calib.) Δ (pp) |
| --- | ---: | ---: |
| humanities | +0.98 | +1.53 |
| other | +1.26 | +1.42 |
| social sciences | +0.75 | +1.49 |
| stem | +0.82 | +2.19 |

**AWQ W4A16-asym (pile-val calib.): largest per-subject drops** (pp; questions lost/gained vs BF16: 617/483)

| Subject | Questions | BF16 acc | Variant acc | Δ (pp) |
| --- | ---: | ---: | ---: | ---: |
| high_school_chemistry | 203 | 76.85% | 71.43% | +5.42 |
| college_mathematics | 100 | 60.00% | 55.00% | +5.00 |
| college_physics | 102 | 65.69% | 60.78% | +4.90 |
| high_school_physics | 151 | 72.85% | 68.21% | +4.64 |
| college_computer_science | 100 | 73.00% | 69.00% | +4.00 |

**GPTQ W4A16-sym (ultrachat calib.): largest per-subject drops** (pp; questions lost/gained vs BF16: 678/447)

| Subject | Questions | BF16 acc | Variant acc | Δ (pp) |
| --- | ---: | ---: | ---: | ---: |
| college_mathematics | 100 | 60.00% | 52.00% | +8.00 |
| global_facts | 100 | 51.00% | 44.00% | +7.00 |
| college_physics | 102 | 65.69% | 58.82% | +6.86 |
| electrical_engineering | 145 | 80.00% | 75.17% | +4.83 |
| college_medicine | 173 | 80.35% | 75.72% | +4.62 |

AWQ and GPTQ used different calibration data, each following its official llm-compressor 0.7.1 example (AWQ: pile-val, 256 x 512 tokens; GPTQ: UltraChat, 512 x 2,048 tokens; ADR-016), so an AWQ-vs-GPTQ difference mixes the method with its calibration set.

Per-subject deltas are noisy: in a 100-question subject one question is 1 pp. The paired test above is the right measure for the overall delta.

Variants come from more than one run (bf16: `full-20260923T142440Z`, awq: `full-20260923T142440Z`, gptq: `full-20260923T175000Z`), all with config sha256 `364e97d9d36a` and identical prompts.
