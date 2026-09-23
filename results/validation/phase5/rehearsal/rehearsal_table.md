| Variant | MMLU 5-shot acc (± stderr) | Δ vs BF16 (pp) | WikiText-2 word perplexity | Δ perplexity |
| --- | ---: | ---: | ---: | ---: |
| tiny A | 25.44% ± 4.21 | — | 1895193.5478 | — |
| tiny B | 28.95% ± 4.56 | -3.51 | 1926674.8842 | +1.66% |

Δ MMLU is BF16 accuracy minus the variant's, in absolute percentage points (positive = lower accuracy). Accuracy is lm-eval's size-weighted mean over all 114 test questions.

**tiny B: largest per-subject drops** (pp; questions lost/gained vs BF16: 29/33)

| Subject | Questions | BF16 acc | Variant acc | Δ (pp) |
| --- | ---: | ---: | ---: | ---: |
| college_biology | 2 | 100.00% | 0.00% | +100.00 |
| elementary_mathematics | 2 | 100.00% | 0.00% | +100.00 |
| professional_medicine | 2 | 100.00% | 0.00% | +100.00 |
| abstract_algebra | 2 | 50.00% | 0.00% | +50.00 |
| business_ethics | 2 | 50.00% | 0.00% | +50.00 |
