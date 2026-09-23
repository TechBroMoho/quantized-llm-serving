# Timing-flake investigation (ADR-019)

Scratch probes run on the laptop (10 cores) on 2026-09-23, kept as written.

- `flake_probe.py`: repeated `measure_timing_accuracy()` with 0–10 spinning
  processes or a 6.3M-object heap, recording gate failures and GC pauses. Run
  against the **pre-ADR-019** code (5 requests × 4 tokens, every-gap rule).
  Output was printed only, not saved; ADR-019 quotes it.
- `delay_probe.py`: per-chunk delivery delay (client receipt − server write).
  Output printed only; quoted in ADR-019.
- `gate_probe.py` → `gate_probe.out`: the **intermediate** gate (p50 + p99 on
  10 × 21), with the old every-gap rule applied to the same runs.
- `gcwatch.py`: a pytest plugin that logged GC pauses during the suite.
- `suite_repeat_*.log`: full-suite repeats under the intermediate gate: 5/6
  passed (first loop, stopped), 22/25 passed (failures in the timing tests),
  and 4/4 passed with GC logging (no GC pause over 0.5 ms during any timing test).

The final gate (unit: strict p50 + a 50 ms bound; full: strict p50 + p99 on
200 × 21) was not repeated in a loop.
