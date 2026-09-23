"""The timing gates on synthetic traces: jitter, bias and tail cases (ADR-019)."""

from __future__ import annotations

from typing import Any

from llmbench.loadtest.metrics import RequestRecord
from llmbench.loadtest.validation import (
    TIMING_FULL,
    TIMING_UNIT,
    TimingPlan,
    evaluate_timing,
)

TTFT, ITL = 0.2, 0.02


def _run(
    plan: TimingPlan,
    delays: dict[tuple[int, int], float] | None = None,
    drift_s: float = 0.0,
) -> dict[str, Any]:
    """Server writes on an exact schedule; the client sees each chunk after a
    0.1 ms delivery delay, plus `delays[(request, chunk)]`, plus `drift_s`
    accumulated per chunk (a biased client clock)."""
    records, traces = [], {}
    for r in range(plan.requests):
        start = 100.0 * r
        writes = [start + 0.001 + TTFT + k * ITL for k in range(plan.output_tokens)]
        row = RequestRecord(f"timing-{r}", "ok", started_at=start)
        for k, write in enumerate(writes):
            extra = (delays or {}).get((r, k), 0.0)
            row.add_text(write + 0.0001 + extra + k * drift_s)
        row.empty_text_chunks = row.usage_events = 1
        row.completion_tokens = plan.output_tokens
        assert row.e2e_s is not None and row.ttft_s is not None
        row.tpot_s = (row.e2e_s - row.ttft_s) / (plan.output_tokens - 1)
        records.append(row)
        traces[row.request_id] = {"start": start + 0.001, "text": writes}
    return evaluate_timing(records, traces, plan=plan)


# --- Unit gate (make check): strict medians + 50 ms bound on every pair ---


def test_unit_exact_client_passes() -> None:
    result = _run(TIMING_UNIT)
    assert result["passed"]
    assert result["gate"]["itl_s"]["samples"] == 200
    assert "p99_relative_error" not in result["gate"]["ttft_s"]


def test_unit_measured_scheduling_delays_pass_but_are_reported() -> None:
    # Measured on the laptop: 6.4 ms on a gap (typical spike) and 33 ms on a
    # first chunk (worst seen, 6 of 10 cores spinning).
    result = _run(TIMING_UNIT, {(3, 7): 0.0064, (5, 0): 0.033})
    assert result["passed"], result["gate"]
    assert result["paired_observations"][3]["over_tolerance"] == ["itl_6", "itl_7"]
    assert result["gate"]["ttft_s"]["paired_over_tolerance"] == 1


def test_unit_one_gross_error_fails_the_tail_bound() -> None:
    result = _run(TIMING_UNIT, {(2, 10): 0.06})
    assert not result["passed"]
    assert result["gate"]["itl_s"]["p50_relative_error"] <= 0.05
    assert not result["gate"]["itl_s"]["passed"]


def test_unit_one_late_request_fails_ttft() -> None:
    # The audit's single corrupt request: 100 ms on every chunk of request 0.
    result = _run(TIMING_UNIT, {(0, k): 0.1 for k in range(21)})
    assert not result["passed"]
    assert not result["gate"]["ttft_s"]["passed"]
    assert result["gate"]["ttft_s"]["p50_relative_error"] <= 0.05


def test_unit_systematic_bias_on_every_gap_fails_the_median() -> None:
    result = _run(TIMING_UNIT, drift_s=0.0012)  # 6% slow on every 20 ms gap
    assert not result["passed"]
    assert result["gate"]["itl_s"]["p50_relative_error"] > 0.05


def test_request_without_a_server_trace_fails() -> None:
    result = evaluate_timing(
        [RequestRecord("timing-0", "ok", 0.0)], {"timing-0": {"text": []}}
    )
    assert not result["passed"] and not result["structural_ok"]


# --- Full gate (mock-validate, in-Modal): strict medians + p99 ---


def test_full_gate_tolerates_an_isolated_spike() -> None:
    result = _run(TIMING_FULL, {(3, 7): 0.0064})
    assert result["passed"], result["gate"]
    assert result["gate"]["itl_s"]["samples"] == 4000
    assert result["gate"]["ttft_s"]["samples"] == 200


def test_full_gate_catches_a_tail_that_the_median_misses() -> None:
    delays = {(r, k): 0.004 for r in range(200) for k in (5, 15)}  # 10% of gaps
    result = _run(TIMING_FULL, delays)
    assert not result["passed"]
    assert result["gate"]["itl_s"]["p50_relative_error"] <= 0.05
    assert result["gate"]["itl_s"]["p99_relative_error"] > 0.05


def test_full_gate_refuses_a_p99_from_too_few_samples() -> None:
    small = TimingPlan(requests=10, output_tokens=21, concurrency=1, full=True)
    result = _run(small)
    assert not result["passed"]
    assert "at least 100 samples" in result["gate"]["ttft_s"]["error"]
    assert result["gate"]["itl_s"]["passed"]  # 200 gaps are enough
