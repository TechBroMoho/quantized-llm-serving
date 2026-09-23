"""Unit tests for clock selection and bounded-window summaries."""

from math import isclose

from llmbench.loadtest.metrics import RequestRecord, percentile, summarize


def test_text_chunks_define_timing_not_empty_or_usage_events() -> None:
    record = RequestRecord("r1", "ok", started_at=10.0)
    record.empty_text_chunks += 1
    record.usage_events += 1
    record.add_text(10.2)
    record.add_text(10.22)
    record.completion_tokens = 2
    record.e2e_s = 0.22
    assert record.ttft_s is not None and isclose(record.ttft_s, 0.2)
    assert len(record.itl_s) == 1 and isclose(record.itl_s[0], 0.02)
    assert record.text_chunks == 2


def test_percentiles_interpolate_and_handle_empty_input() -> None:
    assert percentile([0.0, 10.0], 50) == 5.0
    assert percentile([], 95) is None


def test_late_completions_do_not_count_toward_window_throughput() -> None:
    records = [
        RequestRecord("on-time", "ok", 1.0, completed_at=1.2, completion_tokens=3),
        RequestRecord(
            "late", "late_completion", 1.1, completed_at=2.1, completion_tokens=3
        ),
        RequestRecord("unfinished", "drain_timeout", 1.5, completed_at=3.0),
    ]
    result = summarize(records, window_start=1.0, window_end=2.0)
    assert result["on_time_completed_requests"] == 1
    assert result["late_completed_requests"] == 1
    assert result["in_flight_at_deadline"] == 2
    assert result["request_throughput_per_s"] == 1.0
