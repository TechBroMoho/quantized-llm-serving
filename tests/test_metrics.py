"""Unit tests for clock selection and bounded-window summaries."""

from math import isclose

from llmbench.loadtest.metrics import (
    MIN_TEXT_CHUNKS_PER_TOKEN,
    RequestRecord,
    percentile,
    summarize,
    text_chunk_shortfalls,
)


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


def _counted(request_id: str, status: str, text: int, empty: int, tokens: int):
    row = RequestRecord(request_id, status, 0.0, completion_tokens=tokens)
    row.text_chunks, row.empty_text_chunks = text, empty
    return row


def test_text_chunk_check_flags_streams_that_end_without_text() -> None:
    records = [
        _counted("healthy", "ok", 32, 0, 32),
        _counted("merged-2x", "ok", 16, 0, 32),  # exactly 0.5 per token: allowed
        _counted("skipped-specials", "ok", 2, 30, 32),  # the Phase 3 vLLM rows
        _counted("late-short", "late_completion", 15, 0, 32),
        _counted("errored", "error", 0, 0, 32),  # already invalid; not counted
    ]
    problems = text_chunk_shortfalls(records)
    assert [p.split(":")[0] for p in problems] == ["skipped-specials", "late-short"]
    assert "2 text chunks (30 empty) for 32 completion tokens" in problems[0]
    summary = summarize(records, window_start=0.0, window_end=1.0)
    assert summary["text_chunk_shortfall_requests"] == 2
    assert summary["min_text_chunks_per_token"] == MIN_TEXT_CHUNKS_PER_TOKEN == 0.5
