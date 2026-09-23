"""Request timing and summary calculations for streaming completions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ADR-014: every completed request must stream at least this many text-bearing
# chunks per usage completion token. With skip_special_tokens=false a healthy
# stream has about one text chunk per token (Phase 3 smoke: >= 29 of 32).
# vLLM merges deltas when its API server falls behind the engine, so under load
# one chunk may carry several tokens; 0.5 allows up to two tokens per chunk on
# average. The failure it guards against, text-less trailing tokens that cut
# the last-text E2E short, gave 1-2 text chunks for 32 tokens.
MIN_TEXT_CHUNKS_PER_TOKEN = 0.5


@dataclass
class RequestRecord:
    """One request, including incomplete and invalid outcomes."""

    request_id: str
    status: str
    started_at: float
    completed_at: float | None = None
    ttft_s: float | None = None
    itl_s: list[float] = field(default_factory=list)
    tpot_s: float | None = None
    e2e_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    text_chunks: int = 0
    empty_text_chunks: int = 0
    usage_events: int = 0
    # Diagnostic only (not a metric): request start to the [DONE] event. A gap
    # after e2e_s means tokens arrived without text (ADR-014).
    stream_end_s: float | None = None
    error: str | None = None

    def add_text(self, received_at: float) -> None:
        """Record arrival of one non-empty text chunk using the monotonic clock."""
        if self.ttft_s is None:
            self.ttft_s = received_at - self.started_at
        else:
            previous_offset = self.e2e_s if self.e2e_s is not None else self.ttft_s
            previous_text_at = self.started_at + previous_offset
            self.itl_s.append(received_at - previous_text_at)
        self.e2e_s = received_at - self.started_at
        self.text_chunks += 1

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON serializable record."""
        return asdict(self)


def text_chunk_shortfalls(
    records: list[RequestRecord], minimum: float = MIN_TEXT_CHUNKS_PER_TOKEN
) -> list[str]:
    """Completed requests whose text chunks fall well short of usage tokens.

    Any entry makes the run invalid: its last-text E2E and TPOT may end before
    the last generated token. Empty-chunk counts are included so a failure can
    be told apart from vLLM merging several tokens into one chunk.
    """
    problems = []
    for row in records:
        if row.status not in {"ok", "late_completion"} or not row.completion_tokens:
            continue
        if row.text_chunks < minimum * row.completion_tokens:
            problems.append(
                f"{row.request_id}: {row.text_chunks} text chunks "
                f"({row.empty_text_chunks} empty) for "
                f"{row.completion_tokens} completion tokens, below "
                f"{minimum:g} per token"
            )
    return problems


def percentile(values: list[float], p: float) -> float | None:
    """Return a linearly interpolated percentile, or None for no observations."""
    if not values:
        return None
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(
    records: list[RequestRecord],
    *,
    window_start: float,
    window_end: float,
    offered_requests: int | None = None,
    rejected_requests: int = 0,
    config: dict[str, Any] | None = None,
    versions: dict[str, str] | None = None,
    host: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize on-time results separately from late and incomplete requests."""
    on_time = [
        row
        for row in records
        if row.status == "ok"
        and window_start <= row.started_at <= window_end
        and row.completed_at is not None
        and row.completed_at <= window_end
    ]
    late = [
        row
        for row in records
        if row.status == "late_completion"
        and row.started_at <= window_end
        and row.completed_at is not None
    ]
    in_flight_at_deadline = sum(
        1
        for row in records
        if row.started_at <= window_end
        and (row.completed_at is None or row.completed_at > window_end)
    )
    width = window_end - window_start
    ttft = [row.ttft_s for row in on_time if row.ttft_s is not None]
    tpot = [row.tpot_s for row in on_time if row.tpot_s is not None]
    e2e = [row.e2e_s for row in on_time if row.e2e_s is not None]
    itl = [gap for row in on_time for gap in row.itl_s]
    errors = [row for row in records if row.status not in {"ok", "late_completion"}]
    shortfalls = text_chunk_shortfalls(records)
    return {
        "window_start_monotonic_s": window_start,
        "window_end_monotonic_s": window_end,
        "window_duration_s": width,
        "offered_requests": offered_requests
        if offered_requests is not None
        else len(records),
        "rejected_requests": rejected_requests,
        "on_time_completed_requests": len(on_time),
        "late_completed_requests": len(late),
        "in_flight_at_deadline": in_flight_at_deadline,
        "errored_or_cancelled_requests": len(errors),
        "error_rate": len(errors) / max(1, len(records) + rejected_requests),
        "min_text_chunks_per_token": MIN_TEXT_CHUNKS_PER_TOKEN,
        "text_chunk_shortfall_requests": len(shortfalls),
        "text_chunk_shortfall_examples": shortfalls[:10],
        "request_throughput_per_s": len(on_time) / width if width > 0 else 0.0,
        "output_token_throughput_per_s": sum(
            row.completion_tokens or 0 for row in on_time
        )
        / width
        if width > 0
        else 0.0,
        "ttft_s": _percentiles(ttft),
        "tpot_s": _percentiles(tpot),
        "itl_s": _percentiles(itl),
        "e2e_s": _percentiles(e2e),
        "config": config or {},
        "versions": versions or {},
        "host": host or {},
    }


def _percentiles(values: list[float]) -> dict[str, float | None]:
    return {f"p{p}": percentile(values, p) for p in (50, 90, 95, 99)}
