"""Async OpenAI completions client with chunk-level streaming measurements."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp

from llmbench.loadtest.metrics import RequestRecord, TokenWindow

# Cancellation reason for requests still running when a steady-state window
# closes (ADR-020): recorded as status "window_end", not as an error.
WINDOW_END_REASON = "measurement window ended"


def _fail(record: RequestRecord, error: str) -> None:
    """Mark a usage failure, keeping the first error if there already is one."""
    if record.status == "ok":
        record.status = "error"
        record.error = error


def _consume_event(
    record: RequestRecord,
    data: str,
    received_at: float,
    window: TokenWindow | None = None,
) -> bool:
    """Consume one SSE data field; return true for the stream terminator.

    A usage object on a chunk with choices is cumulative progress
    (`continuous_usage_stats`, ADR-020); the usage on a chunk without choices
    is the final count, and there must be exactly one of those.
    """
    if data == "[DONE]":
        return True
    try:
        payload: dict[str, Any] = json.loads(data)
    except json.JSONDecodeError as exc:
        record.status = "error"
        record.error = f"invalid SSE JSON: {exc.msg}"
        return False
    if not isinstance(payload, dict):
        record.status = "error"
        record.error = "SSE JSON must be an object"
        return False
    if "error" in payload:
        record.status = "error"
        record.error = f"server generation error: {payload['error']}"
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            text = choice.get("text")
            if isinstance(text, str):
                if text:
                    record.add_text(received_at)
                else:
                    record.empty_text_chunks += 1
    usage = payload.get("usage")
    if usage is None:
        return False
    if not isinstance(usage, dict):
        _fail(record, "usage must be an object")
        return False
    counts: list[Any] = [
        usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ]
    if any(type(count) is not int or count < 0 for count in counts):
        _fail(record, "usage counts must be nonnegative integers")
        return False
    if counts[2] != counts[0] + counts[1]:
        _fail(record, "inconsistent total_tokens")
        return False
    if isinstance(choices, list) and choices:
        # Cumulative progress: tokens arrived with this chunk.
        delta = counts[1] - record.streamed_tokens
        record.usage_progress_events += 1
        if delta < 0:
            _fail(record, "cumulative usage decreased")
            return False
        record.streamed_tokens = counts[1]
        if window is not None and delta:
            index = window.bin_of(received_at)
            if index is not None:
                record.window_token_bins[index] += delta
        return False
    record.usage_events += 1
    if record.usage_events != 1:
        _fail(record, "expected one usage object")
    elif record.usage_progress_events and counts[1] != record.streamed_tokens:
        _fail(
            record,
            f"final usage {counts[1]} != streamed usage {record.streamed_tokens}",
        )
    else:
        record.prompt_tokens = counts[0]
        record.completion_tokens = counts[1]
    return False


async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    *,
    request_id: str,
    window: TokenWindow | None = None,
) -> RequestRecord:
    """Stream one request; only non-empty text chunks define text timings."""
    started_at = time.perf_counter()
    record = RequestRecord(request_id=request_id, status="ok", started_at=started_at)
    if window is not None:
        record.window_token_bins = [0] * window.bins
    try:
        async with session.post(
            url, json=payload, headers={"X-Request-ID": request_id}
        ) as response:
            if response.status < 200 or response.status >= 300:
                record.status = "error"
                record.error = f"HTTP {response.status}"
            else:
                done_received = False
                data_lines: list[str] = []
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    elif not line and data_lines:
                        data = "\n".join(data_lines)
                        data_lines.clear()
                        received_at = time.perf_counter()
                        if _consume_event(record, data, received_at, window):
                            record.stream_end_s = received_at - started_at
                            done_received = True
                            break
                if not done_received and record.status == "ok":
                    record.status = "error"
                    record.error = "stream ended without [DONE]"
        if record.status == "ok":
            if record.text_chunks == 0:
                record.status = "error"
                record.error = "stream contained no text-bearing chunks"
            elif record.usage_events == 0 or record.completion_tokens is None:
                record.status = "error"
                record.error = "stream ended without valid usage"
            else:
                expected_output = payload.get("max_tokens")
                if (
                    isinstance(expected_output, int)
                    and record.completion_tokens != expected_output
                ):
                    record.status = "error"
                    record.error = (
                        f"completion_tokens {record.completion_tokens} != "
                        f"max_tokens {expected_output}"
                    )
                prompt = payload.get("prompt")
                if isinstance(prompt, list) and record.prompt_tokens != len(prompt):
                    record.status = "error"
                    record.error = (
                        f"prompt_tokens {record.prompt_tokens} != "
                        f"input tokens {len(prompt)}"
                    )
                if (
                    record.completion_tokens > 1
                    and record.e2e_s is not None
                    and record.ttft_s is not None
                ):
                    record.tpot_s = (record.e2e_s - record.ttft_s) / (
                        record.completion_tokens - 1
                    )
                else:
                    record.tpot_s = None
    except asyncio.CancelledError as exc:
        reason = str(exc)
        record.status = {
            "bounded drain ended": "drain_timeout",
            WINDOW_END_REASON: "window_end",
        }.get(reason, "cancelled")
        record.error = (
            f"request task cancelled: {reason}" if reason else "request task cancelled"
        )
    except TimeoutError:
        record.status = "timeout"
        record.error = "request timed out"
    except aiohttp.ClientError as exc:
        record.status = "error"
        record.error = f"{type(exc).__name__}: {exc}"
    finally:
        record.completed_at = time.perf_counter()
    return record
