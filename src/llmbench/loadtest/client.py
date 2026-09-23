"""Async OpenAI completions client with chunk-level streaming measurements."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp

from llmbench.loadtest.metrics import RequestRecord


def _consume_event(record: RequestRecord, data: str, received_at: float) -> bool:
    """Consume one SSE data field; return true for the stream terminator."""
    if data == "[DONE]":
        return True
    try:
        payload: dict[str, Any] = json.loads(data)
    except json.JSONDecodeError as exc:
        record.status = "error"
        record.error = f"invalid SSE JSON: {exc.msg}"
        return False
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
    if usage is not None:
        record.usage_events += 1
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                record.prompt_tokens = prompt_tokens
                record.completion_tokens = completion_tokens
            else:
                record.status = "error"
                record.error = "usage event is missing integer token counts"
    return False


async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    *,
    request_id: str,
) -> RequestRecord:
    """Stream one request; only non-empty text chunks define text timings."""
    started_at = time.perf_counter()
    record = RequestRecord(request_id=request_id, status="ok", started_at=started_at)
    try:
        async with session.post(url, json=payload) as response:
            if response.status < 200 or response.status >= 300:
                record.status = "error"
                record.error = f"HTTP {response.status}"
            else:
                data_lines: list[str] = []
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                    elif not line and data_lines:
                        data = "\n".join(data_lines)
                        data_lines.clear()
                        if _consume_event(record, data, time.perf_counter()):
                            break
        if record.status == "ok":
            if record.text_chunks == 0:
                record.status = "error"
                record.error = "stream contained no text-bearing chunks"
            elif record.usage_events == 0 or record.completion_tokens is None:
                record.status = "error"
                record.error = "stream ended without valid usage"
            else:
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
        record.status = (
            "drain_timeout" if reason == "bounded drain ended" else "cancelled"
        )
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
