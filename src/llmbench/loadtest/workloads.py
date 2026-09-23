"""Deterministic request construction for the Phase 1 OpenAI API client."""

from __future__ import annotations

from typing import Any


def completions_payload(
    *,
    prompt: str,
    request_index: int,
    output_tokens: int,
    seed: int,
    model: str = "mock-model",
) -> dict[str, Any]:
    """Create a unique prompt and explicit greedy generation parameters."""
    unique_prompt = f"{prompt}\n[request {request_index}]"
    return {
        "model": model,
        "prompt": unique_prompt,
        "stream": True,
        "stream_options": {"include_usage": True},
        "n": 1,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "seed": seed,
    }
