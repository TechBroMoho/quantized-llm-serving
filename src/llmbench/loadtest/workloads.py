"""Deterministic request construction for the OpenAI completions client."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from hashlib import sha256
from typing import Any


def completions_payload(
    *,
    prompt: str | list[int],
    request_index: int,
    output_tokens: int,
    seed: int,
    model: str = "mock-model",
    ignore_eos: bool = False,
) -> dict[str, Any]:
    """Create one request with explicit greedy generation parameters.

    Text prompts get a request-index suffix so they are unique. Token-ID
    prompts are sent unchanged: their uniqueness comes from `TokenPromptPool`,
    and an appended suffix would break the exact input length.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": (
            f"{prompt}\n[request {request_index}]"
            if isinstance(prompt, str)
            else list(prompt)
        ),
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
    if ignore_eos:
        # vLLM's extension; the HF baseline accepts it because it always
        # enforces min_new_tokens == max_new_tokens.
        payload["ignore_eos"] = True
    return payload


def regular_token_ids(vocab_size: int, special_ids: Iterable[int]) -> list[int]:
    """Token IDs usable in synthetic prompts: base vocabulary minus specials."""
    excluded = set(special_ids)
    allowed = [token for token in range(vocab_size) if token not in excluded]
    if not allowed:
        raise ValueError("no regular token IDs available")
    return allowed


class TokenPromptPool:
    """A seeded pool of distinct fixed-length token-ID prompts, each used once.

    Prompts are sampled uniformly from regular token IDs. This is a functional
    smoke workload; the real-text W1 corpus is separate Phase 6 work.
    """

    def __init__(
        self, *, allowed_ids: Sequence[int], count: int, length: int, seed: int
    ) -> None:
        if count < 1 or length < 1:
            raise ValueError("count and length must be positive")
        if not allowed_ids:
            raise ValueError("allowed_ids must be nonempty")
        rng = random.Random(seed)
        self.length = length
        self.seed = seed
        self.prompts = [
            [rng.choice(allowed_ids) for _ in range(length)] for _ in range(count)
        ]
        if len({tuple(prompt) for prompt in self.prompts}) != count:
            raise ValueError("sampled prompts are not unique; choose another seed")
        self.issued: dict[int, int] = {}

    def take(self, request_index: int) -> list[int]:
        """Return the next unused prompt; never reuse a prompt or request index."""
        if request_index in self.issued:
            raise ValueError(f"request index {request_index} already has a prompt")
        position = len(self.issued)
        if position >= len(self.prompts):
            raise IndexError("prompt pool exhausted; prompts are never reused")
        self.issued[request_index] = position
        return self.prompts[position]

    def fingerprint(self) -> str:
        """SHA-256 of the full pool, so a run can prove which prompts it used."""
        return sha256(
            json.dumps(self.prompts, separators=(",", ":")).encode()
        ).hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "seeded_uniform_regular_token_ids",
            "prompt_count": len(self.prompts),
            "prompt_tokens": self.length,
            "seed": self.seed,
            "unique_prompts": len({tuple(prompt) for prompt in self.prompts}),
            "pool_sha256": self.fingerprint(),
            "issued": len(self.issued),
            "request_index_to_prompt": {
                str(index): position for index, position in self.issued.items()
            },
        }
