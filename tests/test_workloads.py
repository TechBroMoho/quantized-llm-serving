"""Token-ID workload construction: exact lengths, uniqueness and no reuse."""

from __future__ import annotations

import pytest

from llmbench.loadtest.workloads import (
    TokenPromptPool,
    completions_payload,
    regular_token_ids,
)


def test_regular_token_ids_exclude_special_tokens() -> None:
    assert regular_token_ids(6, [0, 5]) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        regular_token_ids(2, [0, 1])


def test_pool_is_exact_length_unique_seeded_and_never_reused() -> None:
    pool = TokenPromptPool(allowed_ids=range(3, 100), count=20, length=512, seed=7)
    again = TokenPromptPool(allowed_ids=range(3, 100), count=20, length=512, seed=7)
    other = TokenPromptPool(allowed_ids=range(3, 100), count=20, length=512, seed=8)
    assert pool.fingerprint() == again.fingerprint() != other.fingerprint()
    taken = [pool.take(index) for index in (-1, -5, 0, 1)]
    assert all(len(prompt) == 512 for prompt in taken)
    assert all(3 <= token < 100 for prompt in taken for token in prompt)
    assert len({tuple(prompt) for prompt in taken}) == 4
    with pytest.raises(ValueError, match="already has a prompt"):
        pool.take(0)
    for index in range(2, 18):
        pool.take(index)
    with pytest.raises(IndexError, match="exhausted"):
        pool.take(99)
    described = pool.describe()
    assert described["unique_prompts"] == 20
    assert described["issued"] == 20


def test_pool_rejects_duplicate_prompts() -> None:
    with pytest.raises(ValueError, match="not unique"):
        TokenPromptPool(allowed_ids=[4], count=2, length=3, seed=0)


def test_token_payload_is_unchanged_and_requests_ignore_eos() -> None:
    prompt = [5, 6, 7]
    payload = completions_payload(
        prompt=prompt, request_index=3, output_tokens=32, seed=1, ignore_eos=True
    )
    assert payload["prompt"] == [5, 6, 7]
    assert payload["prompt"] is not prompt
    assert payload["ignore_eos"] is True
    assert payload["max_tokens"] == 32
    text = completions_payload(prompt="p", request_index=3, output_tokens=1, seed=1)
    assert text["prompt"] == "p\n[request 3]"
    assert "ignore_eos" not in text
