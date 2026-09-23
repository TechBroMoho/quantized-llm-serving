"""W1 prompt construction from WikiText-style text (ADR-021)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from llmbench.prompts import (
    NpyPromptPayloads,
    pool_sha256,
    sample_pool,
    token_windows,
    wikitext_articles,
)

LINES = [
    " = Alpha = \n",
    " \n",
    " one two three four five six seven \n",
    " = = Section = = \n",
    " eight nine ten \n",
    " = Beta = \n",
    " a b c \n",
]


def _tokenize(text: str) -> list[int]:
    return [3 + len(word) for word in text.split() if word != "="]


def test_articles_split_only_at_top_level_headings() -> None:
    articles = list(wikitext_articles(LINES))
    assert len(articles) == 2
    assert "Section" in articles[0] and "ten" in articles[0]
    assert articles[1].startswith(" = Beta = ")


def test_windows_are_exact_and_never_span_articles() -> None:
    windows, read = token_windows(
        wikitext_articles(LINES), _tokenize, length=4, needed=2
    )
    assert windows.shape == (3, 4) and windows.dtype == np.int32
    assert read == 1  # the first article alone gave enough windows
    with pytest.raises(ValueError, match="need 9"):
        token_windows(wikitext_articles(LINES), _tokenize, length=4, needed=9)


def test_pool_is_seeded_unique_and_free_of_special_tokens() -> None:
    windows = np.arange(40, dtype=np.int32).reshape(10, 4)
    pool = sample_pool(windows, count=5, seed=1, special_ids=[2])  # drops row 0
    again = sample_pool(windows, count=5, seed=1, special_ids=[2])
    assert pool_sha256(pool) == pool_sha256(again)
    assert pool_sha256(pool) != pool_sha256(
        sample_pool(windows, count=5, seed=2, special_ids=[2])
    )
    assert not np.isin(pool, [2]).any()
    assert len({row.tobytes() for row in pool}) == 5
    with pytest.raises(ValueError, match="distinct windows without special"):
        sample_pool(windows, count=10, seed=1, special_ids=[2])


def test_npy_payloads_map_index_to_row_and_stay_small_when_pickled(
    tmp_path: Path,
) -> None:
    pool = np.arange(3 * 512, dtype=np.int32).reshape(3, 512)
    path = tmp_path / "prompts.npy"
    np.save(path, pool)
    payloads = NpyPromptPayloads(str(path), output_tokens=8, seed=0, model="m")
    assert payloads(2)["prompt"] == pool[2].tolist()
    assert payloads(0)["stream_options"]["continuous_usage_stats"] is True
    assert payloads(0)["skip_special_tokens"] is False
    # The memory-mapped pool is not pickled into worker processes.
    assert len(pickle.dumps(payloads)) < 1000
    assert pickle.loads(pickle.dumps(payloads))(1)["prompt"] == pool[1].tolist()
    with pytest.raises(IndexError, match="never reused"):
        payloads(3)


def test_build_pool_records_what_it_read() -> None:
    from llmbench.prompts import build_pool

    lines = []
    for n in range(12):
        lines += [f" = Article {n} = \n", " " + "w" * (n + 1) + " x" * 7 + " \n"]
    lines += LINES  # a repeated article must not produce repeated prompts
    lines += LINES
    pool, info = build_pool(
        lines, _tokenize, special_ids=[0], count=4, length=4, seed=3, oversample=2
    )
    assert pool.shape == (4, 4)
    assert info["windows_needed"] == 8 and info["windows_available"] >= 8
    assert info["pool_sha256"] == pool_sha256(pool)


def test_duplicate_windows_are_dropped_before_sampling() -> None:
    windows = np.array([[5, 5], [6, 6], [5, 5], [7, 7]], dtype=np.int32)
    pool = sample_pool(windows, count=3, seed=0, special_ids=[])
    assert sorted(map(tuple, pool.tolist())) == [(5, 5), (6, 6), (7, 7)]
    with pytest.raises(ValueError, match="only 3 distinct"):
        sample_pool(windows, count=4, seed=0, special_ids=[])
