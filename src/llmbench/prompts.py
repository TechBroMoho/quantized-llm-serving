"""W1 real-text prompts: exact 512-token windows from WikiText-103 (ADR-021).

Articles are rebuilt from the raw WikiText lines (a top-level heading
` = Title = ` starts a new article), tokenized once with the model's own
tokenizer (no special tokens), and cut into consecutive non-overlapping
windows. A seeded sample of windows becomes the prompt pool, stored as an
int32 array so every client process can memory-map it. Prompts are sent as
token IDs, so the server never re-tokenizes them.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from llmbench.loadtest.workloads import completions_payload

_ARTICLE_HEADING = re.compile(r"^ = [^=].* = $")


def wikitext_articles(lines: Iterable[str]) -> Iterator[str]:
    """Join raw WikiText lines into articles, split at top-level headings."""
    article: list[str] = []
    for line in lines:
        if _ARTICLE_HEADING.match(line.rstrip("\n")) and any(
            part.strip() for part in article
        ):
            yield "".join(article)
            article = []
        article.append(line)
    if any(part.strip() for part in article):
        yield "".join(article)


def token_windows(
    articles: Iterable[str],
    tokenize: Callable[[str], list[int]],
    *,
    length: int,
    needed: int,
) -> tuple[npt.NDArray[np.int32], int]:
    """Cut each article's tokens into windows until `needed` windows exist.

    Returns the windows and the number of articles read. A remainder shorter
    than `length` is dropped, so a window never spans two articles.
    """
    chunks: list[npt.NDArray[np.int32]] = []
    total = 0
    read = 0
    for article in articles:
        read += 1
        ids = np.asarray(tokenize(article), dtype=np.int32)
        usable = len(ids) // length
        if usable:
            chunks.append(ids[: usable * length].reshape(usable, length))
            total += usable
        if total >= needed:
            break
    if total < needed:
        raise ValueError(f"only {total} windows of {length} tokens, need {needed}")
    return np.concatenate(chunks), read


def sample_pool(
    windows: npt.NDArray[np.int32],
    *,
    count: int,
    seed: int,
    special_ids: Iterable[int],
) -> npt.NDArray[np.int32]:
    """A seeded sample of distinct windows with no special tokens.

    Repeated windows (duplicate passages) are dropped first, keeping the
    first occurrence, so the pool can never send the same prompt twice.
    """
    specials = np.asarray(sorted(set(special_ids)), dtype=np.int32)
    clean = windows[~np.isin(windows, specials).any(axis=1)]
    _, first = np.unique(clean, axis=0, return_index=True)
    clean = clean[np.sort(first)]
    if len(clean) < count:
        raise ValueError(f"only {len(clean)} distinct windows without special tokens")
    order = np.random.default_rng(seed).permutation(len(clean))[:count]
    pool = clean[order]
    if len({row.tobytes() for row in pool}) != count:
        raise ValueError("sampled windows are not unique")
    return np.ascontiguousarray(pool, dtype=np.int32)


def pool_sha256(pool: npt.NDArray[np.int32]) -> str:
    """Hash of the exact int32 array (shape and little-endian values)."""
    digest = hashlib.sha256(f"{pool.shape}".encode())
    digest.update(np.ascontiguousarray(pool, dtype="<i4").tobytes())
    return digest.hexdigest()


def load_pool(path: Path) -> npt.NDArray[np.int32]:
    pool: npt.NDArray[np.int32] = np.load(path, mmap_mode="r")
    return pool


@dataclass(frozen=True)
class NpyPromptPayloads:
    """Picklable payload factory over a saved pool: request index i uses
    prompt i. Each process memory-maps the file itself, so the pool is not
    pickled; the runner never repeats an index, so no prompt is reused.
    """

    path: str
    output_tokens: int
    seed: int
    model: str
    ignore_eos: bool = True
    skip_special_tokens: bool | None = False
    continuous_usage: bool = True
    _cache: dict[str, Any] = field(
        default_factory=dict, compare=False, repr=False, hash=False
    )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def prompt(self, request_index: int) -> list[int]:
        if "pool" not in self._cache:
            self._cache["pool"] = load_pool(Path(self.path))
        pool: npt.NDArray[np.int32] = self._cache["pool"]
        if not 0 <= request_index < len(pool):
            raise IndexError(
                f"prompt pool has {len(pool)} prompts; index {request_index} "
                "is out of range (prompts are never reused)"
            )
        return [int(token) for token in pool[request_index]]

    def __call__(self, request_index: int) -> dict[str, Any]:
        return completions_payload(
            prompt=self.prompt(request_index),
            request_index=request_index,
            output_tokens=self.output_tokens,
            seed=self.seed,
            model=self.model,
            ignore_eos=self.ignore_eos,
            skip_special_tokens=self.skip_special_tokens,
            continuous_usage=self.continuous_usage,
        )


def build_pool(
    lines: Iterable[str],
    tokenize: Callable[[str], list[int]],
    *,
    special_ids: Iterable[int],
    count: int,
    length: int,
    seed: int,
    oversample: float,
) -> tuple[npt.NDArray[np.int32], dict[str, Any]]:
    """Articles -> windows (the first `count * oversample`) -> seeded pool."""
    needed = int(count * oversample)
    windows, articles_read = token_windows(
        wikitext_articles(lines), tokenize, length=length, needed=needed
    )
    pool = sample_pool(windows, count=count, seed=seed, special_ids=special_ids)
    distinct = len(np.unique(windows, axis=0))
    return pool, {
        "articles_read": articles_read,
        "windows_available": int(len(windows)),
        "duplicate_windows": int(len(windows) - distinct),
        "windows_needed": needed,
        "prompt_count": count,
        "prompt_tokens": length,
        "seed": seed,
        "pool_sha256": pool_sha256(pool),
    }
