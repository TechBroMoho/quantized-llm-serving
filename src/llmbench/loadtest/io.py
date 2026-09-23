"""Persist per-request JSONL and a self-describing summary JSON."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from llmbench.loadtest.metrics import RequestRecord


def write_results(
    summary_path: Path,
    summary: dict[str, Any],
    records: list[RequestRecord],
    *,
    raw_path: Path | None = None,
) -> Path:
    """Write a summary and sibling JSONL file, optionally gzip-compressed."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path is None:
        raw_path = summary_path.with_suffix(".jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.suffix == ".gz":
        with gzip.open(raw_path, "wt", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record.as_dict(), separators=(",", ":")))
                stream.write("\n")
    else:
        with raw_path.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record.as_dict(), separators=(",", ":")))
                stream.write("\n")
    summary["raw_results_path"] = str(raw_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return raw_path
