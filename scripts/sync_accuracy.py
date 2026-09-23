"""Copy Phase 5 results from a Volume download into results/accuracy/phase5.

lm-eval's per-sample JSONL files (full prompts, tens of MB per variant) stay
on the Volume; `summary.json` keeps their prompt digests and per-question
correctness. Progress-bar redraws are dropped from logs, keeping each bar's
completed (100%) line.

    make sync-accuracy
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

SOURCE = Path(".cache/phase5-sync/phase5")
TARGET = Path("results/accuracy/phase5")
PROGRESS = re.compile(r"(it/s|s/it|toks/s)\]?\s*$|\|\s*\d+/\d+ \[")


def filtered_log(text: str) -> str:
    kept = [
        line
        for line in text.splitlines()
        if not PROGRESS.search(line) or "100%" in line
    ]
    return "\n".join(kept) + "\n"


def main() -> None:
    if not SOURCE.is_dir():
        sys.exit(f"{SOURCE} not found; run `make sync-accuracy`")
    copied = skipped = 0
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file():
            continue
        target = TARGET / path.relative_to(SOURCE)
        if path.name.startswith("samples_") and path.suffix == ".jsonl":
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".log":
            target.write_text(
                filtered_log(path.read_text(encoding="utf-8", errors="replace")),
                encoding="utf-8",
            )
        else:
            shutil.copy2(path, target)
        copied += 1
    print(
        f"copied {copied} files to {TARGET}; left {skipped} sample files on the Volume"
    )


if __name__ == "__main__":
    main()
