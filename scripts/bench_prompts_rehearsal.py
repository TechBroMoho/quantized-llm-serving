"""$0 rehearsal of the Phase 6 prompt build on WikiText-103's small validation file.

    make bench-prompts-rehearsal

Same parquet format, the pinned dataset revision and the pinned Qwen3-8B
tokenizer as `modal_app.bench::prepare`, but the 1.2 MB validation split and a
small pool, so it runs on a laptop. Checks that articles split at real
headings, windows are exact, and prompts decode to ordinary text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from llmbench.prompts import build_pool, wikitext_articles

REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def main() -> None:
    tokenizer_dir, out = sys.argv[1], Path(sys.argv[2])
    path = hf_hub_download(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
        repo_type="dataset",
        revision=REVISION,
    )
    lines = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    articles = list(wikitext_articles(lines))
    pool, info = build_pool(
        lines,
        lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
        special_ids=tokenizer.all_special_ids,
        count=200,
        length=512,
        seed=20260923,
        oversample=1.5,
    )
    assert pool.shape == (200, 512)
    first = tokenizer.decode(pool[0].tolist())
    report = info | {
        "lines": len(lines),
        "articles_in_file": len(articles),
        "first_article_heading": articles[0].splitlines()[0],
        "prompt_0_first_200_chars": first[:200],
        "retokenized_prompt_0_length": len(
            tokenizer(first, add_special_tokens=False)["input_ids"]
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
