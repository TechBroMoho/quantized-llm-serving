"""Measure the exact token lengths lm-eval would send to vLLM, without a GPU.

lm-eval's vLLM backend truncates any request longer than `max_length - 1`
tokens from the left, silently changing the 5-shot prompt. Before any GPU
run we therefore drive lm-eval's own request construction (`simple_evaluate`)
with a stub model that records every tokenized request instead of scoring it.
The stub copies the two pieces of the pinned lm-eval 0.4.11 vLLM backend that
decide lengths: `tok_encode` (plain tokenizer call, no special tokens added
for Qwen3) and the rolling windows of `max_length - 2` tokens used for
perplexity. Everything else (few-shot sampling, prompt templates, the
context/continuation split) is lm-eval's real code.

    python -m llmbench.eval_audit --tokenizer PATH --max-length 4096 --out FILE \
        [--limit mmlu=10 --limit wikitext=5]

The scores it returns are placeholders; only the lengths are meaningful.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from lm_eval.api.model import TemplateLM


class LengthAuditLM(TemplateLM):  # type: ignore[misc]
    """Records request lengths; never computes a log-likelihood."""

    def __init__(self, tokenizer_path: str, max_length: int) -> None:
        super().__init__()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # type: ignore[no-untyped-call]
        self._max_length = max_length
        # (task, request_type, total tokens) for every request sent to the model
        self.requests: list[tuple[str, str, int]] = []
        self._pending_tasks: list[str] = []

    @property
    def eot_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def prefix_token_id(self) -> int:
        # Same rule as the vLLM backend: BOS if the tokenizer has one, else EOS.
        bos = self.tokenizer.bos_token_id
        return int(bos if bos is not None else self.tokenizer.eos_token_id)

    @property
    def max_length(self) -> int:
        return self._max_length

    def tok_encode(self, string: str, **kwargs: Any) -> list[int]:
        if not string:
            return []
        return list(self.tokenizer(string, return_attention_mask=False).input_ids)

    def loglikelihood(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        # TemplateLM encodes the requests in order and passes them on in one
        # call, so the task names line up with `_loglikelihood_tokens` below.
        self._pending_tasks = [request.task_name for request in requests]
        result: list[tuple[float, bool]] = super().loglikelihood(requests, disable_tqdm)
        return result

    def _loglikelihood_tokens(
        self, requests: list[Any], disable_tqdm: bool = False, **kwargs: Any
    ) -> list[tuple[float, bool]]:
        if len(self._pending_tasks) != len(requests):
            raise RuntimeError("request/task bookkeeping is out of step")
        for task, (_, context, continuation) in zip(
            self._pending_tasks, requests, strict=True
        ):
            self.requests.append(
                (task, "loglikelihood", len(context) + len(continuation))
            )
        return [(0.0, False)] * len(requests)

    def loglikelihood_rolling(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[float]:
        from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

        totals = []
        for request in requests:
            (string,) = request.args
            windows = [
                make_disjoint_window(window)
                for window in get_rolling_token_windows(
                    token_list=self.tok_encode(string),
                    prefix_token=self.prefix_token_id,
                    max_seq_len=self.max_length - 2,  # as in the vLLM backend
                    context_len=1,
                )
            ]
            for context, continuation in windows:
                self.requests.append(
                    (
                        request.task_name,
                        "rolling_window",
                        len(context) + len(continuation),
                    )
                )
            totals.append(0.0)
        return totals

    def generate_until(self, requests: list[Any], disable_tqdm: bool = False) -> Any:
        raise NotImplementedError("the accuracy tasks are log-likelihood only")


def summarize(requests: list[tuple[str, str, int]], max_length: int) -> dict[str, Any]:
    """Length statistics and the truncation verdict for recorded requests."""
    # The vLLM backend truncates anything above max_length - 1 tokens.
    limit = max_length - 1
    lengths = [length for _, _, length in requests]
    longest = max(requests, key=lambda item: item[2]) if requests else None
    over = [item for item in requests if item[2] > limit]
    by_type = Counter(kind for _, kind, _ in requests)
    task_max: dict[str, int] = {}
    for task, _, length in requests:
        task_max[task] = max(length, task_max.get(task, 0))
    top = sorted(task_max.items(), key=lambda item: -item[1])[:5]
    return {
        "requests": len(requests),
        "requests_by_type": dict(by_type),
        "total_tokens": sum(lengths),
        "mean_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        "max_tokens": longest[2] if longest else 0,
        "longest_task": longest[0] if longest else None,
        "longest_tasks": dict(top),
        "truncation_limit": limit,
        "requests_over_limit": len(over),
        "fits": not over,
    }


def audit(
    tokenizer_path: str, max_length: int, limits: dict[str, int] | None = None
) -> dict[str, Any]:
    """Run lm-eval's MMLU 5-shot and WikiText request construction.

    `limits` mirrors lm-eval's `--limit` per task (first N documents of every
    subtask), so the probe's exact token volume can be counted too.
    """
    import lm_eval
    from lm_eval.tasks import TaskManager

    lm = LengthAuditLM(tokenizer_path, max_length)
    manager = TaskManager()
    results: dict[str, Any] = {}
    limits = limits or {}
    for label, task, shots in (
        ("mmlu_5shot", "mmlu", 5),
        ("wikitext", "wikitext", None),
    ):
        lm.requests = []
        lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=shots,
            limit=limits.get(task),
            task_manager=manager,
            apply_chat_template=False,
            bootstrap_iters=0,
        )
        results[label] = summarize(lm.requests, max_length)
    return {
        "tokenizer": tokenizer_path,
        "max_length": max_length,
        "limits": limits,
        "lm_eval_version": lm_eval.__version__,
        "tasks": results,
        "fits": all(item["fits"] for item in results.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument(
        "--limit", action="append", default=[], metavar="TASK=N", help="per task"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    limits = {
        task: int(count) for task, count in (item.split("=", 1) for item in args.limit)
    }
    report = audit(args.tokenizer, args.max_length, limits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["fits"]:
        raise SystemExit("some requests exceed max_length - 1 and would be truncated")


if __name__ == "__main__":
    main()
