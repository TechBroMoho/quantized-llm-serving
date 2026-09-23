"""Phase 5 accuracy: lm-eval command lines, result checks and the comparison.

This module never imports lm-eval. It builds the exact `lm-eval run` command
for a variant from the committed config (so every variant gets identical
settings by construction), validates lm-eval's saved JSON (few-shot count,
sample counts, no chat template, no `--limit` in full runs) and compares the
variants. The accuracy delta is BF16 minus quantized, in absolute percentage
points (SPEC §5).

    uv run python -m llmbench.accuracy --run-dir results/accuracy/phase5/<run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

MMLU_GROUP = "mmlu"
WIKITEXT = "wikitext"


def _arg_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if "," in text or "=" in text:
        raise ValueError(f"model_args value {text!r} would be split by lm-eval")
    return text


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def package_drift(installed: Mapping[str, str], constraints_text: str) -> list[str]:
    """Constrained packages whose installed version differs (or is missing).

    A constraint without a local label matches any local build of that
    version, as in pip (`torch==2.8.0` accepts `2.8.0+cu128`).
    """
    have = {_normalize(name): version for name, version in installed.items()}
    drift = []
    for raw in constraints_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        name, sep, wanted = line.partition("==")
        if not sep:
            continue
        got = have.get(_normalize(name))
        if got is None:
            drift.append(f"{name.strip()}: missing (wanted {wanted})")
        elif got != wanted and ("+" in wanted or got.split("+", 1)[0] != wanted):
            drift.append(f"{name.strip()}: {got} (wanted {wanted})")
    return drift


def model_args_string(args: Mapping[str, Any]) -> str:
    """lm-eval's `key=value,key=value` form, in a stable order."""
    return ",".join(f"{key}={_arg_value(value)}" for key, value in args.items())


def variant_model_args(
    settings: Mapping[str, Any], pretrained: str, tokenizer: str
) -> dict[str, Any]:
    """The config's shared model_args; only the checkpoint path differs."""
    return {"pretrained": pretrained, "tokenizer": tokenizer, **settings["model_args"]}


def lm_eval_command(
    settings: Mapping[str, Any],
    *,
    model_args: Mapping[str, Any],
    task: str,
    output_path: str,
    limit: int | None = None,
    executable: str = "lm-eval",
) -> list[str]:
    """The full `lm-eval run` command for one task on one variant."""
    task_settings = settings["tasks"][task]
    if settings.get("apply_chat_template"):
        raise ValueError("Phase 5 evaluates base-style prompts without a chat template")
    command = [
        executable,
        "run",
        "--model",
        str(settings["model"]),
        "--model_args",
        model_args_string(model_args),
        "--tasks",
        task,
        "--batch_size",
        str(settings["batch_size"]),
        "--seed",
        str(settings["seed"]),
        "--output_path",
        output_path,
    ]
    if task_settings.get("num_fewshot") is not None:
        command += ["--num_fewshot", str(task_settings["num_fewshot"])]
    if settings.get("log_samples"):
        command.append("--log_samples")
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def find_results_file(output_dir: Path) -> Path:
    """lm-eval writes one results_<timestamp>.json under a model-named folder."""
    files = sorted(output_dir.rglob("results_*.json"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected one results file in {output_dir}, got {files}"
        )
    return files[0]


def mmlu_subtasks(results: Mapping[str, Any]) -> list[str]:
    """Leaf MMLU subjects (the four category groups are not subjects)."""
    groups = results["group_subtasks"]
    leaves: list[str] = []
    for category in groups.get(MMLU_GROUP, []):
        leaves += groups.get(category, [category])
    return sorted(leaves)


def _stderr(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def summarize_mmlu(results: Mapping[str, Any]) -> dict[str, Any]:
    """Headline and per-subject accuracy from an lm-eval MMLU results file."""
    scores = results["results"]
    subjects: dict[str, dict[str, Any]] = {}
    for task in mmlu_subtasks(results):
        n = results["n-samples"][task]
        subjects[task.removeprefix("mmlu_")] = {
            "acc": float(scores[task]["acc,none"]),
            "acc_stderr": _stderr(scores[task].get("acc_stderr,none")),
            "n": int(n["effective"]),
        }
    categories = {
        name.removeprefix("mmlu_"): float(scores[name]["acc,none"])
        for name in results["group_subtasks"].get(MMLU_GROUP, [])
    }
    accs: list[float] = [item["acc"] for item in subjects.values()]
    return {
        # lm-eval's group score is weighted by subject size (micro average).
        "acc": float(scores[MMLU_GROUP]["acc,none"]),
        "acc_stderr": _stderr(scores[MMLU_GROUP].get("acc_stderr,none")),
        "acc_macro": sum(accs) / len(accs) if accs else math.nan,
        "questions": sum(item["n"] for item in subjects.values()),
        "categories": categories,
        "subjects": subjects,
    }


def summarize_wikitext(results: Mapping[str, Any]) -> dict[str, Any]:
    scores = results["results"][WIKITEXT]
    return {
        "word_perplexity": float(scores["word_perplexity,none"]),
        "byte_perplexity": float(scores["byte_perplexity,none"]),
        "bits_per_byte": float(scores["bits_per_byte,none"]),
        "documents": int(results["n-samples"][WIKITEXT]["effective"]),
    }


def check_results(
    results: Mapping[str, Any],
    task: str,
    task_settings: Mapping[str, Any],
    *,
    model_args: Mapping[str, Any],
    limited: bool,
) -> list[str]:
    """Reasons this lm-eval result is not the evaluation we configured."""
    failures: list[str] = []
    config = results.get("config", {})
    recorded_args = config.get("model_args", {})
    for key, value in model_args.items():
        recorded = recorded_args.get(key)
        if recorded != value:
            failures.append(
                f"model_args[{key}] recorded {recorded!r}, expected {value!r}"
            )
    if results.get("chat_template") is not None:
        failures.append("a chat template was applied")
    if results.get("fewshot_as_multiturn"):
        failures.append("few-shot examples were sent as multi-turn chat")
    if not limited and config.get("limit") is not None:
        failures.append(f"full run used --limit {config.get('limit')}")
    if task == MMLU_GROUP:
        leaves = mmlu_subtasks(results)
        if len(leaves) != task_settings["subtasks"]:
            failures.append(
                f"{len(leaves)} MMLU subjects, expected {task_settings['subtasks']}"
            )
        shots = {results["n-shot"].get(leaf) for leaf in leaves}
        if shots != {task_settings["num_fewshot"]}:
            failures.append(f"MMLU n-shot values {sorted(map(str, shots))}")
        effective = sum(results["n-samples"][leaf]["effective"] for leaf in leaves)
        original = sum(results["n-samples"][leaf]["original"] for leaf in leaves)
        if original != task_settings["expected_samples"]:
            failures.append(
                f"MMLU has {original} questions, "
                f"expected {task_settings['expected_samples']}"
            )
        if not limited and effective != original:
            failures.append(f"evaluated {effective} of {original} MMLU questions")
    elif task == WIKITEXT:
        n = results["n-samples"][WIKITEXT]
        if n["original"] != task_settings["expected_samples"]:
            failures.append(f"WikiText has {n['original']} documents")
        if not limited and n["effective"] != n["original"]:
            failures.append(f"evaluated {n['effective']} of {n['original']} documents")
        if results["n-shot"].get(WIKITEXT) not in (0, None):
            failures.append("WikiText must be zero-shot")
    else:
        failures.append(f"unknown task {task}")
    return failures


def read_samples(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def samples_digest(sample_files: Iterable[Path]) -> dict[str, Any]:
    """Compact per-task evidence: a prompt fingerprint and per-question results.

    Equal prompt digests across variants prove every variant saw byte-identical
    prompts. The per-question correctness string allows paired comparisons.
    """
    digest: dict[str, Any] = {}
    for path in sorted(sample_files):
        task = path.name.removeprefix("samples_").rsplit("_", 1)[0]
        rows = sorted(read_samples(path), key=lambda row: int(row["doc_id"]))
        prompts = "\n".join(f"{row['doc_id']}:{row['prompt_hash']}" for row in rows)
        entry: dict[str, Any] = {
            "n": len(rows),
            "prompt_digest": hashlib.sha256(prompts.encode()).hexdigest(),
        }
        if rows and "acc" in rows[0]:
            entry["correct"] = "".join(
                "1" if float(row["acc"]) else "0" for row in rows
            )
        digest[task] = entry
    return digest


def paired_flips(reference: str, other: str) -> dict[str, int]:
    """Questions that change outcome between two variants."""
    if len(reference) != len(other):
        raise ValueError("per-question strings differ in length")
    lost = sum(a == "1" and b == "0" for a, b in zip(reference, other, strict=True))
    gained = sum(a == "0" and b == "1" for a, b in zip(reference, other, strict=True))
    return {"lost": lost, "gained": gained}


def mcnemar(lost: int, gained: int) -> dict[str, float]:
    """Paired test of whether two variants differ on the same questions.

    Only discordant questions carry information. Chi-square with continuity
    correction, 1 degree of freedom: p = erfc(sqrt(chi2 / 2)). The variants'
    separate lm-eval standard errors ignore the pairing and overstate the
    uncertainty of the difference.
    """
    discordant = lost + gained
    if discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(lost - gained) - 1) ** 2 / discordant if lost != gained else 0.0
    return {"chi2": chi2, "p_value": math.erfc(math.sqrt(chi2 / 2))}


def compare(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    reference: str = "bf16",
    top_n: int = 5,
) -> dict[str, Any]:
    """Deltas vs. the reference: MMLU in percentage points, perplexity ratios."""
    ref = summaries[reference]
    out: dict[str, Any] = {"reference": reference, "variants": {}}
    for name, summary in summaries.items():
        mmlu = summary["mmlu"]
        wiki = summary["wikitext"]
        entry: dict[str, Any] = {
            "mmlu_questions": mmlu["questions"],
            "mmlu_acc": mmlu["acc"],
            "mmlu_acc_stderr": mmlu["acc_stderr"],
            "mmlu_acc_macro": mmlu["acc_macro"],
            "word_perplexity": wiki["word_perplexity"],
        }
        if name != reference:
            entry["mmlu_delta_pp"] = 100 * (ref["mmlu"]["acc"] - mmlu["acc"])
            entry["mmlu_relative_drop_pct"] = (
                100 * (ref["mmlu"]["acc"] - mmlu["acc"]) / ref["mmlu"]["acc"]
            )
            entry["word_perplexity_increase_pct"] = 100 * (
                wiki["word_perplexity"] / ref["wikitext"]["word_perplexity"] - 1
            )
            entry["category_delta_pp"] = {
                category: 100 * (ref["mmlu"]["categories"][category] - acc)
                for category, acc in mmlu["categories"].items()
            }
            subjects = []
            for subject, item in mmlu["subjects"].items():
                base = ref["mmlu"]["subjects"][subject]
                subjects.append(
                    {
                        "subject": subject,
                        "n": item["n"],
                        "bf16_acc": base["acc"],
                        "acc": item["acc"],
                        "delta_pp": 100 * (base["acc"] - item["acc"]),
                    }
                )
            subjects.sort(key=lambda row: (-row["delta_pp"], row["subject"]))
            entry["largest_drops"] = subjects[:top_n]
            entry["largest_gains"] = sorted(
                subjects, key=lambda row: (row["delta_pp"], row["subject"])
            )[:top_n]
            ref_samples = summary_samples(ref)
            samples = summary_samples(summary)
            if ref_samples and samples:
                entry["prompts_identical"] = all(
                    samples[task]["prompt_digest"] == ref_samples[task]["prompt_digest"]
                    for task in ref_samples
                ) and set(samples) == set(ref_samples)
                flips = [
                    paired_flips(ref_samples[task]["correct"], samples[task]["correct"])
                    for task in ref_samples
                    if "correct" in ref_samples[task]
                ]
                lost = sum(item["lost"] for item in flips)
                gained = sum(item["gained"] for item in flips)
                entry["mmlu_questions_lost"] = lost
                entry["mmlu_questions_gained"] = gained
                # Net lost questions must reproduce the headline delta.
                expected_net = entry["mmlu_delta_pp"] / 100 * mmlu["questions"]
                entry["flips_match_delta"] = abs((lost - gained) - expected_net) < 0.5
                entry["mcnemar"] = mcnemar(lost, gained)
        out["variants"][name] = entry
    return out


def summary_samples(summary: Mapping[str, Any]) -> dict[str, Any]:
    samples: dict[str, Any] = summary.get("samples", {}).get(MMLU_GROUP, {})
    return samples


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(
    comparison: Mapping[str, Any],
    labels: Mapping[str, str],
    notes: Sequence[str] = (),
) -> str:
    """The Phase 5 results table (committed next to the raw files)."""
    ref = comparison["reference"]
    lines = [
        "| Variant | MMLU 5-shot acc (± stderr) | Δ vs BF16 (pp) | "
        "WikiText-2 word perplexity | Δ perplexity |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, entry in comparison["variants"].items():
        stderr = entry["mmlu_acc_stderr"]
        acc = _pct(entry["mmlu_acc"]) + (f" ± {100 * stderr:.2f}" if stderr else "")
        delta = "—" if name == ref else f"{entry['mmlu_delta_pp']:+.2f}"
        ppl_delta = (
            "—" if name == ref else f"{entry['word_perplexity_increase_pct']:+.2f}%"
        )
        lines.append(
            f"| {labels.get(name, name)} | {acc} | {delta} | "
            f"{entry['word_perplexity']:.4f} | {ppl_delta} |"
        )
    questions = comparison["variants"][ref]["mmlu_questions"]
    lines += [
        "",
        "Δ MMLU is BF16 accuracy minus the variant's, in absolute percentage "
        "points (positive = lower accuracy). Accuracy is lm-eval's size-weighted "
        f"mean over all {questions:,} test questions.",
    ]
    others = [name for name in comparison["variants"] if name != ref]
    paired = [name for name in others if "mcnemar" in comparison["variants"][name]]
    if paired:
        lines += [
            "",
            f"**Paired per-question comparison vs BF16** (same {questions:,} "
            "prompts; McNemar test with continuity correction)",
            "",
            "| Variant | Lost (BF16 right, variant wrong) | Gained | Net lost | "
            "χ² | p |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in paired:
            entry = comparison["variants"][name]
            test = entry["mcnemar"]
            lost, gained = entry["mmlu_questions_lost"], entry["mmlu_questions_gained"]
            lines.append(
                f"| {labels.get(name, name)} | {lost} | {gained} | {lost - gained} | "
                f"{test['chi2']:.2f} | {test['p_value']:.2g} |"
            )
    if others:
        categories = sorted(comparison["variants"][others[0]]["category_delta_pp"])
        header = " | ".join(f"{labels.get(name, name)} Δ (pp)" for name in others)
        lines += [
            "",
            "**MMLU category deltas vs BF16** (pp)",
            "",
            f"| Category | {header} |",
            "| --- |" + " ---: |" * len(others),
        ]
        for category in categories:
            cells = " | ".join(
                f"{comparison['variants'][name]['category_delta_pp'][category]:+.2f}"
                for name in others
            )
            lines.append(f"| {category.replace('_', ' ')} | {cells} |")
    for name in others:
        entry = comparison["variants"][name]
        lines += [
            "",
            f"**{labels.get(name, name)}: largest per-subject drops** "
            f"(pp; questions lost/gained vs BF16: "
            f"{entry.get('mmlu_questions_lost', '?')}/"
            f"{entry.get('mmlu_questions_gained', '?')})",
            "",
            "| Subject | Questions | BF16 acc | Variant acc | Δ (pp) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in entry["largest_drops"]:
            lines.append(
                f"| {row['subject']} | {row['n']} | {_pct(row['bf16_acc'])} | "
                f"{_pct(row['acc'])} | {row['delta_pp']:+.2f} |"
            )
    for note in notes:
        lines += ["", note]
    return "\n".join(lines) + "\n"


# Log lines that document the resolved engine and the timing of each stage.
LOG_PATTERNS = (
    r"non-default args",
    r"Model loading took",
    r"GPU KV cache size",
    r"Maximum concurrency",
    r"init engine",
    r"max_num_batched_tokens",
    r"Using .* backend",
    r"WNA16|Marlin|[Kk]ernel",
    r"[Ee]nforce.?eager|[Ee]ager mode",
    r"[Pp]refix caching",
    r"Building contexts",
)
TRUNCATION_WARNING = "Truncating context"


def last_progress_line(path: Path, marker: str) -> str | None:
    """The final state of a tqdm bar (tqdm redraws with carriage returns)."""
    if not path.exists():
        return None
    last = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker in line:
            last = line.strip()[:500]
    return last


def run_variant(
    settings: Mapping[str, Any],
    *,
    label: str,
    model_args: Mapping[str, Any],
    out_dir: Path,
    limits: Mapping[str, int | None],
    timeout_s: float,
    executable: str = "lm-eval",
    env: dict[str, str] | None = None,
    on_progress: Any = None,
) -> dict[str, Any]:
    """Run every configured task for one variant; save a checked summary.

    Each task is its own `lm-eval run` process (so `--num_fewshot` can differ)
    with identical model_args. Raw lm-eval output and the full log stay in
    `out_dir/<task>/`; `out_dir/summary.json` is rewritten after every task.
    """
    import resource
    import subprocess
    import time

    from llmbench.smoke import matching_lines, start_server, stop_server, write_json

    limited = any(value is not None for value in limits.values())
    record: dict[str, Any] = {
        "variant": label,
        "model_args": dict(model_args),
        "limits": dict(limits),
        "limited": limited,
        "tasks": {},
        "samples": {},
        "failures": [],
        "state": "started",
    }

    def save(state: str) -> None:
        record["state"] = state
        record["passed"] = state == "finished" and not record["failures"]
        write_json(out_dir / "summary.json", record)
        if on_progress is not None:
            on_progress()

    save("started")
    for task, task_settings in settings["tasks"].items():
        task_dir = out_dir / task
        log_path = task_dir / "lm_eval.log"
        command = lm_eval_command(
            settings,
            model_args=model_args,
            task=task,
            output_path=str(task_dir / "lm_eval_output"),
            limit=limits.get(task),
            executable=executable,
        )
        info: dict[str, Any] = {"command": command}
        record["tasks"][task] = info
        save(f"running {task}")
        started = time.time()
        process = start_server(command, log_path, env)
        try:
            info["returncode"] = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            info["returncode"] = None
            record["failures"].append(f"{task}: timed out after {timeout_s} s")
        finally:
            stop_server(process)
        info["elapsed_s"] = time.time() - started
        # Largest RSS of any finished descendant so far (KiB on Linux), which
        # includes vLLM's engine-core process once lm-eval has reaped it.
        info["children_max_rss"] = resource.getrusage(
            resource.RUSAGE_CHILDREN
        ).ru_maxrss
        info["log_lines"] = matching_lines(log_path, LOG_PATTERNS, limit=80)
        info["loglikelihood_progress"] = last_progress_line(
            log_path, "Running loglikelihood requests"
        )
        truncated = len(matching_lines(log_path, (TRUNCATION_WARNING,), limit=10**6))
        info["truncation_warnings"] = truncated
        if truncated:
            record["failures"].append(f"{task}: {truncated} truncated contexts")
        if info["returncode"] != 0:
            if info["returncode"] is not None:
                record["failures"].append(
                    f"{task}: lm-eval exited {info['returncode']}"
                )
            save(f"failed {task}")
            break
        results_file = find_results_file(task_dir / "lm_eval_output")
        results = json.loads(results_file.read_text(encoding="utf-8"))
        info["results_file"] = str(results_file.relative_to(out_dir))
        info["lm_eval_version"] = results.get("lm_eval_version")
        info["transformers_version"] = results.get("transformers_version")
        info["total_evaluation_time_seconds"] = results.get(
            "total_evaluation_time_seconds"
        )
        record["failures"] += [
            f"{task}: {failure}"
            for failure in check_results(
                results, task, task_settings, model_args=model_args, limited=limited
            )
        ]
        summarize = summarize_mmlu if task == MMLU_GROUP else summarize_wikitext
        record[task] = summarize(results)
        record["samples"][task] = samples_digest(task_dir.rglob("samples_*.jsonl"))
        save(f"finished {task}")
    save("failed" if record["failures"] else "finished")
    return record


def load_summaries(run_dir: Path, variants: Sequence[str]) -> dict[str, Any]:
    return {
        name: json.loads((run_dir / name / "summary.json").read_text(encoding="utf-8"))
        for name in variants
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variants", default="bf16,awq,gptq")
    args = parser.parse_args()
    names = [name.strip() for name in args.variants.split(",") if name.strip()]
    summaries = load_summaries(args.run_dir, names)
    for name, summary in summaries.items():
        if not summary.get("passed"):
            raise SystemExit(
                f"{name} did not pass its checks: {summary.get('failures')}"
            )
    comparison = compare(summaries)
    labels = {
        "bf16": "BF16",
        "awq": "AWQ W4A16-asym (pile-val calib.)",
        "gptq": "GPTQ W4A16-sym (ultrachat calib.)",
    }
    (args.run_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    notes = [
        "AWQ and GPTQ used different calibration data, each following its "
        "official llm-compressor 0.7.1 example (AWQ: pile-val, 256 x 512 tokens; "
        "GPTQ: UltraChat, 512 x 2,048 tokens; ADR-016), so an AWQ-vs-GPTQ "
        "difference mixes the method with its calibration set.",
        "Per-subject deltas are noisy: in a 100-question subject one question is "
        "1 pp. The paired test above is the right measure for the overall delta.",
    ]
    table = render_markdown(comparison, labels, notes)
    (args.run_dir / "accuracy_table.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
