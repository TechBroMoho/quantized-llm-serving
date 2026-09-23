"""Phase 5 accuracy helpers: commands, result checks and the comparison."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from llmbench.accuracy import (
    check_results,
    compare,
    lm_eval_command,
    model_args_string,
    package_drift,
    paired_flips,
    render_markdown,
    samples_digest,
    summarize_mmlu,
    variant_model_args,
)

CONFIG = yaml.safe_load(Path("configs/phase5_accuracy.yaml").read_text())
SETTINGS = CONFIG["lm_eval"]


def _command(pretrained: str, task: str, limit: int | None = None) -> list[str]:
    return lm_eval_command(
        SETTINGS,
        model_args=variant_model_args(SETTINGS, pretrained, "/weights/bf16"),
        task=task,
        output_path="/results/out",
        limit=limit,
    )


def test_committed_settings_are_the_documented_protocol() -> None:
    args = SETTINGS["model_args"]
    assert SETTINGS["model"] == "vllm"
    assert SETTINGS["apply_chat_template"] is False
    assert SETTINGS["tasks"]["mmlu"]["num_fewshot"] == 5
    assert SETTINGS["tasks"]["wikitext"]["num_fewshot"] is None
    assert args["enable_prefix_caching"] is False
    assert args["dtype"] == "bfloat16"
    # `auto` would hold every MMLU request's prompt logprobs in memory at once.
    assert isinstance(SETTINGS["batch_size"], int)
    assert CONFIG["variants"] == ["bf16", "awq", "gptq"]


def test_max_model_len_fits_the_audited_longest_prompt() -> None:
    audit = json.loads(
        Path("results/accuracy/phase5/prompt_lengths_local.json").read_text()
    )
    max_len = SETTINGS["model_args"]["max_model_len"]
    assert audit["max_length"] == max_len
    assert audit["fits"] is True
    # lm-eval's vLLM backend left-truncates anything above max_length - 1.
    for task in audit["tasks"].values():
        assert task["max_tokens"] <= max_len - 1
        assert task["requests_over_limit"] == 0


def test_variants_differ_only_in_the_checkpoint_path() -> None:
    for task in ("mmlu", "wikitext"):
        bf16 = _command("/weights/bf16", task)
        awq = _command("/weights/awq", task)
        diff = [(a, b) for a, b in zip(bf16, awq, strict=True) if a != b]
        assert len(diff) == 1 and "pretrained=/weights/bf16" in diff[0][0]
        assert "tokenizer=/weights/bf16" in diff[0][1]


def test_command_shape() -> None:
    mmlu = _command("/weights/awq", "mmlu")
    assert mmlu[:2] == ["lm-eval", "run"]
    assert mmlu[mmlu.index("--num_fewshot") + 1] == "5"
    assert mmlu[mmlu.index("--tasks") + 1] == "mmlu"
    assert "--apply_chat_template" not in mmlu and "--limit" not in mmlu
    assert "--log_samples" in mmlu
    assert "--num_fewshot" not in _command("/weights/awq", "wikitext")
    limited = _command("/weights/awq", "mmlu", limit=10)
    assert limited[limited.index("--limit") + 1] == "10"


def test_model_args_string() -> None:
    text = model_args_string({"a": 1, "b": False, "c": 0.8, "d": "x"})
    assert text == "a=1,b=false,c=0.8,d=x"
    with pytest.raises(ValueError):
        model_args_string({"a": "x,y"})


def test_chat_template_is_refused() -> None:
    with pytest.raises(ValueError):
        lm_eval_command(
            {**SETTINGS, "apply_chat_template": True},
            model_args={},
            task="mmlu",
            output_path="x",
        )


SUBJECTS = {"mmlu_stem": ["mmlu_a", "mmlu_b"], "mmlu_other": ["mmlu_c"]}


def fake_mmlu(accs: dict[str, float], n: int = 10) -> dict[str, Any]:
    results: dict[str, Any] = {
        leaf: {"acc,none": acc, "acc_stderr,none": 0.01} for leaf, acc in accs.items()
    }
    total = sum(accs.values()) * n / (n * len(accs))
    results["mmlu"] = {"acc,none": total, "acc_stderr,none": 0.004}
    for group, leaves in SUBJECTS.items():
        results[group] = {"acc,none": sum(accs[x] for x in leaves) / len(leaves)}
    return {
        "results": results,
        "group_subtasks": {"mmlu": list(SUBJECTS), **copy.deepcopy(SUBJECTS)},
        "n-shot": dict.fromkeys(accs, 5),
        "n-samples": {leaf: {"original": n, "effective": n} for leaf in accs},
        "config": {
            "model_args": {"pretrained": "/w/x", "dtype": "bfloat16"},
            "limit": None,
        },
        "chat_template": None,
        "fewshot_as_multiturn": False,
    }


MMLU_SETTINGS = {"num_fewshot": 5, "expected_samples": 30, "subtasks": 3}
ARGS = {"pretrained": "/w/x", "dtype": "bfloat16"}


def test_check_results_accepts_the_configured_run() -> None:
    results = fake_mmlu({"mmlu_a": 0.5, "mmlu_b": 0.6, "mmlu_c": 0.7})
    assert (
        check_results(results, "mmlu", MMLU_SETTINGS, model_args=ARGS, limited=False)
        == []
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r["n-shot"].update(mmlu_a=0), "n-shot"),
        (lambda r: r["config"].update(limit=10), "--limit"),
        (lambda r: r.update(chat_template="{{x}}"), "chat template"),
        (lambda r: r.update(fewshot_as_multiturn=True), "multi-turn"),
        (
            lambda r: r["config"]["model_args"].update(dtype="float16"),
            "model_args[dtype]",
        ),
        (lambda r: r["n-samples"]["mmlu_b"].update(effective=9), "evaluated 29 of 30"),
        (lambda r: r["group_subtasks"]["mmlu_other"].clear(), "2 MMLU subjects"),
    ],
)
def test_check_results_rejects_deviations(mutate: Any, expected: str) -> None:
    results = fake_mmlu({"mmlu_a": 0.5, "mmlu_b": 0.6, "mmlu_c": 0.7})
    mutate(results)
    failures = check_results(
        results, "mmlu", MMLU_SETTINGS, model_args=ARGS, limited=False
    )
    assert any(expected in failure for failure in failures), failures


def test_limited_probe_is_allowed_to_subsample() -> None:
    results = fake_mmlu({"mmlu_a": 0.5, "mmlu_b": 0.6, "mmlu_c": 0.7})
    results["config"]["limit"] = 2
    for leaf in ("mmlu_a", "mmlu_b", "mmlu_c"):
        results["n-samples"][leaf]["effective"] = 2
    assert (
        check_results(results, "mmlu", MMLU_SETTINGS, model_args=ARGS, limited=True)
        == []
    )
    assert check_results(results, "mmlu", MMLU_SETTINGS, model_args=ARGS, limited=False)


def test_wikitext_checks() -> None:
    results = {
        "results": {"wikitext": {}},
        "n-shot": {"wikitext": 0},
        "n-samples": {"wikitext": {"original": 62, "effective": 5}},
        "config": {"model_args": ARGS, "limit": 5},
        "chat_template": None,
    }
    wiki = {"num_fewshot": None, "expected_samples": 62}
    assert check_results(results, "wikitext", wiki, model_args=ARGS, limited=True) == []
    failures = check_results(results, "wikitext", wiki, model_args=ARGS, limited=False)
    assert any("5 of 62" in failure for failure in failures)


def summary(accs: dict[str, float], ppl: float) -> dict[str, Any]:
    return {
        "mmlu": summarize_mmlu(fake_mmlu(accs)),
        "wikitext": {"word_perplexity": ppl},
    }


def test_compare_reports_percentage_points_and_largest_drops() -> None:
    summaries = {
        "bf16": summary({"mmlu_a": 0.8, "mmlu_b": 0.6, "mmlu_c": 0.7}, 10.0),
        "awq": summary({"mmlu_a": 0.6, "mmlu_b": 0.6, "mmlu_c": 0.75}, 10.5),
    }
    result = compare(summaries, top_n=2)["variants"]
    awq = result["awq"]
    # micro accuracy 0.7 -> 0.65: 5 points, not 5% or 0.05
    assert awq["mmlu_delta_pp"] == pytest.approx(5.0)
    assert awq["mmlu_relative_drop_pct"] == pytest.approx(100 * 0.05 / 0.7)
    assert awq["word_perplexity_increase_pct"] == pytest.approx(5.0)
    assert [row["subject"] for row in awq["largest_drops"]] == ["a", "b"]
    assert awq["largest_drops"][0]["delta_pp"] == pytest.approx(20.0)
    assert awq["largest_gains"][0]["subject"] == "c"
    assert "mmlu_delta_pp" not in result["bf16"]
    table = render_markdown(compare(summaries), {"awq": "AWQ"})
    assert "| AWQ | 65.00% ± 0.40 | +5.00 |" in table
    assert "all 30 test questions" in table


def _write_samples(path: Path, prompts: list[str], correct: list[int]) -> None:
    rows = [
        {"doc_id": i, "prompt_hash": p, "acc": float(c)}
        for i, (p, c) in reversed(list(enumerate(zip(prompts, correct, strict=True))))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_samples_digest_and_paired_flips(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for directory in (a, b, c):
        directory.mkdir()
    name = "samples_mmlu_a_2026-09-23T06-08-14.242948.jsonl"
    _write_samples(a / name, ["p0", "p1", "p2"], [1, 1, 0])
    _write_samples(b / name, ["p0", "p1", "p2"], [1, 0, 1])
    _write_samples(c / name, ["p0", "pX", "p2"], [1, 1, 0])
    da, db, dc = (samples_digest(d.glob("*.jsonl")) for d in (a, b, c))
    assert list(da) == ["mmlu_a"]
    assert da["mmlu_a"]["correct"] == "110"  # ordered by doc_id
    assert da["mmlu_a"]["prompt_digest"] == db["mmlu_a"]["prompt_digest"]
    assert da["mmlu_a"]["prompt_digest"] != dc["mmlu_a"]["prompt_digest"]
    assert paired_flips("110", "101") == {"lost": 1, "gained": 1}

    base = summary({"mmlu_a": 0.8, "mmlu_b": 0.6, "mmlu_c": 0.7}, 10.0)
    same = copy.deepcopy(base) | {"samples": {"mmlu": db}}
    other = copy.deepcopy(base) | {"samples": {"mmlu": dc}}
    result = compare({"bf16": base | {"samples": {"mmlu": da}}, "x": same, "y": other})
    assert result["variants"]["x"]["prompts_identical"] is True
    assert result["variants"]["x"]["mmlu_questions_lost"] == 1
    # 1 lost and 1 gained: net 0, matching the unchanged accuracy
    assert result["variants"]["x"]["flips_match_delta"] is True
    lower = summary({"mmlu_a": 0.5, "mmlu_b": 0.6, "mmlu_c": 0.7}, 10.0)
    mismatch = compare(
        {
            "bf16": base | {"samples": {"mmlu": da}},
            "z": lower | {"samples": {"mmlu": db}},
        }
    )
    # a 10-point drop over 30 questions needs 3 net lost, not 0
    assert mismatch["variants"]["z"]["flips_match_delta"] is False
    assert result["variants"]["y"]["prompts_identical"] is False


def test_package_drift() -> None:
    constraints = "# comment\ntorch==2.8.0\nvllm==0.10.2\nnumpy==2.2.6\nfoo==1.0+abc\n"
    installed = {
        "torch": "2.8.0+cu128",
        "vllm": "0.10.2",
        "numpy": "2.2.6",
        "Foo": "1.0+abc",
    }
    assert package_drift(installed, constraints) == []
    moved = {**installed, "numpy": "2.3.0", "Foo": "1.0"}
    del moved["vllm"]
    drift = package_drift(moved, constraints)
    assert drift == [
        "vllm: missing (wanted 0.10.2)",
        "numpy: 2.3.0 (wanted 2.2.6)",
        "Foo: 1.0 (wanted 1.0+abc)".replace("Foo", "foo"),
    ]


def _pins(path: str) -> dict[str, str]:
    pins = {}
    for line in Path(path).read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)", line)
        if match:
            pins[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = match.group(2)
    return pins


def test_eval_stack_leaves_every_vllm_image_package_unchanged() -> None:
    image = _pins("requirements/vllm-image-constraints.txt")
    resolved = _pins("requirements/eval-linux.txt")
    assert resolved["lm-eval"] == "0.4.11"
    assert image["vllm"] == "0.10.2" and image["torch"] == "2.8.0"
    changed = {
        k: (image[k], v) for k, v in resolved.items() if k in image and image[k] != v
    }
    assert changed == {}
