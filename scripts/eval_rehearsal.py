"""CPU rehearsal of the Phase 5 evaluation code path (no GPU, $0).

Runs `llmbench.accuracy.run_variant` exactly as the Modal job does, with the
committed task settings (MMLU 5-shot, WikiText, batch size, seeds, samples),
but with lm-eval's `hf` backend on two tiny random Qwen3 models that share the
real Qwen3-8B tokenizer, and with `--limit`. It then builds the comparison and
table. The accuracies are meaningless (random weights); the point is that the
real lm-eval 0.4.11 output passes our parsers and checks end to end.

    make eval-rehearsal
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from llmbench.accuracy import compare, render_markdown, run_variant

OUT = Path("results/validation/phase5/rehearsal")
WORK = Path(".cache/eval-rehearsal")
QWEN = "Qwen/Qwen3-8B"


def tiny_model(tokenizer_dir: str, model_dir: Path, seed: int) -> None:
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(model_dir)
    AutoTokenizer.from_pretrained(tokenizer_dir).save_pretrained(model_dir)


def main() -> None:
    config = yaml.safe_load(Path("configs/phase5_accuracy.yaml").read_text())
    quant = yaml.safe_load(Path("configs/phase4_quantize.yaml").read_text())
    tokenizer_dir = snapshot_download(
        QWEN,
        revision=quant["model"]["revision"],
        allow_patterns=["tokenizer*", "vocab.json", "merges.txt"],
    )
    settings = {
        **config["lm_eval"],
        "model": "hf",  # the only change of backend: vLLM needs a GPU
        # The hf backend materializes batch x length x vocab logits, so the
        # vLLM request-chunk size (1024) would exhaust laptop memory.
        "batch_size": 8,
        # No CUDA here, so lm-eval's hf backend falls back to the CPU.
        "model_args": {"dtype": "float32", "max_length": 4096},
    }
    limits = {"mmlu": 2, "wikitext": 1}
    summaries = {}
    for name, seed in (("bf16", 0), ("awq", 1)):
        model_dir = WORK / f"tiny-{name}"
        if not (model_dir / "config.json").exists():
            tiny_model(tokenizer_dir, model_dir, seed)
        model_args = {"pretrained": str(model_dir), **settings["model_args"]}
        summaries[name] = run_variant(
            settings,
            label=name,
            model_args=model_args,
            out_dir=WORK / "run" / name,
            limits=limits,
            timeout_s=1800,
            # repo-relative, so no local home path lands in committed results
            executable=os.path.relpath(Path(sys.executable).parent / "lm-eval"),
            env={"TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1"},
        )
        print(name, summaries[name]["state"], summaries[name]["failures"])
        if not summaries[name]["passed"]:
            raise SystemExit(f"{name} failed: {summaries[name]['failures']}")
    comparison = compare(summaries)
    table = render_markdown(comparison, {"bf16": "tiny A", "awq": "tiny B"})
    report = {
        "purpose": "CPU rehearsal with random tiny models; accuracies are meaningless",
        "backend": "hf (the Modal run uses vllm)",
        "limits": limits,
        "passed": all(s["passed"] for s in summaries.values())
        and comparison["variants"]["awq"].get("prompts_identical") is True,
        "summaries": {
            name: {
                key: summary[key]
                for key in ("state", "failures", "passed", "limited", "tasks")
            }
            | {
                "mmlu_acc": summary["mmlu"]["acc"],
                "mmlu_questions": summary["mmlu"]["questions"],
                "word_perplexity": summary["wikitext"]["word_perplexity"],
            }
            for name, summary in summaries.items()
        },
        "comparison": comparison,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rehearsal.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "rehearsal_table.md").write_text(table)
    print(table)
    print("passed:", report["passed"])
    if not report["passed"]:
        raise SystemExit("eval rehearsal failed")


if __name__ == "__main__":
    main()
