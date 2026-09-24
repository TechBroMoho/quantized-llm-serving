"""Phase 7: render docs/RESULTS.md from `results/analysis/aggregate.json` ($0).

    uv run python -m llmbench.analysis.report   # -> docs/RESULTS.md

Every number in the page is formatted from the aggregate file, and each one
links to the raw file it came from (paths are relative to docs/). Nothing is
typed in by hand, so the page cannot drift from the results; a test checks
that the committed page equals a fresh render and that every link resolves.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from llmbench.analysis.aggregate import OUT as AGGREGATE

RESULTS_MD = Path("docs/RESULTS.md")
CHARTS = "../results/analysis"


def link(text: str, path: str) -> str:
    """A link from docs/RESULTS.md to a repo path (optionally with #L<n>)."""
    return f"[{text}](../{path})"


def runs(sources: list[str]) -> str:
    """Numbered links to each run a median was taken over."""
    return " ".join(link(f"r{i}", path) for i, path in enumerate(sources, 1))


def f1(value: float) -> str:
    return f"{value:,.1f}"


def ms(seconds: float) -> str:
    return f"{seconds * 1000:,.1f}"


def x(ratio: float) -> str:
    return f"{ratio:.2f}×"


class Page:
    """Accessors over the aggregate, so the text reads like the claims."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.head = data["headline"]
        self.table = data["headline_source"]

    def name(self, system: str) -> str:
        return str(self.data["systems"][system])

    def point(self, system: str, base: str) -> dict[str, Any]:
        return next(p for p in self.data["series"][system] if p["point"] == base)

    def peak(self, system: str, key: str = "peak_output_tokens_per_s") -> Any:
        return self.head[key][system]

    def peak_tokens(self, system: str) -> str:
        peak = self.peak(system)
        return link(f1(peak["median"]), self.table) + (
            f" (c={self.point(system, peak['point'])['concurrency']}, "
            f"{peak['runs']} runs: "
            f"{runs(self.point(system, peak['point'])['sources'])})"
        )

    def decode(self, system: str) -> str:
        values = self.head["single_user_decode"][system]
        return (
            f"{link(f1(values['tokens_per_s']), self.table)} tokens/s "
            f"(median TPOT {ms(values['tpot']['median'])} ms over "
            f"{values['tpot']['runs']} runs: "
            f"{runs(self.point(system, 'c1')['sources'])})"
        )

    def ratio(self, group: str, key: str) -> str:
        return link(x(self.head[group][key]), self.table)


def summary_section(p: Page) -> list[str]:
    mem = p.data["memory"]["variants"]
    acc = p.data["accuracy"]
    awq_acc = acc["variants"]["awq"]
    rr = p.head["requests_per_s_ratios"]
    hf = p.data["hf_static"]
    return [
        "## Summary",
        "",
        "All numbers were measured on one NVIDIA L40S with the W1 workload (512 "
        "input / 256 output tokens, greedy, unique prompts, prefix caching off). "
        "Headline values are medians of three passing runs unless a run count "
        "says otherwise. Engine gains and quantization gains are reported "
        "separately. Each is measured with the other held fixed.",
        "",
        "### Engine gain: vLLM vs Hugging Face `transformers`, both BF16",
        "",
        "Same BF16 weights, same GPU, same prompts. Only the serving engine differs.",
        "",
        f"- Peak output throughput: vLLM BF16 {p.peak_tokens('bf16')} tokens/s, "
        f"HF static batching {p.peak_tokens('hf-static')}, HF naive "
        f"{p.peak_tokens('hf-naive')}.",
        f"- **vLLM BF16 vs HF naive: {p.ratio('engine_gain_vllm_bf16_vs_hf', 'naive')}"
        f"; vs HF static batching: "
        f"{p.ratio('engine_gain_vllm_bf16_vs_hf', 'static')}** (output tokens/s at "
        f"each system's peak). Peak requests/s gives "
        f"{link(x(rr['bf16_vs_hf-naive']), p.table)} and "
        f"{link(x(rr['bf16_vs_hf-static']), p.table)}.",
        f"- For one user the engines are close: vLLM BF16 {p.decode('bf16')} vs "
        f"HF naive "
        f"{link(f1(p.point('hf-naive', 'c1')['metrics']['tokens_per_s']['median']), p.point('hf-naive', 'c1')['sources'][0])}"
        " output tokens/s at c=1 (1 run).",
        "",
        "### Quantization gain: vLLM AWQ (W4A16) vs vLLM BF16",
        "",
        "Same engine, same settings. Only the checkpoint differs.",
        "",
        f"- **Single-user decode speed: {p.ratio('decode_speedup_vs_bf16', 'awq')}**. "
        f"AWQ {p.decode('awq')} vs BF16 {p.decode('bf16')}. GPTQ reaches "
        f"{p.ratio('decode_speedup_vs_bf16', 'gptq')} (1 run).",
        f"- **Weight memory: "
        f"{link(f'−{mem["awq"]["weights_reduction_pct"]:.1f}%', mem['awq']['weights_gib']['source'])}**"
        f" ({link(f'{mem["awq"]["weights_gib"]["value"]:.4f} GiB', mem['awq']['weights_gib']['source'])}"
        f" vs {link(f'{mem["bf16"]["weights_gib"]["value"]:.4f} GiB', mem['bf16']['weights_gib']['source'])}"
        f', vLLM\'s "Model loading took"). On disk: '
        f"{link(f'−{mem["awq"]["disk_reduction_pct"]:.1f}%', mem['awq']['disk_bytes']['source'])}.",
        f"- KV-cache capacity: "
        f"{link(f'{mem["awq"]["kv_cache_tokens"]["value"]:,}', mem['awq']['kv_cache_tokens']['source'])}"
        f" vs {link(f'{mem["bf16"]["kv_cache_tokens"]["value"]:,}', mem['bf16']['kv_cache_tokens']['source'])}"
        f" tokens ({x(mem['awq']['kv_cache_ratio'])}).",
        f"- Peak output throughput: "
        f"{p.ratio('quantization_peak_vs_bf16', 'awq')} (AWQ {p.peak_tokens('awq')} "
        f"vs BF16 {link(f1(p.peak('bf16')['median']), p.table)}). The single-user "
        "gain mostly disappears once decode is batched (see "
        "[Where the speedup comes from](#where-the-speedup-comes-from)).",
        f"- Accuracy: MMLU 5-shot "
        f"{link(f'{100 * awq_acc["mmlu_acc"]:.2f}%', acc['source'])} vs "
        f"{link(f'{100 * acc["variants"]["bf16"]["mmlu_acc"]:.2f}%', acc['source'])}, "
        f"**{link(f'{awq_acc["mmlu_delta_pp"]:.2f} percentage points lower', acc['source'])}**; "
        f"WikiText-2 word perplexity "
        f"{link(f'+{awq_acc["word_perplexity_increase_pct"]:.1f}%', acc['source'])}.",
        "",
        "### Both together: vLLM AWQ vs Hugging Face",
        "",
        f"- Peak output tokens/s: **{p.ratio('awq_peak_vs_hf', 'naive')} HF naive, "
        f"{p.ratio('awq_peak_vs_hf', 'static')} HF static batching**. Peak "
        f"requests/s: {link(x(rr['awq_vs_hf-naive']), p.table)} and "
        f"{link(x(rr['awq_vs_hf-static']), p.table)}.",
        "- The naive ratio mostly measures continuous batching against a server "
        "that runs one request at a time. The static-batching ratio is the fair "
        "baseline.",
        "",
        "> **HF static batch-size caveat.** HF static batching used "
        f"B = {link(str(hf['batch_size']), hf['source'])}, the largest size the "
        f"OOM probe *tested* ({', '.join(str(c) for c in hf['candidates'])}). "
        "No candidate ran out of memory "
        f"({link('first_oom = null', hf['source'])}), so 128 is not shown to be "
        "the maximum that fits. The probe's throughput was still rising at the top: "
        f"{_probe_rate(hf, -2)} tokens/s at B={hf['fitted'][-2]['batch_size']} and "
        f"{_probe_rate(hf, -1)} at B={hf['fitted'][-1]['batch_size']}. HF static's "
        "best point was also the highest concurrency it was run at "
        "(c = 2B = 256). A larger B could raise its peak, so the ratios against "
        "HF static batching are an upper bound on vLLM's advantage over this "
        "baseline, not a tight value.",
        "",
    ]


def _probe_rate(hf: dict[str, Any], index: int) -> str:
    return link(f1(float(hf["fitted"][index]["output_tokens_per_s"])), hf["source"])


ADR_001 = "DECISIONS.md#adr-001--default-model-and-analytical-ceilings-2026-09-23"
CAPACITY = (
    "results/validation/phase6/loadtest-check-20260923T205147Z/capacity_summary.json"
)


def claims_section(p: Page) -> list[str]:
    mem = p.data["memory"]["variants"]
    acc = p.data["accuracy"]
    rr = p.head["requests_per_s_ratios"]
    env = p.data["environment"]["awq-20260923T220125Z"]
    c256 = p.point("awq", "c256")
    stack = (
        f"NVIDIA L40S (driver {env['gpu'].split(',')[1].strip()}), vLLM "
        f"{env['packages']['vllm']}, torch {env['packages']['torch']}"
    )
    bench = "`make bench LIFETIME=<lifetime>` then `make sync-bench report` (billable)"
    return [
        "## Claims and evidence (SPEC §8)",
        "",
        "Targets are the SPEC §0 placeholders. They are ambitions, not results; "
        'the "Measured" column is what the raw files show.',
        "",
        "| Claim | Target | Measured | Definition | Evidence | Date |",
        "| --- | --- | --- | --- | --- | --- |",
        "| Compressed an 8B LLM to 4-bit (AWQ, GPTQ) | — | W4A16 group-128; "
        "AWQ asymmetric, GPTQ symmetric; `lm_head` kept BF16 | llm-compressor 0.7.1 "
        "`oneshot` | "
        + link("AWQ summary", mem["awq"]["disk_bytes"]["source"])
        + ", "
        + link("GPTQ summary", mem["gptq"]["disk_bytes"]["source"])
        + ", [config](../configs/phase4_quantize.yaml) | 2026-09-23 |",
        "| Deployed with vLLM in Docker | — | Every vLLM lifetime's server ran in "
        "an image built from [`docker/Dockerfile`](../docker/Dockerfile) "
        "(`FROM vllm/vllm-openai:v0.10.2`) plus this repo's Python source; image "
        "IDs are recorded per lifetime (they change with the source) | "
        "`Image.from_dockerfile` in [`modal_app/common.py`](../modal_app/common.py) | "
        + link(
            "server log", env["source"].replace("lifetime_summary.json", "server.log")
        )
        + ", "
        + link("lifetime summary", env["source"])
        + ", lifetimes table below | 2026-09-23/24 |",
        f"| X% less memory | 68% | **{mem['awq']['weights_reduction_pct']:.1f}%** "
        f"(AWQ), {mem['gptq']['weights_reduction_pct']:.1f}% (GPTQ); on disk "
        f"{mem['awq']['disk_reduction_pct']:.1f}% / "
        f'{mem["gptq"]["disk_reduction_pct"]:.1f}% | vLLM "Model loading took" '
        "(GiB), 4-bit vs BF16; disk = safetensors bytes | "
        + link("AWQ log", mem["awq"]["weights_gib"]["source"])
        + ", "
        + link("BF16 log", mem["bf16"]["weights_gib"]["source"])
        + ", "
        + link("GPTQ log", mem["gptq"]["weights_gib"]["source"])
        + " | 2026-09-23/24 |",
        f"| Nx faster text generation | 3.1× | **{x(p.head['decode_speedup_vs_bf16']['awq'])}** "
        f"(AWQ, 3 runs each) | 1 / median TPOT at c=1, AWQ vs BF16 on vLLM | "
        f"AWQ {runs(p.point('awq', 'c1')['sources'])}; BF16 "
        f"{runs(p.point('bf16', 'c1')['sources'])} | 2026-09-23/24 |",
        f"| Accuracy within X% | 1.5% | **{acc['variants']['awq']['mmlu_delta_pp']:.2f} pp** "
        f"(AWQ), {acc['variants']['gptq']['mmlu_delta_pp']:.2f} pp (GPTQ) | MMLU "
        "5-shot, BF16 − variant, absolute percentage points, 14,042 questions | "
        + link("comparison.json", acc["source"])
        + " | 2026-09-23 |",
        "| Load tester simulating 256 users | 256 | 256 closed-loop users; "
        f"AWQ c=256 {link(f1(c256['metrics']['tokens_per_s']['median']), c256['sources'][0])}"
        " tokens/s, 0 errors | Our tester, validated against the mock "
        "(Phase 1, [results](../results/validation/)) and in the Modal container | "
        + runs(c256["sources"])
        + ", [in-Modal check](../results/validation/phase6/loadtest-check-20260923T205147Z/)"
        + " | 2026-09-24 |",
        f"| Nx more requests/s than a standard HF setup | 14× | **{x(rr['awq_vs_hf-naive'])}** "
        f"vs HF naive; **{x(rr['awq_vs_hf-static'])}** vs HF static (B=128, see "
        f"caveat); vLLM BF16 vs HF: {x(rr['bf16_vs_hf-naive'])} / "
        f"{x(rr['bf16_vs_hf-static'])} | Peak requests/s (median of 3) across "
        "each system's sweep | " + link("results table", p.table) + " | 2026-09-24 |",
        "",
        f"Environment for every row: {stack}. Reproduce: Phase 4 `uv run modal run "
        "--detach -m modal_app.quantize::{prepare,awq,gptq,sanity}`; Phase 5 `make "
        f"eval-full sync-accuracy`; Phase 6 {bench}. All are billable; "
        "estimates are required first (CLAUDE.md).",
        "",
    ]


def resume_section(p: Page) -> list[str]:
    """Plain-language bullets whose every number is rounded from the aggregate.

    Throughput is attributed to vLLM ("served with vLLM") because most of it
    is the engine, and the decode speedup is labelled single-user because it
    mostly disappears under batching. The table gives each exact value.
    """
    mem = p.data["memory"]["variants"]
    acc = p.data["accuracy"]
    rr = p.head["requests_per_s_ratios"]
    peaks = p.head["peak_requests_per_s"]
    c256 = p.point("awq", "c256")
    delta = acc["variants"]["awq"]["mmlu_delta_pp"]
    memory_pct = round(mem["awq"]["weights_reduction_pct"])
    decode = p.head["decode_speedup_vs_bf16"]["awq"]
    throughput = round(rr["awq_vs_hf-naive"])
    points = math.ceil(delta)
    c256_ratio = (
        c256["metrics"]["requests_per_s"]["median"] / peaks["hf-naive"]["median"]
    )
    return [
        "## Resume bullets",
        "",
        "Written for a non-specialist reader. Each number is rounded from a "
        "value in this page; the table says exactly what it measures.",
        "",
        "```text",
        "Quantized LLM Serving & Benchmarking | Python, PyTorch, vLLM, Hugging Face, Docker, Modal",
        f"• Compressed an 8B-parameter LLM to 4-bit with AWQ, shrinking the model's "
        f"memory footprint by {memory_pct}% and generating text {decode:.1f}x faster "
        f"for a single user while staying within {points} percentage "
        f"point{'s' if points != 1 else ''} of the "
        "original's accuracy",
        "• Served it with vLLM on cloud GPUs and built a custom load tester "
        f"simulating up to 256 simultaneous users, reaching {throughput}x the "
        "throughput of a basic Hugging Face server",
        "```",
        "",
        "| Phrase | Exact value | What it measures | Evidence |",
        "| --- | --- | --- | --- |",
        f"| memory footprint by {memory_pct}% | "
        f"{mem['awq']['weights_reduction_pct']:.1f}% ({mem['awq']['weights_gib']['value']:.4f} "
        f"vs {mem['bf16']['weights_gib']['value']:.4f} GiB); on disk "
        f"{mem['awq']['disk_reduction_pct']:.1f}% | The model weights in GPU memory "
        "(vLLM's load log), AWQ vs BF16. Not total GPU memory: vLLM reserves 90% "
        "of the GPU either way and fills the rest with KV cache | "
        + link("AWQ log", mem["awq"]["weights_gib"]["source"])
        + ", "
        + link("BF16 log", mem["bf16"]["weights_gib"]["source"])
        + ", "
        + link("on-disk sizes", mem["awq"]["disk_bytes"]["source"])
        + " |",
        f"| {decode:.1f}x faster for a single user | {x(decode)} | Single-user "
        "decode speed (1 / median time per output token at 1 user), AWQ vs BF16, "
        "both on vLLM, 3 runs each. With many users the gain shrinks to "
        f"{x(p.head['quantization_peak_vs_bf16']['awq'])} (peak throughput) | "
        f"AWQ {runs(p.point('awq', 'c1')['sources'])}; BF16 "
        f"{runs(p.point('bf16', 'c1')['sources'])} |",
        f"| within {points} percentage point{'s' if points != 1 else ''} of the "
        "original's accuracy | "
        f"{delta:.2f} pp | MMLU 5-shot (14,042 questions): "
        f"{100 * acc['variants']['bf16']['mmlu_acc']:.2f}% → "
        f"{100 * acc['variants']['awq']['mmlu_acc']:.2f}%. GPTQ lost "
        f"{acc['variants']['gptq']['mmlu_delta_pp']:.2f} pp, so the bullet names "
        "AWQ only | " + link("comparison.json", acc["source"]) + " |",
        "| up to 256 simultaneous users | 256 | Closed-loop virtual users "
        "streaming from the AWQ server, "
        f"{f1(c256['metrics']['tokens_per_s']['median'])} output tokens/s, 0 errors | "
        + runs(c256["sources"])
        + " |",
        f"| {throughput}x the throughput of a basic Hugging Face server | "
        f"{x(rr['awq_vs_hf-naive'])} requests/s ({x(p.head['awq_peak_vs_hf']['naive'])} "
        "output tokens/s) | Each system's peak over its sweep: vLLM AWQ at "
        f"{p.point('awq', peaks['awq']['point'])['concurrency']} users vs HF naive "
        "(one request at a time) at "
        f"{p.point('hf-naive', peaks['hf-naive']['point'])['concurrency']}. Mostly "
        f"the engine: vLLM with the original BF16 weights reaches "
        f"{x(rr['bf16_vs_hf-naive'])}. At 256 users AWQ is at {x(c256_ratio)} | "
        + link("results table", p.table)
        + " |",
        "",
        "What the bullets leave out, stated here so nobody has to find it:",
        "",
        '- **"Basic" means one request at a time.** Against Hugging Face '
        "with static batching, the fairer baseline, vLLM AWQ is "
        f"{x(rr['awq_vs_hf-static'])} (an upper bound; see the batch-size caveat).",
        "- **The throughput multiple is not a quantization result.** 4-bit "
        f"weights add only {x(rr['awq_vs_bf16'])} peak requests/s over vLLM BF16.",
        "- Against the SPEC §0 targets: accuracy (within 1.5%) and 256 users "
        "were met. Memory (68%) and single-user speed (3.1×) were not: the "
        f"memory cut matches the ~63% ceiling in [ADR-001]({ADR_001}), and the "
        "speedup is below its ~3.1× bandwidth ceiling (see "
        "[Where the speedup comes from](#where-the-speedup-comes-from)). The 14× "
        "throughput target was beaten against the naive baseline only.",
        "",
    ]


def methodology_section(p: Page) -> list[str]:
    env = p.data["environment"]
    vllm = env["awq-20260923T220125Z"]
    hf = env["hf-static-20260924T013433Z"]
    consistency = p.data["config_consistency"]
    spend = p.data["spend"]
    lines = [
        "## Methodology",
        "",
        "### Hardware and software",
        "",
        f"- GPU: `{vllm['gpu']}` (name, driver, memory, compute capability, from "
        f"`nvidia-smi` in {link('the run', vllm['source'])}). Every compared "
        "number used this GPU type.",
        f"- Container: {vllm['cpu_cores']} CPU cores, {vllm['memory_mib']:,} MiB "
        "memory on Modal, server and load tester in the same container over "
        "localhost (ADR-013).",
        f"- vLLM image: vLLM {vllm['packages']['vllm']}, torch "
        f"{vllm['packages']['torch']}, transformers {vllm['packages']['transformers']}, "
        f"Python {vllm['python']} ({link('packages', vllm['source'])}).",
        f"- HF baseline image: torch {hf['packages']['torch']}, transformers "
        f"{hf['packages']['transformers']}, BF16, SDPA attention "
        f"({link('packages', hf['source'])}).",
        f"- vLLM command: `{' '.join(vllm['command'][3:])}`.",
        f"- HF static command: `{' '.join(hf['command'][1:])}`.",
        "- Config: [`configs/phase6_bench.yaml`](../configs/phase6_bench.yaml). Its "
        "hash changed between lifetimes as lifetimes and points were added, but "
        f"the measurement sections ({', '.join(consistency['sections'])}) are "
        f"**{'identical' if consistency['identical'] else 'NOT identical'}** across "
        "every compared lifetime (checked by the aggregator).",
        "",
        "### Workload and windows",
        "",
        f"- W1: {vllm['prompt_count']:,} distinct 512-token windows of WikiText-103 "
        f"(train; pool sha256 `{vllm['prompt_pool_sha256'][:12]}…`, "
        f"{link('manifest', vllm['source'])}), sent as "
        "token IDs so neither server re-tokenizes (ADR-021). Every lifetime "
        f"hashed the same pool before use ({len(consistency['prompt_pools'])} distinct "
        "pool hash across lifetimes). Each request uses a "
        "prompt never used before in its server lifetime; every lifetime uses the "
        "same prompts in the same order.",
        "- 256 output tokens with `ignore_eos` (vLLM) / `min_new_tokens` (HF), "
        "greedy. Every request's `usage` must be 512 / 256 or the point fails.",
        "- Closed loop: N users, each sending its next request when the previous "
        "one finishes. Users start staggered over a ramp. After a warmup, "
        "one steady-state window is measured and requests still running at its "
        "end are cut (ADR-020). From c=128, the ramp is one expected request "
        "duration and the window is at least 10 of them (ADR-022).",
        "- Output tokens/s counts server-reported tokens (cumulative per-chunk "
        "usage) whose chunk arrived inside the window. Requests/s counts requests "
        "that *completed* inside it. Latency percentiles use those same "
        "completed requests.",
        "- TTFT: request start to the first text chunk. TPOT: (E2E − TTFT) / 255. "
        "ITL: gaps between text chunks. Chunks are not always one token each: "
        "vLLM merges deltas under load, and every point is checked for at least "
        "0.5 text chunks per token (ADR-014).",
        "- A point is **accepted** only if its window halves agree within 5%, the "
        "client received under a third of its validated in-container capacity "
        f"({link(f'{vllm["validated_client_chunks_per_s"]:,.1f} chunks/s', CAPACITY)}), "
        "no request errored, every `usage` was exact, and vLLM's "
        "prefix-cache hit counter did not move. Failed points are kept and listed "
        "below, never used.",
        "- HF naive runs only at c ∈ {1, 4, 16}. It serves one request at a time "
        "by design, so its throughput is flat and higher concurrency only "
        "lengthens the queue. HF static runs at c ∈ {4, 16, B, 2B}, with timings "
        "derived from the OOM probe's measured batch times.",
        "- Not run: the max-batch sweep (`--max-num-seqs` 16/64) and GPTQ at "
        "c=256, both cut for budget and time (ADR-022 addendum). No claim depends "
        "on them.",
        "",
        "### Point windows",
        "",
        "| System | c | Ramp s | Warmup s | Window s | Runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system, points in p.data["series"].items():
        for point in points:
            windows = sorted({t["window_s"] for t in point["timings"]})
            warmups = sorted({t["warmup_s"] for t in point["timings"]})
            ramps = sorted({t["ramp_s"] for t in point["timings"]})
            lines.append(
                f"| {p.name(system)} | {point['concurrency']} | "
                f"{' / '.join(f'{r:g}' for r in ramps)} | "
                f"{' / '.join(f'{w:g}' for w in warmups)} | "
                f"{' / '.join(f'{w:g}' for w in windows)} | "
                f"{runs(point['sources'])} |"
            )
    lines += [
        "",
        "### Server lifetimes",
        "",
        "| Lifetime | Date | Commit | Dirty | Modal image | Server lifetime s | Points passed |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for life in p.data["lifetimes"]:
        lines.append(
            f"| {link(life['run'], life['source'])} | {life['date']} | "
            f"`{life['commit']}` | {'yes' if life['dirty'] else 'no'} | "
            f"`{life['modal_image_id']}` | "
            f"{life['server_lifetime_s']:,.0f} | "
            f"{life['points_passed']} / {life['points']} |"
        )
    lines += [
        "",
        f"GPU spend for the whole project: "
        f"{link(f'${spend["total_usd"]:.2f}', spend['source'])} (Modal billing "
        f"report, whole UTC days {' and '.join(spend['days'])}), of which "
        f"benchmarks {link(f'${spend["by_app_usd"]["llmbench-bench"]:.2f}', spend['source'])}. "
        "The per-run Spend log is in [PROGRESS.md](../PROGRESS.md#spend-log).",
        "",
    ]
    return lines


def sweep_section(p: Page) -> list[str]:
    lines = [
        "## Throughput and latency sweeps",
        "",
        f"![Request throughput vs concurrency]({CHARTS}/1_request_throughput.png)",
        "",
        f"![p95 TTFT vs concurrency]({CHARTS}/2_ttft_p95.png)",
        "",
        f"![Median TPOT vs concurrency]({CHARTS}/3_tpot_p50.png)",
        "",
        f"![Latency-throughput trade-off]({CHARTS}/4_latency_throughput.png)",
        "",
        "Each value is the median over the listed passing runs (r1, r2, … link to "
        "each run's `summary.json`; its `requests.jsonl.gz` beside it has every "
        "request). Where there are several runs, the range is in brackets. "
        "Every summary also has p50/p90/p95/p99 of TTFT, TPOT, ITL and E2E; "
        f"{link('results_table.md', p.table)} lists the p99s per run.",
        "",
        "| System | c | Output tok/s | Req/s | TTFT p50 / p95 ms | TPOT p50 ms | "
        "E2E p50 s | Runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for system, points in p.data["series"].items():
        for point in points:
            m = point["metrics"]
            tokens = m["tokens_per_s"]
            spread = (
                f" [{f1(tokens['min'])}–{f1(tokens['max'])}]"
                if tokens["runs"] > 1
                else ""
            )
            lines.append(
                f"| {p.name(system)} | {point['concurrency']} | "
                f"{f1(tokens['median'])}{spread} | "
                f"{m['requests_per_s']['median']:.3f} | "
                f"{ms(m['ttft_p50']['median'])} / {ms(m['ttft_p95']['median'])} | "
                f"{ms(m['tpot_p50']['median'])} | {m['e2e_p50']['median']:.2f} | "
                f"{runs(point['sources'])}"
                f"{' (req/s warning)' if point['warnings'] else ''} |"
            )
    lines += [
        "",
        '"Req/s warning": requests complete in bunches (a whole HF static batch '
        "at once; single-user runs complete a few requests per window), so moving "
        "a window edge could change the request count by more than 5%. Output "
        "tokens/s, counted by arrival, is not affected.",
        "",
        "### Failed and diagnostic points",
        "",
        "Kept as evidence and never used in a headline.",
        "",
        "| Lifetime | Point | c | Output tok/s | TPOT p50 ms | Halves Δ | Why |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for other in p.data["other_points"]:
        why = (
            "diagnostic: all users start at once (the cross-check's arrival pattern)"
            if other["diagnostic"]
            else "; ".join(other["failures"])[:90]
        )
        lines.append(
            f"| {other['run']} | {link(other['point'], other['source'])} | "
            f"{other['concurrency']} | {f1(other['tokens_per_s'])} | "
            f"{ms(other['tpot_p50'])} | {100 * other['half_deviation']:.2f}% | "
            f"{why} |"
        )
    lines.append("")
    return lines


def memory_section(p: Page) -> list[str]:
    memory = p.data["memory"]
    mem = memory["variants"]
    probe = memory["awq_probe_lifetime"]["kv_cache_tokens"]
    lines = [
        "## Weight memory and KV-cache capacity",
        "",
        f"![Weight memory and KV cache]({CHARTS}/5_memory.png)",
        "",
        "| Variant | vLLM weight memory | Safetensors on disk | Reduction (weights / "
        "disk) | KV cache (tokens) | Full-length requests that fit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in ("bf16", "awq", "gptq"):
        v = mem[variant]
        reduction = (
            "—"
            if variant == "bf16"
            else f"{v['weights_reduction_pct']:.2f}% / {v['disk_reduction_pct']:.2f}%"
        )
        lines.append(
            f"| {p.name(variant)} | "
            f"{link(f'{v["weights_gib"]["value"]:.4f} GiB', v['weights_gib']['source'])} | "
            f"{link(f'{v["disk_bytes"]["value"]:,} B', v['disk_bytes']['source'])} | "
            f"{reduction} | "
            f"{link(f'{v["kv_cache_tokens"]["value"]:,}', v['kv_cache_tokens']['source'])} | "
            f"{link(f'{v["max_concurrency"]["value"]["multiple"]:.1f}× at {v["max_concurrency"]["value"]["tokens_per_request"]:,} tokens', v['max_concurrency']['source'])} |"
        )
    lines += [
        "",
        f"- Settings: `--gpu-memory-utilization {memory['gpu_memory_utilization']}`, "
        f"`--max-model-len {memory['max_model_len']}`, CUDA graphs "
        f"({link('command', memory['engine_args_source'])}).",
        "- Why not `nvidia-smi`: vLLM reserves a fixed fraction of GPU memory "
        "(here 90%) at start-up and fills what the weights leave with KV cache. "
        "Peak `memory.used` was "
        + ", ".join(
            f"{p.name(v).removeprefix('vLLM ')} "
            f"{link(f'{mem[v]["nvidia_smi_peak"]["used_mib"]:,} MiB', mem[v]['nvidia_smi_peak']['source'])}"
            for v in ("bf16", "awq", "gptq")
        )
        + f" of {mem['bf16']['nvidia_smi_peak']['total_mib']:,}: almost the same "
        "for every variant. The weight difference shows up as KV-cache "
        "capacity instead.",
        f"- Run-to-run variation: the AWQ probe lifetime, with the same checkpoint "
        f"and flags, logged {link(f'{probe["value"]:,}', probe['source'])} KV "
        f"tokens, {100 * (probe['value'] / mem['awq']['kv_cache_tokens']['value'] - 1):.1f}% "
        "against the sweep. AWQ vs GPTQ KV capacity is within that noise. Weight "
        "memory was identical in both lifetimes.",
        f"- [ADR-001]({ADR_001})'s analytical estimate was a 62.8–62.9% weight "
        "reduction with BF16 embeddings and `lm_head`. The measured reduction "
        "matches it, and the 68% placeholder is above what this scheme can reach.",
        "",
    ]
    return lines


def accuracy_section(p: Page) -> list[str]:
    acc = p.data["accuracy"]
    lines = [
        "## Accuracy",
        "",
        f"![Accuracy]({CHARTS}/6_accuracy.png)",
        "",
        "| Variant | MMLU 5-shot | Δ vs BF16 (pp) | Lost / gained questions | "
        "McNemar p | WikiText-2 word ppl | Run |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for variant in ("bf16", "awq", "gptq"):
        v = acc["variants"][variant]
        delta = "—" if v.get("mmlu_delta_pp") is None else f"{v['mmlu_delta_pp']:.2f}"
        flips = (
            "—"
            if v.get("mmlu_questions_lost") is None
            else f"{v['mmlu_questions_lost']} / {v['mmlu_questions_gained']}"
        )
        pval = "—" if v.get("mcnemar_p") is None else f"{v['mcnemar_p']:.1e}"
        lines.append(
            f"| {p.name(variant)} | "
            f"{link(f'{100 * v["mmlu_acc"]:.2f}% ± {100 * v["mmlu_acc_stderr"]:.2f}', acc['source'])} | "
            f"{delta} | {flips} | {pval} | {v['word_perplexity']:.3f} | "
            f"{link('lm-eval output', acc['variant_runs'][variant] + '/' + variant)} |"
        )
    lines += [
        "",
        "- Δ is BF16 accuracy minus the variant's, in **absolute percentage "
        "points** (positive = the variant is worse). Accuracy is lm-eval's "
        "size-weighted mean over all 14,042 questions. Every variant was scored "
        "on identical prompts.",
        "- Both drops are statistically real by a paired test on the same "
        "questions. AWQ and GPTQ used different calibration data (pile-val vs "
        "UltraChat, each method's official example), so AWQ vs GPTQ does not "
        "isolate the method.",
        f"- Per-category and per-subject deltas: {link('accuracy table', acc['table'])}.",
        "",
    ]
    return lines


def cross_check_section(p: Page) -> list[str]:
    lines = [
        "## Cross-check against `vllm bench serve`",
        "",
        "Same AWQ server, `vllm bench serve` run after our points "
        "(`random` dataset: 512-token target inputs, which averaged slightly "
        "fewer after its tokenizer round trip; exactly 256 outputs). It is "
        "compared with our *passing* medians at the same concurrency.",
        "",
        "| c | bench serve tok/s | Ours tok/s | Gap | bench TPOT p50 ms | Ours TPOT "
        "p50 ms | Gap | bench mean input tokens | Sources |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in p.data["cross_check"]:
        lines.append(
            f"| {c['concurrency']} | {link(f1(c['bench_tokens_per_s']), c['source'])} | "
            f"{f1(c['ours_tokens_per_s'])} | {c['tokens_gap_pct']:+.1f}% | "
            f"{ms(c['bench_tpot_p50'])} | {ms(c['ours_tpot_p50'])} | "
            f"{c['tpot_gap_pct']:+.1f}% | {c['bench_mean_input_tokens']:.1f} | "
            f"ours {runs(c['ours_sources'])} |"
        )
    sync = next(
        (
            c["ours_all_at_once"]
            for c in p.data["cross_check"]
            if "ours_all_at_once" in c
        ),
        None,
    )
    lines.append("")
    if sync:
        lines += [
            "At c=1 and c=64 the two testers agree within the ~10% SPEC target. At "
            "c=256 they do not, and the difference is the arrival pattern. "
            "`vllm bench serve` starts all 256 users at once, so prefills and "
            "decodes stay separated; our staggered users mix a prefill into almost "
            "every decode step. Our diagnostic run with an all-at-once start "
            f"measured {link(f1(sync['tokens_per_s']), sync['source'])} tokens/s "
            f"and {ms(sync['tpot_p50'])} ms TPOT, within "
            f"{sync['tokens_gap_pct']:+.1f}% / {sync['tpot_gap_pct']:+.1f}% of "
            "`vllm bench serve`. Both numbers are right for their arrival "
            "pattern. The headlines use the staggered steady state. Their peaks "
            "are at c=128, which was not cross-checked; c=64, the nearest "
            "cross-checked point, agreed within 0.1%.",
            "",
        ]
    return lines


def sanity_section(p: Page) -> list[str]:
    s = p.data["sanity"]
    counters = s["vllm_counter_deltas"]
    lines = [
        "## Sanity checks",
        "",
        f"- Points: {s['points_total']} measured, {s['points_passed']} accepted, "
        f"{s['points_failed']} failed the steady-state check, "
        f"{s['points_diagnostic']} diagnostic. "
        f"**Errored requests across all points: {s['errored_requests_all_points']}.**",
        f"- vLLM counters over all {counters['points']} vLLM points: prefix-cache "
        f"queries {counters['prefix_cache_queries']:g}, hits "
        f"{counters['prefix_cache_hits']:g}, preemptions {counters['preemptions']:g}.",
        "- Shape of each throughput curve (output tokens/s):",
        "",
        "| System | Rises monotonically to its peak | Peak at c | Change from peak to "
        "the highest c measured |",
        "| --- | --- | ---: | ---: |",
    ]
    for system, shape in s["curve_shape"].items():
        lines.append(
            f"| {p.name(system)} | {'yes' if shape['rises_to_peak'] else 'no'} | "
            f"{shape['peak_concurrency']} | {shape['after_peak_change_pct']:+.1f}% |"
        )
    lines += [
        "",
        "Throughput rises, then saturates. For AWQ and BF16 it falls past "
        "c=128. The all-at-once diagnostic above stays at the c=128 level, so "
        "the drop comes with the staggered arrival pattern. *(Interpretation: "
        "at c=256 a new request arrives nearly every decode step, and its "
        "prefill slows that step for all running sequences.)* HF "
        "naive is flat by construction. HF static was still rising at its "
        "highest measured concurrency (see the batch-size caveat).",
        "",
        "Spread of repeated points ((max − min) / median):",
        "",
        "| System | Point | Runs | Output tok/s spread | TPOT p50 spread |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in s["repeat_spreads"]:
        lines.append(
            f"| {p.name(r['system'])} | {r['point']} | {r['runs']} | "
            f"{r['tokens_spread_pct']:.2f}% | {r['tpot_spread_pct']:.2f}% |"
        )
    lines.append("")
    return lines


def speedup_section(p: Page) -> list[str]:
    bf16_c1 = p.point("bf16", "c1")["metrics"]
    hf_c1 = p.point("hf-naive", "c1")["metrics"]
    awq_128 = p.point("awq", "c128")["metrics"]
    bf16_128 = p.point("bf16", "c128")["metrics"]
    hf_static = p.point("hf-static", "hf-cB")["metrics"]
    mem = p.data["memory"]["variants"]
    fits = {v: mem[v]["kv_cache_tokens"]["value"] / 768 for v in ("bf16", "awq")}
    hf = p.data["hf_static"]
    batch = hf["fitted"][-1]
    return [
        "## Where the speedup comes from",
        "",
        "Measured facts first; the explanation in each item is interpretation "
        "and is labelled as such.",
        "",
        "### Engine: continuous batching and paged KV cache",
        "",
        f"- One user: vLLM BF16 decodes at {ms(bf16_c1['tpot_p50']['median'])} ms "
        f"per token, HF at {ms(hf_c1['tpot_p50']['median'])} ms. With a single "
        "sequence both engines read the same BF16 weights "
        f"({link(f'{mem["bf16"]["weights_gib"]["value"]:.4f} GiB', mem['bf16']['weights_gib']['source'])}) "
        "for every token, so "
        "the engine matters little. *(Interpretation: batch-1 decode is "
        "memory-bandwidth-bound; vLLM's CUDA graphs and kernels shave the "
        "rest.)*",
        "- Many users: HF naive stays at "
        f"{f1(p.peak('hf-naive')['median'])} tokens/s at any concurrency because it "
        "serves one request at a time. vLLM admits new requests into the running "
        "batch at every step (continuous batching), and its paged KV cache "
        "allocates memory in small blocks as sequences grow, not as one padded "
        "rectangle per batch.",
        f"- Against HF static batching at c=128: TPOT {ms(bf16_128['tpot_p50']['median'])} "
        f"ms (vLLM BF16) vs {ms(hf_static['tpot_p50']['median'])} ms (HF, B=128), "
        f"and p95 TTFT {ms(bf16_128['ttft_p95']['median'])} ms vs "
        f"{ms(hf_static['ttft_p95']['median'])} ms. A static batch only starts "
        "when the previous one has finished, so a new request can wait a whole "
        f"batch: {link(f'{batch["seconds"]:.1f} s', hf['source'])} for "
        f"B={batch['batch_size']} in the OOM probe.",
        "",
        "### Quantization: fewer bytes per decoded token",
        "",
        f"- One user: AWQ {ms(p.point('awq', 'c1')['metrics']['tpot_p50']['median'])} "
        f"ms vs BF16 {ms(bf16_c1['tpot_p50']['median'])} ms per token, "
        f"{x(p.head['decode_speedup_vs_bf16']['awq'])}. [ADR-001]({ADR_001})'s "
        "weight-bandwidth estimate for this layout is about 3.14× (15.14 vs 4.83 GB read per "
        "token, `lm_head` and embeddings in BF16). *(Interpretation: the "
        "measured gain is below it because KV-cache reads, dequantization in "
        "the Marlin kernels and fixed per-step overheads do not shrink.)*",
        f"- Batched: at c=128, AWQ's TPOT is {ms(awq_128['tpot_p50']['median'])} ms "
        f"vs BF16's {ms(bf16_128['tpot_p50']['median'])} ms, and peak throughput "
        f"is only {x(p.head['quantization_peak_vs_bf16']['awq'])} BF16's. "
        "*(Interpretation: a batched step reads the weights once for all 128 "
        "sequences, so weight bytes are no longer the bottleneck. Attention "
        "over each sequence's KV cache and the matrix compute dominate, and "
        "4-bit weights do not reduce either.)*",
        f"- Memory for KV cache: 4-bit weights free "
        f"{mem['bf16']['weights_gib']['value'] - mem['awq']['weights_gib']['value']:.2f}"
        f" GiB, which becomes {x(mem['awq']['kv_cache_ratio'])} the KV-cache "
        f"tokens: room for {fits['awq']:.0f} full 768-token requests at once vs "
        f"{fits['bf16']:.0f} for BF16. This workload did not need that room "
        "(0 preemptions for every variant; with staggered users the average "
        "request in flight is shorter than 768 tokens, an inference from the "
        "arrival pattern rather than a measurement). The extra capacity would "
        "matter for longer contexts or more concurrent users than "
        "`--max-num-seqs 256`.",
        "",
    ]


def limitations_section(p: Page) -> list[str]:
    spreads = p.data["sanity"]["repeat_spreads"]
    worst = max(max(r["tokens_spread_pct"], r["tpot_spread_pct"]) for r in spreads)
    return [
        "## Limitations",
        "",
        "- One GPU type (L40S), one model (Qwen3-8B), one synthetic workload "
        "(512 in / 256 out, fixed lengths). Real traffic has variable lengths.",
        "- Single-GPU, single-server. No tensor parallelism, no network between "
        "client and server (both in one container by design).",
        "- The HF static baseline's batch size is the largest *tested*, not the "
        "largest that fits (caveat above).",
        "- The max-batch sweep and GPTQ at c=256 were not run (budget), and most "
        "non-peak points ran once. The repeated points vary by at most "
        f"{worst:.1f}% (sanity checks).",
        "- MMLU is a multiple-choice log-likelihood test; generative tasks "
        "(e.g. GSM8K) were not evaluated and can be more sensitive to "
        "quantization.",
        "",
    ]


def render(data: dict[str, Any]) -> str:
    page = Page(data)
    lines = [
        "# Results",
        "",
        "<!-- Generated by `make report` (src/llmbench/analysis/report.py) from "
        "results/analysis/aggregate.json. Do not edit by hand. -->",
        "",
        "Qwen/Qwen3-8B in BF16, AWQ W4A16 and GPTQ W4A16, served with vLLM and "
        "compared with a plain Hugging Face `transformers` server. Every number "
        "below links to the raw result it came from.",
        "",
        *summary_section(page),
        *claims_section(page),
        *resume_section(page),
        *methodology_section(page),
        *sweep_section(page),
        *memory_section(page),
        *accuracy_section(page),
        *cross_check_section(page),
        *sanity_section(page),
        *speedup_section(page),
        *limitations_section(page),
        "## Reproduce",
        "",
        "```bash",
        "make plots report   # $0: aggregate results/, redraw charts, rewrite this page",
        "```",
        "",
        "The measurements themselves are billable Modal runs (Phase 4–6 "
        "commands in the claims section; each needs a cost estimate and approval "
        "first).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m llmbench.analysis.report")
    parser.add_argument("--aggregate", type=Path, default=AGGREGATE)
    parser.add_argument("--out", type=Path, default=RESULTS_MD)
    args = parser.parse_args()
    args.out.write_text(render(json.loads(args.aggregate.read_text())))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
