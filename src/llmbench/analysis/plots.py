"""Phase 7 charts from `results/analysis/aggregate.json` ($0).

    uv run python -m llmbench.analysis.plots   # -> results/analysis/*.png

Every plotted value comes from the aggregate file, which links it to its raw
result; RESULTS.md repeats each chart's numbers as a table. Colors follow the
system, not its rank, and every series also has its own marker and a direct
label, so identity never rests on color alone. Two measures with different
units get two panels, never two y-axes.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from llmbench.analysis.aggregate import OUT as AGGREGATE  # noqa: E402

CHART_DIR = Path("results/analysis")

# Categorical slots 1-5 of the validated reference palette, in fixed order
# (validate_palette.js: all adjacent CVD/normal-vision checks pass; three
# slots are below 3:1 on the surface, hence direct labels and tables).
COLORS = {
    "awq": "#2a78d6",
    "gptq": "#eb6834",
    "bf16": "#1baf7a",
    "hf-static": "#eda100",
    "hf-naive": "#e87ba4",
}
MARKERS = {"awq": "o", "gptq": "s", "bf16": "^", "hf-static": "D", "hf-naive": "v"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "text.color": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 11,
        "legend.frameon": False,
    }
)


def subtitle(data: dict[str, Any]) -> str:
    env = data["environment"]["awq-20260923T220125Z"]
    gpu = env["gpu"].split(",")[0]
    return (
        f"{gpu} · Qwen/Qwen3-8B · vLLM {env['packages']['vllm']} · "
        "W1: 512 input / 256 output tokens, greedy,\n"
        "unique prompts, prefix cache off · medians of passing runs"
    )


def _figure(
    data: dict[str, Any], title: str, panels: int = 1
) -> tuple[Figure, list[Axes]]:
    fig, axes = plt.subplots(1, panels, figsize=(8.4, 4.8), squeeze=False)
    fig.suptitle(title, x=0.012, y=0.975, ha="left", fontsize=12, weight="bold")
    fig.text(
        0.012, 0.935, subtitle(data), ha="left", va="top", fontsize=8, color=TEXT_2
    )
    return fig, list(axes[0])


def _save(fig: Figure, out_dir: Path, name: str, note: str | None = None) -> Path:
    bottom = 0.0
    if note:
        wrapped = textwrap.fill(note, 130)
        fig.text(
            0.012, 0.012, wrapped, ha="left", va="bottom", fontsize=7.5, color=TEXT_2
        )
        bottom = 0.035 + 0.03 * wrapped.count("\n")
    fig.tight_layout(rect=(0, bottom, 1, 0.87))
    path = out_dir / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _legend(ax: Axes) -> None:
    """Five series: a legend beside the plot (direct labels would collide)."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)


def _concurrency_axis(ax: Axes, ticks: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Concurrent users (closed loop)")


def _sweep_chart(
    data: dict[str, Any],
    metric: str,
    *,
    scale: float,
    title: str,
    ylabel: str,
    log_y: bool,
    out_dir: Path,
    name: str,
    note: str,
) -> Path:
    fig, (ax,) = _figure(data, title)
    ticks: set[int] = set()
    for system, points in data["series"].items():
        x = [p["concurrency"] for p in points]
        stats = [p["metrics"][metric] for p in points]
        y = [s["median"] * scale for s in stats]
        low = [(s["median"] - s["min"]) * scale for s in stats]
        high = [(s["max"] - s["median"]) * scale for s in stats]
        ticks.update(x)
        ax.errorbar(
            x,
            y,
            yerr=[low, high],
            color=COLORS[system],
            marker=MARKERS[system],
            markersize=6,
            linewidth=2,
            capsize=2.5,
            elinewidth=1,
            label=data["systems"][system],
            zorder=3,
        )
    _concurrency_axis(ax, sorted(ticks))
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    _legend(ax)
    return _save(fig, out_dir, name, note)


def throughput_chart(data: dict[str, Any], out_dir: Path) -> Path:
    return _sweep_chart(
        data,
        "requests_per_s",
        scale=1,
        title="Request throughput vs concurrency",
        ylabel="Completed requests / s (steady-state window)",
        log_y=True,
        out_dir=out_dir,
        name="1_request_throughput.png",
        note="Median of passing runs; bars span min–max where a point was repeated. "
        "HF naive measured at c ∈ {1, 4, 16} only (one request at a time by design).",
    )


def ttft_chart(data: dict[str, Any], out_dir: Path) -> Path:
    return _sweep_chart(
        data,
        "ttft_p95",
        scale=1000,
        title="p95 time to first token vs concurrency",
        ylabel="p95 TTFT (ms, log scale)",
        log_y=True,
        out_dir=out_dir,
        name="2_ttft_p95.png",
        note="Includes queueing: HF naive requests wait for every request ahead of "
        "them, HF static for a free batch.",
    )


def tpot_chart(data: dict[str, Any], out_dir: Path) -> Path:
    return _sweep_chart(
        data,
        "tpot_p50",
        scale=1000,
        title="Median time per output token vs concurrency",
        ylabel="Median TPOT (ms / token)",
        log_y=True,
        out_dir=out_dir,
        name="3_tpot_p50.png",
        note="TPOT = (E2E − TTFT) / (256 − 1) per request. 1 / TPOT at c = 1 is "
        "single-user decode speed.",
    )


# Concurrency labels on the trade-off chart, each with its offset (points).
# GPTQ overlaps AWQ almost exactly, so only AWQ is labelled.
PARETO_LABELS: dict[str, dict[int, tuple[int, int]]] = {
    "awq": {c: (7, -10) for c in (1, 4, 16, 64, 128, 256)},
    "bf16": {1: (4, -12), 128: (-30, 4), 256: (-30, 4)},
    "hf-static": {c: (6, -9) for c in (4, 16, 128, 256)},
}


def pareto_chart(data: dict[str, Any], out_dir: Path) -> Path:
    fig, (ax,) = _figure(data, "Latency–throughput trade-off")
    for system, points in data["series"].items():
        x = [p["metrics"]["tokens_per_s"]["median"] for p in points]
        y = [p["metrics"]["tpot_p50"]["median"] * 1000 for p in points]
        ax.plot(
            x,
            y,
            color=COLORS[system],
            marker=MARKERS[system],
            markersize=6,
            linewidth=2,
            label=data["systems"][system],
            zorder=3,
        )
        labelled = PARETO_LABELS.get(system, {})
        for xi, yi, point in zip(x, y, points, strict=True):
            if point["concurrency"] in labelled:
                ax.annotate(
                    f"c{point['concurrency']}",
                    (xi, yi),
                    xytext=labelled[point["concurrency"]],
                    textcoords="offset points",
                    fontsize=7,
                    color=TEXT_2,
                )
        if system == "hf-naive":  # its three points coincide: one label
            ax.annotate(
                "c1, c4, c16",
                (x[0], y[0]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                color=TEXT_2,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Output tokens / s, all users (log scale)")
    ax.set_ylabel("Median TPOT, ms / token per user (log scale)")
    _legend(ax)
    return _save(
        fig,
        out_dir,
        "4_latency_throughput.png",
        "Each marker is one concurrency level (labels: c = users). Down and to the "
        "right is better. The planned max-batch sweep was not run (ADR-022).",
    )


def memory_chart(data: dict[str, Any], out_dir: Path) -> Path:
    fig, (weights, kv) = _figure(data, "Weight memory and KV-cache capacity", panels=2)
    variants = [v for v in ("bf16", "awq", "gptq") if v in data["memory"]["variants"]]
    names = [data["systems"][v].removeprefix("vLLM ") for v in variants]
    mem = data["memory"]["variants"]
    gib = 1024**3
    loaded = [mem[v]["weights_gib"]["value"] for v in variants]
    disk = [mem[v]["disk_bytes"]["value"] / gib for v in variants]
    positions = range(len(variants))
    width = 0.38
    for offset, values, label, alpha in (
        (-width / 2 - 0.01, loaded, "vLLM 'Model loading took' (GiB)", 1.0),
        (width / 2 + 0.01, disk, "Safetensors on disk (GiB)", 0.45),
    ):
        bars = weights.bar(
            [p + offset for p in positions],
            values,
            width,
            color=[COLORS[v] for v in variants],
            alpha=alpha,
            label=label,
            zorder=3,
        )
        weights.bar_label(bars, fmt="%.2f", fontsize=7, color=TEXT_2, padding=2)
    weights.set_xticks(list(positions), names)
    weights.set_ylabel("GiB")
    weights.set_title("Weights (solid: loaded in vLLM; light: on disk)", fontsize=9)
    weights.grid(axis="x", visible=False)
    tokens = [mem[v]["kv_cache_tokens"]["value"] / 1000 for v in variants]
    bars = kv.bar(
        list(positions), tokens, 0.6, color=[COLORS[v] for v in variants], zorder=3
    )
    kv.bar_label(bars, fmt="%.1fk", fontsize=7, color=TEXT_2, padding=2)
    kv.set_xticks(list(positions), names)
    kv.set_ylabel("Thousand tokens")
    kv.set_title("GPU KV-cache capacity (vLLM log)", fontsize=9)
    kv.grid(axis="x", visible=False)
    return _save(
        fig,
        out_dir,
        "5_memory.png",
        f"gpu_memory_utilization {data['memory']['gpu_memory_utilization']}, "
        f"max_model_len {data['memory']['max_model_len']}, CUDA graphs. Memory freed "
        "by 4-bit weights becomes KV cache.",
    )


def accuracy_chart(data: dict[str, Any], out_dir: Path) -> Path:
    fig, (mmlu, ppl) = _figure(data, "Accuracy cost of 4-bit weights", panels=2)
    acc = data["accuracy"]["variants"]
    variants = [v for v in ("bf16", "awq", "gptq") if v in acc]
    names = [data["systems"][v].removeprefix("vLLM ") for v in variants]
    positions = list(range(len(variants)))
    colors = [COLORS[v] for v in variants]
    values = [100 * acc[v]["mmlu_acc"] for v in variants]
    errors = [100 * acc[v]["mmlu_acc_stderr"] for v in variants]
    # A point estimate with its interval, not a bar: the axis cannot start at
    # zero without hiding ~1 pp differences, and a truncated bar misleads.
    for position, value, error, color in zip(
        positions, values, errors, colors, strict=True
    ):
        mmlu.errorbar(
            [position],
            [value],
            yerr=[error],
            fmt="o",
            color=color,
            markersize=8,
            capsize=4,
            elinewidth=1.5,
            zorder=3,
        )
        mmlu.annotate(
            f"{value:.2f}%",
            (position, value),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TEXT_2,
        )
    mmlu.set_xlim(-0.6, len(variants) - 0.4)
    mmlu.set_xticks(positions, names)
    mmlu.set_ylabel("MMLU 5-shot accuracy (%)")
    mmlu.set_title("MMLU, 14,042 questions (± 1 stderr)", fontsize=9)
    mmlu.grid(axis="x", visible=False)
    ppl_values = [acc[v]["word_perplexity"] for v in variants]
    bars = ppl.bar(positions, ppl_values, 0.6, color=colors, zorder=3)
    ppl.bar_label(bars, fmt="%.2f", fontsize=7, color=TEXT_2, padding=2)
    ppl.set_xticks(positions, names)
    ppl.set_ylabel("WikiText-2 word perplexity (lower is better)")
    ppl.set_title("WikiText-2, 62 documents", fontsize=9)
    ppl.grid(axis="x", visible=False)
    fig.texts[1].set_text(
        "Qwen/Qwen3-8B · lm-evaluation-harness 0.4.11 (vLLM backend), identical"
        " prompts for every variant\nMMLU 5-shot (no chat template) and WikiText-2"
        " word perplexity, all questions and documents"
    )
    return _save(
        fig,
        out_dir,
        "6_accuracy.png",
        "MMLU vs BF16, absolute percentage points: "
        + ", ".join(
            f"{data['systems'][v].removeprefix('vLLM ')} "
            f"{acc[v]['mmlu_delta_pp']:.2f} pp lower"
            for v in variants
            if acc[v].get("mmlu_delta_pp") is not None
        )
        + ".",
    )


CHARTS = (
    throughput_chart,
    ttft_chart,
    tpot_chart,
    pareto_chart,
    memory_chart,
    accuracy_chart,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m llmbench.analysis.plots")
    parser.add_argument("--aggregate", type=Path, default=AGGREGATE)
    parser.add_argument("--out-dir", type=Path, default=CHART_DIR)
    args = parser.parse_args()
    data = json.loads(args.aggregate.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for chart in CHARTS:
        print(f"wrote {chart(data, args.out_dir)}")


if __name__ == "__main__":
    main()
