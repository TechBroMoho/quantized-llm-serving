"""Phase 6 results table from synced lifetime directories ($0).

    uv run python -m llmbench.perf_report results/perf/phase6 \
        --out results/perf/phase6/results_table.md

One row per measured point, then the headline numbers: medians of the
repeated points (the sweep's own run plus -r2 and -r3) and the SPEC §8
ratios. Engine gains (vLLM BF16 vs HF) are kept separate from quantization
gains (AWQ/GPTQ vs vLLM BF16). Only points that passed every check count.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


def load_points(root: Path) -> list[dict[str, Any]]:
    """Every point summary under `<root>/<lifetime>-<stamp>/<label>/`."""
    rows = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        lifetime_dir = summary_path.parent.parent
        match = re.match(r"(.+)-(\d{8}T\d{6}Z)$", lifetime_dir.name)
        if not match or match[1] in {"prepare", "loadtest-check"}:
            continue
        summary = json.loads(summary_path.read_text())
        cpu = summary.get("cpu_in_window") or {}
        rows.append(
            {
                # A follow-up lifetime adds runs to the same variant (ADR-022).
                "lifetime": match[1].removesuffix("-followup"),
                "diagnostic": bool(summary.get("diagnostic")),
                "run": lifetime_dir.name,
                "time": summary.get("window_start_monotonic_s", 0.0),
                "label": summary_path.parent.name,
                "base": re.sub(r"-r\d+$", "", summary_path.parent.name),
                "concurrency": summary["config"]["concurrency"],
                "passed": summary["passed"],
                "failures": summary["failures"],
                "warnings": summary.get("warnings", []),
                "tokens_per_s": summary["output_token_throughput_per_s"],
                "requests_per_s": summary["request_throughput_per_s"],
                "completed": summary["completed_in_window_requests"],
                "ttft_p50": summary["ttft_s"]["p50"],
                "ttft_p99": summary["ttft_s"]["p99"],
                "tpot_p50": summary["tpot_s"]["p50"],
                "tpot_p99": summary["tpot_s"]["p99"],
                "itl_p99": summary["itl_s"]["p99"],
                "e2e_p50": summary["e2e_s"]["p50"],
                "half_deviation": summary["half_window_token_deviation"],
                "headroom": summary.get("client_capacity_headroom"),
                "client_cores": summary.get("window_process_cpu_cores_average"),
                "server_cores": cpu.get("server_cores"),
            }
        )
    return rows


def median_of(rows: list[dict[str, Any]], lifetime: str, base: str, key: str) -> Any:
    values = [
        r[key]
        for r in rows
        if r["lifetime"] == lifetime
        and r["base"] == base
        and r["passed"]
        and not r["diagnostic"]
    ]
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "runs": len(values),
        "min": min(values),
        "max": max(values),
    }


def headline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """SPEC §8: single-user decode speed and peak throughput ratios."""
    decode: dict[str, Any] = {}
    for lifetime in ("bf16", "awq", "gptq"):
        tpot = median_of(rows, lifetime, "c1", "tpot_p50")
        decode[lifetime] = (
            {"tokens_per_s": 1 / tpot["median"], "tpot": tpot} if tpot else None
        )
    peaks: dict[str, Any] = {}
    for lifetime in ("bf16", "awq", "gptq", "hf-naive", "hf-static"):
        best = None
        for base in {r["base"] for r in rows if r["lifetime"] == lifetime}:
            value = median_of(rows, lifetime, base, "tokens_per_s")
            if value and (best is None or value["median"] > best["median"]):
                best = value | {"point": base}
        peaks[lifetime] = best

    def ratio(a: Any, b: Any) -> float | None:
        return a["median"] / b["median"] if a and b else None

    return {
        "single_user_decode": decode,
        "decode_speedup_vs_bf16": {
            v: decode[v]["tokens_per_s"] / decode["bf16"]["tokens_per_s"]
            for v in ("awq", "gptq")
            if decode.get(v) and decode.get("bf16")
        },
        "peak_output_tokens_per_s": peaks,
        "engine_gain_vllm_bf16_vs_hf": {
            "naive": ratio(peaks.get("bf16"), peaks.get("hf-naive")),
            "static": ratio(peaks.get("bf16"), peaks.get("hf-static")),
        },
        "awq_peak_vs_hf": {
            "naive": ratio(peaks.get("awq"), peaks.get("hf-naive")),
            "static": ratio(peaks.get("awq"), peaks.get("hf-static")),
        },
        "quantization_peak_vs_bf16": {
            v: ratio(peaks.get(v), peaks.get("bf16")) for v in ("awq", "gptq")
        },
    }


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value * 1000:.1f}"


def markdown(rows: list[dict[str, Any]], head: dict[str, Any]) -> str:
    lines = [
        "| Lifetime | Point | c | Output tok/s | Req/s | TTFT p50/p99 ms | "
        "TPOT p50/p99 ms | ITL p99 ms | E2E p50 s | Halves Δ | Headroom | "
        "Client / server cores | OK |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['lifetime']} | {r['label']} | {r['concurrency']} | "
            f"{r['tokens_per_s']:.1f} | {r['requests_per_s']:.3f} | "
            f"{_ms(r['ttft_p50'])} / {_ms(r['ttft_p99'])} | "
            f"{_ms(r['tpot_p50'])} / {_ms(r['tpot_p99'])} | {_ms(r['itl_p99'])} | "
            f"{(r['e2e_p50'] or 0):.2f} | {100 * (r['half_deviation'] or 0):.2f}% | "
            f"{'—' if r['headroom'] is None else f'{r["headroom"]:.1f}×'} | "
            f"{(r['client_cores'] or 0):.2f} / "
            f"{'—' if r['server_cores'] is None else f'{r["server_cores"]:.2f}'} | "
            f"{'diagnostic' if r['diagnostic'] else 'yes' if r['passed'] else 'NO'}"
            f"{' (warn)' if r['warnings'] else ''} |"
        )
    return "\n".join(lines) + "\n\n```json\n" + json.dumps(head, indent=2) + "\n```\n"


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m llmbench.perf_report")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = sorted(load_points(args.root), key=lambda r: (r["lifetime"], r["run"]))
    text = markdown(rows, headline(rows))
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
