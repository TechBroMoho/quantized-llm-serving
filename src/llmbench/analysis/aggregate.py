"""Phase 7: collect every reported number from the committed raw results ($0).

    uv run python -m llmbench.analysis.aggregate   # -> results/analysis/aggregate.json

Nothing here measures or estimates. Each value is read from a raw file under
`results/` (a point summary, a server log line, a checkpoint manifest, an
lm-eval comparison, a `vllm bench serve` result) and stored next to the
repo-relative path it came from, so the charts and RESULTS.md can link every
number to its source. Headline values are medians of the passing runs of a
point (SPEC §3); failed and diagnostic points are kept, but only listed.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from llmbench.perf_report import headline, load_points

PHASE6 = Path("results/perf/phase6")
PHASE5 = Path("results/accuracy/phase5")
PHASE4 = Path("results/quantization/phase4")
OUT = Path("results/analysis/aggregate.json")

# Display order and names. vLLM variants first, then the two HF baselines.
SYSTEMS = {
    "awq": "vLLM AWQ (W4A16)",
    "gptq": "vLLM GPTQ (W4A16)",
    "bf16": "vLLM BF16",
    "hf-static": "HF static batching",
    "hf-naive": "HF naive",
}
# The lifetime whose server log gives each variant's weight memory and KV cache.
MEMORY_LIFETIMES = {
    "bf16": "bf16-20260924T003034Z",
    "awq": "awq-20260923T220125Z",
    "gptq": "gptq-trimmed-20260924T010937Z",
}
ACCURACY_RUN = PHASE5 / "full-20260923T142440Z"
CROSS_CHECK_LIFETIME = "awq-20260923T220125Z"
METRICS = (
    "tokens_per_s",
    "requests_per_s",
    "ttft_p50",
    "ttft_p95",
    "tpot_p50",
    "itl_p99",
    "e2e_p50",
    "e2e_p95",
)


def _rel(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "runs": len(values),
        # Relative spread of the repeats: (max - min) / median.
        "spread": (max(values) - min(values)) / statistics.median(values)
        if statistics.median(values)
        else 0.0,
    }


def _timing(summary_path: Path) -> dict[str, float]:
    """The ramp, warmup and window a point actually ran with (ADR-020/022)."""
    point = json.loads(summary_path.read_text())["point"]
    return {k: float(point[k]) for k in ("ramp_s", "warmup_s", "window_s")}


def lifetimes(repo: Path) -> list[dict[str, Any]]:
    """Provenance of every Phase 6 server lifetime (one per directory)."""
    out = []
    for path in sorted((repo / PHASE6).glob("*/lifetime_summary.json")):
        summary = json.loads(path.read_text())
        points = summary.get("points") or []
        out.append(
            {
                "run": path.parent.name,
                "date": re.sub(
                    r".*-(\d{4})(\d{2})(\d{2})T.*", r"\1-\2-\3", path.parent.name
                ),
                "source": _rel(path, repo),
                "commit": summary["git"]["commit"][:7],
                "dirty": summary["git"]["dirty"],
                "config_sha256": summary["config_sha256"][:12],
                "modal_image_id": summary["runtime"].get("modal_image_id"),
                "prompt_pool_sha256": summary["prompt_pool"]["pool_sha256"],
                "server_lifetime_s": summary.get("server_lifetime_s"),
                "points": len(points),
                "points_passed": sum(bool(p["passed"]) for p in points),
                "passed": summary["passed"],
                "failures": summary["failures"][:3],
            }
        )
    return out


BILLING = PHASE6 / "modal_billing_2026-09-24.json"


def spend(repo: Path) -> dict[str, Any]:
    """Actual Modal cost, from the last saved billing report. It covers whole
    UTC days 2026-09-23 and -24, which contain every run in this project."""
    rows = json.loads((repo / BILLING).read_text())
    by_app: dict[str, float] = {}
    for row in rows:
        by_app[row["Description"]] = by_app.get(row["Description"], 0.0) + float(
            row["Cost"]
        )
    return {
        "total_usd": sum(by_app.values()),
        "by_app_usd": dict(sorted(by_app.items())),
        "days": sorted({row["Interval Start"][:10] for row in rows}),
        "source": _rel(repo / BILLING, repo),
    }


MEASUREMENT_SECTIONS = ("model", "prompts", "workload", "vllm", "hf")


def config_consistency(repo: Path) -> dict[str, Any]:
    """The config file changed between lifetimes (new lifetimes and points were
    added), so its hash differs. Every section that affects a measurement must
    still be identical across the compared lifetimes (the probe is excluded)."""
    sections: dict[str, Any] = {}
    differing: dict[str, list[str]] = {}
    for path in sorted((repo / PHASE6).glob("*/lifetime_summary.json")):
        if path.parent.name.startswith("probe-"):
            continue
        config = json.loads(path.read_text())["config"]
        for key in MEASUREMENT_SECTIONS:
            if key not in sections:
                sections[key] = config.get(key)
            elif config.get(key) != sections[key]:
                differing.setdefault(key, []).append(path.parent.name)
    pools = {
        json.loads(path.read_text())["prompt_pool"]["pool_sha256"]
        for path in (repo / PHASE6).glob("*/lifetime_summary.json")
    }
    return {
        "sections": list(MEASUREMENT_SECTIONS),
        "identical": not differing,
        "differing": differing,
        # The prompt pool file itself, hashed by every lifetime before use.
        "prompt_pools": sorted(pools),
    }


def sweep_series(rows: list[dict[str, Any]], repo: Path) -> dict[str, list[Any]]:
    """Per system and concurrency: medians (with min/max) of passing runs."""
    series: dict[str, list[Any]] = {}
    for system in SYSTEMS:
        points = []
        mine = [r for r in rows if r["lifetime"] == system and not r["diagnostic"]]
        for base in sorted({r["base"] for r in mine}):
            runs = [r for r in mine if r["base"] == base and r["passed"]]
            if not runs:
                continue
            points.append(
                {
                    "point": base,
                    "concurrency": runs[0]["concurrency"],
                    "metrics": {
                        m: _stats([float(r[m]) for r in runs if r[m] is not None])
                        for m in METRICS
                    },
                    "sources": [_rel(Path(r["path"]), repo) for r in runs],
                    "timings": [_timing(Path(r["path"])) for r in runs],
                    "warnings": sorted({w for r in runs for w in r["warnings"]}),
                }
            )
        series[system] = sorted(points, key=lambda p: p["concurrency"])
    return series


def other_points(rows: list[dict[str, Any]], repo: Path) -> list[dict[str, Any]]:
    """Failed and diagnostic points: reported, never used for headlines."""
    return [
        {
            "lifetime": r["lifetime"],
            "run": r["run"],
            "point": r["label"],
            "concurrency": r["concurrency"],
            "diagnostic": r["diagnostic"],
            "tokens_per_s": r["tokens_per_s"],
            "tpot_p50": r["tpot_p50"],
            "half_deviation": r["half_deviation"],
            "failures": r["failures"],
            "source": _rel(Path(r["path"]), repo),
        }
        for r in rows
        if r["lifetime"] in SYSTEMS and (r["diagnostic"] or not r["passed"])
    ]


_LOG_PATTERNS = {
    "weights_gib": re.compile(r"Model loading took ([0-9.]+) GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([0-9,]+) tokens"),
    "max_concurrency": re.compile(
        r"Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x"
    ),
}


def server_log_values(log: Path, repo: Path) -> dict[str, Any]:
    """First match of each pattern, with its line number for a source link."""
    found: dict[str, Any] = {}
    for number, line in enumerate(log.read_text(errors="replace").splitlines(), 1):
        for key, pattern in _LOG_PATTERNS.items():
            match = pattern.search(line)
            if match and key not in found:
                if key == "max_concurrency":
                    value: Any = {
                        "tokens_per_request": int(match[1].replace(",", "")),
                        "multiple": float(match[2]),
                    }
                elif key == "kv_cache_tokens":
                    value = int(match[1].replace(",", ""))
                else:
                    value = float(match[1])
                found[key] = {"value": value, "source": f"{_rel(log, repo)}#L{number}"}
    missing = set(_LOG_PATTERNS) - set(found)
    if missing:
        raise ValueError(f"{log}: no {sorted(missing)} line")
    return found


def disk_sizes(repo: Path) -> dict[str, Any]:
    """Safetensors bytes on disk: the Phase 4 download manifest and quantize
    summaries (sha256-verified again by Phase 6 prepare)."""
    manifest = repo / PHASE4 / "download-20260923T102838Z/manifest.json"
    files = json.loads(manifest.read_text())["files"]
    sizes = {
        "bf16": {
            "value": sum(
                f["bytes"] for f in files if f["path"].endswith(".safetensors")
            ),
            "source": _rel(manifest, repo),
        }
    }
    for variant, run in (
        ("awq", "quantize-awq-20260923T103930Z"),
        ("gptq", "quantize-gptq-20260923T121835Z"),
    ):
        summary = repo / PHASE4 / run / "summary.json"
        checkpoint = json.loads(summary.read_text())["checkpoint"]
        sizes[variant] = {
            "value": int(checkpoint["safetensors_bytes"]),
            "source": _rel(summary, repo),
        }
    return sizes


def nvidia_smi_peak(path: Path, repo: Path) -> dict[str, Any]:
    """Largest `memory.used` sampled during a lifetime (1 s `nvidia-smi` loop)."""
    used, total = 0, 0
    for line in path.read_text().splitlines()[1:]:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 5 and fields[1].isdigit():
            used, total = max(used, int(fields[1])), int(fields[2])
    return {"used_mib": used, "total_mib": total, "source": _rel(path, repo)}


def memory(repo: Path) -> dict[str, Any]:
    disk = disk_sizes(repo)
    variants = {}
    for variant, lifetime in MEMORY_LIFETIMES.items():
        values = server_log_values(repo / PHASE6 / lifetime / "server.log", repo)
        variants[variant] = values | {
            "disk_bytes": disk[variant],
            "nvidia_smi_peak": nvidia_smi_peak(
                repo / PHASE6 / lifetime / "nvidia_smi.csv", repo
            ),
        }
    probe = server_log_values(repo / PHASE6 / "probe-20260923T213438Z/server.log", repo)
    bf16 = variants["bf16"]
    for variant in ("awq", "gptq"):
        v = variants[variant]
        v["weights_reduction_pct"] = 100 * (
            1 - v["weights_gib"]["value"] / bf16["weights_gib"]["value"]
        )
        v["disk_reduction_pct"] = 100 * (
            1 - v["disk_bytes"]["value"] / bf16["disk_bytes"]["value"]
        )
        v["kv_cache_ratio"] = (
            v["kv_cache_tokens"]["value"] / bf16["kv_cache_tokens"]["value"]
        )
    summary_path = repo / PHASE6 / MEMORY_LIFETIMES["bf16"] / "lifetime_summary.json"
    command = json.loads(summary_path.read_text())["command"]

    def flag(name: str) -> str:
        return str(command[command.index(name) + 1])

    return {
        "variants": variants,
        # Same AWQ checkpoint and flags in another lifetime: run-to-run spread.
        "awq_probe_lifetime": probe,
        "gpu_memory_utilization": float(flag("--gpu-memory-utilization")),
        "max_model_len": int(flag("--max-model-len")),
        "engine_args_source": _rel(summary_path, repo),
    }


def accuracy(repo: Path) -> dict[str, Any]:
    path = repo / ACCURACY_RUN / "comparison.json"
    data = json.loads(path.read_text())
    table_path = repo / ACCURACY_RUN / "accuracy_table.md"
    out: dict[str, Any] = {
        "source": _rel(path, repo),
        "table": _rel(table_path, repo),
        "variant_runs": {
            v: _rel(repo / src, repo) for v, src in data["sources"].items()
        },
        "variants": {},
    }
    for variant, values in data["variants"].items():
        keep = {
            k: values.get(k)
            for k in (
                "mmlu_questions",
                "mmlu_acc",
                "mmlu_acc_stderr",
                "word_perplexity",
                "mmlu_delta_pp",
                "word_perplexity_increase_pct",
                "mmlu_questions_lost",
                "mmlu_questions_gained",
            )
        }
        keep["mcnemar_p"] = (values.get("mcnemar") or {}).get("p_value")
        out["variants"][variant] = keep
    return out


def cross_check(
    series: dict[str, list[Any]], rows: list[dict[str, Any]], repo: Path
) -> list[Any]:
    """`vllm bench serve` vs our passing AWQ medians at the same concurrency."""
    directory = repo / PHASE6 / CROSS_CHECK_LIFETIME / "vllm_bench_serve"
    ours = {p["concurrency"]: p for p in series["awq"]}
    sync = next(
        (r for r in rows if r["lifetime"] == "awq" and r["label"] == "c256-sync"), None
    )
    checks = []
    for path in sorted(directory.glob("vllm_bench_serve_c*.json")):
        data = json.loads(path.read_text())
        concurrency = int(data["max_concurrency"])
        point = ours[concurrency]
        entry = {
            "concurrency": concurrency,
            "source": _rel(path, repo),
            "bench_tokens_per_s": data["output_throughput"],
            "bench_tpot_p50": data["median_tpot_ms"] / 1000,
            "bench_ttft_p50": data["median_ttft_ms"] / 1000,
            "bench_mean_input_tokens": data["total_input_tokens"] / data["completed"],
            "bench_completed": data["completed"],
            "ours_tokens_per_s": point["metrics"]["tokens_per_s"]["median"],
            "ours_tpot_p50": point["metrics"]["tpot_p50"]["median"],
            "ours_ttft_p50": point["metrics"]["ttft_p50"]["median"],
            "ours_runs": point["metrics"]["tokens_per_s"]["runs"],
            "ours_sources": point["sources"],
        }
        entry["tokens_gap_pct"] = 100 * (
            entry["bench_tokens_per_s"] / entry["ours_tokens_per_s"] - 1
        )
        entry["tpot_gap_pct"] = 100 * (
            entry["bench_tpot_p50"] / entry["ours_tpot_p50"] - 1
        )
        if concurrency == 256 and sync is not None:
            entry["ours_all_at_once"] = {
                "tokens_per_s": sync["tokens_per_s"],
                "tpot_p50": sync["tpot_p50"],
                "source": _rel(Path(sync["path"]), repo),
                "tokens_gap_pct": 100
                * (entry["bench_tokens_per_s"] / sync["tokens_per_s"] - 1),
                "tpot_gap_pct": 100 * (entry["bench_tpot_p50"] / sync["tpot_p50"] - 1),
            }
        checks.append(entry)
    return sorted(checks, key=lambda c: c["concurrency"])


def hf_static(repo: Path) -> dict[str, Any]:
    run = repo / PHASE6 / "hf-static-20260924T013433Z"
    probe = json.loads((run / "oom_probe.json").read_text())
    return {
        "batch_size": probe["chosen_batch_size"],
        "first_oom": probe["first_oom"],
        "candidates": probe["candidates"],
        "fitted": probe["fitted"],
        "source": _rel(run / "oom_probe.json", repo),
        "plan_source": _rel(run / "static_plan.json", repo),
    }


COUNTERS = {
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "preemptions": "vllm:num_preemptions_total",
}


def vllm_counter_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Server counters over each vLLM point (after - before), summed."""
    totals = dict.fromkeys(COUNTERS, 0.0)
    points = 0
    for row in rows:
        if row["lifetime"] not in {"awq", "gptq", "bf16", "probe"}:
            continue
        summary = json.loads(Path(row["path"]).read_text())
        before = summary["server_before"]["counters"]
        after = summary["server_after"]["counters"]
        for key, name in COUNTERS.items():
            totals[key] += after[name] - before[name]  # KeyError if absent
        points += 1
    return {"points": points} | totals


def sanity(series: dict[str, list[Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """SPEC §7 Phase 7 checks, as data: the shape of each throughput curve,
    error totals, and the spread of repeated points."""
    shapes = {}
    for system, points in series.items():
        tps = [p["metrics"]["tokens_per_s"]["median"] for p in points]
        peak = max(range(len(tps)), key=tps.__getitem__)
        shapes[system] = {
            "concurrency": [p["concurrency"] for p in points],
            "rises_to_peak": all(
                a < b for a, b in zip(tps[:peak], tps[1 : peak + 1], strict=True)
            ),
            "peak_concurrency": points[peak]["concurrency"],
            # Change from the peak to the highest concurrency measured.
            "after_peak_change_pct": 100 * (tps[-1] / tps[peak] - 1),
        }
    counted = [r for r in rows if r["lifetime"] in SYSTEMS]
    repeats = [
        {
            "system": system,
            "point": p["point"],
            "runs": p["metrics"]["tokens_per_s"]["runs"],
            "tokens_spread_pct": 100 * p["metrics"]["tokens_per_s"]["spread"],
            "tpot_spread_pct": 100 * p["metrics"]["tpot_p50"]["spread"],
        }
        for system, points in series.items()
        for p in points
        if p["metrics"]["tokens_per_s"]["runs"] > 1
    ]
    return {
        "curve_shape": shapes,
        "points_total": len(counted),
        "points_passed": sum(r["passed"] for r in counted if not r["diagnostic"]),
        "points_failed": sum(not r["passed"] for r in counted if not r["diagnostic"]),
        "points_diagnostic": sum(r["diagnostic"] for r in counted),
        "errored_requests_all_points": sum(int(r["errors"] or 0) for r in counted),
        "vllm_counter_deltas": vllm_counter_deltas(rows),
        "repeat_spreads": repeats,
    }


def environment(repo: Path) -> dict[str, Any]:
    out = {}
    for name in ("awq-20260923T220125Z", "hf-static-20260924T013433Z"):
        path = repo / PHASE6 / name / "lifetime_summary.json"
        summary = json.loads(path.read_text())
        runtime = summary["runtime"]
        out[name] = {
            "source": _rel(path, repo),
            "gpu": runtime["gpu"]["nvidia_smi"]["stdout"],
            "packages": runtime["packages"],
            "python": runtime["python"],
            "modal_image_id": runtime.get("modal_image_id"),
            "cpu_cores": summary["resources"]["cpu_cores"],
            "memory_mib": summary["resources"]["memory_mib"],
            "command": summary["command"],
            "config_sha256": summary["config_sha256"],
            "git": summary["git"],
            "validated_client_chunks_per_s": summary["config"]["workload"][
                "validated_client_chunks_per_s"
            ],
            "prompt_count": summary["prompt_pool"]["prompt_count"],
            "prompt_pool_sha256": summary["prompt_pool"]["pool_sha256"],
        }
    return out


def collect(repo: Path) -> dict[str, Any]:
    rows = sorted(load_points(repo / PHASE6), key=lambda r: (r["lifetime"], r["run"]))
    series = sweep_series(rows, repo)
    head = headline(rows)
    return {
        "systems": SYSTEMS,
        "series": series,
        "other_points": other_points(rows, repo),
        "headline": head,
        "headline_source": _rel(repo / PHASE6 / "results_table.md", repo),
        "memory": memory(repo),
        "accuracy": accuracy(repo),
        "cross_check": cross_check(series, rows, repo),
        "hf_static": hf_static(repo),
        "sanity": sanity(series, rows),
        "environment": environment(repo),
        "lifetimes": lifetimes(repo),
        "config_consistency": config_consistency(repo),
        "spend": spend(repo),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m llmbench.analysis.aggregate")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    data = collect(args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
