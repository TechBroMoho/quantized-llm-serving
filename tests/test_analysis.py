"""Phase 7: the aggregate, the charts and RESULTS.md match the committed raw
results, and every number's link points at a file (and line) that exists."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from llmbench.analysis import aggregate, plots, report

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def _results_links() -> list[tuple[str, str]]:
    text = (DOCS / "RESULTS.md").read_text()
    return [(t, target) for t, target in LINK.findall(text) if "://" not in target]


def test_committed_aggregate_and_results_page_are_current() -> None:
    fresh = json.loads(json.dumps(aggregate.collect(REPO)))
    committed = json.loads((REPO / aggregate.OUT).read_text())
    assert fresh == committed, "run `make report plots` and commit the output"
    assert (DOCS / "RESULTS.md").read_text() == report.render(committed)


def test_every_results_link_resolves() -> None:
    links = _results_links()
    assert len(links) > 150  # every number is linked
    for text, target in links:
        path, _, anchor = target.partition("#")
        if not path:  # an in-page anchor
            continue
        resolved = (DOCS / path).resolve()
        assert resolved.exists(), f"{text!r} -> {target}: missing"
        if anchor.startswith("L"):
            lines = resolved.read_text(errors="replace").splitlines()
            line = lines[int(anchor[1:]) - 1]
            # A log-line link lands on the line that states the value.
            assert re.search(
                r"Model loading took|GPU KV cache size|Maximum concurrency", line
            ), (target, line)
            number = re.match(r"[−+]?([\d,]+(?:\.\d+)?)(?: GiB)?$", text)
            if number:
                assert number[1].replace(",", "") in line.replace(",", ""), (
                    text,
                    line,
                )


def test_decisions_anchors_match_headings() -> None:
    headings = (DOCS / "DECISIONS.md").read_text().splitlines()
    slugs = {
        re.sub(r"[^\w\- ]", "", h.lstrip("# ").lower()).replace(" ", "-")
        for h in headings
        if h.startswith("## ")
    }
    for _, target in _results_links():
        path, _, anchor = target.partition("#")
        if path == "DECISIONS.md":
            assert anchor in slugs, anchor


def test_headline_values_come_from_passing_runs_only() -> None:
    data = json.loads((REPO / aggregate.OUT).read_text())
    for points in data["series"].values():
        for point in points:
            for source in point["sources"]:
                summary = json.loads((REPO / source).read_text())
                assert summary["passed"] and not summary.get("diagnostic")
    # The failed AWQ c=1 and c=256 points are listed, not used.
    failed = {(o["run"], o["point"]) for o in data["other_points"]}
    assert ("awq-20260923T220125Z", "c1") in failed
    awq_c1 = next(p for p in data["series"]["awq"] if p["point"] == "c1")
    assert not any(
        s.endswith("awq-20260923T220125Z/c1/summary.json") for s in awq_c1["sources"]
    )


def test_curve_shape_flags_a_dip_before_the_peak() -> None:
    def point(c: int, tps: float) -> dict:
        stats = {"median": tps, "min": tps, "max": tps, "runs": 1, "spread": 0.0}
        return {
            "point": f"c{c}",
            "concurrency": c,
            "metrics": {"tokens_per_s": stats, "tpot_p50": stats},
        }

    series = {
        "awq": [
            point(1, 10),
            point(4, 30),
            point(16, 20),
            point(64, 50),
            point(128, 40),
        ]
    }
    shape = aggregate.sanity(series, [])["curve_shape"]["awq"]
    assert not shape["rises_to_peak"]
    assert shape["peak_concurrency"] == 64
    assert shape["after_peak_change_pct"] == pytest.approx(-20.0)


def test_server_log_values_need_every_line(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "noise\nINFO Model loading took 5.7088 GiB and 3.1 s\n"
        "INFO GPU KV cache size: 245,312 tokens\n"
    )
    with pytest.raises(ValueError, match="max_concurrency"):
        aggregate.server_log_values(log, tmp_path)
    log.write_text(
        log.read_text() + "Maximum concurrency for 1,024 tokens per request: 239.56x\n"
    )
    values = aggregate.server_log_values(log, tmp_path)
    assert values["weights_gib"] == {"value": 5.7088, "source": "server.log#L2"}
    assert values["kv_cache_tokens"]["value"] == 245312
    assert values["max_concurrency"]["value"]["multiple"] == 239.56


def test_nvidia_smi_peak_skips_the_header_and_bad_rows(tmp_path: Path) -> None:
    csv = tmp_path / "nvidia_smi.csv"
    csv.write_text(
        "timestamp, memory.used [MiB], memory.total [MiB], utilization.gpu [%], "
        "power.draw [W]\n"
        "t, 5, 46068, 0, 33.2\nt, 41472, 46068, 97, 350.1\nt, [N/A], 46068, 0, 0\n"
    )
    peak = aggregate.nvidia_smi_peak(csv, tmp_path)
    assert peak["used_mib"] == 41472 and peak["total_mib"] == 46068


def test_charts_render_from_the_committed_aggregate(tmp_path: Path) -> None:
    data = json.loads((REPO / aggregate.OUT).read_text())
    paths = [chart(data, tmp_path) for chart in plots.CHARTS]
    assert len(paths) == 6
    for path in paths:
        assert (
            path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            and path.stat().st_size > 20_000
        )
    committed = {p.name for p in (REPO / plots.CHART_DIR).glob("*.png")}
    assert committed == {p.name for p in paths}


def test_committed_results_table_is_current() -> None:
    # RESULTS.md links its headline ratios to this file, so it must match too.
    from llmbench.perf_report import headline, load_points, markdown

    root = REPO / "results/perf/phase6"
    rows = sorted(load_points(root), key=lambda r: (r["lifetime"], r["run"]))
    assert (root / "results_table.md").read_text() == markdown(rows, headline(rows))
