"""The hand-written README cannot drift from the generated RESULTS.md: its
links resolve, its headline numbers exist there, and its resume bullets are
the generated ones word for word."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text()
RESULTS = (REPO / "docs" / "RESULTS.md").read_text()
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _slugs(markdown: str) -> set[str]:
    """GitHub heading anchors: lowercase, punctuation dropped, spaces -> '-'."""
    return {
        re.sub(r"[^\w\- ]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
        for line in markdown.splitlines()
        if line.startswith("#")
    }


def _section(markdown: str, heading: str) -> str:
    start = markdown.index(f"\n{heading}\n")
    end = markdown.find("\n## ", start + 1)
    return markdown[start : end if end != -1 else None]


def _words(text: str) -> str:
    return " ".join(text.replace(">", " ").replace("•", "-").split())


def test_readme_links_and_anchors_resolve() -> None:
    targets = [t for t in LINK.findall(README) if "://" not in t]
    assert len(targets) > 20
    for target in targets:
        path, _, anchor = target.partition("#")
        resolved = REPO / path
        assert resolved.exists(), target
        if anchor and resolved.suffix == ".md":
            assert anchor in _slugs(resolved.read_text()), target


def test_readme_result_numbers_come_from_results_page() -> None:
    numbers = re.findall(r"\d[\d,]*\.\d+", _section(README, "## Results"))
    assert len(numbers) > 20
    missing = [n for n in numbers if n not in RESULTS]
    assert not missing, f"not in docs/RESULTS.md: {missing}"


def test_readme_resume_bullets_match_results_page() -> None:
    quoted = _section(README, "## Results").split("### Resume bullets")[1]
    readme_bullets = [
        _words(b) for b in _words(quoted.split("\n> ", 1)[1]).split("- ")[1:]
    ]
    block = _section(RESULTS, "## Resume bullets").split("```text")[1].split("```")[0]
    results_bullets = [
        _words(line)[2:] for line in block.splitlines() if line.startswith("•")
    ]
    assert len(results_bullets) == 2
    assert readme_bullets == results_bullets
