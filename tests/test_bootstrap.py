"""Verify that the Phase 0 package and instruction copy are usable."""

from importlib.metadata import version
from pathlib import Path

import llmbench


def test_installed_package() -> None:
    assert version("llmbench") == "0.1.0"
    assert llmbench.__doc__


def test_instructions_identical() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "CLAUDE.md").read_bytes() == (root / "AGENTS.md").read_bytes()
