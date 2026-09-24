"""The verify-checkpoint command compares local files with Phase 4 evidence."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from llmbench.cli import main


def _repo(tmp_path: Path, content: bytes) -> tuple[Path, Path]:
    """A fake repo: bench config -> evidence file -> one checkpoint file."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "results").mkdir()
    config = tmp_path / "configs" / "bench.yaml"
    config.write_text("expected_checkpoints:\n  awq: results/summary.json\n")
    evidence = {
        "checkpoint": {
            "files": [
                {
                    "path": "model.safetensors",
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            ]
        }
    }
    (tmp_path / "results" / "summary.json").write_text(json.dumps(evidence))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    return config, checkpoint


def _run(monkeypatch: pytest.MonkeyPatch, config: Path, checkpoint: Path) -> None:
    argv = ["llmbench", "verify-checkpoint", "--variant", "awq"]
    argv += ["--dir", str(checkpoint), "--config", str(config)]
    monkeypatch.setattr(sys, "argv", argv)
    main()


def test_verify_checkpoint_passes_on_identical_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, checkpoint = _repo(tmp_path, b"weights")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    _run(monkeypatch, config, checkpoint)
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] and result["files_checked"] == 1
    assert result["evidence"].endswith("results/summary.json")


@pytest.mark.parametrize("content", [b"weightz", b"weights!", None])
def test_verify_checkpoint_fails_on_changed_or_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes | None
) -> None:
    config, checkpoint = _repo(tmp_path, b"weights")
    if content is not None:
        (checkpoint / "model.safetensors").write_bytes(content)
    with pytest.raises(SystemExit) as failed:
        _run(monkeypatch, config, checkpoint)
    assert failed.value.code == 1
