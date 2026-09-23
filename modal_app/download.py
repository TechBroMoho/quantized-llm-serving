"""CPU-only model download into the weights Volume (never while holding a GPU).

uv run modal run --detach -m modal_app.download --config phase3_smoke.yaml
"""

from __future__ import annotations

import json
from typing import Any

import modal

from modal_app.common import (
    DOWNLOAD_IMAGE,
    DOWNLOAD_RESOURCES,
    HF_SECRET,
    RESULTS,
    RESULTS_PATH,
    WEIGHTS,
    WEIGHTS_PATH,
    load_config,
    manifest_path,
    model_dir,
    run_stamp,
)

app = modal.App("llmbench-download")


@app.function(
    image=DOWNLOAD_IMAGE,
    volumes={WEIGHTS_PATH: WEIGHTS, RESULTS_PATH: RESULTS},
    secrets=[HF_SECRET],
    **DOWNLOAD_RESOURCES.function_kwargs(),
)
def download_model(model_id: str, revision: str, stamp: str) -> dict[str, Any]:
    """Download one immutable revision and verify it against Hub metadata."""
    import hashlib
    import time
    from importlib.metadata import version
    from pathlib import Path

    from huggingface_hub import HfApi, snapshot_download

    target = Path(model_dir(model_id, revision))
    started = time.perf_counter()
    snapshot_download(repo_id=model_id, revision=revision, local_dir=target)
    elapsed = time.perf_counter() - started

    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Hub resolved {info.sha}, expected {revision}")
    files = []
    failures = []
    for sibling in info.siblings or []:
        path = target / sibling.rfilename
        if not path.is_file():
            failures.append(f"missing {sibling.rfilename}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        entry = {
            "path": sibling.rfilename,
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
        if sibling.size is not None and sibling.size != entry["bytes"]:
            failures.append(f"size mismatch {sibling.rfilename}")
        if sibling.lfs is not None:
            entry["hub_lfs_sha256"] = sibling.lfs.sha256
            if sibling.lfs.sha256 != entry["sha256"]:
                failures.append(f"sha256 mismatch {sibling.rfilename}")
        files.append(entry)
    manifest = {
        "model_id": model_id,
        "revision": revision,
        "local_dir": str(target),
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
        "download_seconds": elapsed,
        "huggingface_hub": version("huggingface-hub"),
        "verified": not failures,
        "failures": failures,
        "stamp": stamp,
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if not failures:
        # Only a verified download gets the manifest that GPU runs require.
        manifest_file = Path(WEIGHTS_PATH) / manifest_path(model_id, revision)
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(text)
    WEIGHTS.commit()
    result_file = Path(RESULTS_PATH) / "phase3" / f"download-{stamp}" / "manifest.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(text)
    RESULTS.commit()
    if failures:
        raise RuntimeError(f"download verification failed: {failures}")
    return manifest


@app.local_entrypoint()
def main(config: str = "phase3_smoke.yaml") -> None:
    settings, _ = load_config(config)
    manifest = download_model.remote(
        settings["model"]["id"], settings["model"]["revision"], run_stamp()
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    for item in manifest["files"]:
        print(f"{item['bytes']:>12}  {item['sha256'][:16]}  {item['path']}")
