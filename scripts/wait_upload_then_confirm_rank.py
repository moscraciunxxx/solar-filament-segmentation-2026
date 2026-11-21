#!/usr/bin/env python3
"""After a real PROMOTE upload succeeds, snapshot official rank. Never submits.

Waits for artifacts/upload_if_{promote,resnet_promote,patch_promote,3ch_promote}.json
with kaggle_submit_called true, upload_flag true, and returncode 0, then runs
scripts/confirm_kaggle_rank.py. Does not call kaggle competitions submit itself.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = (
    ROOT / "artifacts" / "upload_if_promote.json",
    ROOT / "artifacts" / "upload_if_resnet_promote.json",
    ROOT / "artifacts" / "upload_if_patch_promote.json",
    ROOT / "artifacts" / "upload_if_3ch_promote.json",
)
PYTHON = Path("~/.venvs/filament/bin/python")
CONFIRM = ROOT / "scripts" / "confirm_kaggle_rank.py"


def _file_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        blob = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(blob, dict):
        return False
    if blob.get("kaggle_submit_called") is not True:
        return False
    if blob.get("upload_flag") is not True:
        return False
    return int(blob.get("returncode") or 1) == 0


def _ready() -> bool:
    return any(_file_ready(path) for path in UPLOADS)


def main() -> None:
    names = " or ".join(path.name for path in UPLOADS)
    print(f"waiting for successful upload in {names}", flush=True)
    while not _ready():
        time.sleep(20)
    print("upload succeeded; snapshotting official public rank", flush=True)
    proc = subprocess.run([str(PYTHON), str(CONFIRM)], check=False)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
