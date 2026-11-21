#!/usr/bin/env python3
"""After UTC midnight, snapshot remaining Kaggle quota. Never submits."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RESET = datetime(2026, 9, 11, 0, 5, tzinfo=timezone.utc)
KAGGLE = Path("~/.local/bin/kaggle")


def main() -> None:
    now = datetime.now(timezone.utc)
    wait = (RESET - now).total_seconds()
    if wait > 0:
        print(f"waiting {wait:.0f}s until {RESET.isoformat()}", flush=True)
        time.sleep(wait)
    proc = subprocess.run(
        [str(KAGGLE), "competitions", "submissions", "-c", "filament-segmentation-2026"],
        capture_output=True,
        text=True,
        check=False,
    )
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-1000:],
        "do_not_submit": True,
        "rule": "Do not upload faint 0.412, merckle 0.036, or U-Net before PROMOTE_LOOK.",
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "kaggle_quota_after_reset.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"path": str(path), "returncode": proc.returncode}, indent=2))
    parse = ROOT / "scripts" / "parse_kaggle_quota.py"
    if parse.exists():
        subprocess.run([str(Path("~/.venvs/filament/bin/python")), str(parse)], check=False)


if __name__ == "__main__":
    main()
