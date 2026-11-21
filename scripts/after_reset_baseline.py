#!/usr/bin/env python3
"""After remaining-looks JSON exists, snapshot official LB. Never submits.

Waits for artifacts/kaggle_quota_remaining.json, then runs
scripts/confirm_kaggle_rank.py. Writes artifacts/quota_reset_verified.json.
Does not call kaggle competitions submit.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REMAINING = ARTIFACTS / "kaggle_quota_remaining.json"
PYTHON = Path("~/.venvs/filament/bin/python")
CONFIRM = ROOT / "scripts" / "confirm_kaggle_rank.py"


def main() -> None:
    print(f"waiting for {REMAINING}", flush=True)
    while not REMAINING.exists():
        time.sleep(5)
    rem = json.loads(REMAINING.read_text())
    remaining = rem.get("remaining")
    print("snapshotting official public rank (no submit)", flush=True)
    proc = subprocess.run([str(PYTHON), str(CONFIRM)], check=False)
    lb = {}
    latest = ARTIFACTS / "kaggle_lb_latest.json"
    if latest.exists():
        lb = json.loads(latest.read_text())
    us = lb.get("us") or {}
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "remaining": remaining,
        "n_used_today": rem.get("n_used_today"),
        "used_today": rem.get("used_today"),
        "expected_remaining": 5,
        "remaining_matches_expected": remaining == 5,
        "rank": us.get("rank"),
        "public": us.get("public"),
        "rank_le_10": bool(lb.get("rank_le_10")),
        "goal_top10": lb.get("goal_top10", "unproven"),
        "spend_only_if": "PROMOTE_LOOK",
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "confirm_returncode": proc.returncode,
    }
    dest = ARTIFACTS / "quota_reset_verified.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
