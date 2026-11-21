#!/usr/bin/env python3
"""Parse Kaggle submissions into remaining daily looks. Never submits."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
KAGGLE = Path("~/.local/bin/kaggle")
COMP = "filament-segmentation-2026"
DAILY_CAP = 5


def _rows_from_text(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        m = re.match(
            r"^\s*(\d+)\s+(\S+)\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}\S*)\s+.*?"
            r"(SubmissionStatus\.\w+)\s+(\d+\.\d+)?",
            line,
        )
        if not m:
            continue
        rows.append(
            {
                "ref": int(m.group(1)),
                "file": m.group(2),
                "date": m.group(3),
                "time": m.group(4),
                "status": m.group(5),
                "public": float(m.group(6)) if m.group(6) else None,
            }
        )
    return rows


def _fetch_live() -> tuple[str, int]:
    proc = subprocess.run(
        [str(KAGGLE), "competitions", "submissions", "-c", COMP],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout or "", int(proc.returncode)


def _text_from_snapshot() -> str:
    raw_path = ARTIFACTS / "kaggle_quota_after_reset.json"
    if not raw_path.exists():
        return ""
    try:
        blob = json.loads(raw_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(blob, dict):
        return ""
    return str(blob.get("stdout") or blob.get("stdout_tail") or "")


def main() -> None:
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    live_rc = None
    text, live_rc = _fetch_live()
    source = "kaggle_live"
    if not text.strip():
        text = _text_from_snapshot()
        source = "snapshot_fallback"
    rows = _rows_from_text(text)
    used_today = [r for r in rows if r["date"] == today]
    remaining = max(0, DAILY_CAP - len(used_today))
    out = {
        "written": now.isoformat(),
        "utc_day": today,
        "source": source,
        "live_returncode": live_rc,
        "daily_cap": DAILY_CAP,
        "used_today": used_today,
        "n_used_today": len(used_today),
        "remaining": remaining,
        "do_not_submit": True,
        "spend_only_if": "artifacts/unet_slot1_merge_fold0_official.json decision == PROMOTE_LOOK",
        "never_spend_on": [
            "faint merge 0.412",
            "merckle 0.036",
            "U-Net standalone",
            "conf clones",
            "phys / tiles / all-annotator",
        ],
        "locked_finals": [56148686, 56148889],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "kaggle_quota_remaining.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"path": str(path), "remaining": remaining, "n_used_today": len(used_today)}, indent=2))


if __name__ == "__main__":
    main()
