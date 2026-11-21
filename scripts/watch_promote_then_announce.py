#!/usr/bin/env python3
"""When promote_look_if_ready.json appears, write the upload cue. Never submits.

Overwatch may run upload_if_promote.py --upload only after this file says
built_csv is true AND remaining >= 1 AND gate is PROMOTE_LOOK. This script
does not call kaggle competitions submit.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PROMOTE = ARTIFACTS / "promote_look_if_ready.json"
REMAINING = ARTIFACTS / "kaggle_quota_remaining.json"
OUT = ARTIFACTS / "promote_csv_ready.json"
PYTHON = "~/.venvs/filament/bin/python"
UPLOAD = ROOT / "scripts" / "upload_if_promote.py"
CSV = (
    Path.home()
    / ".cache"
    / "filament-segmentation-2026"
    / "submissions"
    / "submission_slot1_unet_merged_PROMOTE.csv"
)


def main() -> None:
    print(f"waiting for {PROMOTE}", flush=True)
    while not PROMOTE.exists():
        time.sleep(10)
    blob = json.loads(PROMOTE.read_text())
    remaining = None
    if REMAINING.exists():
        remaining = json.loads(REMAINING.read_text()).get("remaining")
    built = bool(blob.get("built_csv"))
    decision = blob.get("decision")
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "source": str(PROMOTE),
        "decision": decision,
        "built_csv": built,
        "csv_exists": CSV.exists(),
        "csv": str(CSV),
        "remaining": remaining,
        "auto_upload": False,
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "upload_now": False,
        "slot2": 56148889,
    }
    if built and CSV.exists() and remaining is not None and int(remaining) >= 1 and decision == "PROMOTE_LOOK":
        out["overwatch_may_upload_once"] = True
        out["command"] = f"{PYTHON} {UPLOAD} --upload"
        out["note"] = (
            "CSV is built. Overwatch runs the command once. This script does not upload. "
            "Year-shift 56148889 stays Slot 2. Climb look, not certified top-10."
        )
    else:
        out["overwatch_may_upload_once"] = False
        out["command"] = None
        out["note"] = (
            "No upload. If decision is not PROMOTE_LOOK, read overwatch_next_action.json "
            "and failpath_start_playbook.json. Do not auto-start the next family."
        )
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
