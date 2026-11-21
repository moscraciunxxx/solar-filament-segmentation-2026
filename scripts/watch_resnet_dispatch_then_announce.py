#!/usr/bin/env python3
"""When post_resnet_dispatch.json appears, write Overwatch next action. Never trains/submits."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DISPATCH = ARTIFACTS / "post_resnet_dispatch.json"
OUT = ARTIFACTS / "overwatch_resnet_next_action.json"


def main() -> None:
    print(f"waiting for {DISPATCH}", flush=True)
    while not DISPATCH.exists():
        time.sleep(10)
    dispatch = json.loads(DISPATCH.read_text())
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "decision": dispatch.get("decision"),
        "dispatch": str(DISPATCH),
        "step": dispatch.get("plan"),
        "auto_start": False,
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "slot2": 56148889,
        "note": "Overwatch starts the named script only after reading this file. No auto-start.",
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
