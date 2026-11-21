#!/usr/bin/env python3
"""When post_3ch_dispatch.json appears, write Overwatch next action. Never trains/submits."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISPATCH = ROOT / "artifacts" / "post_3ch_dispatch.json"
OUT = ROOT / "artifacts" / "overwatch_3ch_next_action.json"


def main() -> None:
    print(f"waiting for {DISPATCH}", flush=True)
    while not DISPATCH.exists():
        time.sleep(10)
    dispatch = json.loads(DISPATCH.read_text())
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "decision": dispatch.get("decision"),
        "step": dispatch.get("plan"),
        "auto_start": False,
        "do_not_submit": True,
        "slot2": 56148889,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
