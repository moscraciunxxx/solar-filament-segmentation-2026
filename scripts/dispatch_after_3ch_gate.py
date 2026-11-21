#!/usr/bin/env python3
"""When the 3ch merge gate appears, record the next action. Never trains/submits."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

from gate_unet_family import CH3_GATE, dispatch_plan_3ch, read_gate


def main() -> None:
    print(f"waiting for {CH3_GATE}", flush=True)
    while not CH3_GATE.exists():
        time.sleep(15)
    decision, payload = read_gate(CH3_GATE)
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "gate": str(CH3_GATE),
        "decision": decision,
        "plan": dispatch_plan_3ch(decision),
        "do_not_auto_train": True,
        "do_not_submit": True,
        "slot2": 56148889,
        "gate_vs_slot1": (payload.get("gate") or {}).get("vs_slot1") if isinstance(payload, dict) else None,
    }
    dest = CH3_GATE.parent / "post_3ch_dispatch.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
