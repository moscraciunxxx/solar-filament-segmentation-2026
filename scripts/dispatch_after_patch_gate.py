#!/usr/bin/env python3
"""When the patch merge gate appears, record the next action. Never trains/submits."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

from gate_unet_family import PATCH_GATE, dispatch_plan_patch, read_gate


def main() -> None:
    print(f"waiting for {PATCH_GATE}", flush=True)
    while not PATCH_GATE.exists():
        time.sleep(15)
    decision, payload = read_gate(PATCH_GATE)
    plan = dispatch_plan_patch(decision)
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "gate": str(PATCH_GATE),
        "decision": decision,
        "plan": plan,
        "do_not_auto_train": True,
        "do_not_submit": True,
        "slot2": 56148889,
        "note": "Overwatch starts any next family only after this file is read. No auto-start.",
        "gate_vs_slot1": (payload.get("gate") or {}).get("vs_slot1") if isinstance(payload, dict) else None,
    }
    dest = PATCH_GATE.parent / "post_patch_dispatch.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
