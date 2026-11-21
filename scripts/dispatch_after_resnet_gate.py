#!/usr/bin/env python3
"""When the ResNet34 merge gate appears, record the next family. Never trains/submits."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

from gate_unet_family import RESNET_GATE, dispatch_plan_resnet, read_gate


def main() -> None:
    print(f"waiting for {RESNET_GATE}", flush=True)
    while not RESNET_GATE.exists():
        time.sleep(15)
    decision, payload = read_gate(RESNET_GATE)
    plan = dispatch_plan_resnet(decision)
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "gate": str(RESNET_GATE),
        "decision": decision,
        "plan": plan,
        "do_not_auto_train": True,
        "do_not_submit": True,
        "slot2": 56148889,
        "note": "Overwatch starts the named script only after this file is read. No auto-start.",
        "gate_vs_slot1": (payload.get("gate") or {}).get("vs_slot1") if isinstance(payload, dict) else None,
    }
    dest = RESNET_GATE.parent / "post_resnet_dispatch.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
