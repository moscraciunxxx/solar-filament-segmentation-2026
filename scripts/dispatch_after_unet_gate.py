#!/usr/bin/env python3
"""When the v1 merge gate appears, record the next family. Never trains. Never submits."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

from gate_unet_family import GATE, dispatch_plan, read_gate


def main() -> None:
    print(f"waiting for {GATE}", flush=True)
    while not GATE.exists():
        time.sleep(15)
    decision, payload = read_gate()
    plan = dispatch_plan(decision)
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "gate": str(GATE),
        "decision": decision,
        "plan": plan,
        "do_not_auto_train": True,
        "do_not_submit": True,
        "note": "Overwatch starts the named script only after this file is read. No auto-start.",
        "gate_vs_slot1": (payload.get("gate") or {}).get("vs_slot1") if isinstance(payload, dict) else None,
    }
    dest = GATE.parent / "post_unet_dispatch.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
