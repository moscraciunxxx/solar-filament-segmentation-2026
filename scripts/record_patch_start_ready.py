#!/usr/bin/env python3
"""When ResNet gate allows patch and refuse_patch() is None, record ready. Never starts train."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "scripts"))

from gate_unet_family import (
    PROMOTE,
    RESNET_GATE,
    START_PATCH_AFTER_RESNET,
    refuse_patch,
    resolve_start_gate,
    read_gate,
)

OUT = ROOT / "artifacts" / "patch_may_start.json"


def main() -> None:
    print(f"waiting for {RESNET_GATE}", flush=True)
    while not RESNET_GATE.exists():
        time.sleep(15)
    decision, _ = read_gate(RESNET_GATE)
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "resnet_decision": decision,
        "auto_start": False,
        "did_not_start_train_unet_patch": True,
        "do_not_submit": True,
        "kaggle_submit_called": False,
    }
    if decision == PROMOTE:
        out["may_start"] = False
        out["reason"] = "ResNet PROMOTE_LOOK — one merged look, no second family"
        OUT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2), flush=True)
        return
    if decision not in START_PATCH_AFTER_RESNET:
        out["may_start"] = False
        out["reason"] = f"ResNet gate is {decision!r}, not a patch-start decision"
        OUT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2), flush=True)
        return
    print("ResNet gate allows patch; waiting until refuse_patch() is None", flush=True)
    reason = refuse_patch(resolve_start_gate())
    while reason:
        time.sleep(20)
        reason = refuse_patch(resolve_start_gate())
    out["written"] = datetime.now(timezone.utc).isoformat()
    out["may_start"] = True
    out["refuse_patch"] = None
    out["note"] = "Overwatch may start train_unet_patch.py + watch_patch_then_score.py. This script does not start them."
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
