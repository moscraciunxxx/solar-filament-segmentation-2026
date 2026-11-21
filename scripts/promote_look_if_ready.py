#!/usr/bin/env python3
"""After v1 merge gate is PROMOTE_LOOK, build the merged test CSV. Never uploads.

Waits for artifacts/unet_slot1_merge_fold0_official.json. If decision is not
PROMOTE_LOOK, records that and exits. Does not call kaggle competitions submit.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS
from gate_unet_family import GATE, PROMOTE, live_mps_owner, read_gate

PYTHON = Path("~/.venvs/filament/bin/python")
MAKE = ROOT / "scripts" / "make_unet_merged_submission.py"
POLICY = ARTIFACTS / "quota_reset_look_policy.json"


def main() -> None:
    print(f"waiting for {GATE}", flush=True)
    while not GATE.exists():
        time.sleep(15)
    decision, payload = read_gate()
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "do_not_submit": True,
        "kaggle_submit_called": False,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS / "promote_look_if_ready.json"
    if decision != PROMOTE:
        out["built_csv"] = False
        out["reason"] = f"gate is {decision!r}, not PROMOTE_LOOK — no test infer, no upload"
        dest.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2), flush=True)
        return
    owner = live_mps_owner()
    while owner:
        print(f"PROMOTE_LOOK set; waiting for {owner} to leave MPS", flush=True)
        time.sleep(20)
        owner = live_mps_owner()
    print("building Slot1+UNet merged test CSV; will not upload", flush=True)
    proc = subprocess.run([str(PYTHON), str(MAKE)], check=False)
    out["make_returncode"] = proc.returncode
    out["built_csv"] = proc.returncode == 0
    out["upload"] = False
    out["policy"] = str(POLICY)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"make_unet_merged_submission failed: {proc.returncode}")


if __name__ == "__main__":
    main()
