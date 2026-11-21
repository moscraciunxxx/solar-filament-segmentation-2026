#!/usr/bin/env python3
"""CPU overwatch for ResNet extras crop-CNN fold-0 merge. Never uploads."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gate_unet_family import (
    EXTRA_FILTER_CROP_GATE,
    EXTRA_FILTER_CTX_GATE,
    EXTRA_FILTER_GATE,
    EXTRA_FILTER_HGB_GATE,
    PROMOTE,
    live_mps_owner,
    pgrep,
)

GATE = EXTRA_FILTER_CROP_GATE
OUT = ROOT / "artifacts" / "resnet_crop_extra_filter_eval_watch.json"
WATCH_LIMIT_S = 7200
POLL_S = 30


def _locked(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("locked_row") or {}


def main() -> None:
    started = time.time()
    polls = 0
    while time.time() - started < WATCH_LIMIT_S:
        polls += 1
        eval_pids = pgrep("eval_unet_resnet_extra_filter_crop_merge.py")
        if GATE.exists():
            payload = json.loads(GATE.read_text())
            if payload.get("scored") and isinstance(payload.get("gate"), dict):
                locked = payload.get("locked_row") or {}
                gate = payload["gate"]
                decision = gate.get("decision")
                vs = gate.get("vs_slot1") or {}
                ctx_row = _locked(EXTRA_FILTER_CTX_GATE)
                log_row = _locked(EXTRA_FILTER_GATE)
                out = {
                    "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "status": "GATE_READY",
                    "role": "cpu_only_resnet_crop_extra_filter_eval_overwatch",
                    "decision": decision,
                    "decision_source": str(GATE.relative_to(ROOT)),
                    "copied_from_file_not_invented": True,
                    "did_not_invent_PROMOTE_LOOK": True,
                    "recommend_upload": False,
                    "recommend_PROMOTE_local": decision == PROMOTE,
                    "do_not_submit": True,
                    "did_not_upload": True,
                    "did_not_upload_ctx_0.4155_HOLD": True,
                    "locked_row": locked,
                    "pq": locked.get("pq"),
                    "tp": locked.get("tp"),
                    "fp": locked.get("fp"),
                    "vs_slot1": vs,
                    "vs_ctx_hold": {
                        "hold_pq": ctx_row.get("pq"),
                        "d_pq": float(locked.get("pq") or 0) - float(ctx_row.get("pq") or 0),
                        "d_tp": int(locked.get("tp") or 0) - int(ctx_row.get("tp") or 0),
                        "do_not_upload": True,
                    },
                    "vs_logistic_hold": {
                        "hold_pq": log_row.get("pq"),
                        "do_not_upload": True,
                    },
                    "desert_2017_2020_drop": gate.get("desert_2017_2020_drop"),
                    "promote_needs": {
                        "pq_ge_0.42": float(locked.get("pq") or 0) >= 0.42,
                        "desert_drop_lt_0.05": float(gate.get("desert_2017_2020_drop") or 1) < 0.05,
                        "not_fp_exploded": not bool(vs.get("fp_exploded")),
                    },
                    "observed_merge_pids": eval_pids,
                    "xor": "not a look unless parent hidden-exam on copied PROMOTE_LOOK",
                }
                needs = out["promote_needs"]
                needs["all_three"] = all(
                    [needs["pq_ge_0.42"], needs["desert_drop_lt_0.05"], needs["not_fp_exploded"]]
                )
                OUT.write_text(json.dumps(out, indent=2) + "\n")
                print(json.dumps({"status": "GATE_READY", "decision": decision, "pq": locked.get("pq")}))
                return
        time.sleep(POLL_S)
    OUT.write_text(
        json.dumps(
            {
                "status": "WATCH_TIMEOUT",
                "recommend_upload": False,
                "mps_owner": live_mps_owner(),
                "watcher_pid": os.getpid(),
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(f"watch timeout waiting for scored {GATE}")


if __name__ == "__main__":
    main()
