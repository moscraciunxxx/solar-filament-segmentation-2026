#!/usr/bin/env python3
"""CPU overwatch for ResNet extras + Slot1-context fold-0 merge. Never uploads.

Copies gate.decision from the scored official JSON. Does not invent PROMOTE_LOOK.
Does not upload the logistic 0.414 HOLD or the HGB 0.414 HOLD.
"""

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
    EXTRA_FILTER_CTX_GATE,
    EXTRA_FILTER_GATE,
    EXTRA_FILTER_HGB_GATE,
    PROMOTE,
    live_mps_owner,
    pgrep,
)

GATE = EXTRA_FILTER_CTX_GATE
OUT = ROOT / "artifacts" / "resnet_ctx_extra_filter_eval_watch.json"
LOG = ROOT / "artifacts" / "resnet_ctx_extra_filter_eval.out"
WATCH_LIMIT_S = 5400
POLL_S = 30


def _locked(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("locked_row") or {}


def main() -> None:
    started = time.time()
    polls = 0
    while time.time() - started < WATCH_LIMIT_S:
        polls += 1
        eval_pids = pgrep("eval_unet_resnet_extra_filter_ctx_merge.py")
        if GATE.exists():
            payload = json.loads(GATE.read_text())
            if payload.get("scored") and isinstance(payload.get("gate"), dict):
                locked = payload.get("locked_row") or {}
                gate = payload["gate"]
                decision = gate.get("decision")
                vs = gate.get("vs_slot1") or {}
                log_row = _locked(EXTRA_FILTER_GATE)
                hgb_row = _locked(EXTRA_FILTER_HGB_GATE)
                out = {
                    "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "status": "GATE_READY",
                    "role": "cpu_only_resnet_ctx_extra_filter_eval_overwatch",
                    "cpu_only": True,
                    "fold0_diagnosis": True,
                    "elapsed_s": time.time() - started,
                    "polls": polls,
                    "source": str(GATE.relative_to(ROOT)),
                    "decision": decision,
                    "decision_source": str(GATE.relative_to(ROOT)),
                    "copied_from_file_not_invented": True,
                    "did_not_invent_PROMOTE_LOOK": True,
                    "recommend_upload": False,
                    "recommend_PROMOTE_local": decision == PROMOTE,
                    "kaggle_submit_called": False,
                    "did_not_submit": True,
                    "do_not_submit": True,
                    "did_not_train": True,
                    "did_not_mine": True,
                    "did_not_eval_restart": True,
                    "did_not_upload": True,
                    "did_not_upload_resnet_extra_filter_0.414_HOLD": True,
                    "did_not_upload_resnet_hgb_HOLD": True,
                    "locked": locked.get("thr"),
                    "locked_row": locked,
                    "pq": locked.get("pq"),
                    "tp": locked.get("tp"),
                    "fp": locked.get("fp"),
                    "fn": locked.get("fn"),
                    "d_pq": vs.get("d_pq"),
                    "d_tp": vs.get("d_tp"),
                    "fp_exploded": vs.get("fp_exploded"),
                    "n_extras_kept": locked.get("n_extras_kept"),
                    "rejector_lock": (payload.get("rejector") or {}).get("foldne0_lock"),
                    "vs_slot1": vs,
                    "vs_resnet_logistic_extra_filter_hold": {
                        "hold_pq": log_row.get("pq"),
                        "hold_tp": log_row.get("tp"),
                        "hold_fp": log_row.get("fp"),
                        "d_pq": float(locked.get("pq") or 0) - float(log_row.get("pq") or 0),
                        "d_tp": int(locked.get("tp") or 0) - int(log_row.get("tp") or 0),
                        "do_not_upload": True,
                    },
                    "vs_resnet_hgb_extra_filter_hold": {
                        "hold_pq": hgb_row.get("pq"),
                        "hold_tp": hgb_row.get("tp"),
                        "hold_fp": hgb_row.get("fp"),
                        "d_pq": float(locked.get("pq") or 0) - float(hgb_row.get("pq") or 0),
                        "d_tp": int(locked.get("tp") or 0) - int(hgb_row.get("tp") or 0),
                        "do_not_upload": True,
                    },
                    "desert_2017_2020_drop": gate.get("desert_2017_2020_drop"),
                    "promote_needs": {
                        "pq_ge_0.42": float(locked.get("pq") or 0) >= 0.42,
                        "desert_drop_lt_0.05": float(gate.get("desert_2017_2020_drop") or 1) < 0.05,
                        "not_fp_exploded": not bool(vs.get("fp_exploded")),
                    },
                    "observed_merge_pids": eval_pids,
                    "log": str(LOG.relative_to(ROOT)),
                    "xor": (
                        "not a look unless parent runs hidden-exam on a copied PROMOTE_LOOK; "
                        "do not upload logistic 0.414 or HGB HOLD; do not invert rejector_thr"
                    ),
                }
                needs = out["promote_needs"]
                needs["all_three"] = all(
                    [needs["pq_ge_0.42"], needs["desert_drop_lt_0.05"], needs["not_fp_exploded"]]
                )
                OUT.write_text(json.dumps(out, indent=2) + "\n")
                print(json.dumps({"status": "GATE_READY", "decision": decision, "pq": locked.get("pq")}))
                return
        time.sleep(POLL_S)
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "WATCH_TIMEOUT",
        "elapsed_s": time.time() - started,
        "polls": polls,
        "recommend_upload": False,
        "did_not_upload": True,
        "mps_owner": live_mps_owner(),
        "watcher_pid": os.getpid(),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    raise SystemExit(f"watch timeout waiting for scored {GATE}")


if __name__ == "__main__":
    main()
