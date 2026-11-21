#!/usr/bin/env python3
"""CPU overwatch for ResNet extras HistGB fold-0 merge. Never trains, mines, or uploads.

Copies gate.decision from the scored official JSON. Does not invent PROMOTE_LOOK.
Does not upload the logistic 0.414 HOLD.
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

from gate_unet_family import EXTRA_FILTER_GATE, EXTRA_FILTER_HGB_GATE, PROMOTE, live_mps_owner, pgrep

GATE = EXTRA_FILTER_HGB_GATE
LOGISTIC_GATE = EXTRA_FILTER_GATE
OUT = ROOT / "artifacts" / "resnet_hgb_extra_filter_eval_watch.json"
LOG = ROOT / "artifacts" / "resnet_hgb_extra_filter_eval.out"
WATCH_LIMIT_S = 5400
POLL_S = 30


def _copy_decision(payload: dict) -> str | None:
    gate = payload.get("gate") if isinstance(payload, dict) else None
    if not isinstance(gate, dict):
        return None
    decision = gate.get("decision")
    return str(decision) if decision is not None else None


def main() -> None:
    started = time.time()
    polls = 0
    eval_pids = pgrep("eval_unet_resnet_extra_filter_hgb_merge.py")
    while time.time() - started < WATCH_LIMIT_S:
        polls += 1
        eval_pids = pgrep("eval_unet_resnet_extra_filter_hgb_merge.py")
        if GATE.exists():
            payload = json.loads(GATE.read_text())
            if payload.get("scored") and isinstance(payload.get("gate"), dict):
                locked = payload.get("locked_row") or {}
                gate = payload["gate"]
                decision = _copy_decision(payload)
                vs = gate.get("vs_slot1") or {}
                logistic = {}
                if LOGISTIC_GATE.exists():
                    log_payload = json.loads(LOGISTIC_GATE.read_text())
                    log_row = log_payload.get("locked_row") or {}
                    logistic = {
                        "hold_pq": log_row.get("pq"),
                        "hold_tp": log_row.get("tp"),
                        "hold_fp": log_row.get("fp"),
                        "d_pq": float(locked.get("pq", 0)) - float(log_row.get("pq") or 0),
                        "d_tp": int(locked.get("tp", 0)) - int(log_row.get("tp") or 0),
                        "d_fp": int(locked.get("fp", 0)) - int(log_row.get("fp") or 0),
                        "do_not_upload": True,
                    }
                out = {
                    "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "status": "GATE_READY",
                    "role": "cpu_only_resnet_hgb_extra_filter_eval_overwatch",
                    "cpu_only": True,
                    "fold0_diagnosis": True,
                    "elapsed_s": time.time() - started,
                    "polls": polls,
                    "watch_limit_s": WATCH_LIMIT_S,
                    "source": str(GATE.relative_to(ROOT)),
                    "this_is_fold0_diagnosis": True,
                    "legal_test_csv": False,
                    "scored": True,
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
                    "did_not_execute_make_unet_resnet_extra_filter_hgb_merged_submission": True,
                    "did_not_start_second_infer": True,
                    "did_not_upload": True,
                    "did_not_upload_resnet_extra_filter_0.414_HOLD": True,
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
                    "n_extras_before_filter": locked.get("n_extras_before_filter"),
                    "rejector_lock": (payload.get("rejector") or {}).get("foldne0_lock"),
                    "vs_slot1": vs,
                    "vs_resnet_logistic_extra_filter_hold": logistic,
                    "desert_2017_2020_drop": gate.get("desert_2017_2020_drop"),
                    "promote_needs": {
                        "pq_ge_0.42": float(locked.get("pq") or 0) >= 0.42,
                        "desert_drop_lt_0.05": float(gate.get("desert_2017_2020_drop") or 1) < 0.05,
                        "not_fp_exploded": not bool(vs.get("fp_exploded")),
                    },
                    "observed_merge_pids": eval_pids,
                    "processes_at_gate": (
                        f"{eval_pids} still live" if eval_pids else "hgb eval exited after writing scored official"
                    ),
                    "log": str(LOG.relative_to(ROOT)),
                    "xor": (
                        "not a look unless parent runs hidden-exam on a copied PROMOTE_LOOK; "
                        "do not upload logistic 0.414 HOLD; do not invert rejector_thr from fold-0"
                    ),
                }
                out["promote_needs"]["all_three"] = all(
                    [
                        out["promote_needs"]["pq_ge_0.42"],
                        out["promote_needs"]["desert_drop_lt_0.05"],
                        out["promote_needs"]["not_fp_exploded"],
                    ]
                )
                OUT.write_text(json.dumps(out, indent=2) + "\n")
                print(json.dumps({"status": "GATE_READY", "decision": decision, "pq": locked.get("pq")}, indent=2))
                return
        time.sleep(POLL_S)
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "WATCH_TIMEOUT",
        "elapsed_s": time.time() - started,
        "polls": polls,
        "source": str(GATE.relative_to(ROOT)),
        "gate_exists": GATE.exists(),
        "eval_pids": pgrep("eval_unet_resnet_extra_filter_hgb_merge.py"),
        "mps_owner": live_mps_owner(),
        "recommend_upload": False,
        "did_not_upload": True,
        "do_not_submit": True,
        "watcher_pid": os.getpid(),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    raise SystemExit(f"watch timeout waiting for scored {GATE}")


if __name__ == "__main__":
    main()
