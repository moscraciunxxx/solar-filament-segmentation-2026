#!/usr/bin/env python3
"""Fail-closed Wave13-D finish verdict. CPU only. Never infers. Never submits.

Reads official fold-0 PQ if present. Parent bar is locked erode 0.4365.
If official is missing, writes waiting. If pq <= bar, refuses a look.
If pq > bar, marks one look eligible but still do_not_submit.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OFFICIAL = ARTIFACTS / "yolo11l_seg_erode_gt_fold0_official.json"
PARENT_BAR_FILE = ARTIFACTS / "yolo11l_erode_gt_parent_bar.json"
OUT = ARTIFACTS / "wave13_d_finish_verdict.json"
ERODE_BAR = 0.4365
LOCKED_CONF = 0.35
SLOT2 = 56148889
SLOT1 = 56173362
GAP = 0.1065


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    verdict = {
        "written": now,
        "family": "yolo11l_seg_erode_gt",
        "skill": "hidden-exam-discipline",
        "cpu_only": True,
        "did_not_train": True,
        "did_not_infer": True,
        "did_not_submit": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "parent_bar": ERODE_BAR,
        "locked_conf_a_priori": LOCKED_CONF,
        "slot2_stays": SLOT2,
        "locked_pair_if_no_look": [SLOT1, SLOT2],
        "certified_top10": False,
        "climb_look_is_not_certified_top10": True,
        "forecast_rule": "official_pq - 0.1065",
        "observed_gap": GAP,
        "do_not_auto_apply_test_time_erode": True,
        "hidden_exam_finish": "artifacts/hidden_exam_wave13_d_finish.json",
        "do_not_upload_wave12_0.3987": True,
        "do_not_upload_wave12_0.4353": True,
        "do_not_upload_wave12_0.4234": True,
        "do_not_upload_0.4352": True,
        "do_not_upload_0.44033": True,
    }
    if not OFFICIAL.exists():
        verdict.update(
            {
                "status": "waiting_official",
                "official_pq": None,
                "upload_now": False,
                "invented_look": False,
                "remaining_looks": 2,
                "note": "Train/watch still own the score path. Do not invent PQ.",
            }
        )
        OUT.write_text(json.dumps(verdict, indent=2) + "\n")
        print(json.dumps(verdict, indent=2))
        return

    payload = json.loads(OFFICIAL.read_text())
    locked = payload.get("locked_row") or {}
    pq = float(locked.get("pq", 0.0))
    beats = pq > ERODE_BAR
    parent_bar_payload = None
    if PARENT_BAR_FILE.exists():
        parent_bar_payload = json.loads(PARENT_BAR_FILE.read_text())
    verdict.update(
        {
            "status": "scored",
            "official_path": str(OFFICIAL),
            "official_pq": pq,
            "tp": locked.get("tp"),
            "fp": locked.get("fp"),
            "fn": locked.get("fn"),
            "beats_erode_bar": beats,
            "score_family_gate": payload.get("gate"),
            "ignore_score_family_PROMOTE_vs_0.38": True,
            "parent_bar_file": parent_bar_payload,
            "forecast_public": round(pq - GAP, 4),
            "upload_now": False,
            "invented_look": False,
        }
    )
    if beats:
        verdict.update(
            {
                "eligible_one_look": True,
                "looks_if_parent_uploads": 1,
                "remaining_looks_if_uploaded": 1,
                "pair_if_uploaded": ["new_best_or_56173362", SLOT2],
                "survivor_if_public_lucky": SLOT2,
                "parent_must_still_run_hidden_exam": True,
                "note": (
                    "Official beats 0.4365. One look eligible, not auto-upload. "
                    "Forecast public = pq - 0.1065. Climb ≠ certified top-10. "
                    "Slot 2 stays year-shift. No test-time erode."
                ),
            }
        )
    else:
        verdict.update(
            {
                "eligible_one_look": False,
                "remaining_looks": 2,
                "pair_stays": [SLOT1, SLOT2],
                "next_id": None,
                "recommend_start": False,
                "note": (
                    "Official misses parent bar 0.4365. Do not invent a look. "
                    "Do not upload wave12 0.3987 / 0.4353 / 0.4234, "
                    "faint+erode 0.4352, or erode+ctx 0.44033. remaining_looks stay 2."
                ),
            }
        )
    OUT.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
