#!/usr/bin/env python3
"""CPU year-bin + bar copy after D official exists. Never infers. Never submits.

Implements the five legal_after_d_measurements in artifacts/transfer_gap_after_d.json.
Refuses if official json is missing. Does not invent PQ.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OFFICIAL = ARTIFACTS / "yolo11l_seg_erode_gt_fold0_official.json"
SLOT1 = ARTIFACTS / "slot1_erode_foldne0_lock_fold0_official.json"
C_OFF = ARTIFACTS / "yolo11m_seg_erode_gt_fold0_official.json"
OUT = ARTIFACTS / "transfer_gap_after_d_measured.json"
ERODE_BAR = 0.4365
GAP = 0.1065
RANK10_PUBLIC = 0.55
SLOT2 = 56148889
SLOT1_REF = 56173362
BINS = ("all", "pre-2017", "2017-2020", "2020+")


def _slice(payload: dict, name: str) -> dict | None:
    sl = (payload.get("year_slice") or {}).get(name)
    if not sl:
        return None
    return {k: sl.get(k) for k in ("n_images", "n_pred", "pq", "tp", "fp", "fn", "pred_per_img")}


def _diff(a: dict | None, b: dict | None) -> dict | None:
    if not a or not b:
        return None
    out = {}
    for k in ("pq", "tp", "fp", "fn", "pred_per_img"):
        if a.get(k) is None or b.get(k) is None:
            continue
        out[k] = a[k] - b[k]
    return out


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {
        "written": now,
        "role": "cpu_only_d_year_bin_after_official",
        "skill": "hidden-exam-discipline",
        "cpu_only": True,
        "did_not_train": True,
        "did_not_infer": True,
        "did_not_submit": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "recommend_start": False,
        "next_id": None,
        "shrinks_public_gap": False,
        "can_hit_rank10": False,
        "parent_bar": ERODE_BAR,
        "slot2_stays": SLOT2,
    }
    if not OFFICIAL.exists():
        base.update(
            {
                "status": "waiting_official",
                "official_pq": None,
                "invented_official_pq": False,
                "note": "Official missing. Do not invent PQ. Do not infer.",
            }
        )
        OUT.write_text(json.dumps(base, indent=2) + "\n")
        print(json.dumps(base, indent=2))
        raise SystemExit(0)

    d = json.loads(OFFICIAL.read_text())
    locked = d.get("locked_row") or {}
    pq = float(locked.get("pq", 0.0))
    beats = pq > ERODE_BAR
    forecast = round(pq - GAP, 4)
    slot1 = json.loads(SLOT1.read_text()) if SLOT1.exists() else {}
    cpay = json.loads(C_OFF.read_text()) if C_OFF.exists() else {}
    d_bins = {name: _slice(d, name) for name in BINS}
    slot1_bins = {name: _slice(slot1, name) for name in BINS}
    c_bins = {name: _slice(cpay, name) for name in BINS}
    base.update(
        {
            "status": "scored",
            "official_path": str(OFFICIAL),
            "official_pq": pq,
            "tp": locked.get("tp"),
            "fp": locked.get("fp"),
            "fn": locked.get("fn"),
            "beats_erode_bar": beats,
            "ignore_score_family_PROMOTE_vs_0.38": True,
            "d_year_slice": d_bins,
            "vs_slot1_erode": {name: _diff(d_bins[name], slot1_bins[name]) for name in BINS},
            "vs_c_m_erode_gt": {name: _diff(d_bins[name], c_bins[name]) for name in BINS},
            "forecast_public": forecast,
            "forecast_rule": "official_pq - 0.1065",
            "rank10_public": RANK10_PUBLIC,
            "forecast_hits_rank10": forecast >= RANK10_PUBLIC,
            "upload_now": False,
            "invented_look": False,
        }
    )
    if beats:
        base.update(
            {
                "eligible_one_look": True,
                "pair_if_uploaded": ["new_best_or_56173362", SLOT2],
                "note": (
                    "Official beats 0.4365. Year bins are local diagnosis, not a builder. "
                    "One look eligible, not auto-upload. Climb ≠ certified top-10."
                ),
            }
        )
    else:
        base.update(
            {
                "eligible_one_look": False,
                "remaining_looks": 2,
                "pair_stays": [SLOT1_REF, SLOT2],
                "note": (
                    "Official misses 0.4365. Do not invent a look. "
                    "Year bins explain local mix only. Score D then stop."
                ),
            }
        )
    OUT.write_text(json.dumps(base, indent=2) + "\n")
    print(json.dumps(base, indent=2))


if __name__ == "__main__":
    main()
