#!/usr/bin/env python3
"""Apply locked erode 3x3 x1 to the frozen Slot1 test CSV. CPU. Never submits."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_slot1_nearmiss_morph import morph_rles
from filament.paths import ARTIFACTS, FAST_SUBMISSION_DIR

SRC = FAST_SUBMISSION_DIR / "submission_yolo11m_1280_conf035.csv"
OUT = FAST_SUBMISSION_DIR / "submission_slot1_erode_3x1_PROMOTE.csv"
META = ARTIFACTS / "submission_slot1_erode_3x1_PROMOTE_meta.json"
OFFICIAL = ARTIFACTS / "slot1_erode_foldne0_lock_fold0_official.json"


def main() -> None:
    gate = json.loads(OFFICIAL.read_text())
    if gate.get("gate", {}).get("decision") != "PROMOTE_LOOK":
        raise SystemExit(f"refusing: gate is {gate.get('gate')}")
    if not gate.get("foldne0", {}).get("lock_ok"):
        raise SystemExit("refusing: fold!=0 lock_ok is false")
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    df = pd.read_csv(SRC)
    rows = []
    for fid, g in df.groupby(df["filament_id"].str.rsplit("_", n=1).str[0], sort=False):
        rles = g["segmentation_rle"].astype(str).tolist()
        grown = morph_rles(rles, "erode", 3, 1)
        for i, rle in enumerate(grown, start=1):
            rows.append({"filament_id": f"{fid}_{i}", "segmentation_rle": rle})
    out_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    FAST_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    meta = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "src": str(SRC),
        "path": str(OUT),
        "n_src_rows": int(len(df)),
        "n_out_rows": int(len(out_df)),
        "op": "erode",
        "k": 3,
        "iters": 1,
        "kaggle_submit_called": False,
        "do_not_submit_from_this_script": True,
        "leakage": "test JPEGs only via locked Slot1 CSV; no MAGFiLO public-test labels",
        "year_shift_stays": 56148889,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"path": str(OUT), "n_src": len(df), "n_out": len(out_df)}, indent=2))


if __name__ == "__main__":
    main()
