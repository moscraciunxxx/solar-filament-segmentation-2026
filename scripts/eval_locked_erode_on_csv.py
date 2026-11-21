#!/usr/bin/env python3
"""Official fold-0: locked erode 3x3x1 on an existing pred CSV.

CPU. No morph sweep. No dest write. No submit. Parent bar is Slot1 erode 0.4365.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_slot1_erode_foldne0_lock import apply_erode, score_frames
from eval_unet_merge import gate_unet_merge
from filament.coco_io import grouped_folds, load_coco
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from score_faint_merge import _stem_frames
from score_family import year_slices

ERODE_BAR = {
    "pq": 0.4365241382777018,
    "tp": 949,
    "fp": 385,
    "fn": 640,
    "mean_tp_iou": 0.672265572279095,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise SystemExit(f"missing {pred_path}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    fold0 = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    ident = _stem_frames(pd.read_csv(pred_path), fold0)
    eroded = apply_erode(ident)
    ident_row = score_frames(fold0, ident, views, by_file)
    erode_row = score_frames(fold0, eroded, views, by_file)
    slices = year_slices(fold0, eroded, views, by_file)
    ident_slices = year_slices(fold0, ident, views, by_file)
    gate = gate_unet_merge(erode_row, ident_row, slices["2017-2020"], ident_slices["2017-2020"])
    vs_bar = {
        "d_pq": erode_row["pq"] - ERODE_BAR["pq"],
        "d_tp": erode_row["tp"] - ERODE_BAR["tp"],
        "beats_erode_bar": erode_row["pq"] > ERODE_BAR["pq"],
    }
    pred_csv = FAST_ARTIFACTS / f"{args.family}_erode3x1_fold0_pred.csv"
    pd.concat(eroded.values(), ignore_index=True).to_csv(pred_csv, index=False)
    official = ARTIFACTS / f"{args.family}_erode3x1_fold0_official.json"
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "family": f"{args.family}_erode3x1",
        "locked_morph": ["erode", 3, 1],
        "source_pred": str(pred_path),
        "identity": ident_row,
        "locked_row": erode_row,
        "vs_erode_bar": vs_bar,
        "year_slice": slices,
        "gate_vs_own_identity": gate,
        "pred_csv": str(pred_csv),
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "slot2_stays": 56148889,
        "parent_bar": 0.4365,
        "recommend_upload": bool(
            gate.get("decision") == "PROMOTE_LOOK" and vs_bar["beats_erode_bar"]
        ),
    }
    official.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"path": str(official), "locked": erode_row, "vs_bar": vs_bar, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
