#!/usr/bin/env python3
"""CPU replay: v1 locked merge extras filtered by min area. Never trains. Never submits.

Uses saved Slot-1 + merged pred CSVs. No U-Net, no MPS, no test labels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import decode_rle
from score_family import year_slices
from score_faint_merge import SLOT1_PRED_CSV, _stem_frames

MERGE_PRED = FAST_ARTIFACTS / "unet_slot1_merge_fold0_thr0.50_pred.csv"
MIN_AREAS = (64, 200, 642, 1034)


def extra_rles(slot_rles: list[str], merged_rles: list[str]) -> list[str]:
    slot_set = set(slot_rles)
    return [r for r in merged_rles if r not in slot_set]


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    slot1_frames = _stem_frames(pd.read_csv(SLOT1_PRED_CSV), val_files)
    merge_frames = _stem_frames(pd.read_csv(MERGE_PRED), val_files)
    slot1_df = pd.concat(slot1_frames.values(), ignore_index=True)
    slot1_br = score_pq(gt_df, slot1_df)
    slot1_row = {"pq": slot1_br.pq, "tp": slot1_br.tp, "fp": slot1_br.fp, "fn": slot1_br.fn}
    slot1_slices = year_slices(val_files, slot1_frames, views, by_file)

    extras_area: list[int] = []
    rows = []
    for min_area in MIN_AREAS:
        frames = {}
        n_kept = 0
        n_dropped = 0
        for file_name in val_files:
            slot_rles = slot1_frames[file_name]["segmentation_rle"].astype(str).tolist()
            merged_rles = merge_frames[file_name]["segmentation_rle"].astype(str).tolist()
            extras = extra_rles(slot_rles, merged_rles)
            kept = []
            for rle in extras:
                area = int(decode_rle(rle).sum())
                if min_area == MIN_AREAS[0]:
                    extras_area.append(area)
                if area >= min_area:
                    kept.append(rle)
                    n_kept += 1
                else:
                    n_dropped += 1
            frames[file_name] = pred_to_df(Path(file_name).stem, slot_rles + kept)
        pred_df = pd.concat(frames.values(), ignore_index=True)
        br = score_pq(gt_df, pred_df)
        slices = year_slices(val_files, frames, views, by_file)
        desert = slices["2017-2020"]
        desert_drop = float(slot1_slices["2017-2020"]["pq"]) - float(desert["pq"])
        fp_exploded = int(br.fp) >= 2 * int(slot1_br.fp)
        rows.append(
            {
                "min_area": min_area,
                "pq": br.pq,
                "tp": br.tp,
                "fp": br.fp,
                "fn": br.fn,
                "d_pq": br.pq - slot1_br.pq,
                "d_tp": br.tp - slot1_br.tp,
                "n_extras_kept": n_kept,
                "n_extras_dropped": n_dropped,
                "fp_exploded": fp_exploded,
                "desert_2017_2020_drop": desert_drop,
                "desert_pq": desert["pq"],
            }
        )
        print(rows[-1], flush=True)

    extras_area.sort()
    n = len(extras_area)
    out = {
        "written": "2026-09-11T00:46Z",
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "source_merge_pred": str(MERGE_PRED),
        "slot1": slot1_row,
        "n_extras_at_min64": n,
        "extra_area_quartiles": {
            "q1": extras_area[n // 4] if n else None,
            "q2": extras_area[n // 2] if n else None,
            "q3": extras_area[(3 * n) // 4] if n else None,
            "min": extras_area[0] if n else None,
            "max": extras_area[-1] if n else None,
        },
        "replays": rows,
        "note": "CPU RLE filter of the already-locked v1 merge. Not a public invert. ResNet merge should use min_area=642 if 642 kills the hose without wiping d_tp.",
    }
    dest = ARTIFACTS / "unet_v1_minarea_replay.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"path": str(dest), "n_extras": n}, indent=2))


if __name__ == "__main__":
    main()
