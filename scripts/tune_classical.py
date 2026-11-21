#!/usr/bin/env python3
"""Small PQ grid on a grouped holdout. Picks the config that raises PQ, not Dice."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.classical import ClassicalConfig, detect_classical
from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import masks_to_rles
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR
from filament.preprocess import read_gray


def main() -> None:
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)[:48]
    gt_df = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))

    grid = list(
        itertools.product(
            (90.0, 92.5, 94.5),
            (28.0, 36.0, 48.0),
            (96, 160),
            (7, 11),
        )
    )
    rows = []
    for dark_p, sigma, min_area, merge_gap in tqdm(grid, desc="grid"):
        cfg = ClassicalConfig(
            dark_percentile=dark_p,
            blur_sigma=sigma,
            min_area=min_area,
            merge_gap=merge_gap,
        )
        frames = []
        for file_name in val_files:
            image = read_gray(TRAIN_IMG_DIR / file_name)
            rles = masks_to_rles(detect_classical(image, cfg))
            frames.append(pred_to_df(Path(file_name).stem, rles))
        pred_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["filament_id", "segmentation_rle"]
        )
        br = score_pq(gt_df, pred_df)
        rows.append(
            {
                "dark_percentile": dark_p,
                "blur_sigma": sigma,
                "min_area": min_area,
                "merge_gap": merge_gap,
                "pq": br.pq,
                "tp": br.tp,
                "fp": br.fp,
                "fn": br.fn,
                "mean_tp_iou": br.mean_tp_iou,
                "n_pred": int(len(pred_df)),
            }
        )
        print(rows[-1])

    rows.sort(key=lambda r: r["pq"], reverse=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "classical_grid.json").write_text(json.dumps(rows, indent=2))
    print("BEST", rows[0])


if __name__ == "__main__":
    main()
