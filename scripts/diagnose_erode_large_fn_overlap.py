#!/usr/bin/env python3
"""CPU: how many Slot1-erode large-identifiable FNs other pred CSVs already hit.

Diagnosis only. Existing fold-0 pred CSVs. No train, no MPS, no submit.
Matching: IoU>0.5, all annotator-images on fold-0 JPEGs. Not a leftover-to-start.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_slot1_large_fn_for_unet import IOU_THR, is_large_identifiable
from filament.coco_io import grouped_folds, load_coco, rasterize_polygons
from filament.metric import overlap_matrices
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED
from filament.rle import encode_rle, rles_to_layers

ERODE_CSV = FAST_ARTIFACTS / "slot1_erode_3x1_fold0_pred.csv"
OUT = ARTIFACTS / "erode_large_fn_overlap_after_d.json"
CANDIDATES = [
    ("D_yolo11l_seg_erode_gt", FAST_ARTIFACTS / "yolo11l_seg_erode_gt_fold0_conf0.35_pred.csv"),
    ("C_yolo11m_seg_erode_gt", FAST_ARTIFACTS / "yolo11m_seg_erode_gt_fold0_conf0.35_pred.csv"),
    ("B_yolo11l_erode3x1", FAST_ARTIFACTS / "yolo11l_seg_1280_erode3x1_fold0_pred.csv"),
    ("A_yolo11l_identity", FAST_ARTIFACTS / "yolo11l_seg_1280_fold0_conf0.35_pred.csv"),
    ("faint_merged_erode", FAST_ARTIFACTS / "yolo11s_seg_faint_merged_erode3x1_fold0_pred.csv"),
    ("unet_resnet_slot1_merge", FAST_ARTIFACTS / "unet_slot1_merge_fold0_thr0.50_pred.csv"),
]


def _by_stem(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    out: dict[str, list[str]] = defaultdict(list)
    for fid, rle in zip(df["filament_id"], df["segmentation_rle"]):
        out[str(fid).rsplit("_", 1)[0]].append(str(rle))
    return out


def _layers(rles: list[str]) -> torch.Tensor:
    if not rles:
        return torch.zeros((0, 2048, 2048), dtype=torch.float32)
    return torch.from_numpy(rles_to_layers(rles))


def main() -> None:
    if not ERODE_CSV.exists():
        raise SystemExit(f"missing {ERODE_CSV}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    erode = _by_stem(ERODE_CSV)
    others = []
    for name, path in CANDIDATES:
        if path.exists():
            others.append((name, str(path), _by_stem(path)))
        else:
            others.append((name, str(path), None))

    n_erode_fn = 0
    n_large = 0
    hits = {name: 0 for name, _, _ in others if True}
    for file_name in val_files:
        stem = Path(file_name).stem
        erode_layers = _layers(erode.get(stem, []))
        other_layers = {
            name: _layers(mp.get(stem, [])) if mp is not None else None
            for name, _, mp in others
        }
        for image_id in by_file[file_name]:
            view = views[image_id]
            if not view.filaments:
                continue
            fils = list(view.filaments)
            gt_rles = [encode_rle(rasterize_polygons([fil.polygon])) for fil in fils]
            gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
            if erode_layers.shape[0] == 0:
                erode_max = torch.zeros(len(fils))
            else:
                iou, _ = overlap_matrices(gt_layers, erode_layers)
                erode_max = iou.max(dim=1).values
            other_max = {}
            for name, layers in other_layers.items():
                if layers is None:
                    continue
                if layers.shape[0] == 0:
                    other_max[name] = torch.zeros(len(fils))
                else:
                    oi, _ = overlap_matrices(gt_layers, layers)
                    other_max[name] = oi.max(dim=1).values
            for i, fil in enumerate(fils):
                if float(erode_max[i]) > IOU_THR:
                    continue
                n_erode_fn += 1
                if not is_large_identifiable(fil):
                    continue
                n_large += 1
                for name, mx in other_max.items():
                    if float(mx[i]) > IOU_THR:
                        hits[name] += 1

    missing = [name for name, path, mp in others if mp is None]
    rows = []
    for name, path, mp in others:
        if mp is None:
            rows.append({"id": name, "csv": path, "missing": True})
            continue
        covered = hits[name]
        rows.append(
            {
                "id": name,
                "csv": path,
                "missing": False,
                "large_fn_hit_iou_gt_0.5": covered,
                "uncovered_if_this_csv_only": n_large - covered,
                "frac": (covered / n_large) if n_large else None,
            }
        )

    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "cpu_only_erode_large_fn_overlap",
        "skill": "hidden-exam-discipline",
        "diagnosis_not_a_builder": True,
        "did_not_train": True,
        "did_not_start_mps": True,
        "did_not_submit": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
        "recommend_start": False,
        "next_id": None,
        "slot1": "erode_3x1",
        "erode_fn_total": n_erode_fn,
        "erode_large_identifiable_fn": n_large,
        "rule": "area>642 AND category_id!=3; FN vs Slot1 erode IoU<=0.5",
        "candidates": rows,
        "missing_csvs": missing,
        "why": (
            "Identity 274 pool was vs old m@1280 Slot1, not erode. This remasures vs "
            "public-best erode. Hits are diagnosis; OR extras on public-best stays vetoed."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"wrote": str(OUT), "erode_fn": n_erode_fn, "large": n_large, "hits": hits}, indent=2))


if __name__ == "__main__":
    main()
