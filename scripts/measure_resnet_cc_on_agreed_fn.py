#!/usr/bin/env python3
"""Diagnosis: do ResNet CC extras cover agreed large true-miss, and does Slot1 NMS eat them?

Read-only. Does not overwrite unet_resnet34. Does not lock a cut. Never submits.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_unet import decode_instances, infer_prob
from filament.coco_io import grouped_folds, load_coco, rasterize_polygons, semantic_iou
from filament.infer import pick_device
from filament.merge import merge_rles
from filament.metric import overlap_matrices
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import encode_rle, rles_to_layers
from filament.unet_resnet import build_resnet34_unet

SLOT1_CSV = FAST_ARTIFACTS / "yolo_yolo11m_seg_1280_fold0_conf0.35_sz1280_m0_base_m1280_pq.csv"
WEIGHTS = WEIGHTS_DIR / "unet_resnet34" / "best.pt"
AREA_P25 = 642.0
UNIDENT = 3
IOU_THR = 0.5
NEAR_LO = 0.3
COMPARE = 512


def other_rater_hit(fil, other_views) -> bool:
    scale = COMPARE / 2048.0
    mine = rasterize_polygons([fil.polygon * scale], size=COMPARE)
    for view in other_views:
        for other in view.filaments:
            other_m = rasterize_polygons([other.polygon * scale], size=COMPARE)
            if semantic_iou(mine, other_m) > IOU_THR:
                return True
    return False


def main() -> None:
    pred_df = pd.read_csv(SLOT1_CSV)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    pred_by_stem: dict[str, list[str]] = defaultdict(list)
    for fid, rle in zip(pred_df["filament_id"], pred_df["segmentation_rle"]):
        pred_by_stem[str(fid).rsplit("_", 1)[0]].append(str(rle))

    agreed_true = []
    for file_name in val_files:
        stem = Path(file_name).stem
        n_raters = len(by_file[file_name])
        pred_rles = pred_by_stem.get(stem, [])
        pred_layers = (
            torch.from_numpy(rles_to_layers(pred_rles))
            if pred_rles
            else torch.zeros((0, 2048, 2048), dtype=torch.float32)
        )
        for image_id in by_file[file_name]:
            view = views[image_id]
            if not view.filaments:
                continue
            fils = list(view.filaments)
            gt_rles = [encode_rle(rasterize_polygons([fil.polygon])) for fil in fils]
            gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
            row_max = (
                [0.0] * len(fils)
                if pred_layers.shape[0] == 0
                else overlap_matrices(gt_layers, pred_layers)[0].max(dim=1).values.tolist()
            )
            others = [views[oid] for oid in by_file[file_name] if oid != image_id]
            for fil, mx in zip(fils, row_max):
                faint = float(fil.area) <= AREA_P25 or int(fil.category_id) == UNIDENT
                if mx > NEAR_LO or faint:
                    continue
                if n_raters < 2 or not other_rater_hit(fil, others):
                    continue
                agreed_true.append((file_name, fil, encode_rle(rasterize_polygons([fil.polygon]))))

    need = sorted({fn for fn, _, _ in agreed_true})
    device = pick_device("auto")
    model = build_resnet34_unet(in_ch=3, pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.to(device)
    model.eval()
    probs = {fn: infer_prob(model, TRAIN_IMG_DIR / fn, 1024, device) for fn in need}

    cuts = ((0.5, 1034), (0.45, 642), (0.50, 642))
    cut_stats = {}
    for thr, min_area in cuts:
        covered = 0
        nms_eaten = 0
        kept_new = 0
        for file_name, _fil, gt_rle in agreed_true:
            extras = decode_instances(probs[file_name], thr, min_area=min_area)
            if not extras:
                continue
            gt_l = torch.from_numpy(rles_to_layers([gt_rle]))
            ex_l = torch.from_numpy(rles_to_layers(extras))
            iou_gt = overlap_matrices(gt_l, ex_l)[0].max().item()
            if iou_gt <= IOU_THR:
                continue
            covered += 1
            slot = pred_by_stem.get(Path(file_name).stem, [])
            merged, _added = merge_rles(slot, extras)
            mer_l = torch.from_numpy(rles_to_layers(merged))
            if overlap_matrices(gt_l, mer_l)[0].max().item() > IOU_THR:
                kept_new += 1
            else:
                nms_eaten += 1
        cut_stats[f"thr{thr}_area{min_area}"] = {
            "n_gt": len(agreed_true),
            "cc_covers_iou50": covered,
            "nms_eaten": nms_eaten,
            "kept_as_new_after_nms": kept_new,
        }

    out = {
        "do_not_submit": True,
        "did_not_lock_a_cut": True,
        "n_agreed_large_true_miss": len(agreed_true),
        "cuts": cut_stats,
        "note": "Diagnosis only. Do not retune rejector_thr from these fold-0 counts.",
    }
    dest = ARTIFACTS / "resnet_cc_on_agreed_true_miss.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
