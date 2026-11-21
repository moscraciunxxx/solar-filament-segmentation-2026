#!/usr/bin/env python3
"""Diagnosis: 3ch vs ResNet CC cover of agreed large true-miss. XOR extras only.

Read-only. Does not overwrite unet_3ch_disk or unet_resnet34. Never submits.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_unet import decode_instances, infer_prob
from eval_unet_3ch import infer_prob_3ch
from filament.coco_io import grouped_folds, load_coco, rasterize_polygons, semantic_iou
from filament.infer import pick_device
from filament.merge import merge_rles
from filament.metric import overlap_matrices
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import encode_rle, rles_to_layers
from filament.unet import build_fulldisk_unet
from filament.unet_resnet import build_resnet34_unet

SLOT1_CSV = FAST_ARTIFACTS / "yolo_yolo11m_seg_1280_fold0_conf0.35_sz1280_m0_base_m1280_pq.csv"
RESNET = WEIGHTS_DIR / "unet_resnet34" / "best.pt"
CH3 = WEIGHTS_DIR / "unet_3ch_disk" / "best.pt"
AREA_P25 = 642.0
UNIDENT = 3
IOU_THR = 0.5
NEAR_LO = 0.3
COMPARE = 512
THR = 0.5
MIN_AREA = 1034


def other_rater_hit(fil, other_views) -> bool:
    scale = COMPARE / 2048.0
    mine = rasterize_polygons([fil.polygon * scale], size=COMPARE)
    for view in other_views:
        for other in view.filaments:
            other_m = rasterize_polygons([other.polygon * scale], size=COMPARE)
            if semantic_iou(mine, other_m) > IOU_THR:
                return True
    return False


def covers(prob, gt_rle, slot_rles) -> tuple[bool, bool]:
    extras = decode_instances(prob, THR, min_area=MIN_AREA)
    if not extras:
        return False, False
    gt_l = torch.from_numpy(rles_to_layers([gt_rle]))
    ex_l = torch.from_numpy(rles_to_layers(extras))
    if overlap_matrices(gt_l, ex_l)[0].max().item() <= IOU_THR:
        return False, False
    merged, _ = merge_rles(slot_rles, extras)
    mer_l = torch.from_numpy(rles_to_layers(merged))
    kept = overlap_matrices(gt_l, mer_l)[0].max().item() > IOU_THR
    return True, kept


def main() -> None:
    pred_df = pd.read_csv(SLOT1_CSV)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    pred_by_stem: dict[str, list[str]] = defaultdict(list)
    for fid, rle in zip(pred_df["filament_id"], pred_df["segmentation_rle"]):
        pred_by_stem[str(fid).rsplit("_", 1)[0]].append(str(rle))

    agreed = []
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
                agreed.append((file_name, encode_rle(rasterize_polygons([fil.polygon]))))

    need = sorted({fn for fn, _ in agreed})
    device = pick_device("auto")
    r_model = build_resnet34_unet(in_ch=3, pretrained=False)
    r_model.load_state_dict(torch.load(RESNET, map_location=device))
    r_model.to(device).eval()
    c_model = build_fulldisk_unet(in_ch=3, base=32)
    c_model.load_state_dict(torch.load(CH3, map_location=device))
    c_model.to(device).eval()
    r_probs = {fn: infer_prob(r_model, TRAIN_IMG_DIR / fn, 1024, device) for fn in need}
    c_probs = {fn: infer_prob_3ch(c_model, TRAIN_IMG_DIR / fn, 1024, device) for fn in need}

    r_kept = c_kept = both = r_only = c_only = 0
    for file_name, gt_rle in agreed:
        slot = pred_by_stem.get(Path(file_name).stem, [])
        _rc, rk = covers(r_probs[file_name], gt_rle, slot)
        _cc, ck = covers(c_probs[file_name], gt_rle, slot)
        r_kept += int(rk)
        c_kept += int(ck)
        both += int(rk and ck)
        r_only += int(rk and not ck)
        c_only += int(ck and not rk)

    out = {
        "do_not_submit": True,
        "did_not_overwrite_locks": True,
        "n_agreed_large_true_miss": len(agreed),
        "cut": {"thr": THR, "min_area": MIN_AREA},
        "resnet_kept_after_nms": r_kept,
        "ch3_kept_after_nms": c_kept,
        "both": both,
        "resnet_only": r_only,
        "ch3_only": c_only,
        "union_keep": r_kept + c_only,
        "note": "ch3_only extras are a different proposal set. Union of two HOLD keep-sets is still forbidden. Diagnosis only.",
    }
    dest = ARTIFACTS / "ch3_resnet_xor_agreed_true_miss.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
