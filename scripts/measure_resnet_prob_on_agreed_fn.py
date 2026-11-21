#!/usr/bin/env python3
"""Diagnosis: ResNet@1024 probability on Slot1 agreed large true-miss GTs.

Read-only weights. Does not overwrite unet_resnet34. Does not lock a cut. Never submits.
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

from eval_unet import infer_prob
from filament.coco_io import grouped_folds, load_coco, rasterize_polygons, semantic_iou
from filament.infer import pick_device
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


def _stats(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "frac_mean_ge_0.45": float((arr >= 0.45).mean()),
        "frac_mean_ge_0.50": float((arr >= 0.50).mean()),
    }


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"missing {WEIGHTS}")
    pred_df = pd.read_csv(SLOT1_CSV)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    pred_by_stem: dict[str, list[str]] = defaultdict(list)
    for fid, rle in zip(pred_df["filament_id"], pred_df["segmentation_rle"]):
        pred_by_stem[str(fid).rsplit("_", 1)[0]].append(str(rle))

    agreed_true = []
    tp_refs = []
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
            if pred_layers.shape[0] == 0:
                row_max = [0.0] * len(fils)
            else:
                iou, _ = overlap_matrices(gt_layers, pred_layers)
                row_max = iou.max(dim=1).values.tolist()
            others = [views[oid] for oid in by_file[file_name] if oid != image_id]
            for fil, mx in zip(fils, row_max):
                faint = float(fil.area) <= AREA_P25 or int(fil.category_id) == UNIDENT
                if mx > IOU_THR and not faint and len(tp_refs) < 120:
                    tp_refs.append((file_name, fil))
                    continue
                if mx > NEAR_LO or faint:
                    continue
                if n_raters < 2 or not other_rater_hit(fil, others):
                    continue
                agreed_true.append((file_name, fil))

    need = sorted({fn for fn, _ in agreed_true} | {fn for fn, _ in tp_refs})
    device = pick_device("auto")
    model = build_resnet34_unet(in_ch=3, pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.to(device)
    model.eval()
    probs = {}
    for file_name in need:
        probs[file_name] = infer_prob(model, TRAIN_IMG_DIR / file_name, 1024, device)

    def mean_on(fil, file_name: str) -> float:
        mask = rasterize_polygons([fil.polygon]).astype(bool)
        p = probs[file_name][mask]
        return float(p.mean()) if p.size else 0.0

    fn_means = [mean_on(fil, fn) for fn, fil in agreed_true]
    tp_means = [mean_on(fil, fn) for fn, fil in tp_refs]
    bg = []
    for file_name in need[:40]:
        p = probs[file_name]
        disk = p > 0
        rng = np.random.default_rng(20260910)
        ys, xs = np.where(disk)
        if ys.size == 0:
            continue
        idx = rng.choice(ys.size, size=min(2000, ys.size), replace=False)
        bg.extend(p[ys[idx], xs[idx]].tolist())

    out = {
        "do_not_submit": True,
        "did_not_overwrite_unet_resnet34": True,
        "did_not_lock_a_cut": True,
        "weights": str(WEIGHTS),
        "n_agreed_large_true_miss": len(agreed_true),
        "n_tp_refs": len(tp_refs),
        "n_images_inferred": len(need),
        "agreed_true_miss_mean_prob": _stats(fn_means),
        "slot1_tp_large_mean_prob": _stats(tp_means),
        "background_sample": _stats(bg),
        "note": (
            "If agreed-FN mean prob is near TP, a decode leftover may exist. "
            "If near background, the encoder is blind. Do not pick thr from this file."
        ),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS / "resnet_prob_on_agreed_true_miss.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
