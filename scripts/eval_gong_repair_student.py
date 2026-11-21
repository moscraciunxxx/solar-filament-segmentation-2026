#!/usr/bin/env python3
"""Score the repaired student. Fold 0 once, pycocotools GT. Does not submit.

Alone at confidence 0.35, then 8x3 TTA. IoU 0.5 is frozen. Min votes are
re-picked on folds 1-4 and that rank is in-sample. The gate is single-model
TTA against the existing single-model TTA predictions rescored with the same
rasterizer, plus 0.01. Two more seeds are proposed only, never trained here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.infer import run_yolo_infer
from filament.metric import gt_annotations_to_df, overlap_matrices, pred_to_df, score_pq
from filament.paths import FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR
from filament.rle import rles_to_layers

spec = importlib.util.spec_from_file_location("parent_tta", ROOT / "scripts" / "eval_parent_tta.py")
parent_tta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_tta)
repair = importlib.util.spec_from_file_location("repair", ROOT / "scripts" / "repair_gong_warp.py")
repair_mod = importlib.util.module_from_spec(repair)
repair.loader.exec_module(repair_mod)

OUT = ROOT / "artifacts" / "phaseB_repair"
WEIGHTS = (
    Path.home()
    / ".cache/filament-segmentation-2026/weights/yolo11s_seg_gong_repair_s20260910/weights/score_last.pt"
)
CACHE = FAST_ARTIFACTS / "tta_gong_repair_s20260910"
BASELINE_CSV = FAST_ARTIFACTS / "yolo11s_seg_erode_gt_2048_tta_fold0_pred.csv"
CHAMPION_COCO = 0.4655531057360983
VOTE_GRID = (2, 4, 8)
IOU = 0.5


def _score_frame(files: list[str], pred: pd.DataFrame, views, by_file) -> dict:
    gt = gt_annotations_to_df(gt_records_for_files(views, by_file, files, rasterizer="coco"))
    scored = score_pq(gt, pred)
    denom = scored.tp + 0.5 * scored.fp + 0.5 * scored.fn
    return {
        "rasterizer": "coco",
        "pq": scored.pq,
        "sq": scored.mean_tp_iou,
        "rq": (scored.tp / denom) if denom else 0.0,
        "tp_pairs": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "n_pred": int(len(pred)),
        "mean_tp_iou": scored.mean_tp_iou,
    }


def _unique_and_bins(files: list[str], pred: pd.DataFrame, views, by_file) -> dict:
    gt = gt_annotations_to_df(gt_records_for_files(views, by_file, files, rasterizer="coco"))
    gt_keys = gt["filament_id"].str.split("_", n=1).str[0]
    pred_keys = pred["filament_id"].str.split("_", n=1).str[0]
    areas = []
    records = []
    disks = {}
    for file_name in files:
        gray_path = TRAIN_IMG_DIR / file_name
        import cv2

        gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        disks[Path(file_name).stem] = repair_mod.measure_disk(gray) if gray is not None else None
    for annotator_image in gt_keys.unique():
        image_id = annotator_image.split("-", maxsplit=1)[1]
        gt_rles = gt.loc[gt_keys == annotator_image, "segmentation_rle"].tolist()
        pred_rles = pred.loc[pred_keys == image_id, "segmentation_rle"].tolist()
        gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
        if pred_rles:
            pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
            iou, _dice = overlap_matrices(gt_layers, pred_layers)
            hit = iou > 0.5
            gt_n = hit.sum(dim=1)
            pred_n = hit.sum(dim=0)
        else:
            gt_n = torch.zeros(len(gt_rles))
            pred_n = torch.zeros(0)
            pred_layers = torch.zeros((0, IMAGE_SIZE, IMAGE_SIZE))
        disk = disks.get(image_id)
        for index in range(gt_layers.shape[0]):
            mask = gt_layers[index].numpy() > 0
            geom = _geom(mask, disk)
            areas.append(geom["area"])
            records.append({"kind": "tp" if int(gt_n[index]) > 0 else "fn", **geom})
        if annotator_image == sorted(set(gt_keys[gt_keys.str.endswith("-" + image_id)]))[0]:
            for index in range(pred_layers.shape[0]):
                if int(pred_n[index]) == 0:
                    records.append({"kind": "fp", **_geom(pred_layers[index].numpy() > 0, disk)})
    edges = [float(np.quantile(areas, q)) for q in (1 / 3, 2 / 3)] if areas else [0, 0]
    unique_hits = sum(1 for row in records if row["kind"] == "tp")
    return {
        "unique_gt_hits": unique_hits,
        "unique_gt_misses": sum(1 for row in records if row["kind"] == "fn"),
        "pair_note": "unique hits count GT filaments with at least one IoU>0.5 prediction",
        "area_tercile_edges": edges,
        "by_area": _count_bins(records, "area", edges),
        "by_radius": _count_bins(records, "radius_frac", [0.5, 0.75, 0.9]),
        "radius_note": "fraction of the measured limb on that image. False-positive bins use the first annotator of each image; pair FP in the score counts every annotator.",
    }


def _geom(mask: np.ndarray, disk: dict | None) -> dict:
    ys, xs = np.nonzero(mask)
    area = int(ys.size)
    if area == 0 or disk is None:
        return {"area": area, "radius_frac": None}
    cy = float(ys.mean())
    cx = float(xs.mean())
    dist = ((cy - disk["cy"]) ** 2 + (cx - disk["cx"]) ** 2) ** 0.5
    return {"area": area, "radius_frac": dist / disk["radius"]}


def _count_bins(records: list[dict], key: str, edges: list[float]) -> dict:
    labels = [f"<{edges[0]:.3g}"] + [f"{edges[i]:.3g}-{edges[i+1]:.3g}" for i in range(len(edges) - 1)] + [f">={edges[-1]:.3g}"]
    counts = {kind: [0] * len(labels) for kind in ("tp", "fn", "fp")}
    for row in records:
        value = row.get(key)
        if value is None:
            continue
        index = 0
        while index < len(edges) and value >= edges[index]:
            index += 1
        counts[row["kind"]][index] += 1
    return {"labels": labels, "counts": counts}


def _baseline(views, by_file, files: list[str]) -> dict:
    pred = pd.read_csv(BASELINE_CSV)
    scored = _score_frame(files, pred, views, by_file)
    scored["pred_csv"] = str(BASELINE_CSV)
    scored["role"] = "existing single-model 8x3 TTA, votes frozen at 8, rescored with pycocotools GT"
    return scored


def _cache_one(model, file_name: str) -> dict:
    dest = CACHE / f"{Path(file_name).stem}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    payload = parent_tta._predict_views(model, file_name, image_dir=TRAIN_IMG_DIR, device="mps")
    dest.write_text(json.dumps(payload))
    return payload


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"missing {WEIGHTS}")
    if CACHE.resolve() == (FAST_ARTIFACTS / "tta_parent_2048").resolve():
        raise SystemExit("refusing to write the champion TTA cache")
    CACHE.mkdir(parents=True, exist_ok=True)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    fold0 = sorted(name for name, fold in fold_of.items() if fold == 0)
    tune = sorted(name for name, fold in fold_of.items() if fold in {1, 2, 3, 4})
    baseline = _baseline(views, by_file, fold0)
    (OUT / "single_model_tta_baseline_coco.json").write_text(json.dumps(baseline, indent=2) + "\n")
    print("baseline", json.dumps(baseline), flush=True)

    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    alone_frames = []
    for index, name in enumerate(fold0, start=1):
        rles = run_yolo_infer(model, TRAIN_IMG_DIR / name, conf=0.35, imgsz=2048, merge_gap=0, device="mps")
        alone_frames.append(pred_to_df(Path(name).stem, rles))
        if index == 1 or index % 20 == 0:
            print(f"alone {index}/{len(fold0)}", flush=True)
    alone_pred = pd.concat(alone_frames, ignore_index=True)
    alone_csv = FAST_ARTIFACTS / "yolo11s_seg_gong_repair_s20260910_alone_fold0_pred.csv"
    alone_pred.to_csv(alone_csv, index=False)
    alone = _score_frame(fold0, alone_pred, views, by_file)
    alone["unique"] = _unique_and_bins(fold0, alone_pred, views, by_file)
    alone["pred_csv"] = str(alone_csv)
    (OUT / "student_alone_fold0_coco.json").write_text(json.dumps(alone, indent=2) + "\n")
    print("alone", json.dumps({k: v for k, v in alone.items() if k != "unique"}), flush=True)

    for index, name in enumerate(tune + fold0, start=1):
        _cache_one(model, name)
        if index == 1 or index % 20 == 0:
            print(f"tta_cached {index}/{len(tune) + len(fold0)}", flush=True)
    tune_payloads = {name: json.loads((CACHE / f"{Path(name).stem}.json").read_text()) for name in tune}
    rows = []
    for votes in VOTE_GRID:
        pred = parent_tta._frames(tune, tune_payloads, IOU, votes)
        scored = score_pq(
            gt_annotations_to_df(gt_records_for_files(views, by_file, tune, rasterizer="coco")),
            pred,
        )
        row = {"iou_thr": IOU, "min_votes": votes, "pq": scored.pq, "tp": scored.tp, "fp": scored.fp, "fn": scored.fn, "tune_is_in_sample": True}
        rows.append(row)
        print("tune_in_sample", json.dumps(row), flush=True)
    frozen = parent_tta.select_frozen(rows)
    frozen["tune_is_in_sample"] = True
    (OUT / "student_tta_tune_folds14_insample.json").write_text(json.dumps({"grid": rows, "frozen": frozen}, indent=2) + "\n")
    fold0_payloads = {name: json.loads((CACHE / f"{Path(name).stem}.json").read_text()) for name in fold0}
    pred = parent_tta._frames(fold0, fold0_payloads, IOU, int(frozen["min_votes"]))
    pred_csv = FAST_ARTIFACTS / "yolo11s_seg_gong_repair_s20260910_tta_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    scored = _score_frame(fold0, pred, views, by_file)
    scored["unique"] = _unique_and_bins(fold0, pred, views, by_file)
    scored["frozen_votes"] = frozen
    scored["votes_picked_on"] = "folds 1-4, in-sample"
    scored["baseline_pq"] = baseline["pq"]
    scored["gate"] = baseline["pq"] + 0.01
    scored["passes_gate"] = bool(scored["pq"] > baseline["pq"] + 0.01)
    scored["champion_coco_pq"] = CHAMPION_COCO
    scored["pred_csv"] = str(pred_csv)
    scored["kaggle_submit_called"] = False
    (OUT / "student_tta_fold0_coco.json").write_text(json.dumps(scored, indent=2) + "\n")
    proposal = {
        "passes_gate": scored["passes_gate"],
        "student_tta_pq": scored["pq"],
        "baseline_pq": baseline["pq"],
        "gate": scored["gate"],
        "champion_coco_pq": CHAMPION_COCO,
        "action": "propose two more seeds, do not train them" if scored["passes_gate"] else "stop; do not train more seeds",
        "seeds_proposed": [20260911, 20260912] if scored["passes_gate"] else [],
        "seeds_trained": [],
    }
    (OUT / "seed_proposal.json").write_text(json.dumps(proposal, indent=2) + "\n")
    print(json.dumps(proposal, indent=2), flush=True)


if __name__ == "__main__":
    main()
