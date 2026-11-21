#!/usr/bin/env python3
"""Tile-inference add-on for the 3-seed champion. Folds 1-4 tune, fold 0 once.

1024 and 1536 crops, 25% overlap, upscaled to 2048. Three seeds, views
id/h/v. Detections are stored at conf 0.15. A tile instance is added only
when it has votes from at least two seeds, meets the tuned vote and area
cuts, and its IoU with every champion instance is below 0.2.

A second candidate flattens the limb and runs only tiles whose centers sit
in 0.80-0.95 of the limb radius. The tune grid picks one setting. Fold 0
is scored once. Never submits. Never reads test labels.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.coco_io import grouped_folds, load_coco, rasterize_polygons
from filament.flatfield import center_to_limb_flat
from filament.infer import _collect_yolo_instances
from filament.metric import overlap_matrices, pred_to_df, score_pq
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from filament.rle import decode_rle, encode_rle, rles_to_layers

spec = importlib.util.spec_from_file_location("eval_parent_tta", ROOT / "scripts" / "eval_parent_tta.py")
tta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tta)
pool_spec = importlib.util.spec_from_file_location("eval_pool_3s_plus_m", ROOT / "scripts" / "eval_pool_3s_plus_m.py")
pool = importlib.util.module_from_spec(pool_spec)
pool_spec.loader.exec_module(pool)

CHAMPION_PQ = 0.4741684688395591
KEEP_DELTA = 0.01
CONF_FLOOR = 0.15
YOLO_IOU = 0.60
OVERLAP = 0.25
ADD_IOU = 0.2
TILES = (1024, 1536)
CONF_GRID = (0.15, 0.25)
VOTE_GRID = (2, 3)
AREA_GRID = (96, 400)
MIN_SEEDS = 2
TIE_PQ = 0.005
VIEW_NAMES = ("id", "h", "v")
SEEDS = (
    ("s20260910", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt", FAST_ARTIFACTS / "tta_parent_2048"),
    ("s20260911", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260911" / "weights" / "best.pt", FAST_ARTIFACTS / "tta_s20260911"),
    ("s20260912", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260912" / "weights" / "best.pt", FAST_ARTIFACTS / "tta_s20260912"),
)
ROOT_CACHE = FAST_ARTIFACTS / "tile_addon"
CHAMPION_DIR = ROOT_CACHE / "champion"
TUNE_JSON = ARTIFACTS / "tile_addon_tune_folds14.json"
FOLD0_JSON = ARTIFACTS / "tile_addon_fold0.json"


def _origins(tile: int) -> list[int]:
    stride = max(int(round(tile * (1.0 - OVERLAP))), 1)
    origins = list(range(0, IMAGE_SIZE - tile + 1, stride))
    last = IMAGE_SIZE - tile
    if not origins or origins[-1] != last:
        origins.append(last)
    return origins


def _limb_radius(gray: np.ndarray) -> float:
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    cy = (small.shape[0] - 1) / 2.0
    cx = (small.shape[1] - 1) / 2.0
    yy, xx = np.mgrid[0 : small.shape[0], 0 : small.shape[1]]
    rho = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = max(int(min(cy, cx)), 2)
    r_int = np.clip(rho.astype(np.int32), 0, r_max - 1)
    sums = np.bincount(r_int.ravel(), weights=small.ravel().astype(np.float64), minlength=r_max)
    counts = np.maximum(np.bincount(r_int.ravel(), minlength=r_max).astype(np.float64), 1.0)
    profile = sums / counts
    outer = int(r_max * 0.55)
    drop = np.diff(profile)
    limb = outer + int(np.argmin(drop[outer:])) if outer < len(drop) else r_max - 1
    return max(int(limb), 1) * (gray.shape[0] / 512.0)


def _outer(origin: tuple[int, int], tile: int, limb: float) -> bool:
    cy = origin[1] + tile / 2.0
    cx = origin[0] + tile / 2.0
    mid = (IMAGE_SIZE - 1) / 2.0
    rho = math.hypot(cy - mid, cx - mid) / max(limb, 1.0)
    return 0.80 <= rho < 0.95


def _cache_path(kind: str, seed: str, file_name: str) -> Path:
    return ROOT_CACHE / kind / seed / f"{Path(file_name).stem}.json"


def _champion_path(file_name: str) -> Path:
    return CHAMPION_DIR / f"{Path(file_name).stem}.json"


def _read_champion_views(file_name: str) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    stem = Path(file_name).stem
    for name, _, cache in SEEDS:
        one = json.loads((cache / f"{stem}.json").read_text())
        for key, rles in one.items():
            payload[f"{name}_{key}"] = rles
    return payload


def _ensure_champion(files: list[str]) -> None:
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    missing = [fn for fn in files if not _champion_path(fn).exists()]
    print(f"champion missing {len(missing)}/{len(files)}", flush=True)
    t0 = time.perf_counter()
    for i, file_name in enumerate(missing, start=1):
        masks = tta._decode_all(_read_champion_views(file_name))
        groups = pool._groups_fast(masks, 0.5)
        inst = pool.ens._instances_for_votes(masks, groups, 24)
        _champion_path(file_name).write_text(json.dumps([encode_rle(m) for m in inst]))
        if i == 1 or i % 25 == 0 or i == len(missing):
            print(f"champion {i}/{len(missing)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}", flush=True)


def _views():
    return [view for view in tta.VIEWS if view[0] in VIEW_NAMES]


def _predict_np(model, image: np.ndarray):
    results = model.predict(
        source=image,
        conf=CONF_FLOOR,
        iou=YOLO_IOU,
        imgsz=IMAGE_SIZE,
        retina_masks=True,
        augment=False,
        verbose=False,
        device="mps",
    )
    return _collect_yolo_instances(results[0] if results else None)


def _to_full(mask: np.ndarray, origin: tuple[int, int], tile: int) -> np.ndarray:
    small = cv2.resize(mask.astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    x, y = origin
    full[y : y + tile, x : x + tile] = small
    return full


def _infer_file(model, file_name: str, kind: str) -> list[dict]:
    gray = cv2.imread(str(TRAIN_IMG_DIR / file_name), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(file_name)
    if gray.shape[0] != IMAGE_SIZE:
        gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    limb = _limb_radius(gray)
    flat = None
    if kind == "limb":
        flat_f = center_to_limb_flat(gray)
        flat = cv2.cvtColor(np.clip(flat_f * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    views = _views()
    rows = []
    for tile in TILES if kind == "plain" else (1024,):
        for y in _origins(tile):
            for x in _origins(tile):
                if kind == "limb" and not _outer((x, y), tile, limb):
                    continue
                crop = (flat if kind == "limb" else color)[y : y + tile, x : x + tile]
                up = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
                for name, forward, inverse in views:
                    view = forward(up)
                    for mask, score in _predict_np(model, view):
                        restored = inverse(mask)
                        if restored.shape[0] != IMAGE_SIZE:
                            restored = cv2.resize(restored, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                        full = _to_full(restored, (x, y), tile)
                        if int(full.sum()) < 96:
                            continue
                        rows.append(
                            {
                                "rle": encode_rle(full),
                                "score": float(score),
                                "view": name,
                                "x": x,
                                "y": y,
                                "tile": tile,
                            }
                        )
    return rows


def _ensure_tiles(files: list[str], kind: str) -> None:
    from ultralytics import YOLO

    for seed, weights, _cache in SEEDS:
        missing = [fn for fn in files if not _cache_path(kind, seed, fn).exists()]
        print(f"{kind} {seed} missing {len(missing)}/{len(files)}", flush=True)
        if not missing:
            continue
        if not weights.is_file():
            raise SystemExit(f"missing {weights}")
        model = YOLO(str(weights))
        t0 = time.perf_counter()
        for i, file_name in enumerate(missing, start=1):
            rows = _infer_file(model, file_name, kind)
            path = _cache_path(kind, seed, file_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows))
            if i == 1 or i % 10 == 0 or i == len(missing):
                print(
                    f"{kind} {seed} {i}/{len(missing)} dets {len(rows)} "
                    f"elapsed_min {(time.perf_counter() - t0) / 60:.1f}",
                    flush=True,
                )
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _load_dets(kind: str, file_name: str, tile: int, conf: float) -> tuple[list[np.ndarray], list[str]]:
    masks = []
    seeds = []
    for seed, _weights, _cache in SEEDS:
        rows = json.loads(_cache_path(kind, seed, file_name).read_text())
        for row in rows:
            if int(row["tile"]) != tile or float(row["score"]) < conf:
                continue
            masks.append(decode_rle(row["rle"]))
            seeds.append(seed)
    return masks, seeds


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(a.sum()) + int(b.sum()) - inter
    return inter / union if union else 0.0


def _families(settings: list[dict]) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for setting in settings:
        grouped.setdefault((setting["kind"], int(setting["tile"]), float(setting["conf"])), []).append(setting)
    return grouped


def _extras_for_image(file_name: str, settings: list[dict], champions: list[np.ndarray]) -> dict[str, list[str]]:
    from filament.infer import yolo_masks_to_instances

    out = { _key(setting): [] for setting in settings }
    prepared = {}
    for kind, tile, conf in _families(settings):
        masks, seeds = _load_dets(kind, file_name, tile, conf)
        small = [cv2.resize(m, (512, 512), interpolation=cv2.INTER_NEAREST) for m in masks]
        prepared[(kind, tile, conf)] = (masks, seeds, pool._groups_fast(small, 0.5) if masks else [])
    for (kind, tile, conf), group_settings in _families(settings).items():
        masks, seeds, groups = prepared[(kind, tile, conf)]
        merged_groups = []
        for group in groups:
            if len({seeds[k] for k in group}) < MIN_SEEDS:
                continue
            vote = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)
            for k in group:
                vote += masks[k]
            merged = (vote >= max(1, (len(group) + 1) // 2)).astype(np.uint8)
            if any(_iou(merged, champ) >= ADD_IOU for champ in champions):
                continue
            merged_groups.append((len(group), merged))
        for setting in group_settings:
            kept = [mask for votes, mask in merged_groups if votes >= int(setting["votes"]) and int(mask.sum()) >= int(setting["min_area"])]
            out[_key(setting)] = [encode_rle(m) for m in yolo_masks_to_instances(kept, merge_gap=0)]
    return out


def _champions(file_name: str) -> list[np.ndarray]:
    rles = json.loads(_champion_path(file_name).read_text())
    return [decode_rle(rle) for rle in rles]


def _settings() -> list[dict]:
    rows = []
    for tile in TILES:
        for conf in CONF_GRID:
            for votes in VOTE_GRID:
                for area in AREA_GRID:
                    rows.append({"kind": "plain", "tile": tile, "conf": conf, "votes": votes, "min_area": area})
    for conf in CONF_GRID:
        for votes in VOTE_GRID:
            for area in AREA_GRID:
                rows.append({"kind": "limb", "tile": 1024, "conf": conf, "votes": votes, "min_area": area})
    return rows


def _key(setting: dict) -> str:
    return f"{setting['kind']}_t{setting['tile']}_c{setting['conf']}_v{setting['votes']}_a{setting['min_area']}"


def _assemble_path(file_name: str) -> Path:
    return ROOT_CACHE / "assembled" / f"{Path(file_name).stem}.json"


def _frames(files: list[str], settings: list[dict]) -> dict[str, pd.DataFrame]:
    keys = [_key(s) for s in settings]
    buckets = {key: [] for key in keys}
    t0 = time.perf_counter()
    for i, file_name in enumerate(files, start=1):
        path = _assemble_path(file_name)
        champ_rles = json.loads(_champion_path(file_name).read_text())
        if path.exists():
            saved = json.loads(path.read_text())
            extras = saved if all(key in saved for key in keys) else None
        else:
            extras = None
        if extras is None:
            extras = _extras_for_image(file_name, settings, _champions(file_name))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(extras))
            tmp.replace(path)
        stem = Path(file_name).stem
        for key in keys:
            buckets[key].append(pred_to_df(stem, champ_rles + extras[key]))
        if i == 1 or i % 25 == 0:
            print(f"assembled {i}/{len(files)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}", flush=True)
    return {key: pd.concat(frames, ignore_index=True) for key, frames in buckets.items()}


def _select(rows: list[dict]) -> dict:
    best = max(float(r["pq"]) for r in rows)
    band = [r for r in rows if best - float(r["pq"]) <= TIE_PQ]
    chosen = dict(max(band, key=lambda r: (float(r["conf"]), int(r["votes"]), int(r["min_area"]))))
    chosen["selection"] = "highest conf, then votes, then min area, among folds 1-4 PQ within 0.005 of the best"
    chosen["best_tune_pq"] = best
    chosen["n_in_band"] = len(band)
    return chosen


def _hit_ids(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> set[str]:
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_keys = pred_df["filament_id"].str.split("_", n=1).str[0]
    hits = set()
    for annotator_image in gt_keys.unique():
        image_id = annotator_image.split("-", maxsplit=1)[1]
        gt_rows = gt_df.loc[gt_keys == annotator_image]
        pred_rles = pred_df.loc[pred_keys == image_id, "segmentation_rle"].tolist()
        if not len(gt_rows) or not pred_rles:
            continue
        gt_layers = torch.from_numpy(rles_to_layers(gt_rows["segmentation_rle"].tolist()))
        pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
        iou, _ = overlap_matrices(gt_layers, pred_layers)
        flags = (iou > 0.5).any(dim=1).tolist()
        hits.update(fid for fid, flag in zip(gt_rows["filament_id"].tolist(), flags) if flag)
    return hits


def _new_buckets(views, by_file, val_files, new_ids: set[str]) -> dict:
    counts = {"small_lt_400": 0, "faint_contrast_lt_15": 0, "outer_0.80_0.95": 0, "new_tp": len(new_ids)}
    wanted = {}
    for fid in new_ids:
        image_id, index = fid.rsplit("_", 1)
        wanted.setdefault(image_id, set()).add(int(index))
    for file_name in val_files:
        gray = cv2.imread(str(TRAIN_IMG_DIR / file_name), cv2.IMREAD_GRAYSCALE)
        limb = _limb_radius(gray)
        mid = (IMAGE_SIZE - 1) / 2.0
        for image_id in by_file[file_name]:
            if image_id not in wanted:
                continue
            view = views[image_id]
            for idx, filament in enumerate(view.filaments, start=1):
                if idx not in wanted[image_id]:
                    continue
                mask = rasterize_polygons([filament.polygon])
                area = int(mask.sum())
                ys, xs = np.nonzero(mask)
                rho = math.hypot(ys.mean() - mid, xs.mean() - mid) / max(limb, 1.0) if len(xs) else 0.0
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
                ring = (cv2.dilate(mask, kernel) > 0) & (mask == 0)
                contrast = float(gray[ring].mean() - gray[mask > 0].mean()) if int(ring.sum()) >= 20 else 0.0
                if area < 400:
                    counts["small_lt_400"] += 1
                if contrast < 15:
                    counts["faint_contrast_lt_15"] += 1
                if 0.80 <= rho < 0.95:
                    counts["outer_0.80_0.95"] += 1
    return counts


def main() -> None:
    if FOLD0_JSON.exists():
        raise SystemExit(f"fold 0 already scored: {FOLD0_JSON}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    tune_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    if len(tune_files) != 565 or len(val_files) != 142:
        raise SystemExit(f"unexpected split tune {len(tune_files)} val {len(val_files)}")
    files = tune_files + val_files
    _ensure_tiles(files, "plain")
    _ensure_tiles(files, "limb")
    _ensure_champion(files)
    settings = _settings()
    if TUNE_JSON.exists():
        saved = json.loads(TUNE_JSON.read_text())
        tune_rows = saved["grid"]
        frozen = saved["frozen"]
        print("tune_cached", json.dumps(frozen), flush=True)
    else:
        from filament.coco_io import gt_records_for_files
        from filament.metric import gt_annotations_to_df

        gt_tune = gt_annotations_to_df(gt_records_for_files(views, by_file, tune_files))
        preds = _frames(tune_files, settings)
        tune_rows = []
        for setting in settings:
            key = _key(setting)
            scored = score_pq(gt_tune, preds[key])
            row = {
                **setting,
                "pq": scored.pq,
                "tp": scored.tp,
                "fp": scored.fp,
                "fn": scored.fn,
                "n_pred": int(len(preds[key])),
                "mean_tp_iou": scored.mean_tp_iou,
            }
            tune_rows.append(row)
            print("tune", json.dumps(row), flush=True)
        frozen = _select(tune_rows)
        TUNE_JSON.write_text(json.dumps({"grid": tune_rows, "frozen": frozen, "test_labels_used": False}, indent=2) + "\n")
        print("frozen", json.dumps(frozen), flush=True)

    from filament.coco_io import gt_records_for_files
    from filament.metric import gt_annotations_to_df

    gt_val = gt_annotations_to_df(gt_records_for_files(views, by_file, val_files))
    chosen = {k: frozen[k] for k in ("kind", "tile", "conf", "votes", "min_area")}
    pred = _frames(val_files, [chosen])[_key(chosen)]
    pred_csv = FAST_ARTIFACTS / "tile_addon_fold0_pred.csv"
    pred.to_csv(pred_csv, index=False)
    scored = score_pq(gt_val, pred)
    champion_pred = pd.concat(
        [pred_to_df(Path(fn).stem, json.loads(_champion_path(fn).read_text())) for fn in val_files],
        ignore_index=True,
    )
    champ_hits = _hit_ids(gt_val, champion_pred)
    new_hits = _hit_ids(gt_val, pred)
    only_new = new_hits - champ_hits
    buckets = _new_buckets(views, by_file, val_files, only_new)
    kept = bool(scored.pq > CHAMPION_PQ + KEEP_DELTA)
    payload = {
        "family": "tile_addon",
        "frozen": frozen,
        "fold": 0,
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "n_pred": int(len(pred)),
        "mean_tp_iou": scored.mean_tp_iou,
        "champion_pq": CHAMPION_PQ,
        "delta_pq": scored.pq - CHAMPION_PQ,
        "new_tp": len(only_new),
        "lost_tp": len(champ_hits - new_hits),
        "new_tp_buckets": buckets,
        "kept": kept,
        "pred_csv": str(pred_csv),
        "views": list(VIEW_NAMES),
        "test_labels_used": False,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    FOLD0_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print("FOLD0", json.dumps(payload), flush=True)
    print(
        f"DONE tile pq {scored.pq} delta {scored.pq - CHAMPION_PQ} new_tp {len(only_new)} kept {kept}",
        flush=True,
    )


if __name__ == "__main__":
    main()
