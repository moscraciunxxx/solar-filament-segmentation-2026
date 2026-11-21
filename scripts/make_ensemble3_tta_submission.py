#!/usr/bin/env python3
"""Three-seed parent-recipe TTA on test JPEGs only.

Seeds 20260910, 20260911, 20260912.
8 geometric views x scales 1792/2048/2304 (72 voters).
Frozen fusion from folds 1-4: IoU 0.5, min votes 24.
Test JPEGs only. Does not open any annotation JSON. Does not submit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.infer import yolo_masks_to_instances
from filament.metric import pred_to_df
from filament.paths import FAST_ARTIFACTS, FAST_SUBMISSION_DIR, IMAGE_SIZE, TEST_IMG_DIR, WEIGHTS_DIR
from filament.rle import encode_rle

spec = importlib.util.spec_from_file_location("eval_parent_tta", ROOT / "scripts" / "eval_parent_tta.py")
tta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tta)

OUT_NAME = "submission_yolo11s_erode_gt_2048_ens3_tta8x3_v24.csv"
PARENT_CSV = FAST_SUBMISSION_DIR / "submission_yolo11s_erode_gt_2048_conf035.csv"
SINGLE_TTA_CSV = FAST_SUBMISSION_DIR / "submission_yolo11s_erode_gt_2048_tta8x3_v8.csv"
IOU = 0.5
VOTES = 24

MODELS = (
    (
        "s20260910",
        WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt",
        FAST_ARTIFACTS / "tta_parent_2048_test",
    ),
    (
        "s20260911",
        WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260911" / "weights" / "best.pt",
        FAST_ARTIFACTS / "tta_s20260911_test",
    ),
    (
        "s20260912",
        WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260912" / "weights" / "best.pt",
        FAST_ARTIFACTS / "tta_s20260912_test",
    ),
)


def _assert_test_jpegs(images: list[Path]) -> None:
    if not images:
        raise SystemExit(f"no test JPEGs in {TEST_IMG_DIR}")
    root = TEST_IMG_DIR.resolve()
    for path in images:
        resolved = path.resolve()
        if resolved.parent != root or resolved.suffix.lower() != ".jpeg":
            raise SystemExit(f"refusing non-test-jpeg path {path}")
        if "magfilo" in resolved.name.lower() or resolved.suffix.lower() == ".json":
            raise SystemExit(f"refusing label-like path {path}")


def _ensure_test_cache(model_name: str, weights: Path, cache: Path, images: list[Path], device: str = "mps") -> None:
    if not weights.exists():
        raise SystemExit(f"missing {weights}")
    cache.mkdir(parents=True, exist_ok=True)
    missing = [p for p in images if not (cache / f"{p.stem}.json").exists()]
    print(f"{model_name} test missing {len(missing)}/{len(images)}", flush=True)
    if not missing:
        return
    from ultralytics import YOLO

    model = YOLO(str(weights))
    t0 = time.perf_counter()
    for i, path in enumerate(missing, start=1):
        dest = cache / f"{path.stem}.json"
        payload = tta._predict_views(model, path.name, image_dir=TEST_IMG_DIR, device=device)
        dest.write_text(json.dumps(payload))
        if i == 1 or i % 10 == 0 or i == len(missing):
            print(
                f"{model_name} test_inferred {i}/{len(missing)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}",
                flush=True,
            )
    del model


def _read_merged_test(stem: str) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    for name, _, cache in MODELS:
        one = json.loads((cache / f"{stem}.json").read_text())
        for key, rles in one.items():
            payload[f"{name}_{key}"] = rles
    return payload


def fuse_masks_fast(masks: list[np.ndarray], iou_thr: float, min_votes: int) -> list[np.ndarray]:
    if not masks:
        return []
    boxes = [tta._box(m) for m in masks]
    areas = [int(m.sum()) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    used = [False] * len(masks)
    out: list[np.ndarray] = []
    for i in order:
        if used[i] or boxes[i] is None:
            continue
        group = [i]
        used[i] = True
        ax0, ay0, ax1, ay1 = boxes[i]
        ai = areas[i]
        mi = masks[i]
        for j in order:
            if used[j] or boxes[j] is None:
                continue
            bx0, by0, bx1, by1 = boxes[j]
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            ix0 = max(ax0, bx0)
            iy0 = max(ay0, by0)
            ix1 = min(ax1, bx1) + 1
            iy1 = min(ay1, by1) + 1
            inter = int(np.logical_and(mi[iy0:iy1, ix0:ix1], masks[j][iy0:iy1, ix0:ix1]).sum())
            union = ai + areas[j] - inter
            if union and (inter / union) >= iou_thr:
                group.append(j)
                used[j] = True
        if len(group) < min_votes:
            continue
        vote = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for k in group:
            vote += masks[k]
        merged = (vote >= max(1, (len(group) + 1) // 2)).astype(np.uint8)
        if int(merged.sum()) >= 96:
            out.append(merged)
    return yolo_masks_to_instances(out, merge_gap=0)


def _check(df: pd.DataFrame, images: list[Path]) -> dict:
    parent = pd.read_csv(PARENT_CSV)
    single_tta = pd.read_csv(SINGLE_TTA_CSV) if SINGLE_TTA_CSV.exists() else None
    if list(df.columns) != list(parent.columns) or list(df.columns) != ["filament_id", "segmentation_rle"]:
        raise SystemExit(f"column mismatch {list(df.columns)} vs {list(parent.columns)}")
    stems = {p.stem for p in images}
    ids = df["filament_id"].astype(str)
    if not ids.str.contains(r"_\d+$").all():
        raise SystemExit("filament_id is not stem_index")
    id_stems = set(ids.str.rsplit("_", n=1).str[0])
    extra = id_stems - stems
    if extra:
        raise SystemExit(f"predictions for non-test stems: {sorted(extra)[:5]}")
    if df["segmentation_rle"].isna().any() or (df["segmentation_rle"].astype(str).str.len() == 0).any():
        raise SystemExit("empty RLE")
    return {
        "parent_rows": int(len(parent)),
        "single_tta_rows": int(len(single_tta)) if single_tta is not None else None,
        "ens3_tta_rows": int(len(df)),
        "parent_columns": list(parent.columns),
        "ens3_tta_columns": list(df.columns),
        "test_jpegs": len(images),
        "ens3_images_with_preds": len(id_stems),
        "parent_images_with_preds": int(parent["filament_id"].astype(str).str.rsplit("_", n=1).str[0].nunique()),
        "format_match": True,
        "test_labels_opened": False,
        "magfilo_opened": False,
    }


def main() -> None:
    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    _assert_test_jpegs(images)
    for name, weights, cache in MODELS:
        _ensure_test_cache(name, weights, cache, images, device="mps")

    t0 = time.perf_counter()
    frames = []
    for i, path in enumerate(images, start=1):
        payload = _read_merged_test(path.stem)
        masks = fuse_masks_fast(tta._decode_all(payload), IOU, VOTES)
        if masks:
            frames.append(pred_to_df(path.stem, [encode_rle(m) for m in masks]))
        if i == 1 or i % 25 == 0 or i == len(images):
            print(f"test_fused {i}/{len(images)} elapsed_min {(time.perf_counter() - t0) / 60:.2f}", flush=True)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    report = _check(df, images)
    FAST_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out = FAST_SUBMISSION_DIR / OUT_NAME
    df.to_csv(out, index=False)
    report.update(
        {
            "path": str(out),
            "seeds": [20260910, 20260911, 20260912],
            "device": "mps",
            "scales": list(tta.SCALES),
            "n_geom_views": len(tta.VIEWS),
            "vote_pool": 72,
            "iou_thr": IOU,
            "min_votes": VOTES,
            "fold0_pq": 0.4741684688395591,
            "kaggle_submit_called": False,
        }
    )
    (FAST_ARTIFACTS / f"{out.stem}_meta.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"DONE csv {out} rows {report['ens3_tta_rows']}", flush=True)


if __name__ == "__main__":
    main()
