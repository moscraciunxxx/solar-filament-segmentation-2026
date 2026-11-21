#!/usr/bin/env python3
"""Parent-weight TTA on test JPEGs only.

8 geometric views x scales 1792/2048/2304. Fusion IoU 0.5, min votes 8.
Device is CPU so the fold-1-4 ensemble can keep MPS. Does not submit.
Does not open any annotation JSON.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.metric import pred_to_df
from filament.paths import FAST_ARTIFACTS, FAST_SUBMISSION_DIR, TEST_IMG_DIR

spec = importlib.util.spec_from_file_location("eval_parent_tta", ROOT / "scripts" / "eval_parent_tta.py")
tta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tta)

OUT_NAME = "submission_yolo11s_erode_gt_2048_tta8x3_v8.csv"
PARENT_CSV = FAST_SUBMISSION_DIR / "submission_yolo11s_erode_gt_2048_conf035.csv"
CACHE = FAST_ARTIFACTS / "tta_parent_2048_test"
IOU = 0.5
VOTES = 8


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


def _cache_one(model, path: Path) -> dict:
    dest = CACHE / f"{path.stem}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    payload = tta._predict_views(model, path.name, image_dir=TEST_IMG_DIR, device="cpu")
    dest.write_text(json.dumps(payload))
    return payload


def _check(df: pd.DataFrame, images: list[Path]) -> dict:
    parent = pd.read_csv(PARENT_CSV)
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
    report = {
        "parent_rows": int(len(parent)),
        "tta_rows": int(len(df)),
        "parent_columns": list(parent.columns),
        "tta_columns": list(df.columns),
        "test_jpegs": len(images),
        "tta_images_with_preds": len(id_stems),
        "parent_images_with_preds": int(parent["filament_id"].astype(str).str.rsplit("_", n=1).str[0].nunique()),
        "format_match": True,
        "row_count_equal": int(len(df)) == int(len(parent)),
        "test_labels_opened": False,
        "magfilo_opened": False,
    }
    return report


def main() -> None:
    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    _assert_test_jpegs(images)
    if not tta.WEIGHTS.exists():
        raise SystemExit(f"missing {tta.WEIGHTS}")
    CACHE.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    model = YOLO(str(tta.WEIGHTS))
    t0 = time.perf_counter()
    payloads = {}
    for i, path in enumerate(images, start=1):
        payloads[path.name] = _cache_one(model, path)
        if i == 1 or i % 10 == 0:
            print(f"test_cached {i}/{len(images)} elapsed_min {(time.perf_counter() - t0) / 60:.1f}", flush=True)
    del model
    frames = []
    for path in images:
        masks = tta.fuse_masks(tta._decode_all(payloads[path.name]), IOU, VOTES)
        if not masks:
            continue
        from filament.rle import encode_rle

        frames.append(pred_to_df(path.stem, [encode_rle(m) for m in masks]))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    report = _check(df, images)
    FAST_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out = FAST_SUBMISSION_DIR / OUT_NAME
    df.to_csv(out, index=False)
    report["path"] = str(out)
    report["weights"] = str(tta.WEIGHTS)
    report["device"] = "cpu"
    report["scales"] = list(tta.SCALES)
    report["n_geom_views"] = len(tta.VIEWS)
    report["iou_thr"] = IOU
    report["min_votes"] = VOTES
    report["kaggle_submit_called"] = False
    (FAST_ARTIFACTS / f"{out.stem}_meta.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"DONE csv {out} rows {report['tta_rows']}", flush=True)


if __name__ == "__main__":
    main()
