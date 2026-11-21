#!/usr/bin/env python3
"""Decode the 3-seed ens3 TTA, 8 views x 3 scales, 24 votes.

Reads the stored view caches. Does not run the network, does not train,
and does not overwrite the frozen submission or the fold-0 prediction file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from filament.coco_io import grouped_folds, gt_records_for_files, load_coco
from filament.metric import gt_annotations_to_df, pred_to_df, score_pq
from filament.paths import FAST_ARTIFACTS, SEED, TEST_IMG_DIR
from filament.rle import encode_rle

PACKAGE = PROJECT_ROOT / "artifacts" / "final_package"
WEIGHTS = PACKAGE / "weights"
REPRO = PACKAGE / "reproduce"
FRESH_ROOT = REPRO / "fresh_views"
FROZEN_SHA = "56ae033451164a1d8c74b1d90ef1655d9810a42bff3d191139c8320e7a2c90c5"
FOLD0_PQ = 0.4741684688395591
IOU = 0.5
VOTES = 24
SEEDS = (
    ("s20260910", "s20260910", "tta_parent_2048", "tta_parent_2048_test"),
    ("s20260911", "s20260911", "tta_s20260911", "tta_s20260911_test"),
    ("s20260912", "s20260912", "tta_s20260912", "tta_s20260912_test"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ens():
    spec = importlib.util.spec_from_file_location(
        "ens3", PROJECT_ROOT / "scripts" / "make_ensemble3_tta_submission.py"
    )
    ens = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ens)
    return ens


def _check_weights() -> dict:
    manifest = json.loads((WEIGHTS / "SHA256S.json").read_text())
    for row in manifest["files"]:
        path = WEIGHTS / row["file"]
        digest = _sha256(path)
        if digest != row["sha256"]:
            raise SystemExit(f"hash mismatch {path}")
    return manifest


def _payload(stem: str, split: str, fresh: bool = False) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    for name, folder, train_cache, test_cache in SEEDS:
        if fresh and split == "test":
            cache = FRESH_ROOT / folder
        else:
            cache = FAST_ARTIFACTS / (train_cache if split == "train" else test_cache)
        path = cache / f"{stem}.json"
        if not path.exists():
            raise SystemExit(f"missing view cache {path}")
        one = json.loads(path.read_text())
        for key, rles in one.items():
            payload[f"{name}_{key}"] = rles
    return payload


def _fuse_split(ens, files: list[str], split: str, fresh: bool = False) -> tuple[list, float]:
    frames = []
    t0 = time.perf_counter()
    for index, name in enumerate(files, start=1):
        stem = Path(name).stem
        masks = ens.fuse_masks_fast(ens.tta._decode_all(_payload(stem, split, fresh)), IOU, VOTES)
        if masks:
            frames.append(pred_to_df(stem, [encode_rle(mask) for mask in masks]))
        if index == 1 or index % 20 == 0 or index == len(files):
            print(f"{split}_fused {index}/{len(files)}", flush=True)
    return frames, time.perf_counter() - t0


def fold0(ens=None) -> dict:
    ens = ens or _load_ens()
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    files = sorted(name for name, fold in fold_of.items() if fold == 0)
    frames, elapsed = _fuse_split(ens, files, "train")
    import pandas as pd

    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    REPRO.mkdir(parents=True, exist_ok=True)
    csv_path = REPRO / "fold0_ens3_tta8x3_v24.csv"
    pred.to_csv(csv_path, index=False)
    gt = gt_annotations_to_df(gt_records_for_files(views, by_file, files))
    scored = score_pq(gt, pred)
    return {
        "pq": scored.pq,
        "tp": scored.tp,
        "fp": scored.fp,
        "fn": scored.fn,
        "n_pred": int(len(pred)),
        "rasterizer": "cv2",
        "target_pq": FOLD0_PQ,
        "matches_0.4742": round(scored.pq, 4) == 0.4742,
        "delta_vs_known": scored.pq - FOLD0_PQ,
        "elapsed_sec": elapsed,
        "csv": str(csv_path),
    }


def test_csv(ens=None, fresh: bool = False) -> dict:
    ens = ens or _load_ens()
    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    ens._assert_test_jpegs(images)
    frames, elapsed = _fuse_split(ens, [path.name for path in images], "test", fresh)
    import pandas as pd

    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    REPRO.mkdir(parents=True, exist_ok=True)
    csv_path = REPRO / ("test_fresh_ens3_tta8x3_v24.csv" if fresh else "test_ens3_tta8x3_v24.csv")
    pred.to_csv(csv_path, index=False)
    digest = _sha256(csv_path)
    frozen = Path.home() / ".cache/filament-segmentation-2026/submissions/submission_yolo11s_erode_gt_2048_ens3_tta8x3_v24.csv"
    frozen_bytes = frozen.read_bytes()
    rebuilt_bytes = csv_path.read_bytes()
    return {
        "rows": int(len(pred)),
        "sha256": digest,
        "frozen_sha256": _sha256(frozen),
        "target_sha256": FROZEN_SHA,
        "byte_identical": rebuilt_bytes == frozen_bytes and digest == FROZEN_SHA,
        "elapsed_sec": elapsed,
        "csv": str(csv_path),
        "frozen_csv_unchanged": True,
    }


def main() -> None:
    fresh = "--fresh" in sys.argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    which = args[0] if args else "both"
    _check_weights()
    ens = _load_ens()
    t0 = time.perf_counter()
    report = {
        "project_root": str(PROJECT_ROOT),
        "inference_rerun": False,
        "fresh_test_caches": fresh,
        "kaggle_submit_called": False,
    }
    if which in {"both", "fold0"}:
        report["fold0"] = fold0(ens)
    if which in {"both", "test"}:
        report["test"] = test_csv(ens, fresh=fresh)
    report["elapsed_sec"] = time.perf_counter() - t0
    report["match"] = bool(report.get("fold0", {}).get("matches_0.4742", True) and report.get("test", {}).get("byte_identical", True))
    REPRO.mkdir(parents=True, exist_ok=True)
    (REPRO / "reproduce_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
