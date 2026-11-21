#!/usr/bin/env python3
"""Run the three packaged seeds on raw test JPEGs into fresh view caches.

Does not write the champion caches or the frozen submission. Does not submit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from config import PROJECT_ROOT
from decode_ens3 import FRESH_ROOT, SEEDS, _check_weights

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from filament.paths import FAST_ARTIFACTS, TEST_IMG_DIR
from filament.rle import decode_rle

STORED = {
    "s20260910": FAST_ARTIFACTS / "tta_parent_2048_test",
    "s20260911": FAST_ARTIFACTS / "tta_s20260911_test",
    "s20260912": FAST_ARTIFACTS / "tta_s20260912_test",
}
WEIGHTS = {
    "s20260910": PROJECT_ROOT / "artifacts" / "final_package" / "weights" / "s20260910" / "best.pt",
    "s20260911": PROJECT_ROOT / "artifacts" / "final_package" / "weights" / "s20260911" / "best.pt",
    "s20260912": PROJECT_ROOT / "artifacts" / "final_package" / "weights" / "s20260912" / "best.pt",
}


def _load_tta():
    spec = importlib.util.spec_from_file_location("parent_tta", PROJECT_ROOT / "scripts" / "eval_parent_tta.py")
    tta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tta)
    return tta


def _device() -> str:
    import torch

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS required")
    return "mps"


def infer(images: list[Path], tta) -> float:
    from ultralytics import YOLO

    device = _device()
    t0 = time.perf_counter()
    for name, _folder, _train, _test in SEEDS:
        weights = WEIGHTS[name]
        dest_dir = FRESH_ROOT / name
        if dest_dir.resolve().is_relative_to(FAST_ARTIFACTS.resolve()):
            raise SystemExit(f"refusing to write a champion cache {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        missing = [path for path in images if not (dest_dir / f"{path.stem}.json").exists()]
        print(f"{name} missing {len(missing)}/{len(images)}", flush=True)
        if not missing:
            continue
        model = YOLO(str(weights))
        for index, path in enumerate(missing, start=1):
            payload = tta._predict_views(model, path.name, image_dir=TEST_IMG_DIR, device=device)
            (dest_dir / f"{path.stem}.json").write_text(json.dumps(payload))
            if index == 1 or index % 10 == 0 or index == len(missing):
                print(f"{name} inferred {index}/{len(missing)}", flush=True)
        del model
    return time.perf_counter() - t0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(a.sum()) + int(b.sum()) - inter
    return inter / union if union else 0.0


def _compare_rles(fresh: list[str], stored: list[str]) -> dict:
    left = [decode_rle(item) > 0 for item in fresh]
    right = [decode_rle(item) > 0 for item in stored]
    pairs = [(_iou(a, b), i, j) for i, a in enumerate(left) for j, b in enumerate(right)]
    pairs.sort(reverse=True)
    used_l, used_r = set(), set()
    ious = []
    abs_diffs = []
    for score, i, j in pairs:
        if i in used_l or j in used_r:
            continue
        if score <= 0 and (left[i].any() or right[j].any()):
            continue
        used_l.add(i)
        used_r.add(j)
        ious.append(score)
        abs_diffs.append(int(np.logical_xor(left[i], right[j]).sum()))
    for i, mask in enumerate(left):
        if i not in used_l:
            abs_diffs.append(int(mask.sum()))
            ious.append(0.0)
    for j, mask in enumerate(right):
        if j not in used_r:
            abs_diffs.append(int(mask.sum()))
            ious.append(0.0)
    return {
        "n_fresh": len(left),
        "n_stored": len(right),
        "min_iou": min(ious) if ious else None,
        "max_abs_diff": max(abs_diffs) if abs_diffs else 0,
    }


def verify(images: list[Path]) -> dict:
    rows = []
    worst = 0
    min_iou = 1.0
    for path in images:
        for name, _folder, _train, _test in SEEDS:
            fresh = json.loads((FRESH_ROOT / name / f"{path.stem}.json").read_text())
            stored = json.loads((STORED[name] / f"{path.stem}.json").read_text())
            keys = sorted(set(fresh) | set(stored))
            for key in keys:
                stats = _compare_rles(fresh.get(key, []), stored.get(key, []))
                stats.update(image=path.name, seed=name, view=key)
                rows.append(stats)
                worst = max(worst, stats["max_abs_diff"])
                if stats["min_iou"] is not None:
                    min_iou = min(min_iou, stats["min_iou"])
    from decode_ens3 import _load_ens, _payload

    ens = _load_ens()
    from filament.metric import pred_to_df
    from filament.rle import encode_rle
    import pandas as pd

    frames = []
    for path in images:
        masks = ens.fuse_masks_fast(ens.tta._decode_all(_payload(path.stem, "test", fresh=True)), 0.5, 24)
        if masks:
            frames.append(pred_to_df(path.stem, [encode_rle(mask) for mask in masks]))
    fresh_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    frozen = pd.read_csv(
        Path.home() / ".cache/filament-segmentation-2026/submissions/submission_yolo11s_erode_gt_2048_ens3_tta8x3_v24.csv"
    )
    stems = {path.stem for path in images}
    frozen_sub = frozen[frozen["filament_id"].astype(str).str.rsplit("_", n=1).str[0].isin(stems)]
    identical = fresh_df["filament_id"].tolist() == frozen_sub["filament_id"].tolist() and (
        fresh_df["segmentation_rle"].astype(str).tolist() == frozen_sub["segmentation_rle"].astype(str).tolist()
    )
    report = {
        "images": [path.name for path in images],
        "views": len(rows),
        "max_abs_diff": worst,
        "min_mask_iou": min_iou,
        "final_rles_identical": bool(identical),
        "fresh_rows": int(len(fresh_df)),
        "frozen_rows": int(len(frozen_sub)),
    }
    out = PROJECT_ROOT / "artifacts" / "final_package" / "reproduce" / "fresh_verify_10.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    limit = 0
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    _check_weights()
    tta = _load_tta()
    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    tta_assert = getattr(tta, "_assert_test_jpegs", None)
    if tta_assert:
        pass
    from decode_ens3 import _load_ens

    ens = _load_ens()
    ens._assert_test_jpegs(images)
    chosen = images[:limit] if limit else images
    elapsed = infer(chosen, tta)
    per = elapsed / max(len(chosen), 1)
    eta_all = per * len(images)
    print(json.dumps({"inferred": len(chosen), "elapsed_sec": elapsed, "eta_full_sec": eta_all}), flush=True)
    if limit:
        verify(chosen)
        (PROJECT_ROOT / "artifacts" / "final_package" / "reproduce" / "fresh_eta.json").write_text(
            json.dumps({"per_image_sec": per, "eta_full_sec": eta_all, "under_2h": eta_all < 7200}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
