#!/usr/bin/env python3
"""Pseudo-label propagated GONG frames with the 3-seed champion.

Eight geometries times three scales times three seeds. An instance is kept
when its vote fraction is at least 0.5. Test JPEGs are not read. Fold 0
source frames are not labeled for student training.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.paths import FAST_ARTIFACTS, IMAGE_SIZE, WEIGHTS_DIR
from filament.rle import decode_rle
from filament.yolo_export import polygon_to_yolo_line

spec = importlib.util.spec_from_file_location("eval_parent_tta", ROOT / "scripts" / "eval_parent_tta.py")
tta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tta)

JPG_DIR = ROOT / "data" / "gong_propagate" / "jpg"
PSEUDO_DIR = ROOT / "data" / "gong_propagate" / "pseudo"
WARPED = ROOT / "artifacts" / "gong_propagate_warped.jsonl"
MIN_FRACTION = 0.5
N_VOTERS = 72
SEEDS = (
    ("s20260910", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048" / "weights" / "best.pt"),
    ("s20260911", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260911" / "weights" / "best.pt"),
    ("s20260912", WEIGHTS_DIR / "yolo11s_seg_erode_gt_2048_s20260912" / "weights" / "best.pt"),
)


def _student_files() -> list[str]:
    names = []
    seen = set()
    if not WARPED.exists():
        return names
    for line in WARPED.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("use_in_student_train"):
            continue
        name = row["gong_file"]
        if name in seen:
            continue
        if not (JPG_DIR / name).exists():
            continue
        seen.add(name)
        names.append(name)
    return names


def _load_or_infer(model, cache: Path, file_name: str) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{Path(file_name).stem}.json"
    if path.exists():
        return json.loads(path.read_text())
    payload = tta._predict_views(model, file_name, image_dir=JPG_DIR)
    path.write_text(json.dumps(payload))
    return payload


def _fuse(masks: list[np.ndarray]) -> list[tuple[np.ndarray, int]]:
    if not masks:
        return []
    boxes = [tta._box(mask) for mask in masks]
    areas = [int(mask.sum()) for mask in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    used = [False] * len(masks)
    kept = []
    min_votes = int(np.ceil(MIN_FRACTION * N_VOTERS))
    for i in order:
        if used[i] or boxes[i] is None:
            continue
        group = [i]
        used[i] = True
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in order:
            if used[j] or boxes[j] is None:
                continue
            bx0, by0, bx1, by1 = boxes[j]
            if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
                continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            if union and (inter / union) >= 0.5:
                group.append(j)
                used[j] = True
        if len(group) < min_votes:
            continue
        vote = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint16)
        for k in group:
            vote += masks[k]
        merged = (vote >= max(1, (len(group) + 1) // 2)).astype(np.uint8)
        if int(merged.sum()) >= 96:
            kept.append((merged, len(group)))
    return kept


def _mask_to_line(mask: np.ndarray) -> str | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    poly = contour.reshape(-1, 2).astype(np.float32)
    return polygon_to_yolo_line(poly)


def main() -> None:
    files = _student_files()
    print(f"student_frames {len(files)}", flush=True)
    if not files:
        raise SystemExit("no propagated frames for folds 1-4")
    from ultralytics import YOLO

    for seed, weights in SEEDS:
        if not weights.exists():
            raise SystemExit(f"missing {weights}")
        cache = FAST_ARTIFACTS / f"tta_gong_{seed}"
        model = YOLO(str(weights))
        for index, name in enumerate(files, start=1):
            _load_or_infer(model, cache, name)
            if index % 25 == 0:
                print(f"{seed} cached {index}/{len(files)}", flush=True)
        del model
    PSEUDO_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, name in enumerate(files, start=1):
        dest = PSEUDO_DIR / f"{Path(name).stem}.txt"
        meta = PSEUDO_DIR / f"{Path(name).stem}.votes.json"
        if dest.exists() and meta.exists():
            continue
        masks = []
        for seed, _weights in SEEDS:
            payload = json.loads((FAST_ARTIFACTS / f"tta_gong_{seed}" / f"{Path(name).stem}.json").read_text())
            for rles in payload.values():
                for rle in rles:
                    masks.append(decode_rle(rle))
        fused = _fuse(masks)
        lines = []
        votes = []
        for mask, count in fused:
            line = _mask_to_line(mask)
            if not line:
                continue
            lines.append(line)
            votes.append({"votes": count, "fraction": count / N_VOTERS})
        dest.write_text("\n".join(lines) + ("\n" if lines else ""))
        meta.write_text(json.dumps(votes))
        written += 1
        if index % 25 == 0:
            print(f"pseudo {index}/{len(files)}", flush=True)
    print(f"DONE gong_pseudo written {written} frames {len(files)}", flush=True)


if __name__ == "__main__":
    main()
