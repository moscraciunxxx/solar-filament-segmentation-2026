#!/usr/bin/env python3
"""GT-to-GT control: disk rescale only, no differential rotation. CPU only."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id
from filament.paths import IMAGE_SIZE, SEED, TRAIN_IMG_DIR

spec = importlib.util.spec_from_file_location("repair", ROOT / "scripts" / "repair_gong_warp.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

OUT = ROOT / "artifacts" / "phaseB_repair"
WINDOW = timedelta(minutes=10)
WARP_MEDIAN = 0.3764898194938627


def _stamp(name: str) -> datetime:
    return datetime.strptime(Path(name).stem[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _site(name: str) -> str:
    stem = Path(name).stem
    return stem[14:-1] if stem.endswith("h") else stem[14:]


def _rescale(poly: np.ndarray, source: dict, target: dict) -> np.ndarray:
    dx = (poly[:, 0] - source["cx"]) / source["radius"]
    dy = (source["cy"] - poly[:, 1]) / source["radius"]
    x = target["cx"] + dx * target["radius"]
    y = target["cy"] - dy * target["radius"]
    return np.column_stack([x, y])


def _raster(polys: list[np.ndarray]) -> list[np.ndarray]:
    masks = []
    for poly in polys:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        pts = np.rint(poly).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
        if int(mask.sum()) >= 16:
            masks.append(mask.astype(bool))
    return masks


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(a.sum()) + int(b.sum()) - inter
    return inter / union if union else 0.0


def _match(left_masks: list[np.ndarray], right_masks: list[np.ndarray]) -> list[tuple[float, int, int]]:
    scores = [(_iou(left, right), i, j) for i, left in enumerate(left_masks) for j, right in enumerate(right_masks)]
    scores.sort(reverse=True)
    used_left, used_right = set(), set()
    matched = []
    for score, i, j in scores:
        if score <= 0 or i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matched.append((score, i, j))
    return matched


def _radius(poly: np.ndarray, disk: dict) -> float:
    return float(np.hypot(poly[:, 0].mean() - disk["cx"], disk["cy"] - poly[:, 1].mean()) / disk["radius"])


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _kept(polys: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    kept, masks = [], []
    for poly in polys:
        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        pts = np.rint(poly).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
        if int(mask.sum()) >= 16:
            kept.append(poly)
            masks.append(mask.astype(bool))
    return kept, masks


def _compare(source_polys: list[np.ndarray], target_polys: list[np.ndarray], source_disk: dict, target_disk: dict) -> list[dict]:
    moved, left = _kept([_rescale(poly, source_disk, target_disk) for poly in source_polys])
    target, right = _kept(target_polys)
    rows = []
    for score, i, j in _match(left, right):
        boundary = 0.5 * (
            repair._edge_distance(moved[i], target[j]) + repair._edge_distance(target[j], moved[i])
        )
        radius = _radius(moved[i], target_disk)
        rows.append({"iou": score, "boundary_px": boundary, "radius_frac": radius, "outer": radius > 0.75})
    return rows


def main() -> None:
    views, by_file = load_coco()
    medoid = {name: medoid_view_id(views, by_file[name]) for name in by_file}
    polys = {
        name: [np.asarray(f.polygon, dtype=np.float64) for f in views[medoid[name]].filaments if len(f.polygon) >= 3]
        for name in by_file
    }
    names = sorted(p.name for p in TRAIN_IMG_DIR.glob("*.jpeg") if p.name in polys)
    order = sorted(names, key=_stamp)
    disks: dict[str, dict | None] = {}
    for name in order:
        disks[name] = repair._disk_for(TRAIN_IMG_DIR / name, disks)
    pairs = []
    for i, left in enumerate(order):
        for right in order[i + 1 :]:
            delta = _stamp(right) - _stamp(left)
            if delta > WINDOW:
                break
            if _site(left) == _site(right):
                continue
            if disks[left] is None or disks[right] is None:
                continue
            pairs.append((left, right, delta.total_seconds()))
    print(f"train_pairs {len(pairs)}", flush=True)

    filament_rows = []
    pair_rows = []
    for index, (left, right, gap) in enumerate(pairs, start=1):
        matched = _compare(polys[left], polys[right], disks[left], disks[right])
        filament_rows.extend(matched)
        pair_rows.append(
            {
                "source": left,
                "target": right,
                "gap_seconds": gap,
                "sites": f"{_site(left)}->{_site(right)}",
                "n_matched": len(matched),
                "median_iou": _median([row["iou"] for row in matched]),
                "median_boundary_px": _median([row["boundary_px"] for row in matched]),
            }
        )
        if index % 20 == 0:
            print(f"pairs {index}/{len(pairs)}", flush=True)

    gate = json.loads((OUT / "alignment_gate.json").read_text())
    anchor_rows = []
    for frame in gate["frames"]:
        source = frame["ancestor"]
        target = frame["anchor"]
        if disks.get(source) is None or disks.get(target) is None or source not in polys or target not in polys:
            anchor_rows.append({"gong_file": frame["gong_file"], "status": "missing"})
            continue
        matched = _compare(polys[source], polys[target], disks[source], disks[target])
        anchor_rows.append(
            {
                "gong_file": frame["gong_file"],
                "source": source,
                "target": target,
                "offset_hours": frame["offset_hours"],
                "n_matched": len(matched),
                "median_iou": _median([row["iou"] for row in matched]),
                "median_boundary_px": _median([row["boundary_px"] for row in matched]),
                "filaments": matched,
            }
        )
    anchor_filaments = [item for row in anchor_rows for item in row.get("filaments", [])]
    control_median = _median([row["iou"] for row in filament_rows])
    anchor_median = _median([row["iou"] for row in anchor_filaments])
    if control_median is None:
        verdict = "no_pairs"
    elif 0.35 <= control_median <= 0.45:
        threshold = control_median - 0.05
        verdict = "warp_adds_no_error_gate_passed" if WARP_MEDIAN >= threshold else "control_in_band_but_warp_below_threshold"
    elif control_median >= 0.55:
        threshold = None
        verdict = "warp_broken_close_phase_b"
    else:
        threshold = None
        verdict = "between_bands"
    summary = {
        "method": "medoid GT rescaled by measured centre and radius only; no rotation; no erosion",
        "n_pairs": len(pairs),
        "n_matched_filaments": len(filament_rows),
        "median_iou": control_median,
        "median_boundary_px": _median([row["boundary_px"] for row in filament_rows]),
        "inner_median_iou": _median([row["iou"] for row in filament_rows if not row["outer"]]),
        "outer_median_iou": _median([row["iou"] for row in filament_rows if row["outer"]]),
        "same_14_anchor_pairs": {
            "n_frames": len(anchor_rows),
            "n_matched_filaments": len(anchor_filaments),
            "median_iou": anchor_median,
            "median_boundary_px": _median([row["boundary_px"] for row in anchor_filaments]),
        },
        "warp_median_iou": WARP_MEDIAN,
        "threshold": None if control_median is None or not (0.35 <= control_median <= 0.45) else control_median - 0.05,
        "verdict": verdict,
        "pairs": pair_rows,
        "student_started": False,
    }
    (OUT / "zero_offset_control.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "pairs"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
