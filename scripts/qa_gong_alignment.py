#!/usr/bin/env python3
"""Alignment QA for repaired GONG warps. CPU only. Does not train or infer.

A frame counts when another train image, not the ancestor, falls within
10 minutes of the target time. Warped labels are compared with that image's
medoid GT mapped into the target disk. The round-trip number is not a gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id
from filament.paths import FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR
from filament.yolo_export import erode_instance_polygon

spec = importlib.util.spec_from_file_location("repair", ROOT / "scripts" / "repair_gong_warp.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

OUT = ROOT / "artifacts" / "phaseB_repair"
PROVENANCE = OUT / "provenance.jsonl"
JPG_DIR = ROOT / "data" / "gong_propagate" / "jpg"
QA = OUT / "qa"
WINDOW = timedelta(minutes=10)


def _stamp(name: str) -> datetime:
    return datetime.strptime(Path(name).stem[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _site(name: str) -> str:
    stem = Path(name).stem
    return stem[14:-1] if stem.endswith("h") else stem[14:]


def _load_yolo(path: Path) -> list[np.ndarray]:
    polys = []
    for line in path.read_text().splitlines():
        vals = line.split()
        if len(vals) < 7:
            continue
        xy = np.asarray(list(map(float, vals[1:])), dtype=np.float64).reshape(-1, 2)
        polys.append(xy * IMAGE_SIZE)
    return polys


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


def _match(warped: list[np.ndarray], gt: list[np.ndarray]) -> list[tuple[float, int, int]]:
    scores = []
    for i, left in enumerate(warped):
        for j, right in enumerate(gt):
            scores.append((_iou(left, right), i, j))
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
    dist = np.hypot(poly[:, 0].mean() - disk["cx"], disk["cy"] - poly[:, 1].mean())
    return float(dist / disk["radius"])


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _passes(ious: list[float], inner: list[float], outer: list[float]) -> dict:
    med = _median(ious)
    inner_m = _median(inner)
    outer_m = _median(outer)
    gap_ok = True
    if inner_m is not None and outer_m is not None:
        gap_ok = outer_m >= inner_m - 0.1
    return {
        "n": len(ious),
        "median_iou": med,
        "inner_n": len(inner),
        "inner_median_iou": inner_m,
        "outer_n": len(outer),
        "outer_median_iou": outer_m,
        "outer_within_0.1_of_inner": gap_ok,
        "passes": bool(med is not None and med >= 0.6 and gap_ok),
    }


def _nearest(when: datetime, exclude: str, times: list[tuple[datetime, str]]) -> tuple[str, float] | None:
    best = None
    for stamp, name in times:
        if name == exclude:
            continue
        delta = abs(when - stamp)
        if delta <= WINDOW and (best is None or delta < best[1]):
            best = (name, delta.total_seconds())
    return best


def main() -> None:
    kept = [json.loads(line) for line in PROVENANCE.read_text().splitlines() if line.strip()]
    kept = [row for row in kept if row.get("status") == "kept"]
    train_names = sorted(p.name for p in TRAIN_IMG_DIR.glob("*.jpeg"))
    times = sorted((_stamp(name), name) for name in train_names)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    medoid = {name: medoid_view_id(views, by_file[name]) for name in by_file}
    disk_cache: dict[str, dict | None] = {}
    anchors = []
    for row in kept:
        hit = _nearest(datetime.fromisoformat(row["gong_time"]), row["train_file"], times)
        if hit:
            anchors.append((row, hit[0], hit[1]))
    print(f"gt_anchor_frames {len(anchors)}", flush=True)

    pairs = []
    frame_rows = []
    for rank, (row, anchor_name, gap_s) in enumerate(anchors, start=1):
        anchor_path = TRAIN_IMG_DIR / anchor_name
        anchor_disk = repair._disk_for(anchor_path, disk_cache)
        target_disk = {"cx": row["target_cx"], "cy": row["target_cy"], "radius": row["target_radius"]}
        if anchor_disk is None:
            frame_rows.append({"gong_file": row["gong_file"], "status": "anchor_disk_failed"})
            continue
        t0 = Time(_stamp(anchor_name))
        t1 = Time(datetime.fromisoformat(row["gong_time"]))
        gt_polys = []
        for filament in views[medoid[anchor_name]].filaments:
            poly = np.asarray(filament.polygon, dtype=np.float64)
            if len(poly) < 3:
                continue
            mapped = repair.rotate_polygon(poly, t0, t1, anchor_disk, target_disk)
            if mapped is None:
                continue
            eroded = erode_instance_polygon(mapped)
            if eroded is not None:
                gt_polys.append(eroded)
        warped_polys = _load_yolo(Path(row["label_path"]))
        warped_masks = _raster(warped_polys)
        gt_masks = _raster(gt_polys)
        matched = _match(warped_masks, gt_masks)
        frame_pairs = []
        for score, i, j in matched:
            radius = _radius(warped_polys[i], target_disk)
            boundary = 0.5 * (
                repair._edge_distance(warped_polys[i], gt_polys[j]) + repair._edge_distance(gt_polys[j], warped_polys[i])
            )
            item = {
                "iou": score,
                "boundary_px": boundary,
                "radius_frac": radius,
                "outer": radius > 0.75,
                "offset_hours": int(row["offset_hours"]),
                "gong_file": row["gong_file"],
            }
            pairs.append(item)
            frame_pairs.append(item)
        image = cv2.imread(str(JPG_DIR / row["gong_file"]), cv2.IMREAD_COLOR)
        if image is not None:
            for poly in gt_polys:
                cv2.polylines(image, [np.rint(poly).astype(np.int32).reshape(-1, 1, 2)], True, (40, 180, 60), 2)
            for poly in warped_polys:
                cv2.polylines(image, [np.rint(poly).astype(np.int32).reshape(-1, 1, 2)], True, (40, 60, 220), 2)
            title = f"GT o{int(row['offset_hours']):+d} {_site(row['gong_file'])} {Path(row['gong_file']).stem[:4]}"
            cv2.putText(image, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(str(QA / f"align_{rank:02d}_gt_{Path(row['gong_file']).stem}.png"), image)
        frame_rows.append(
            {
                "kind": "train_gt",
                "gong_file": row["gong_file"],
                "ancestor": row["train_file"],
                "anchor": anchor_name,
                "anchor_fold": int(fold_of[anchor_name]),
                "source_fold": int(row["source_fold"]),
                "offset_hours": int(row["offset_hours"]),
                "gap_seconds": gap_s,
                "year": _stamp(row["gong_file"]).year,
                "site": _site(row["gong_file"]),
                "n_warped": len(warped_polys),
                "n_gt": len(gt_polys),
                "n_matched": len(frame_pairs),
                "median_iou": _median([item["iou"] for item in frame_pairs]),
                "median_boundary_px": _median([item["boundary_px"] for item in frame_pairs]),
            }
        )
        print(f"gt {rank}/{len(anchors)} matched {len(frame_pairs)}", flush=True)

    def _slice(pred) -> list[dict]:
        return [item for item in pairs if pred(item)]

    summary = {
        "method": "warped label vs medoid GT of the nearest other train image within 10 min, mapped with the measured disks",
        "round_trip_gate": "retired; it is ~0 px by construction",
        "n_anchor_frames": len(frame_rows),
        "n_matched_filaments": len(pairs),
        "median_boundary_px": _median([item["boundary_px"] for item in pairs]),
        "all": _passes([item["iou"] for item in pairs], [item["iou"] for item in pairs if not item["outer"]], [item["iou"] for item in pairs if item["outer"]]),
        "offset_abs_6": _passes(
            [item["iou"] for item in _slice(lambda item: abs(item["offset_hours"]) == 6)],
            [item["iou"] for item in _slice(lambda item: abs(item["offset_hours"]) == 6 and not item["outer"])],
            [item["iou"] for item in _slice(lambda item: abs(item["offset_hours"]) == 6 and item["outer"])],
        ),
        "offset_abs_le_3": _passes(
            [item["iou"] for item in _slice(lambda item: abs(item["offset_hours"]) <= 3)],
            [item["iou"] for item in _slice(lambda item: abs(item["offset_hours"]) <= 3 and not item["outer"])],
            [item["iou"] for item in _slice(lambda item: abs(item["offset_hours"]) <= 3 and item["outer"])],
        ),
        "frames": frame_rows,
        "mps_inference": False,
    }
    six_ok = summary["offset_abs_6"]["passes"]
    three_ok = summary["offset_abs_le_3"]["passes"]
    if summary["all"]["passes"] and six_ok:
        decision = "keep_offsets_1_to_6"
    elif (not six_ok) and three_ok:
        decision = "cut_to_abs_offset_le_3"
    else:
        decision = "fail_do_not_build_student"
    summary["decision"] = decision
    (OUT / "alignment_gate.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("n_anchor_frames", "n_matched_filaments", "median_boundary_px", "all", "offset_abs_6", "offset_abs_le_3", "decision")}), flush=True)


if __name__ == "__main__":
    main()
