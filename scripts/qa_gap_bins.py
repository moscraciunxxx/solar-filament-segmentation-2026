#!/usr/bin/env python3
"""Pre-registered gap filter on the 14 anchor matches. Does not train."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.time import Time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import load_coco, medoid_view_id
from filament.paths import TRAIN_IMG_DIR
from filament.yolo_export import erode_instance_polygon

qa_spec = importlib.util.spec_from_file_location("qa", ROOT / "scripts" / "qa_gong_alignment.py")
qa = importlib.util.module_from_spec(qa_spec)
qa_spec.loader.exec_module(qa)

THRESHOLD = 0.455
MIN_N = 8
GAP_ORDER = ("<=1h", "1-2h", "2-4h", ">4h")
DISK_ORDER = ("inner", "outer")


def _gap(hours: int) -> str:
    value = abs(int(hours))
    if value <= 1:
        return "<=1h"
    if value <= 2:
        return "1-2h"
    if value <= 4:
        return "2-4h"
    return ">4h"


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def main() -> None:
    kept = [json.loads(line) for line in qa.PROVENANCE.read_text().splitlines() if line.strip()]
    kept = [row for row in kept if row.get("status") == "kept"]
    times = sorted((qa._stamp(path.name), path.name) for path in TRAIN_IMG_DIR.glob("*.jpeg"))
    views, by_file = load_coco()
    medoid = {name: medoid_view_id(views, by_file[name]) for name in by_file}
    disk_cache: dict[str, dict | None] = {}
    pairs = []
    for row in kept:
        hit = qa._nearest(datetime.fromisoformat(row["gong_time"]), row["train_file"], times)
        if hit is None:
            continue
        anchor_name, _gap_s = hit
        anchor_disk = qa.repair._disk_for(TRAIN_IMG_DIR / anchor_name, disk_cache)
        target_disk = {"cx": row["target_cx"], "cy": row["target_cy"], "radius": row["target_radius"]}
        if anchor_disk is None:
            continue
        t0 = Time(qa._stamp(anchor_name))
        t1 = Time(datetime.fromisoformat(row["gong_time"]))
        gt_polys = []
        for filament in views[medoid[anchor_name]].filaments:
            poly = np.asarray(filament.polygon, dtype=np.float64)
            if len(poly) < 3:
                continue
            mapped = qa.repair.rotate_polygon(poly, t0, t1, anchor_disk, target_disk)
            if mapped is None:
                continue
            eroded = erode_instance_polygon(mapped)
            if eroded is not None:
                gt_polys.append(eroded)
        warped_polys = qa._load_yolo(Path(row["label_path"]))
        warped_kept, warped_masks = [], []
        gt_kept, gt_masks = [], []
        for poly, bucket, masks in (
            (warped_polys, warped_kept, warped_masks),
            (gt_polys, gt_kept, gt_masks),
        ):
            for one in poly:
                mask = qa._raster([one])
                if mask:
                    bucket.append(one)
                    masks.append(mask[0])
        for score, i, j in qa._match(warped_masks, gt_masks):
            radius = qa._radius(warped_kept[i], target_disk)
            pairs.append(
                {
                    "iou": score,
                    "gap": _gap(row["offset_hours"]),
                    "offset_hours": int(row["offset_hours"]),
                    "disk": "outer" if radius > 0.75 else "inner",
                    "radius_frac": radius,
                    "gong_file": row["gong_file"],
                }
            )
    table = []
    qualified = []
    for gap in GAP_ORDER:
        for disk in DISK_ORDER:
            rows = [item for item in pairs if item["gap"] == gap and item["disk"] == disk]
            ious = [item["iou"] for item in rows]
            median = _median(ious)
            ok = bool(median is not None and median >= THRESHOLD and len(rows) >= MIN_N)
            record = {"gap": gap, "disk": disk, "n": len(rows), "median_iou": median, "qualifies": ok}
            table.append(record)
            if ok:
                qualified.append(record)
    summary = {
        "rule": "median IoU >= 0.455 and n >= 8, gap x disk, on the 14 anchor matches",
        "threshold": THRESHOLD,
        "min_n": MIN_N,
        "n_anchor_matches": len(pairs),
        "all_median_iou": _median([item["iou"] for item in pairs]),
        "table": table,
        "qualified_bins": qualified,
        "decision": "close_phase_b" if not qualified else "build_qualified_bins",
        "student_started": False,
        "kaggle_submit_called": False,
    }
    (qa.OUT / "gap_bins.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not qualified:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
