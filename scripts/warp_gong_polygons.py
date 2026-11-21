#!/usr/bin/env python3
"""Warp train medoid polygons onto downloaded GONG frames.

Uses SunPy's sidereal differential-rotation rate and Astropy solar
ephemerides. Drops an instance whose warped centroid is outside 0.9 R.
Frames whose source image is fold 0 are warped and recorded, then marked
so student training can leave them out. Does not read test labels.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.coordinates.sun import angular_radius
import sunpy.sun.models as sun_models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id
from filament.paths import IMAGE_SIZE, SEED
from filament.yolo_export import erode_instance_polygon, polygon_to_yolo_line

MANIFEST = ROOT / "artifacts" / "gong_propagate_manifest.jsonl"
OUT_MANIFEST = ROOT / "artifacts" / "gong_propagate_warped.jsonl"
LABEL_DIR = ROOT / "data" / "gong_propagate" / "labels"
RADIUS_PX = 1000.0
MAX_R = 0.9
CENTER = (IMAGE_SIZE - 1) / 2.0


def _pix_to_coord(points: np.ndarray, obstime: Time) -> SkyCoord:
    rsun = angular_radius(obstime)
    dx = (points[:, 0] - CENTER) / RADIUS_PX
    dy = (CENTER - points[:, 1]) / RADIUS_PX
    return SkyCoord(
        dx * rsun,
        dy * rsun,
        frame=Helioprojective(observer="earth", obstime=obstime),
    )


def _coord_to_pix(coord: SkyCoord, obstime: Time) -> np.ndarray:
    rsun = angular_radius(obstime).to_value(u.arcsec)
    tx = coord.Tx.to_value(u.arcsec)
    ty = coord.Ty.to_value(u.arcsec)
    x = CENTER + (tx / rsun) * RADIUS_PX
    y = CENTER - (ty / rsun) * RADIUS_PX
    return np.column_stack([x, y])


def _rotate(points: np.ndarray, t0: Time, t1: Time) -> np.ndarray | None:
    coord = _pix_to_coord(points, t0)
    hgs = coord.transform_to(HeliographicStonyhurst(obstime=t0))
    if np.any(~np.isfinite(hgs.lon.degree)) or np.any(~np.isfinite(hgs.lat.degree)):
        return None
    interval = (t1 - t0).to(u.s)
    drot = sun_models.differential_rotation(interval, hgs.lat, frame_time="sidereal")
    rotated = SkyCoord(
        hgs.lon + drot,
        hgs.lat,
        obstime=t1,
        frame=HeliographicStonyhurst,
    )
    out = rotated.transform_to(Helioprojective(observer="earth", obstime=t1))
    if np.any(~np.isfinite(out.Tx)) or np.any(~np.isfinite(out.Ty)):
        return None
    return _coord_to_pix(out, t1)


def _centroid_r(points: np.ndarray) -> float:
    dx = (points[:, 0].mean() - CENTER) / RADIUS_PX
    dy = (CENTER - points[:, 1].mean()) / RADIUS_PX
    return float(np.hypot(dx, dy))


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"missing {MANIFEST}")
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    medoid = {name: medoid_view_id(views, by_file[name]) for name in by_file}
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT_MANIFEST.exists():
        for line in OUT_MANIFEST.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["train_file"], row["offset_hours"]))
    written = 0
    with OUT_MANIFEST.open("a") as handle:
        for line in MANIFEST.read_text().splitlines():
            if not line.strip():
                continue
            try:
                job = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (job["train_file"], job["offset_hours"])
            if key in done or job.get("status") != "downloaded":
                continue
            t0 = Time(datetime.fromisoformat(job["train_time"]))
            t1 = Time(datetime.fromisoformat(job["gong_time"]))
            source = job["train_file"]
            polygons = []
            dropped_limb = 0
            dropped_off = 0
            for filament in views[medoid[source]].filaments:
                if len(filament.polygon) < 3:
                    continue
                warped = _rotate(filament.polygon, t0, t1)
                if warped is None:
                    dropped_off += 1
                    continue
                if _centroid_r(warped) > MAX_R:
                    dropped_limb += 1
                    continue
                eroded = erode_instance_polygon(warped)
                if eroded is None:
                    dropped_off += 1
                    continue
                polygons.append(eroded)
            lines = [polygon_to_yolo_line(poly) for poly in polygons]
            lines = [ln for ln in lines if ln]
            label_path = LABEL_DIR / f"{Path(job['gong_file']).stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            row = {
                "train_file": source,
                "source_fold": int(fold_of[source]),
                "offset_hours": job["offset_hours"],
                "gong_file": job["gong_file"],
                "gong_time": job["gong_time"],
                "n_kept": len(lines),
                "n_dropped_off_disk": dropped_off,
                "n_dropped_outer": dropped_limb,
                "use_in_student_train": int(fold_of[source]) != 0,
                "label_path": str(label_path),
                "test_labels_used": False,
            }
            handle.write(json.dumps(row) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"warped {written}", flush=True)
    print(f"DONE gong_warp {written}", flush=True)


if __name__ == "__main__":
    main()
