#!/usr/bin/env python3
"""Rebuild GONG warps in a new directory. Does not touch original labels or caches.

One label file per target frame and ancestor. Targets with any fold-0 ancestor
are dropped, as are the shared fold-1/fold-4 targets. Disk centre and radius
are measured per image. Differential rotation uses that radius. The 0.9 R
centroid cut and the one-day test exclusion are kept.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective
from sunpy.coordinates.sun import angular_radius
import sunpy.sun.models as sun_models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco, medoid_view_id
from filament.paths import IMAGE_SIZE, SEED, TEST_IMG_DIR, TRAIN_IMG_DIR
from filament.yolo_export import erode_instance_polygon, polygon_to_yolo_line

ORIGINAL_LABELS = (ROOT / "data" / "gong_propagate" / "labels").resolve()
ORIGINAL_WARPED = (ROOT / "artifacts" / "gong_propagate_warped.jsonl").resolve()
DOWNLOAD_MANIFEST = ROOT / "artifacts" / "gong_propagate_manifest.jsonl"
JPG_DIR = ROOT / "data" / "gong_propagate" / "jpg"
OUT_LABELS = ROOT / "data" / "gong_propagate_repair" / "labels"
OUT_DIR = ROOT / "artifacts" / "phaseB_repair"
MANIFEST_PATH = OUT_DIR / "provenance.jsonl"
DISK_PATH = OUT_DIR / "disk_fits.json"
QA_DIR = OUT_DIR / "qa"
SUMMARY_PATH = OUT_DIR / "repair_summary.json"
MAX_R = 0.9
EXCLUSION = timedelta(days=1)
DROP_FOLD_PAIRS = {(1, 4)}
NAMED_DROP = {
    "20141013135834Ch.jpg",
    "20141014005834Lh.jpg",
    "20141014015834Lh.jpg",
    "20120330013134Lh.jpg",
    "20120330123134Ch.jpg",
}


def _parse_stamp(name: str) -> datetime | None:
    stem = Path(name).stem
    if len(stem) < 14 or not stem[:14].isdigit():
        return None
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fit_circle(points: np.ndarray) -> tuple[float, float, float]:
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack([2 * x, 2 * y, np.ones(len(points))])
    sol, *_ = np.linalg.lstsq(design, x * x + y * y, rcond=None)
    cx, cy, c = (float(v) for v in sol)
    radius = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
    return cx, cy, radius


def measure_disk(gray: np.ndarray) -> dict | None:
    """Fit the limb. Search stays near 900 px so the image-corner cliff is ignored."""
    height, width = gray.shape
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = 902.0
    angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    for _ in range(3):
        radii = np.linspace(max(820.0, radius - 40.0), min(980.0, radius + 40.0), 140)
        xs = cx + np.cos(angles)[:, None] * radii[None, :]
        ys = cy + np.sin(angles)[:, None] * radii[None, :]
        xi = np.clip(np.rint(xs).astype(np.int32), 0, width - 1)
        yi = np.clip(np.rint(ys).astype(np.int32), 0, height - 1)
        profile = gray[yi, xi].astype(np.float32)
        slope = np.diff(profile, axis=1)
        index = np.argmin(slope, axis=1)
        drop = slope[np.arange(angles.size), index]
        keep = drop < -0.8
        if int(keep.sum()) < 60:
            return None
        chosen = radii[index]
        points = np.column_stack(
            [cx + np.cos(angles) * chosen, cy + np.sin(angles) * chosen]
        )[keep]
        cx, cy, radius = _fit_circle(points)
    center_shift = float(np.hypot(cx - (width - 1) / 2.0, cy - (height - 1) / 2.0))
    if not (870.0 <= radius <= 940.0 and center_shift <= 40.0):
        return None
    return {"cx": cx, "cy": cy, "radius": radius, "center_shift": center_shift}


def _disk_for(path: Path, cache: dict[str, dict]) -> dict | None:
    key = str(path)
    if key in cache:
        return cache[key]
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        cache[key] = None
        return None
    cache[key] = measure_disk(gray)
    return cache[key]


def _pix_to_coord(points: np.ndarray, obstime: Time, disk: dict) -> SkyCoord:
    rsun = angular_radius(obstime)
    dx = (points[:, 0] - disk["cx"]) / disk["radius"]
    dy = (disk["cy"] - points[:, 1]) / disk["radius"]
    return SkyCoord(dx * rsun, dy * rsun, frame=Helioprojective(observer="earth", obstime=obstime))


def _coord_to_pix(coord: SkyCoord, obstime: Time, disk: dict) -> np.ndarray:
    rsun = angular_radius(obstime).to_value(u.arcsec)
    tx = coord.Tx.to_value(u.arcsec)
    ty = coord.Ty.to_value(u.arcsec)
    x = disk["cx"] + (tx / rsun) * disk["radius"]
    y = disk["cy"] - (ty / rsun) * disk["radius"]
    return np.column_stack([x, y])


def rotate_polygon(points: np.ndarray, t0: Time, t1: Time, source_disk: dict, target_disk: dict) -> np.ndarray | None:
    coord = _pix_to_coord(points, t0, source_disk)
    hgs = coord.transform_to(HeliographicStonyhurst(obstime=t0))
    if np.any(~np.isfinite(hgs.lon.degree)) or np.any(~np.isfinite(hgs.lat.degree)):
        return None
    interval = (t1 - t0).to(u.s)
    drot = sun_models.differential_rotation(interval, hgs.lat, frame_time="sidereal")
    rotated = SkyCoord(hgs.lon + drot, hgs.lat, obstime=t1, frame=HeliographicStonyhurst)
    out = rotated.transform_to(Helioprojective(observer="earth", obstime=t1))
    if np.any(~np.isfinite(out.Tx)) or np.any(~np.isfinite(out.Ty)):
        return None
    return _coord_to_pix(out, t1, target_disk)


def _centroid_r(points: np.ndarray, disk: dict) -> float:
    dx = (points[:, 0].mean() - disk["cx"]) / disk["radius"]
    dy = (disk["cy"] - points[:, 1].mean()) / disk["radius"]
    return float(np.hypot(dx, dy))


def _edge_distance(points: np.ndarray, source: np.ndarray) -> float:
    if len(points) == 0 or len(source) < 2:
        return 1e9
    closed = np.vstack([source, source[0]])
    starts = closed[:-1]
    ends = closed[1:]
    dists = []
    for point in points:
        ab = ends - starts
        denom = np.einsum("ij,ij->i", ab, ab) + 1e-12
        t = np.clip(np.einsum("ij,ij->i", point - starts, ab) / denom, 0.0, 1.0)
        proj = starts + ab * t[:, None]
        dists.append(float(np.linalg.norm(proj - point, axis=1).min()))
    return float(np.median(dists))


def _drop_reason(gong_file: str, folds: set[int]) -> str | None:
    if 0 in folds:
        return "fold0_ancestor"
    if tuple(sorted(folds)) in DROP_FOLD_PAIRS or (1 in folds and 4 in folds):
        return "fold1_fold4_shared_target"
    if gong_file in NAMED_DROP and (0 in folds or (1 in folds and 4 in folds)):
        return "named_drop"
    return None


def _draw_qa(path: Path, image_path: Path, polygons: list[np.ndarray], title: str) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    for poly in polygons:
        pts = np.rint(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], isClosed=True, color=(40, 60, 220), thickness=2)
    cv2.putText(image, title[:90], (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(path), image)


def main() -> None:
    if OUT_LABELS.resolve() == ORIGINAL_LABELS or MANIFEST_PATH.resolve() == ORIGINAL_WARPED:
        raise SystemExit("refusing to write an original GONG path")
    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    download = {
        (row["train_file"], int(row["offset_hours"])): row
        for row in _load_jsonl(DOWNLOAD_MANIFEST)
        if row.get("status") == "downloaded"
    }
    warped = _load_jsonl(ORIGINAL_WARPED)
    by_target: dict[str, list[dict]] = {}
    for row in warped:
        by_target.setdefault(row["gong_file"], []).append(row)

    test_times = [t for name in TEST_IMG_DIR.glob("*.jpeg") if (t := _parse_stamp(name.name))]
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    medoid = {name: medoid_view_id(views, by_file[name]) for name in by_file}

    jobs = []
    dropped = []
    for gong_file, ancestors in by_target.items():
        folds = {int(row["source_fold"]) for row in ancestors}
        reason = _drop_reason(gong_file, folds)
        if reason:
            dropped.append({"gong_file": gong_file, "reason": reason, "source_folds": sorted(folds), "n_ancestors": len(ancestors)})
            continue
        for row in ancestors:
            key = (row["train_file"], int(row["offset_hours"]))
            src = download.get(key)
            if src is None:
                dropped.append({"gong_file": gong_file, "reason": "missing_download_row", "train_file": row["train_file"]})
                continue
            gong_time = datetime.fromisoformat(src["gong_time"])
            near = next((t for t in test_times if abs(gong_time - t) <= EXCLUSION), None)
            if near is not None:
                dropped.append({"gong_file": gong_file, "reason": "test_exclusion_1day", "train_file": row["train_file"]})
                continue
            if int(fold_of[row["train_file"]]) != int(row["source_fold"]):
                raise SystemExit(f"fold mismatch {row['train_file']}")
            jobs.append((row, src))

    disk_cache: dict[str, dict | None] = {}
    if DISK_PATH.exists():
        disk_cache.update(json.loads(DISK_PATH.read_text()))

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if limit:
        jobs = jobs[:limit]

    written = 0
    kept_rows = []
    qa_candidates = []
    with MANIFEST_PATH.open("w") as handle:
        for index, (row, src) in enumerate(jobs, start=1):
            source_path = TRAIN_IMG_DIR / row["train_file"]
            target_path = JPG_DIR / row["gong_file"]
            source_disk = _disk_for(source_path, disk_cache)
            target_disk = _disk_for(target_path, disk_cache)
            record = {
                "gong_file": row["gong_file"],
                "train_file": row["train_file"],
                "source_fold": int(row["source_fold"]),
                "offset_hours": int(row["offset_hours"]),
                "train_time": src["train_time"],
                "gong_time": src["gong_time"],
                "time_delta_hours": int(row["offset_hours"]),
                "teacher_pseudo_used": False,
                "test_labels_used": False,
            }
            if source_disk is None or target_disk is None:
                record["status"] = "dropped_disk_fit"
                handle.write(json.dumps(record) + "\n")
                continue
            t0 = Time(datetime.fromisoformat(src["train_time"]))
            t1 = Time(datetime.fromisoformat(src["gong_time"]))
            source_polys = []
            warped_polys = []
            centroid_rs = []
            roundtrip = []
            for filament in views[medoid[row["train_file"]]].filaments:
                poly = np.asarray(filament.polygon, dtype=np.float64)
                if len(poly) < 3:
                    continue
                source_polys.append(poly)
                warped = rotate_polygon(poly, t0, t1, source_disk, target_disk)
                if warped is None or _centroid_r(warped, target_disk) > MAX_R:
                    continue
                back = rotate_polygon(warped, t1, t0, target_disk, source_disk)
                if back is not None:
                    roundtrip.append(_edge_distance(back, poly))
                eroded = erode_instance_polygon(warped)
                if eroded is None:
                    continue
                warped_polys.append(eroded)
                centroid_rs.append(_centroid_r(eroded, target_disk))
            stem = Path(row["gong_file"]).stem
            ancestor = Path(row["train_file"]).stem
            label_name = f"{stem}__f{int(row['source_fold'])}__{ancestor}__o{int(row['offset_hours']):+d}.txt"
            label_path = OUT_LABELS / label_name
            if label_path.resolve().is_relative_to(ORIGINAL_LABELS):
                raise SystemExit(f"label escaped into originals: {label_path}")
            lines = [polygon_to_yolo_line(poly) for poly in warped_polys]
            lines = [line for line in lines if line]
            if not lines:
                record["status"] = "dropped_empty"
                record["source_radius"] = source_disk["radius"]
                record["target_radius"] = target_disk["radius"]
                handle.write(json.dumps(record) + "\n")
                continue
            label_path.write_text("\n".join(lines) + "\n")
            record.update(
                {
                    "status": "kept",
                    "label_path": str(label_path),
                    "n_kept": len(lines),
                    "source_cx": source_disk["cx"],
                    "source_cy": source_disk["cy"],
                    "source_radius": source_disk["radius"],
                    "target_cx": target_disk["cx"],
                    "target_cy": target_disk["cy"],
                    "target_radius": target_disk["radius"],
                    "centroid_r_min": float(min(centroid_rs)),
                    "centroid_r_max": float(max(centroid_rs)),
                    "roundtrip_median_px": float(np.median(roundtrip)) if roundtrip else None,
                }
            )
            handle.write(json.dumps(record) + "\n")
            kept_rows.append(record)
            qa_candidates.append((record, target_path, warped_polys, source_path, source_polys))
            written += 1
            if index == 1 or index % 200 == 0:
                print(f"repaired {index}/{len(jobs)} kept {written}", flush=True)
                DISK_PATH.write_text(json.dumps({k: v for k, v in disk_cache.items() if v}) + "\n")

    DISK_PATH.write_text(json.dumps({k: v for k, v in disk_cache.items() if v}) + "\n")
    rng = np.random.default_rng(SEED)
    bins = {
        "m1_inner": [item for item in qa_candidates if abs(item[0]["offset_hours"]) == 1 and item[0]["centroid_r_min"] < 0.5],
        "m1_outer": [item for item in qa_candidates if abs(item[0]["offset_hours"]) == 1 and item[0]["centroid_r_max"] > 0.7],
        "m6_inner": [item for item in qa_candidates if abs(item[0]["offset_hours"]) == 6 and item[0]["centroid_r_min"] < 0.5],
        "m6_outer": [item for item in qa_candidates if abs(item[0]["offset_hours"]) == 6 and item[0]["centroid_r_max"] > 0.7],
    }
    chosen = []
    used = set()

    def _take(pool: list, bucket: str, count: int) -> None:
        avail = [item for item in pool if item[0]["label_path"] not in used]
        if not avail:
            return
        order = rng.permutation(len(avail))
        for pos in order[:count]:
            if len(chosen) >= 10:
                return
            pick = avail[int(pos)]
            used.add(pick[0]["label_path"])
            chosen.append((bucket, pick))

    for name, pool in bins.items():
        _take(pool, name, 3)
    _take(qa_candidates, "fill", 10)
    qa_errors = []
    for rank, (bucket, (record, target_path, polys, _source_path, _source_polys)) in enumerate(chosen, start=1):
        title = f"{bucket} o{record['offset_hours']:+d} r{record['centroid_r_min']:.2f}-{record['centroid_r_max']:.2f}"
        _draw_qa(QA_DIR / f"{rank:02d}_{bucket}_{Path(record['gong_file']).stem}.png", target_path, polys, title)
        if record["roundtrip_median_px"] is not None:
            qa_errors.append(record["roundtrip_median_px"])
    all_errors = [row["roundtrip_median_px"] for row in kept_rows if row["roundtrip_median_px"] is not None]
    qa_median = float(np.median(qa_errors)) if qa_errors else None
    all_median = float(np.median(all_errors)) if all_errors else None
    summary = {
        "targets_seen": len(by_target),
        "targets_dropped_by_rule": len({row["gong_file"] for row in dropped if "gong_file" in row}),
        "drop_reasons": {reason: sum(1 for row in dropped if row.get("reason") == reason) for reason in sorted({row.get("reason") for row in dropped})},
        "ancestor_jobs": len(jobs) if not limit else len(jobs),
        "labels_kept": written,
        "qa_frames": len(chosen),
        "qa_bins": {name: len(pool) for name, pool in bins.items()},
        "qa_median_roundtrip_px": qa_median,
        "all_kept_median_roundtrip_px": all_median,
        "gate_px": 5.0,
        "gate_definition": "median over QA frames of the per-frame median distance from round-tripped vertices to the source polygon edge, before erosion",
        "passed_gate": bool(qa_median is not None and qa_median <= 5.0 and (all_median is None or all_median <= 5.0)),
        "originals_modified": False,
        "teacher_pseudo_used": False,
        "limit": limit,
        "named_drop_absent_from_kept": all(name not in {row["gong_file"] for row in kept_rows} for name in NAMED_DROP),
        "unique_label_names": len({Path(row["label_path"]).name for row in kept_rows}),
        "shared_label_filenames": len(kept_rows) - len({Path(row["label_path"]).name for row in kept_rows}),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if limit:
        return
    if not summary["passed_gate"]:
        raise SystemExit("STOP median registration error > 5 px")


if __name__ == "__main__":
    main()
