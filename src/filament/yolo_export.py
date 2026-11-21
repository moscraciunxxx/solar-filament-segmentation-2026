from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from .coco_io import AnnotatorView, medoid_view_id, rasterize_polygons
from .paths import (
    FAST_CACHE,
    IMAGE_SIZE,
    SEED,
    TEST_IMG_DIR,
    WEIGHTS_DIR,
    YOLO_ALL_DIR,
    YOLO_CLAHE_DIR,
    YOLO_DIR,
    YOLO_ERODE_GT_DIR,
    YOLO_FAINT_DIR,
    YOLO_PHYS_DIR,
    YOLO_RESIDUAL_DIR,
    YOLO_PQ_ANY_DIR,
    YOLO_RATER_DIR,
    YOLO_SHIFT_DIR,
    YOLO_TILE_DIR,
)


def polygon_to_yolo_line(polygon, class_id: int = 0, size: int = IMAGE_SIZE) -> str | None:
    if len(polygon) < 3:
        return None
    xs = polygon[:, 0] / size
    ys = polygon[:, 1] / size
    xs = xs.clip(0.0, 1.0)
    ys = ys.clip(0.0, 1.0)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
    return f"{class_id} {coords}"


def write_yolo_split(
    views: dict[str, AnnotatorView],
    by_file: dict[str, list[str]],
    train_files: list[str],
    val_files: list[str],
    image_dir: Path,
    out_dir: Path = YOLO_DIR,
) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def export(files: list[str], split: str) -> int:
        n = 0
        for file_name in files:
            view_id = medoid_view_id(views, by_file[file_name])
            view = views[view_id]
            src = image_dir / file_name
            dst = out_dir / "images" / split / file_name
            shutil.copy2(src, dst)
            lines = []
            for filament in view.filaments:
                line = polygon_to_yolo_line(filament.polygon)
                if line:
                    lines.append(line)
            label_path = out_dir / "labels" / split / f"{Path(file_name).stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
        return n

    n_train = export(train_files, "train")
    n_val = export(val_files, "val")
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "filament"},
        "nc": 1,
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    (out_dir / "split_counts.txt").write_text(f"train={n_train} val={n_val}\n")
    return yaml_path


# Locked Slot1 erode postprocess: cv2.MORPH_ELLIPSE 3x3, iterations=1.
ERODE_GT_K = 3
ERODE_GT_ITERS = 1
_CLOSED_ERODE_GT_NAMES = frozenset(
    {
        "yolo_seg",
        "yolo_seg_all",
        "yolo_seg_rater",
        "yolo_seg_phys",
        "yolo_seg_residual",
        "yolo_seg_shift",
        "yolo_seg_faint",
        "yolo_seg_tiles",
        "yolo_seg_clahe",
        "yolo_seg_union",
        "yolo_seg_pq_any",
    }
)


def erode_instance_polygon(
    polygon: np.ndarray,
    size: int = IMAGE_SIZE,
    k: int = ERODE_GT_K,
    iters: int = ERODE_GT_ITERS,
) -> np.ndarray | None:
    """Rasterize one instance, apply Slot1 binary erode, return largest remaining contour.

    Empty after erode → None (drop the instance; do not invent pixels).
    """
    import cv2

    mask = rasterize_polygons([polygon], size=size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask, kernel, iterations=iters)
    if int(eroded.sum()) == 0:
        return None
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3 or float(cv2.contourArea(contour)) <= 0.0:
        return None
    return contour.reshape(-1, 2).astype(np.float32)


def _refuse_erode_gt_out_dir(out_dir: Path) -> None:
    target = out_dir.resolve()
    protected = {
        YOLO_DIR.resolve(),
        YOLO_ALL_DIR.resolve(),
        YOLO_RATER_DIR.resolve(),
        YOLO_PHYS_DIR.resolve(),
        YOLO_RESIDUAL_DIR.resolve(),
        YOLO_SHIFT_DIR.resolve(),
        YOLO_FAINT_DIR.resolve(),
        YOLO_TILE_DIR.resolve(),
        YOLO_CLAHE_DIR.resolve(),
        WEIGHTS_DIR.resolve(),
    }
    if target == FAST_CACHE.resolve():
        raise ValueError(f"Refusing to write eroded GT into cache root {out_dir}")
    if target in protected or target.name in _CLOSED_ERODE_GT_NAMES:
        raise ValueError(f"Refusing to write eroded GT into closed dest {out_dir}")
    if any(target == p or p in target.parents for p in protected):
        raise ValueError(f"Refusing to write eroded GT under closed dest {out_dir}")
    if any(target in p.parents for p in protected):
        raise ValueError(f"Refusing to write eroded GT onto an ancestor of a closed dest {out_dir}")
    if target == WEIGHTS_DIR.resolve() or WEIGHTS_DIR.resolve() in target.parents:
        raise ValueError(f"Refusing to write eroded GT under live weights {out_dir}")


def write_yolo_erode_gt(
    views: dict[str, AnnotatorView],
    by_file: dict[str, list[str]],
    train_files: list[str],
    val_files: list[str],
    image_dir: Path,
    out_dir: Path = YOLO_ERODE_GT_DIR,
    k: int = ERODE_GT_K,
    iters: int = ERODE_GT_ITERS,
) -> tuple[Path, dict]:
    """Same grouped medoid split as write_yolo_split, labels from Slot1 erode 3x3x1.

    Copies or hardlinks JPEGs. Vanished instances are dropped. Never rmtree
    live yolo_seg* dests or weights/*.
    """
    _refuse_erode_gt_out_dir(out_dir)
    _refuse_live_yolo_dirs(out_dir)
    train_files = sorted(train_files)
    val_files = sorted(val_files)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def image_src(file_name: str, split: str) -> Path:
        cached = YOLO_DIR / "images" / split / file_name
        if cached.exists():
            return cached
        return image_dir / file_name

    def export(files: list[str], split: str) -> dict:
        n_source = 0
        n_src_instances = 0
        n_kept = 0
        n_dropped = 0
        n_empty = 0
        for i, file_name in enumerate(files):
            view = views[medoid_view_id(views, by_file[file_name])]
            src = image_src(file_name, split)
            if not src.exists():
                raise FileNotFoundError(src)
            _link_or_copy(src, out_dir / "images" / split / file_name)
            lines: list[str] = []
            for filament in view.filaments:
                n_src_instances += 1
                eroded = erode_instance_polygon(filament.polygon, k=k, iters=iters)
                if eroded is None:
                    n_dropped += 1
                    continue
                line = polygon_to_yolo_line(eroded)
                if line:
                    lines.append(line)
                    n_kept += 1
                else:
                    n_dropped += 1
            (out_dir / "labels" / split / f"{Path(file_name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            if not lines:
                n_empty += 1
            n_source += 1
            if (i + 1) % 50 == 0 or i + 1 == len(files):
                print(f"yolo_erode_gt {split} {i + 1}/{len(files)}", flush=True)
        return {
            "n_files": n_source,
            "n_src_instances": n_src_instances,
            "n_kept_instances": n_kept,
            "n_dropped_instances": n_dropped,
            "n_empty_label_images": n_empty,
        }

    train_stats = export(train_files, "train")
    val_stats = export(val_files, "val")
    yaml_path = _write_yaml(out_dir)
    (out_dir / "split_counts.txt").write_text(
        f"train={train_stats['n_files']} val={val_stats['n_files']} "
        f"kept={train_stats['n_kept_instances'] + val_stats['n_kept_instances']} "
        f"dropped={train_stats['n_dropped_instances'] + val_stats['n_dropped_instances']} "
        f"erode={k}x{iters}\n"
    )
    img_train, lbl_train = _split_stems(out_dir, "train")
    img_val, lbl_val = _split_stems(out_dir, "val")
    test_stems = _test_stems()
    yolo_val = (
        {p.name for p in (YOLO_DIR / "images" / "val").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        if (YOLO_DIR / "images" / "val").exists()
        else set()
    )
    stats = {
        "mode": "medoid_erode_gt",
        "family": "yolo_seg_erode_gt",
        "op": "erode",
        "k": k,
        "iters": iters,
        "kernel": "MORPH_ELLIPSE",
        "same_as_slot1_erode": True,
        "n_train": train_stats["n_files"],
        "n_val": val_stats["n_files"],
        "train": train_stats,
        "val": val_stats,
        "images_labels_match": img_train == lbl_train and img_val == lbl_val,
        "test_stem_overlap": len((img_train | img_val) & test_stems),
        "val_files_match_yolo_seg": img_val == {Path(n).stem for n in yolo_val} if yolo_val else False,
        "yaml": str(yaml_path),
        "data_yaml_exists": yaml_path.exists(),
        "out_dir": str(out_dir.resolve()),
        "dest": str(out_dir.resolve()),
        "do_not_train_now": True,
        "do_not_submit": True,
        "did_not_touch": [
            str(YOLO_DIR),
            str(YOLO_ALL_DIR),
            str(YOLO_RATER_DIR),
            str(YOLO_PHYS_DIR),
            str(YOLO_RESIDUAL_DIR),
            str(YOLO_SHIFT_DIR),
            str(YOLO_FAINT_DIR),
            str(YOLO_TILE_DIR),
            str(WEIGHTS_DIR),
        ],
        "closed_dest_written": False,
    }
    (out_dir / "erode_gt_stats.json").write_text(json.dumps(stats, indent=2))
    return yaml_path, stats


def write_yolo_all_annotators(
    views: dict[str, AnnotatorView],
    by_file: dict[str, list[str]],
    train_files: list[str],
    val_files: list[str],
    image_dir: Path,
    out_dir: Path = YOLO_ALL_DIR,
) -> Path:
    """One YOLO sample per annotator-view. Same pixels, independent labels.

    This covers faint filaments that the medoid annotator dropped, at the cost
    of label noise. Val still uses one medoid view so PQ-style selection stays
    grouped by physical JPEG.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def export_train(files: list[str]) -> int:
        n = 0
        for file_name in files:
            src = image_dir / file_name
            for view_id in by_file[file_name]:
                view = views[view_id]
                alias = f"{view.annotator}-{file_name}"
                shutil.copy2(src, out_dir / "images" / "train" / alias)
                lines = [polygon_to_yolo_line(f.polygon) for f in view.filaments]
                lines = [ln for ln in lines if ln]
                (out_dir / "labels" / "train" / f"{Path(alias).stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else "")
                )
                n += 1
        return n

    def export_val(files: list[str]) -> int:
        n = 0
        for file_name in files:
            view = views[medoid_view_id(views, by_file[file_name])]
            shutil.copy2(image_dir / file_name, out_dir / "images" / "val" / file_name)
            lines = [polygon_to_yolo_line(f.polygon) for f in view.filaments]
            lines = [ln for ln in lines if ln]
            (out_dir / "labels" / "val" / f"{Path(file_name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            n += 1
        return n

    n_train = export_train(train_files)
    n_val = export_val(val_files)
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "filament"},
                "nc": 1,
            },
            sort_keys=False,
        )
    )
    (out_dir / "split_counts.txt").write_text(f"train={n_train} val={n_val}\n")
    return yaml_path


def _refuse_live_yolo_dirs(out_dir: Path) -> None:
    live = {YOLO_DIR.resolve(), YOLO_ALL_DIR.resolve()}
    if out_dir.resolve() in live:
        raise ValueError(f"Refusing to overwrite live train dir {out_dir}")


def write_yolo_random_rater(
    views: dict[str, AnnotatorView],
    by_file: dict[str, list[str]],
    train_files: list[str],
    val_files: list[str],
    image_dir: Path,
    out_dir: Path = YOLO_RATER_DIR,
    seed: int = SEED,
) -> tuple[Path, dict]:
    """One YOLO sample per physical JPEG. Train = one random rater; val = medoid.

    Random-rater (not raster-union) is the Method-1 expected-PQ export: each train
    JPEG keeps a real annotator's instance polygons. That creates NEW objects vs
    medoid whenever the draw is not the medoid (minority filaments the consensus
    dropped). A union-then-contour mask would also add pixels, but it merges
    adjacent filaments into one instance — host PQ has no Hungarian rescue.

    Val stays the same 142 grouped fold-0 files and the same medoid labels as
    ``yolo_seg`` so the holdout recipe stays comparable to frozen m@1280 0.410.
    Extra val annotators are not written.

    Does not rmtree ``yolo_seg`` or ``yolo_seg_all``. Train JSON only.
    """
    _refuse_live_yolo_dirs(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_files = sorted(train_files)
    val_files = sorted(val_files)
    rng = np.random.default_rng(seed)
    assignments: list[dict] = []
    n_train_multi = 0
    n_differ = 0

    def write_sample(view: AnnotatorView, file_name: str, split: str) -> None:
        shutil.copy2(image_dir / file_name, out_dir / "images" / split / file_name)
        lines = [polygon_to_yolo_line(f.polygon) for f in view.filaments]
        lines = [ln for ln in lines if ln]
        (out_dir / "labels" / split / f"{Path(file_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )

    for file_name in train_files:
        image_ids = list(by_file[file_name])
        medoid_id = medoid_view_id(views, image_ids)
        choice_ids = sorted(image_ids)
        picked_id = choice_ids[int(rng.integers(0, len(choice_ids)))]
        multi = len(image_ids) > 1
        differ = picked_id != medoid_id
        if multi:
            n_train_multi += 1
        if differ:
            n_differ += 1
        write_sample(views[picked_id], file_name, "train")
        assignments.append(
            {
                "file_name": file_name,
                "picked_view": picked_id,
                "medoid_view": medoid_id,
                "n_raters": len(image_ids),
                "differ_from_medoid": differ,
            }
        )

    for file_name in val_files:
        image_ids = list(by_file[file_name])
        write_sample(views[medoid_view_id(views, image_ids)], file_name, "val")

    n_train = len(train_files)
    n_val = len(val_files)
    n_more = n_fewer = n_same_n = added = removed = 0
    for row in assignments:
        if not row["differ_from_medoid"]:
            continue
        n_p = len(views[row["picked_view"]].filaments)
        n_m = len(views[row["medoid_view"]].filaments)
        delta = n_p - n_m
        if delta > 0:
            n_more += 1
            added += delta
        elif delta < 0:
            n_fewer += 1
            removed += -delta
        else:
            n_same_n += 1
    yaml_path = _write_yaml(out_dir)
    (out_dir / "split_counts.txt").write_text(
        f"train={n_train} val={n_val} differ_from_medoid={n_differ} "
        f"train_multi={n_train_multi} mode=random_rater seed={seed}\n"
    )
    stats = {
        "mode": "random_rater",
        "choice": (
            "random_rater not union: real annotator instances (new objects vs "
            "medoid on non-medoid draws) without contour-merging filaments"
        ),
        "val_labels": "medoid",
        "val_note": (
            "Same grouped fold-0 JPEGs and medoid labels as yolo_seg. Extra val "
            "annotators are withheld so holdout stays comparable to m@1280 0.410."
        ),
        "n_train_files": n_train,
        "n_val_files": n_val,
        "n_train_multi_annotator": n_train_multi,
        "n_train_differ_from_medoid": n_differ,
        "n_train_more_instances": n_more,
        "n_train_fewer_instances": n_fewer,
        "n_train_same_count_different_polygons": n_same_n,
        "n_instances_added": added,
        "n_instances_removed": removed,
        "n_dataset_multi_annotator": sum(1 for ids in by_file.values() if len(ids) > 1),
        "seed": seed,
        "yaml": str(yaml_path),
        "out_dir": str(out_dir.resolve()),
        "train_assignments": assignments,
    }
    (out_dir / "rater_stats.json").write_text(json.dumps(stats, indent=2))
    return yaml_path, stats


# Match-any: one real rater outline per filament. IoU>0.5 links raters.
# Not a pixel union and not one YOLO line per rater (that export scored ~0.359).
PQ_ANY_IOU = 0.5
PQ_ANY_COMPARE = 256


def _pq_any_bbox_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _pq_any_mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(a.sum()) + int(b.sum()) - inter
    return float(inter / union) if union else 0.0


def _refuse_pq_any_out_dir(out_dir: Path) -> None:
    if out_dir.name != "yolo_seg_pq_any":
        raise ValueError(f"Refusing match-any export into unexpected dir {out_dir}")
    target = out_dir.resolve()
    protected = {
        YOLO_DIR.resolve(),
        YOLO_ALL_DIR.resolve(),
        YOLO_RATER_DIR.resolve(),
        YOLO_PHYS_DIR.resolve(),
        YOLO_RESIDUAL_DIR.resolve(),
        YOLO_SHIFT_DIR.resolve(),
        YOLO_FAINT_DIR.resolve(),
        YOLO_TILE_DIR.resolve(),
        YOLO_CLAHE_DIR.resolve(),
        YOLO_ERODE_GT_DIR.resolve(),
        (FAST_CACHE / "yolo_seg_union").resolve(),
        WEIGHTS_DIR.resolve(),
    }
    if target in protected or WEIGHTS_DIR.resolve() in target.parents:
        raise ValueError(f"Refusing to write match-any labels onto a closed dest {out_dir}")
    if target != YOLO_PQ_ANY_DIR.resolve():
        raise ValueError(f"Refusing match-any export outside {YOLO_PQ_ANY_DIR}")


def match_any_polygons(
    views: dict[str, AnnotatorView],
    image_ids: list[str],
    iou_thr: float = PQ_ANY_IOU,
    compare_size: int = PQ_ANY_COMPARE,
) -> tuple[list[np.ndarray], dict]:
    """Collapse cross-rater IoU>0.5 outlines into one filament.

    The emitted mask is one annotator's polygon (the member that already
    matches the most other raters at IoU>thr). Pixel-union is never built.
    On a multi-rater image, a filament only one rater marked is dropped:
    that extra instance is an FP against every other annotator under host PQ.
    A single-rater image keeps every filament.
    """
    scale = compare_size / float(IMAGE_SIZE)
    items: list[dict] = []
    for view_id in image_ids:
        view = views[view_id]
        for filament in view.filaments:
            if len(filament.polygon) < 3:
                continue
            items.append(
                {
                    "ann": view.annotator,
                    "polygon": filament.polygon,
                    "bbox": filament.bbox,
                    "mask": None,
                }
            )
    n_raw = len(items)
    n_raters = len(image_ids)
    if n_raw == 0:
        return [], {
            "n_raw": 0,
            "n_kept": 0,
            "n_dropped_singleton": 0,
            "n_agreed": 0,
            "n_raters": n_raters,
        }

    def mask_of(i: int) -> np.ndarray:
        cached = items[i]["mask"]
        if cached is None:
            cached = rasterize_polygons([items[i]["polygon"] * scale], size=compare_size)
            items[i]["mask"] = cached
        return cached

    pair_iou: dict[tuple[int, int], float] = {}
    parent = list(range(n_raw))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ann_of = {i: {items[i]["ann"]} for i in range(n_raw)}
    pairs: list[tuple[float, int, int]] = []
    for i in range(n_raw):
        for j in range(i + 1, n_raw):
            if items[i]["ann"] == items[j]["ann"]:
                continue
            if not _pq_any_bbox_overlap(items[i]["bbox"], items[j]["bbox"]):
                continue
            iou = _pq_any_mask_iou(mask_of(i), mask_of(j))
            pair_iou[(i, j)] = iou
            if iou > iou_thr:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    for _iou, i, j in pairs:
        ri, rj = find(i), find(j)
        if ri == rj or ann_of[ri] & ann_of[rj]:
            continue
        parent[rj] = ri
        ann_of[ri] |= ann_of[rj]
        del ann_of[rj]

    clusters: dict[int, list[int]] = {}
    for i in range(n_raw):
        clusters.setdefault(find(i), []).append(i)

    kept: list[np.ndarray] = []
    n_dropped = 0
    n_agreed = 0
    n_hit_links = 0
    n_rep_links = 0
    for members in clusters.values():
        anns = {items[i]["ann"] for i in members}
        if n_raters >= 2 and len(anns) < 2:
            n_dropped += 1
            continue
        n_agreed += 1

        def iou_ij(a: int, b: int) -> float:
            if a > b:
                a, b = b, a
            if a == b:
                return 1.0
            cached = pair_iou.get((a, b))
            if cached is None:
                if items[a]["ann"] == items[b]["ann"]:
                    return 0.0
                if not _pq_any_bbox_overlap(items[a]["bbox"], items[b]["bbox"]):
                    return 0.0
                cached = _pq_any_mask_iou(mask_of(a), mask_of(b))
                pair_iou[(a, b)] = cached
            return cached

        best_i = members[0]
        best_key = (-1, -1.0)
        for i in members:
            others = [j for j in members if j != i]
            if not others:
                key = (1, 1.0)
            else:
                ious = [iou_ij(i, j) for j in others]
                key = (sum(v > iou_thr for v in ious), float(sum(ious)))
            if key > best_key:
                best_key = key
                best_i = i
        for j in members:
            if j == best_i:
                continue
            n_rep_links += 1
            if iou_ij(best_i, j) > iou_thr:
                n_hit_links += 1
        kept.append(items[best_i]["polygon"])
    return kept, {
        "n_raw": n_raw,
        "n_kept": len(kept),
        "n_dropped_singleton": n_dropped,
        "n_agreed": n_agreed,
        "n_raters": n_raters,
        "n_rep_links": n_rep_links,
        "n_hit_links": n_hit_links,
    }


def write_yolo_pq_any(
    views: dict[str, AnnotatorView],
    by_file: dict[str, list[str]],
    train_files: list[str],
    val_files: list[str],
    image_dir: Path,
    out_dir: Path = YOLO_PQ_ANY_DIR,
) -> tuple[Path, dict]:
    """H-alpha JPEGs, one box per agreed filament, real rater polygon.

    Train and YOLO-internal val both use match-any labels. Official fold-0
    PQ is scored later by score_family against every annotator, not these files.
    Does not rmtree medoid, union, erode, shift, CLAHE, or weights.
    """
    _refuse_pq_any_out_dir(out_dir)
    _refuse_live_yolo_dirs(out_dir)
    train_files = sorted(train_files)
    val_files = sorted(val_files)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def image_src(file_name: str, split: str) -> Path:
        cached = YOLO_DIR / "images" / split / file_name
        if cached.exists():
            return cached
        return image_dir / file_name

    totals = {
        "n_raw": 0,
        "n_kept": 0,
        "n_dropped_singleton": 0,
        "n_agreed": 0,
        "n_files": 0,
        "n_multi": 0,
        "n_raw_multi": 0,
        "n_kept_multi": 0,
        "n_medoid": 0,
        "n_rep_links": 0,
        "n_hit_links": 0,
        "n_empty": 0,
    }

    def export(files: list[str], split: str) -> None:
        for i, file_name in enumerate(files):
            image_ids = list(by_file[file_name])
            src = image_src(file_name, split)
            if not src.exists():
                raise FileNotFoundError(src)
            _link_or_copy(src, out_dir / "images" / split / file_name)
            polygons, info = match_any_polygons(views, image_ids)
            lines = []
            for polygon in polygons:
                line = polygon_to_yolo_line(polygon)
                if line:
                    lines.append(line)
            (out_dir / "labels" / split / f"{Path(file_name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            medoid = views[medoid_view_id(views, image_ids)]
            totals["n_raw"] += info["n_raw"]
            totals["n_kept"] += info["n_kept"]
            totals["n_dropped_singleton"] += info["n_dropped_singleton"]
            totals["n_agreed"] += info["n_agreed"]
            totals["n_files"] += 1
            totals["n_medoid"] += len(medoid.filaments)
            totals["n_rep_links"] += info["n_rep_links"]
            totals["n_hit_links"] += info["n_hit_links"]
            if info["n_raters"] >= 2:
                totals["n_multi"] += 1
                totals["n_raw_multi"] += info["n_raw"]
                totals["n_kept_multi"] += info["n_kept"]
            if not lines:
                totals["n_empty"] += 1
            if (i + 1) % 50 == 0 or i + 1 == len(files):
                print(f"yolo_pq_any {split} {i + 1}/{len(files)}", flush=True)

    # Split counters: export() accumulates into totals, so snapshot per split.
    export(train_files, "train")
    train_snap = dict(totals)
    export(val_files, "val")
    for key, value in list(totals.items()):
        totals[key] = value - train_snap[key]
    val_snap = dict(totals)
    totals = {key: train_snap[key] + val_snap[key] for key in train_snap}

    yaml_path = _write_yaml(out_dir)
    (out_dir / "split_counts.txt").write_text(
        f"train_files={len(train_files)} val_files={len(val_files)} "
        f"kept={totals['n_kept']} raw={totals['n_raw']} "
        f"dropped_singleton={totals['n_dropped_singleton']} mode=match_any\n"
    )
    img_train, lbl_train = _split_stems(out_dir, "train")
    img_val, lbl_val = _split_stems(out_dir, "val")
    test_overlap = len((img_train | img_val) & _test_stems())
    yolo_val = (
        {p.name for p in (YOLO_DIR / "images" / "val").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        if (YOLO_DIR / "images" / "val").exists()
        else set()
    )
    multi_ratio = (
        train_snap["n_kept_multi"] / train_snap["n_raw_multi"] if train_snap["n_raw_multi"] else None
    )
    stats = {
        "mode": "match_any_rater",
        "family": "yolo_seg_pq_any",
        "choice": (
            "one real annotator polygon per filament cluster linked at IoU>0.5; "
            "singleton filaments dropped on multi-rater images; not a pixel union "
            "and not one YOLO line per rater"
        ),
        "iou_thr": PQ_ANY_IOU,
        "compare_size": PQ_ANY_COMPARE,
        "val_labels": "match_any",
        "official_score": "score_family all-annotator PQ, not these label files",
        "pixels": "halpha_jpeg",
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "train": train_snap,
        "val": val_snap,
        "multi_rater_kept_over_raw_train": multi_ratio,
        "images_labels_match": img_train == lbl_train and img_val == lbl_val,
        "test_stem_overlap": test_overlap,
        "test_labels_used": False,
        "val_files_match_yolo_seg": img_val == {Path(n).stem for n in yolo_val} if yolo_val else False,
        "yaml": str(yaml_path),
        "out_dir": str(out_dir.resolve()),
        "extras_or": False,
        "did_not_touch": [
            str(YOLO_DIR),
            str(YOLO_ALL_DIR),
            str(YOLO_RATER_DIR),
            str(YOLO_SHIFT_DIR),
            str(YOLO_ERODE_GT_DIR),
            str(YOLO_CLAHE_DIR),
            str(FAST_CACHE / "yolo_seg_union"),
            str(WEIGHTS_DIR),
        ],
    }
    (out_dir / "pq_any_stats.json").write_text(json.dumps(stats, indent=2))
    if test_overlap:
        raise RuntimeError(f"pq_any export overlaps test stems: {test_overlap}")
    if not stats["images_labels_match"]:
        raise RuntimeError("pq_any images and labels diverged")
    if multi_ratio is not None and multi_ratio > 0.85:
        raise RuntimeError(
            f"pq_any did not collapse multi-rater duplicates (kept/raw={multi_ratio:.3f}); refusing a union-like set"
        )
    return yaml_path, stats


def _write_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "filament"},
                "nc": 1,
            },
            sort_keys=False,
        )
    )
    return yaml_path


def _copy_medoid_sample(
    views: dict[str, AnnotatorView],
    by_file: dict[str, list[str]],
    file_name: str,
    image_dir: Path,
    dst_image: Path,
    dst_label: Path,
) -> None:
    view = views[medoid_view_id(views, by_file[file_name])]
    shutil.copy2(image_dir / file_name, dst_image)
    lines = [polygon_to_yolo_line(f.polygon) for f in view.filaments]
    lines = [ln for ln in lines if ln]
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""))


def year_upsample_copies(file_name: str) -> int:
    """Extra train copies so late-cycle / test-like years are not drowned by 2012-2016."""
    year = int(file_name[:4])
    if year >= 2020:
        return 3
    if year >= 2018:
        return 1
    return 0


def write_yolo_year_upsample(
    views: dict[str, AnnotatorView],
    by_file: dict[str, list[str]],
    train_files: list[str],
    val_files: list[str],
    image_dir: Path,
    out_dir: Path = YOLO_SHIFT_DIR,
) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = 0
    extra = 0
    for file_name in train_files:
        stem = Path(file_name).stem
        _copy_medoid_sample(
            views,
            by_file,
            file_name,
            image_dir,
            out_dir / "images" / "train" / file_name,
            out_dir / "labels" / "train" / f"{stem}.txt",
        )
        n_train += 1
        for k in range(year_upsample_copies(file_name)):
            alias = f"{stem}_yup{k}.jpeg"
            _copy_medoid_sample(
                views,
                by_file,
                file_name,
                image_dir,
                out_dir / "images" / "train" / alias,
                out_dir / "labels" / "train" / f"{Path(alias).stem}.txt",
            )
            n_train += 1
            extra += 1

    n_val = 0
    for file_name in val_files:
        stem = Path(file_name).stem
        _copy_medoid_sample(
            views,
            by_file,
            file_name,
            image_dir,
            out_dir / "images" / "val" / file_name,
            out_dir / "labels" / "val" / f"{stem}.txt",
        )
        n_val += 1

    yaml_path = _write_yaml(out_dir)
    (out_dir / "split_counts.txt").write_text(f"train={n_train} val={n_val} extra={extra}\n")
    return yaml_path


def write_yolo_clahe(
    src_dir: Path = YOLO_DIR,
    out_dir: Path = YOLO_CLAHE_DIR,
    clip_limit: float = 2.0,
    tile: int = 8,
) -> Path:
    """Medoid split with CLAHE pixels. Never rmtree yolo_seg — copy labels only."""
    import cv2

    from .preprocess import apply_clahe, read_gray

    if not (src_dir / "data.yaml").exists():
        raise FileNotFoundError(f"Need an existing medoid export first: {src_dir}")
    if src_dir.resolve() == YOLO_DIR.resolve() and out_dir.resolve() == YOLO_DIR.resolve():
        raise ValueError("Refusing to overwrite live yolo_seg")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for src in sorted((src_dir / "images" / split).iterdir()):
            if src.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            enhanced = apply_clahe(read_gray(src), clip_limit=clip_limit, tile=tile)
            bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(out_dir / "images" / split / src.name), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            label_src = src_dir / "labels" / split / f"{src.stem}.txt"
            label_dst = out_dir / "labels" / split / f"{src.stem}.txt"
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
            else:
                label_dst.write_text("")
            n += 1
        counts[split] = n
    yaml_path = _write_yaml(out_dir)
    (out_dir / "split_counts.txt").write_text(
        f"train={counts['train']} val={counts['val']} clahe=clip{clip_limit}_tile{tile}\n"
    )
    return yaml_path


def write_yolo_phys_stack(
    src_dir: Path = YOLO_DIR,
    out_dir: Path = YOLO_PHYS_DIR,
) -> Path:
    """Medoid split with gray+CLAHE+dark-residual BGR. Never rmtree yolo_seg."""
    import cv2

    from .classical import phys_stack
    from .preprocess import read_gray

    if not (src_dir / "data.yaml").exists():
        raise FileNotFoundError(f"Need an existing medoid export first: {src_dir}")
    if out_dir.resolve() == YOLO_DIR.resolve():
        raise ValueError("Refusing to overwrite live yolo_seg")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for src in sorted((src_dir / "images" / split).iterdir()):
            if src.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stacked = phys_stack(read_gray(src))
            cv2.imwrite(str(out_dir / "images" / split / src.name), stacked, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            label_src = src_dir / "labels" / split / f"{src.stem}.txt"
            label_dst = out_dir / "labels" / split / f"{src.stem}.txt"
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
            else:
                label_dst.write_text("")
            n += 1
        counts[split] = n
    yaml_path = _write_yaml(out_dir)
    (out_dir / "split_counts.txt").write_text(
        f"train={counts['train']} val={counts['val']} phys=gray_clahe_residual\n"
    )
    return yaml_path


_CLOSED_RESIDUAL_DESTS = frozenset(
    {
        YOLO_DIR.resolve(),
        YOLO_PHYS_DIR.resolve(),
        YOLO_CLAHE_DIR.resolve(),
        YOLO_ALL_DIR.resolve(),
        YOLO_SHIFT_DIR.resolve(),
        YOLO_FAINT_DIR.resolve(),
        YOLO_RATER_DIR.resolve(),
        YOLO_TILE_DIR.resolve(),
    }
)


def write_yolo_residual_stack(
    src_dir: Path = YOLO_DIR,
    out_dir: Path = YOLO_RESIDUAL_DIR,
) -> Path:
    """Medoid split with flatten+raw+Sobel BGR. Never rmtree yolo_seg or yolo_seg_phys."""
    import cv2

    from .classical import flatten_raw_sobel_stack
    from .preprocess import read_gray

    if not (src_dir / "data.yaml").exists():
        raise FileNotFoundError(f"Need an existing medoid export first: {src_dir}")
    if out_dir.resolve() in _CLOSED_RESIDUAL_DESTS:
        raise ValueError(f"Refusing to overwrite closed YOLO dest {out_dir}")
    if out_dir.resolve() == YOLO_PHYS_DIR.resolve():
        raise ValueError("Refusing to overwrite yolo_seg_phys")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for src in sorted((src_dir / "images" / split).iterdir()):
            if src.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stacked = flatten_raw_sobel_stack(read_gray(src))
            cv2.imwrite(str(out_dir / "images" / split / src.name), stacked, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            label_src = src_dir / "labels" / split / f"{src.stem}.txt"
            label_dst = out_dir / "labels" / split / f"{src.stem}.txt"
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
            else:
                label_dst.write_text("")
            n += 1
        counts[split] = n
    yaml_path = _write_yaml(out_dir)
    (out_dir / "split_counts.txt").write_text(
        f"train={counts['train']} val={counts['val']} residual=flatten_raw_sobel\n"
    )
    return yaml_path


UNIDENTIFIABLE_CATEGORY_ID = 3


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _refuse_faint_out_dir(out_dir: Path) -> None:
    target = out_dir.resolve()
    protected = {
        YOLO_DIR.resolve(),
        YOLO_ALL_DIR.resolve(),
        YOLO_SHIFT_DIR.resolve(),
        YOLO_PHYS_DIR.resolve(),
        YOLO_RATER_DIR.resolve(),
        YOLO_CLAHE_DIR.resolve(),
        YOLO_TILE_DIR.resolve(),
        WEIGHTS_DIR.resolve(),
    }
    if target in protected:
        raise ValueError(f"Refusing to write faint labels into live dir {out_dir}")
    if target == WEIGHTS_DIR.resolve() or WEIGHTS_DIR.resolve() in target.parents:
        raise ValueError(f"Refusing to write faint labels under live weights {out_dir}")


def write_yolo_faint(
    views: dict[str, AnnotatorView],
    by_file: dict[str, list[str]],
    train_files: list[str],
    val_files: list[str],
    image_dir: Path,
    out_dir: Path = YOLO_FAINT_DIR,
) -> tuple[Path, dict]:
    """Medoid JPEGs, labels kept only if area<=train-p25 or Unidentifiable.

    Trainer holdout uses the same filter. Official PQ is still all-annotator
    on the 142 JPEGs after NMS-merge with frozen Slot 1. Never rmtree yolo_seg.
    """
    _refuse_faint_out_dir(out_dir)
    train_files = sorted(train_files)
    val_files = sorted(val_files)
    train_areas: list[float] = []
    for file_name in train_files:
        view = views[medoid_view_id(views, by_file[file_name])]
        train_areas.extend(float(f.area) for f in view.filaments)
    if not train_areas:
        raise ValueError("No train medoid filaments; cannot compute faint p25")
    area_p25 = float(np.percentile(np.asarray(train_areas, dtype=np.float64), 25))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def image_src(file_name: str, split: str) -> Path:
        cached = YOLO_DIR / "images" / split / file_name
        if cached.exists():
            return cached
        return image_dir / file_name

    def export(files: list[str], split: str) -> dict:
        n_instances = 0
        n_small = 0
        n_unident = 0
        n_unident_above_p25 = 0
        n_empty = 0
        n_source = 0
        for file_name in files:
            view = views[medoid_view_id(views, by_file[file_name])]
            kept = []
            for filament in view.filaments:
                small = filament.area <= area_p25
                unident = int(filament.category_id) == UNIDENTIFIABLE_CATEGORY_ID
                if not (small or unident):
                    continue
                kept.append(filament)
                n_instances += 1
                if small:
                    n_small += 1
                if unident:
                    n_unident += 1
                if unident and not small:
                    n_unident_above_p25 += 1
            src = image_src(file_name, split)
            if not src.exists():
                raise FileNotFoundError(src)
            _link_or_copy(src, out_dir / "images" / split / file_name)
            lines = [polygon_to_yolo_line(f.polygon) for f in kept]
            lines = [ln for ln in lines if ln]
            (out_dir / "labels" / split / f"{Path(file_name).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            if not lines:
                n_empty += 1
            n_source += 1
        return {
            "n_files": n_source,
            "n_instances": n_instances,
            "n_small": n_small,
            "n_unidentifiable": n_unident,
            "n_unidentifiable_above_p25": n_unident_above_p25,
            "n_empty_label_images": n_empty,
        }

    train_stats = export(train_files, "train")
    val_stats = export(val_files, "val")

    img_train = {p.stem for p in (out_dir / "images" / "train").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    lbl_train = {p.stem for p in (out_dir / "labels" / "train").glob("*.txt")}
    img_val = {p.stem for p in (out_dir / "images" / "val").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    lbl_val = {p.stem for p in (out_dir / "labels" / "val").glob("*.txt")}
    test_stems = {p.stem for p in TEST_IMG_DIR.glob("*")} if TEST_IMG_DIR.exists() else set()
    yolo_val = {p.stem for p in (YOLO_DIR / "images" / "val").glob("*")} if (YOLO_DIR / "images" / "val").exists() else set()

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "filament"},
                "nc": 1,
            },
            sort_keys=False,
        )
    )
    (out_dir / "split_counts.txt").write_text(
        f"train={train_stats['n_files']} val={val_stats['n_files']} "
        f"train_instances={train_stats['n_instances']} area_p25={area_p25:.3f}\n"
    )
    stats = {
        "mode": "faint_specialist_medoid",
        "family": "faint_specialist_s1024",
        "area_p25": area_p25,
        "unidentifiable_category_id": UNIDENTIFIABLE_CATEGORY_ID,
        "keep_rule": "area<=train_medoid_p25 OR category_id==3 Unidentifiable",
        "n_train_files": train_stats["n_files"],
        "n_val_files": val_stats["n_files"],
        "train": train_stats,
        "val": val_stats,
        "images_labels_match": img_train == lbl_train and img_val == lbl_val,
        "test_stem_overlap": len((img_train | img_val) & test_stems),
        "val_files_match_yolo_seg": img_val == yolo_val if yolo_val else False,
        "out_dir": str(out_dir.resolve()),
        "did_not_touch": [str(YOLO_DIR), str(YOLO_ALL_DIR)],
        "infer": "NMS-merge with frozen Slot 1 (m@1280 conf 0.35) into ONE csv. Never upload specialist alone.",
        "official_pq": "all annotators on 142 JPEGs after merge, not trainer faint-val mAP",
    }
    (out_dir / "faint_export_contract.json").write_text(json.dumps(stats, indent=2))
    return yaml_path, stats


# Native-2048 tiles for later tile_native_s640 (imgsz=640). Export only — do not train.
TILE_SIZE_DEFAULT = 768
TILE_OVERLAP_DEFAULT = 192
HARD_NEG_PER_JPEG = 2
_LIVE_TILE_FORBIDDEN_NAMES = frozenset(
    {
        "yolo_seg",
        "yolo_seg_all",
        "yolo_seg_shift",
        "yolo_seg_phys",
        "yolo_seg_residual",
        "yolo_seg_rater",
        "yolo_seg_clahe",
        "yolo_seg_faint",
        "yolo11s_seg_all",
    }
)


def tile_axis_origins(image_size: int = IMAGE_SIZE, tile: int = TILE_SIZE_DEFAULT, overlap: int = TILE_OVERLAP_DEFAULT) -> list[int]:
    """Flush last tile so 2048 is covered. Infer must use this same axis."""
    if tile <= overlap or tile > image_size:
        raise ValueError(f"Need 0 <= overlap < tile <= image_size, got tile={tile} overlap={overlap}")
    stride = tile - overlap
    last = image_size - tile
    xs = list(range(0, last + 1, stride))
    if xs[-1] != last:
        xs.append(last)
    return xs


def stitch_eval_contract(tile: int = TILE_SIZE_DEFAULT, overlap: int = TILE_OVERLAP_DEFAULT) -> dict:
    """Official PQ for this family is stitch-then-score, not tile mAP."""
    origins = tile_axis_origins(IMAGE_SIZE, tile, overlap)
    return {
        "family": "tile_native_s640",
        "do_not_train_in_this_job": True,
        "official_metric": (
            "Stitch tile preds back to native 2048, then official host PQ "
            "(IoU>0.5, not Hungarian, all annotators) on the same grouped fold-0 "
            "142 JPEGs as yolo_seg. YOLO trainer tile mAP is not official PQ."
        ),
        "tile": tile,
        "overlap": overlap,
        "stride": tile - overlap,
        "image_size": IMAGE_SIZE,
        "later_imgsz": 640,
        "axis_origins": origins,
        "n_infer_tiles_per_jpeg": len(origins) * len(origins),
        "infer_steps": [
            "Load the native 2048 JPEG (train or Kaggle test pixels only; never MAGFiLO test GT).",
            f"Tile with tile={tile}, overlap={overlap}, last-tile flush; axis origins {origins}.",
            "Run YOLO11s-seg at imgsz=640, retina_masks=True, on each tile (optional skip if disk_frac<0.15).",
            "Map each instance mask/polygon from tile coords to full-frame: (x+x0, y+y0).",
            "Stitch by greedy mask-IoU cluster (IoU>=0.5), keep highest-conf or majority-vote union.",
            "resolve_overlaps on the 2048 canvas.",
            "merge_gap=0. Do not merge_nearby. Do not CC-split a YOLO mask (fragments are FPs).",
            "Score with scripts/score_family.py / eval_yolo.py on the stitched full-frame instances.",
        ],
        "merge_gap": 0,
        "cc_split": False,
        "val_export_note": (
            "Val tiles in this export use the same GT-centric + 2 hard-neg recipe as train "
            "on the fold-0 142 JPEGs. That is a trainer holdout, not stitch-eval. At infer, "
            "tile the full disk with the axis_origins grid — never GT-centric (no GT on test)."
        ),
        "forbidden": [
            "MAGFiLO public-test / Harvard Dataverse test labels",
            "CC-split of YOLO masks",
            "merge_gap>0 as the only lever",
            "selecting this family from tile mAP or public invert",
        ],
    }


def _refuse_tile_out_dir(out_dir: Path) -> None:
    target = out_dir.resolve()
    protected = {
        YOLO_DIR.resolve(),
        YOLO_ALL_DIR.resolve(),
        YOLO_SHIFT_DIR.resolve(),
        YOLO_PHYS_DIR.resolve(),
        YOLO_RATER_DIR.resolve(),
        YOLO_CLAHE_DIR.resolve(),
        WEIGHTS_DIR.resolve(),
    }
    if target in protected or target.name in _LIVE_TILE_FORBIDDEN_NAMES:
        raise ValueError(f"Refusing to write tiles into live dir {out_dir}")
    if target == WEIGHTS_DIR.resolve() or WEIGHTS_DIR.resolve() in target.parents:
        raise ValueError(f"Refusing to write tiles under live weights {out_dir}")


def _clip_intersect(p1: tuple[float, float], p2: tuple[float, float], axis: str, value: float) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    if axis == "x":
        if abs(x2 - x1) < 1e-9:
            return (value, y1)
        t = (value - x1) / (x2 - x1)
        return (value, y1 + t * (y2 - y1))
    if abs(y2 - y1) < 1e-9:
        return (x1, value)
    t = (value - y1) / (y2 - y1)
    return (x1 + t * (x2 - x1), value)


def _clip_axis(pts: list[tuple[float, float]], axis: str, value: float, keep_high: bool) -> list[tuple[float, float]]:
    if not pts:
        return []
    out: list[tuple[float, float]] = []
    prev = pts[-1]
    for cur in pts:
        prev_v = prev[0] if axis == "x" else prev[1]
        cur_v = cur[0] if axis == "x" else cur[1]
        prev_in = prev_v >= value if keep_high else prev_v <= value
        cur_in = cur_v >= value if keep_high else cur_v <= value
        if cur_in:
            if not prev_in:
                out.append(_clip_intersect(prev, cur, axis, value))
            out.append(cur)
        elif prev_in:
            out.append(_clip_intersect(prev, cur, axis, value))
        prev = cur
    return out


def clip_polygon_to_tile(polygon: np.ndarray, x0: int, y0: int, tile: int) -> np.ndarray | None:
    """Sutherland–Hodgman clip of a full-frame polygon into tile-local xy."""
    import cv2

    if len(polygon) < 3:
        return None
    pts = [(float(x - x0), float(y - y0)) for x, y in polygon]
    tmax = float(tile)
    pts = _clip_axis(pts, "x", 0.0, keep_high=True)
    pts = _clip_axis(pts, "x", tmax, keep_high=False)
    pts = _clip_axis(pts, "y", 0.0, keep_high=True)
    pts = _clip_axis(pts, "y", tmax, keep_high=False)
    if len(pts) < 3:
        return None
    clipped = np.asarray(pts, dtype=np.float32)
    area = float(cv2.contourArea(clipped))
    if area < 8.0:
        return None
    return clipped


def _filament_bbox(filament) -> tuple[float, float, float, float]:
    if filament.bbox and filament.bbox[2] > 0 and filament.bbox[3] > 0:
        return (float(filament.bbox[0]), float(filament.bbox[1]), float(filament.bbox[2]), float(filament.bbox[3]))
    poly = filament.polygon
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    return (float(x0), float(y0), float(max(x1 - x0, 1.0)), float(max(y1 - y0, 1.0)))


def _axis_cover(p0: float, p1: float, tile: int, overlap: int, image_size: int) -> list[int]:
    last = image_size - tile
    stride = tile - overlap
    start = int(np.clip(np.floor(p0), 0, last))
    origins = [start]
    pos = start
    while pos + tile < min(p1, image_size) and pos < last:
        pos = min(pos + stride, last)
        if origins[-1] == pos:
            break
        origins.append(int(pos))
    flush = int(np.clip(np.ceil(p1) - tile, 0, last))
    if flush not in origins and flush + tile >= min(p1, image_size):
        origins.append(flush)
    return origins


def gt_centric_origins(filaments, tile: int, overlap: int, image_size: int = IMAGE_SIZE) -> list[tuple[int, int]]:
    """One centered tile per filament that fits; overlapping cover tiles if it does not."""
    last = image_size - tile
    seen: set[tuple[int, int]] = set()
    origins: list[tuple[int, int]] = []
    for filament in filaments:
        x, y, w, h = _filament_bbox(filament)
        if w <= tile and h <= tile:
            x0 = int(np.clip(round(x + w / 2.0 - tile / 2.0), 0, last))
            y0 = int(np.clip(round(y + h / 2.0 - tile / 2.0), 0, last))
            cand = [(x0, y0)]
        else:
            xs = _axis_cover(x, x + w, tile, overlap, image_size)
            ys = _axis_cover(y, y + h, tile, overlap, image_size)
            if w <= tile:
                xs = [int(np.clip(round(x + w / 2.0 - tile / 2.0), 0, last))]
            if h <= tile:
                ys = [int(np.clip(round(y + h / 2.0 - tile / 2.0), 0, last))]
            cand = [(xi, yi) for yi in ys for xi in xs]
        for origin in cand:
            if origin not in seen:
                seen.add(origin)
                origins.append(origin)
    return origins


def _label_lines_for_tile(filaments, x0: int, y0: int, tile: int) -> list[str]:
    lines: list[str] = []
    for filament in filaments:
        clipped = clip_polygon_to_tile(filament.polygon, x0, y0, tile)
        if clipped is None:
            continue
        line = polygon_to_yolo_line(clipped, size=tile)
        if line:
            lines.append(line)
    return lines


def _read_source_bgr(image_dir: Path, file_name: str, split: str) -> np.ndarray:
    import cv2

    src = image_dir / file_name
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        fallback = YOLO_DIR / "images" / split / file_name
        image = cv2.imread(str(fallback), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(src)
    if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return image


def _hard_neg_origins(
    gray: np.ndarray,
    filaments,
    occupied: set[tuple[int, int]],
    tile: int,
    overlap: int,
    n: int,
    rng: np.random.Generator,
    image_size: int = IMAGE_SIZE,
) -> list[tuple[int, int]]:
    import cv2

    from .preprocess import solar_disk_mask

    if n <= 0:
        return []
    disk = solar_disk_mask(gray)
    gt = rasterize_polygons([f.polygon for f in filaments], size=image_size)
    if gt.any():
        gt = cv2.dilate(gt, np.ones((24, 24), np.uint8))
    last = image_size - tile
    min_disk = 0.45

    def ok(x0: int, y0: int) -> bool:
        if (x0, y0) in occupied:
            return False
        patch_gt = gt[y0 : y0 + tile, x0 : x0 + tile]
        if patch_gt.any():
            return False
        disk_frac = float(disk[y0 : y0 + tile, x0 : x0 + tile].mean())
        return disk_frac >= min_disk

    found: list[tuple[int, int]] = []
    grid = [(x0, y0) for y0 in tile_axis_origins(image_size, tile, overlap) for x0 in tile_axis_origins(image_size, tile, overlap)]
    candidates = [xy for xy in grid if ok(*xy)]
    if candidates:
        pick = rng.choice(len(candidates), size=min(n, len(candidates)), replace=False)
        for i in pick:
            found.append(candidates[int(i)])
            occupied.add(candidates[int(i)])
    tries = 0
    while len(found) < n and tries < 80:
        tries += 1
        radius = 1000.0 * float(np.sqrt(rng.random()))
        theta = float(rng.random() * 2.0 * np.pi)
        cx = (image_size - 1) / 2.0 + radius * np.cos(theta)
        cy = (image_size - 1) / 2.0 + radius * np.sin(theta)
        x0 = int(np.clip(round(cx - tile / 2.0), 0, last))
        y0 = int(np.clip(round(cy - tile / 2.0), 0, last))
        if ok(x0, y0):
            found.append((x0, y0))
            occupied.add((x0, y0))
    return found


def _write_one_tile(
    image: np.ndarray,
    filaments,
    x0: int,
    y0: int,
    tile: int,
    out_dir: Path,
    split: str,
    stem: str,
    kind: str,
) -> tuple[str, list[str]]:
    import cv2

    crop = image[y0 : y0 + tile, x0 : x0 + tile]
    if crop.shape[0] != tile or crop.shape[1] != tile:
        raise ValueError(f"Bad crop {crop.shape} at {(x0, y0)} tile={tile}")
    name = f"{stem}_x{x0:04d}_y{y0:04d}.jpeg"
    cv2.imwrite(str(out_dir / "images" / split / name), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    lines = _label_lines_for_tile(filaments, x0, y0, tile) if kind != "hard_neg" else []
    (out_dir / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return name, lines


def _test_stems() -> set[str]:
    if not TEST_IMG_DIR.exists():
        return set()
    return {p.stem for p in TEST_IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}


def _split_stems(out_dir: Path, split: str) -> tuple[set[str], set[str]]:
    img_dir = out_dir / "images" / split
    lab_dir = out_dir / "labels" / split
    images = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    labels = {p.stem for p in lab_dir.iterdir() if p.suffix == ".txt"}
    return images, labels


def write_yolo_tiles(
    views: dict[str, AnnotatorView],
    by_file: dict[str, list[str]],
    train_files: list[str],
    val_files: list[str],
    image_dir: Path,
    out_dir: Path = YOLO_TILE_DIR,
    tile: int = TILE_SIZE_DEFAULT,
    overlap: int = TILE_OVERLAP_DEFAULT,
    hard_neg_per_jpeg: int = HARD_NEG_PER_JPEG,
    seed: int = SEED,
) -> tuple[Path, dict]:
    """GT-centric native-2048 tiles + 2 hard-neg empty-disk tiles per JPEG.

    Labels are medoid polygons in tile coords. Never rmtree yolo_seg / yolo_seg_all
    (or the other live families). Train JSON only. merge_gap / CC-split are infer-time.
    """
    import cv2

    _refuse_tile_out_dir(out_dir)
    _refuse_live_yolo_dirs(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_files = sorted(train_files)
    val_files = sorted(val_files)
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    n_hard_neg = {"train": 0, "val": 0}
    n_gt = {"train": 0, "val": 0}
    n_source = {"train": 0, "val": 0}

    def export_files(files: list[str], split: str) -> None:
        for i, file_name in enumerate(files):
            view = views[medoid_view_id(views, by_file[file_name])]
            image = _read_source_bgr(image_dir, file_name, split)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            stem = Path(file_name).stem
            occupied: set[tuple[int, int]] = set()
            gt_origins = gt_centric_origins(view.filaments, tile, overlap)
            for x0, y0 in gt_origins:
                occupied.add((x0, y0))
                name, lines = _write_one_tile(image, view.filaments, x0, y0, tile, out_dir, split, stem, "gt")
                if not lines:
                    (out_dir / "images" / split / name).unlink(missing_ok=True)
                    (out_dir / "labels" / split / f"{Path(name).stem}.txt").unlink(missing_ok=True)
                    occupied.discard((x0, y0))
                    continue
                n_gt[split] += 1
                records.append(
                    {
                        "split": split,
                        "source": file_name,
                        "tile": name,
                        "x0": x0,
                        "y0": y0,
                        "kind": "gt",
                        "n_polys": len(lines),
                    }
                )
            for x0, y0 in _hard_neg_origins(gray, view.filaments, occupied, tile, overlap, hard_neg_per_jpeg, rng):
                name, _ = _write_one_tile(image, view.filaments, x0, y0, tile, out_dir, split, stem, "hard_neg")
                n_hard_neg[split] += 1
                records.append(
                    {
                        "split": split,
                        "source": file_name,
                        "tile": name,
                        "x0": x0,
                        "y0": y0,
                        "kind": "hard_neg",
                        "n_polys": 0,
                    }
                )
            n_source[split] += 1
            if (i + 1) % 50 == 0 or i + 1 == len(files):
                print(f"yolo_tiles {split} {i + 1}/{len(files)}", flush=True)

    export_files(train_files, "train")
    export_files(val_files, "val")

    yaml_path = _write_yaml(out_dir)
    n_train = n_gt["train"] + n_hard_neg["train"]
    n_val = n_gt["val"] + n_hard_neg["val"]
    train_img, train_lab = _split_stems(out_dir, "train")
    val_img, val_lab = _split_stems(out_dir, "val")
    if train_img != train_lab or val_img != val_lab:
        raise RuntimeError("Tile image/label stems do not match")
    test_stems = _test_stems()
    overlap_test = sorted((train_img | val_img) & test_stems)
    if overlap_test:
        raise RuntimeError(f"Tile stems overlap test stems: {overlap_test[:8]}")
    source_overlap = set(train_files) & set(val_files)
    if source_overlap:
        raise RuntimeError(f"Train/val source JPEG leak: {sorted(source_overlap)[:4]}")

    yolo_val_match = None
    yolo_val_dir = YOLO_DIR / "images" / "val"
    if yolo_val_dir.exists():
        yolo_val = {p.name for p in yolo_val_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        yolo_val_match = yolo_val == set(val_files)

    contract = stitch_eval_contract(tile, overlap)
    (out_dir / "split_counts.txt").write_text(
        f"train={n_train} val={n_val} hard_neg={n_hard_neg['train']} "
        f"tile={tile} overlap={overlap} source={n_source['train']}/{n_source['val']}\n"
    )
    stats = {
        "mode": "tile_native_medoid",
        "family": "tile_native_s640",
        "n_train_tiles": n_train,
        "n_val_tiles": n_val,
        "n_hard_neg": n_hard_neg["train"],
        "n_val_hard_neg": n_hard_neg["val"],
        "n_train_gt_tiles": n_gt["train"],
        "n_val_gt_tiles": n_gt["val"],
        "n_train_source_jpegs": n_source["train"],
        "n_val_source_jpegs": n_source["val"],
        "tile": tile,
        "overlap": overlap,
        "stride": tile - overlap,
        "later_imgsz": 640,
        "labels": "medoid polygons in tile coords",
        "hard_neg_per_train_jpeg": hard_neg_per_jpeg,
        "val_recipe": "same GT-centric cover + 2 hard-neg empty-disk tiles as train",
        "images_labels_match": True,
        "test_stem_overlap": 0,
        "test_labels_used": False,
        "val_files_match_yolo_seg": yolo_val_match,
        "yaml": str(yaml_path),
        "out_dir": str(out_dir.resolve()),
        "seed": seed,
        "merge_gap": 0,
        "cc_split": False,
        "stitch_eval_contract": contract,
        "did_not_touch": [
            str(YOLO_DIR),
            str(YOLO_ALL_DIR),
            str(YOLO_SHIFT_DIR),
            str(YOLO_PHYS_DIR),
            str(YOLO_RATER_DIR),
            str(WEIGHTS_DIR / "yolo11s_seg_all"),
        ],
    }
    (out_dir / "tile_stats.json").write_text(json.dumps({k: v for k, v in stats.items()}, indent=2))
    (out_dir / "tile_map.json").write_text(json.dumps(records, indent=2))
    (out_dir / "stitch_eval_contract.json").write_text(json.dumps(contract, indent=2))
    return yaml_path, stats
