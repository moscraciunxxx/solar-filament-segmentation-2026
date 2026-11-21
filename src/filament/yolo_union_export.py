"""YOLO-seg export: one JPEG per physical, train = stacked annotator instances.

This is climber-plan rank 4 (union_mask_s1024). It is NOT raster-union +
contour-merge (that would glue adjacent filaments into one instance). It is
NOT all-annotator (926 contradictory copies of the same pixels). Train JSON
only. Val is fold-0 medoid, byte-comparable to yolo_seg.

Never rmtree yolo_seg / yolo_seg_all / yolo_seg_shift / yolo_seg_phys /
yolo_seg_rater. Do not import write_* helpers from yolo_export (tile agent
owns that file).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from .coco_io import AnnotatorView, medoid_view_id, rasterize_polygons
from .paths import FAST_CACHE, IMAGE_SIZE, SEED, TEST_IMG_DIR

# Hardcoded so we do not contest paths.py. Same cache root as the live exports.
YOLO_UNION_DIR = FAST_CACHE / "yolo_seg_union"
YOLO_SEG_DIR = FAST_CACHE / "yolo_seg"

PROTECTED_DIRS = (
    FAST_CACHE / "yolo_seg",
    FAST_CACHE / "yolo_seg_all",
    FAST_CACHE / "yolo_seg_shift",
    FAST_CACHE / "yolo_seg_phys",
    FAST_CACHE / "yolo_seg_rater",
)

DUPLICATE_IOU = 0.5
DUPLICATE_COMPARE_SIZE = 256


def polygon_to_yolo_line(polygon, class_id: int = 0, size: int = IMAGE_SIZE) -> str | None:
    if len(polygon) < 3:
        return None
    xs = np.clip(polygon[:, 0] / size, 0.0, 1.0)
    ys = np.clip(polygon[:, 1] / size, 0.0, 1.0)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
    return f"{class_id} {coords}"


def _refuse_protected(out_dir: Path) -> None:
    target = out_dir.resolve()
    for live in PROTECTED_DIRS:
        if target == live.resolve():
            raise ValueError(f"Refusing to overwrite live train dir {out_dir}")


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


def _label_lines(filaments) -> list[str]:
    lines: list[str] = []
    for filament in filaments:
        line = polygon_to_yolo_line(filament.polygon)
        if line:
            lines.append(line)
    return lines


def _write_label(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _union_lines(
    views: dict[str, AnnotatorView],
    image_ids: list[str],
) -> list[str]:
    """Every annotator polygon is its own YOLO instance. Overlaps stay.

    Two raters marking the same filament produce two lines. That is expected
    and must be counted — we do not NMS, raster-OR, or contour-merge.
    """
    lines: list[str] = []
    for view_id in sorted(image_ids):
        lines.extend(_label_lines(views[view_id].filaments))
    return lines


def _instance_masks(view: AnnotatorView, size: int = DUPLICATE_COMPARE_SIZE) -> list[np.ndarray]:
    scale = size / float(IMAGE_SIZE)
    masks: list[np.ndarray] = []
    for filament in view.filaments:
        if len(filament.polygon) < 3:
            continue
        masks.append(rasterize_polygons([filament.polygon * scale], size=size))
    return masks


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def count_cross_rater_duplicates(
    views: dict[str, AnnotatorView],
    image_ids: list[str],
    iou_thr: float = DUPLICATE_IOU,
) -> int:
    """Pairs of instances from *different* raters with IoU >= thr.

    Documents 'same filament marked twice'. Does not change the exported labels.
    """
    if len(image_ids) < 2:
        return 0
    packed = [(vid, _instance_masks(views[vid])) for vid in sorted(image_ids)]
    n = 0
    for i, (id_i, masks_i) in enumerate(packed):
        for id_j, masks_j in packed[i + 1 :]:
            if id_i == id_j:
                continue
            for mi in masks_i:
                for mj in masks_j:
                    if _mask_iou(mi, mj) >= iou_thr:
                        n += 1
    return n


def compare_val_labels_to_yolo_seg(
    out_dir: Path,
    ref_dir: Path = YOLO_SEG_DIR,
) -> dict:
    """Byte-compare union val labels to the live yolo_seg medoid val set."""
    union_val = out_dir / "labels" / "val"
    ref_val = ref_dir / "labels" / "val"
    union_files = sorted(p.name for p in union_val.glob("*.txt"))
    ref_files = sorted(p.name for p in ref_val.glob("*.txt"))
    identical = 0
    mismatched: list[str] = []
    missing_in_ref: list[str] = []
    extra_in_union: list[str] = []
    union_set = set(union_files)
    ref_set = set(ref_files)
    extra_in_union = sorted(union_set - ref_set)
    missing_in_ref = sorted(ref_set - union_set)
    for name in sorted(union_set & ref_set):
        if (union_val / name).read_bytes() == (ref_val / name).read_bytes():
            identical += 1
        else:
            mismatched.append(name)
    n_union = len(union_files)
    n_ref = len(ref_files)
    ok = (
        n_union == n_ref
        and identical == n_ref
        and not mismatched
        and not extra_in_union
        and not missing_in_ref
    )
    return {
        "n_union_val_labels": n_union,
        "n_yolo_seg_val_labels": n_ref,
        "n_identical": identical,
        "n_val_labels_identical_to_yolo_seg": f"{identical}/{n_ref}",
        "val_labels_byte_identical_to_yolo_seg": ok,
        "n_mismatched": len(mismatched),
        "mismatched": mismatched[:20],
        "extra_in_union": extra_in_union[:20],
        "missing_in_ref": missing_in_ref[:20],
    }


def test_stem_overlap(file_names: list[str], test_dir: Path = TEST_IMG_DIR) -> int:
    """Stems that appear in both the train JSON split and the unlabeled test JPEGs."""
    if not test_dir.exists():
        return 0
    test_stems = {
        p.stem for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    split_stems = {Path(fn).stem for fn in file_names}
    return len(test_stems & split_stems)


def write_yolo_union(
    views: dict[str, AnnotatorView],
    by_file: dict[str, list[str]],
    train_files: list[str],
    val_files: list[str],
    image_dir: Path,
    out_dir: Path = YOLO_UNION_DIR,
    seed: int = SEED,
) -> tuple[Path, dict]:
    """One physical JPEG per sample. Train = all-annotator polygons as instances."""
    _refuse_protected(out_dir)
    if out_dir.name != "yolo_seg_union":
        raise ValueError(f"Refusing to write union export to unexpected dir {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_files = sorted(train_files)
    val_files = sorted(val_files)

    n_train_instances = 0
    n_train_instances_medoid = 0
    n_train_multi = 0
    n_train_union_gt_medoid = 0
    n_duplicate_pairs = 0
    n_files_with_duplicate = 0
    per_file: list[dict] = []

    for file_name in train_files:
        image_ids = list(by_file[file_name])
        medoid_id = medoid_view_id(views, image_ids)
        union_lines = _union_lines(views, image_ids)
        medoid_lines = _label_lines(views[medoid_id].filaments)
        shutil.copy2(image_dir / file_name, out_dir / "images" / "train" / file_name)
        _write_label(out_dir / "labels" / "train" / f"{Path(file_name).stem}.txt", union_lines)

        n_u = len(union_lines)
        n_m = len(medoid_lines)
        n_train_instances += n_u
        n_train_instances_medoid += n_m
        multi = len(image_ids) > 1
        if multi:
            n_train_multi += 1
        if n_u > n_m:
            n_train_union_gt_medoid += 1
        dup = count_cross_rater_duplicates(views, image_ids) if multi else 0
        n_duplicate_pairs += dup
        if dup:
            n_files_with_duplicate += 1
        per_file.append(
            {
                "file_name": file_name,
                "n_raters": len(image_ids),
                "n_union_instances": n_u,
                "n_medoid_instances": n_m,
                "n_cross_rater_iou50_pairs": dup,
                "medoid_view": medoid_id,
            }
        )

    for file_name in val_files:
        image_ids = list(by_file[file_name])
        view = views[medoid_view_id(views, image_ids)]
        shutil.copy2(image_dir / file_name, out_dir / "images" / "val" / file_name)
        _write_label(
            out_dir / "labels" / "val" / f"{Path(file_name).stem}.txt",
            _label_lines(view.filaments),
        )

    n_train = len(train_files)
    n_val = len(val_files)
    n_val_instances = 0
    for file_name in val_files:
        n_val_instances += len(
            _label_lines(views[medoid_view_id(views, by_file[file_name])].filaments)
        )

    yaml_path = _write_yaml(out_dir)
    val_cmp = compare_val_labels_to_yolo_seg(out_dir)
    overlap = test_stem_overlap(train_files + val_files)
    delta = n_train_instances - n_train_instances_medoid

    (out_dir / "split_counts.txt").write_text(
        f"train={n_train} val={n_val} train_instances={n_train_instances} "
        f"medoid_instances={n_train_instances_medoid} delta={delta:+d} "
        f"mode=union_instances seed={seed}\n"
    )

    stats = {
        "mode": "union_instances",
        "choice": (
            "UNION of all annotator polygons as separate YOLO instances on one "
            "physical JPEG. Not raster-union+contour (would merge adjacent "
            "filaments). Not 926 contradictory all-annotator copies."
        ),
        "val_labels": "medoid",
        "val_note": (
            "Same grouped fold-0 JPEGs and medoid labels as yolo_seg "
            f"(seed {seed}). Extra val annotators are withheld."
        ),
        "duplicate_note": (
            "Two raters marking the same filament is expected and counted as "
            f"two instances. Cross-rater pairs with mask IoU>={DUPLICATE_IOU} "
            f"at {DUPLICATE_COMPARE_SIZE}px: {n_duplicate_pairs} pairs on "
            f"{n_files_with_duplicate} train JPEGs. Those pairs stay as "
            "overlapping YOLO instances; we do not drop or merge them."
        ),
        "n_train_files": n_train,
        "n_val_files": n_val,
        "n_train_instances": n_train_instances,
        "n_train_instances_medoid": n_train_instances_medoid,
        "n_train_instances_delta": delta,
        "n_val_instances": n_val_instances,
        "n_train_multi_annotator": n_train_multi,
        "n_train_files_union_gt_medoid": n_train_union_gt_medoid,
        "n_cross_rater_iou50_pairs": n_duplicate_pairs,
        "n_train_files_with_cross_rater_duplicate": n_files_with_duplicate,
        "duplicate_iou_threshold": DUPLICATE_IOU,
        "n_dataset_multi_annotator": sum(1 for ids in by_file.values() if len(ids) > 1),
        "val_compare": val_cmp,
        "test_overlap": overlap,
        "test_labels_used": False,
        "seed": seed,
        "yaml": str(yaml_path),
        "out_dir": str(out_dir.resolve()),
        "did_not_touch": [str(p.resolve()) for p in PROTECTED_DIRS],
        "train_file_stats": per_file,
    }
    (out_dir / "union_stats.json").write_text(json.dumps(stats, indent=2))
    return yaml_path, stats
