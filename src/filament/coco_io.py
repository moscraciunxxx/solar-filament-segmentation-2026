from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .paths import IMAGE_SIZE, SEED, TRAIN_JSON
from .rle import encode_rle


@dataclass
class Filament:
    polygon: np.ndarray
    bbox: tuple[float, float, float, float]
    area: float
    category_id: int


@dataclass
class AnnotatorView:
    image_id: str
    file_name: str
    annotator: str
    filaments: list[Filament] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return Path(self.file_name).stem


def polygon_to_xy(segmentation: list) -> np.ndarray:
    coords = segmentation[0] if segmentation and isinstance(segmentation[0], list) else segmentation
    arr = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    return arr


def rasterize_polygons(polygons: list[np.ndarray], size: int = IMAGE_SIZE) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def polygon_to_coco_rle(polygon: np.ndarray, size: int = IMAGE_SIZE) -> str:
    """Rasterize one polygon with pycocotools, matching the host RLE decoder.

    The self-evaluation notebook decodes ``segmentation_rle`` with
    ``pycocotools.mask``. OpenCV ``fillPoly`` is a different boundary rule.
    """
    from pycocotools import mask as mask_utils

    flat = np.asarray(polygon, dtype=np.float64).reshape(-1)
    if flat.size < 6:
        return encode_rle(np.zeros((size, size), dtype=np.uint8))
    rles = mask_utils.frPyObjects([flat.tolist()], size, size)
    counts = rles[0]["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return counts


def load_coco(path: Path = TRAIN_JSON) -> tuple[dict[str, AnnotatorView], dict[str, list[str]]]:
    with path.open() as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    views: dict[str, AnnotatorView] = {}
    for img in coco["images"]:
        annotator, _, _ = img["id"].partition("-")
        views[img["id"]] = AnnotatorView(
            image_id=img["id"],
            file_name=img["file_name"],
            annotator=annotator,
        )
    for ann in coco["annotations"]:
        view = views[ann["image_id"]]
        view.filaments.append(
            Filament(
                polygon=polygon_to_xy(ann["segmentation"]),
                bbox=tuple(ann["bbox"]),
                area=float(ann["area"]),
                category_id=int(ann["category_id"]),
            )
        )

    by_file: dict[str, list[str]] = defaultdict(list)
    for image_id, view in views.items():
        by_file[view.file_name].append(image_id)
    return views, dict(by_file)


def semantic_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def _scale_polygons(polygons: list[np.ndarray], scale: float) -> list[np.ndarray]:
    return [poly * scale for poly in polygons]


def medoid_view_id(
    views: dict[str, AnnotatorView],
    image_ids: list[str],
    compare_size: int = 512,
) -> str:
    """Pick the annotator whose union mask is closest to the others.

    Agreement is measured on a downsampled canvas. Instance polygons themselves
    stay at native 2048 resolution when we export YOLO labels.
    """
    if len(image_ids) == 1:
        return image_ids[0]
    scale = compare_size / float(IMAGE_SIZE)
    masks = []
    for image_id in image_ids:
        polys = _scale_polygons([f.polygon for f in views[image_id].filaments], scale)
        masks.append(rasterize_polygons(polys, size=compare_size))
    scores = []
    for i, mask_i in enumerate(masks):
        others = [semantic_iou(mask_i, mask_j) for j, mask_j in enumerate(masks) if i != j]
        scores.append(float(np.mean(others)) if others else 0.0)
    return image_ids[int(np.argmax(scores))]


def grouped_folds(
    by_file: dict[str, list[str]],
    views: dict[str, AnnotatorView],
    n_folds: int = 5,
    seed: int = SEED,
) -> dict[str, int]:
    """Assign each physical JPEG to a fold, stratified by year and filament-count bin."""
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for file_name, image_ids in by_file.items():
        year = int(file_name[:4])
        n_fil = int(np.median([len(views[i].filaments) for i in image_ids]))
        if n_fil <= 3:
            bin_id = 0
        elif n_fil <= 6:
            bin_id = 1
        elif n_fil <= 10:
            bin_id = 2
        else:
            bin_id = 3
        buckets[(year, bin_id)].append(file_name)

    fold_of: dict[str, int] = {}
    fold_sizes = np.zeros(n_folds, dtype=int)
    for key in sorted(buckets):
        files = buckets[key]
        rng.shuffle(files)
        for file_name in files:
            fold = int(np.argmin(fold_sizes))
            fold_of[file_name] = fold
            fold_sizes[fold] += 1
    return fold_of


def gt_records_for_files(
    views: dict[str, AnnotatorView],
    by_file: dict[str, list[str]],
    file_names: list[str],
    rasterizer: str = "cv2",
) -> list[dict]:
    """Build official-schema GT rows.

    rasterizer ``cv2`` is the historical OpenCV fill. ``coco`` uses
    pycocotools.frPyObjects, the same library the host notebook decodes with.
    """
    if rasterizer not in {"cv2", "coco"}:
        raise ValueError(rasterizer)
    records = []
    for file_name in file_names:
        for image_id in by_file[file_name]:
            view = views[image_id]
            for idx, filament in enumerate(view.filaments, start=1):
                if rasterizer == "coco":
                    rle = polygon_to_coco_rle(filament.polygon)
                else:
                    rle = encode_rle(rasterize_polygons([filament.polygon]))
                records.append(
                    {
                        "annotator_image": image_id,
                        "index": idx,
                        "rle": rle,
                        "file_name": file_name,
                    }
                )
    return records
