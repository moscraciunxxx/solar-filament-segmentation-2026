"""CLV flat-field images and the soft mean of every annotator. Not the medoid."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .coco_io import rasterize_polygons
from .flatfield import center_to_limb_flat
from .paths import IMAGE_SIZE
from .preprocess import read_gray

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def soft_annotator_mean(views, image_ids: list[str], size: int = IMAGE_SIZE) -> np.ndarray:
    """Mean of every annotator's raster. Values lie in [0, 1]."""
    if not image_ids:
        return np.zeros((size, size), dtype=np.float32)
    scale = size / float(IMAGE_SIZE)
    total = np.zeros((size, size), dtype=np.float32)
    for image_id in image_ids:
        polygons = [filament.polygon * scale for filament in views[image_id].filaments]
        total += rasterize_polygons(polygons, size=size)
    return total / float(len(image_ids))


class SoftClvDataset(Dataset):
    def __init__(self, views, by_file, files: list[str], image_dir: Path, imgsz: int, augment: bool) -> None:
        self.views = views
        self.by_file = by_file
        self.files = list(files)
        self.image_dir = Path(image_dir)
        self.imgsz = imgsz
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        name = self.files[index]
        gray = read_gray(self.image_dir / name)
        flat = center_to_limb_flat(gray)
        target = soft_annotator_mean(self.views, self.by_file[name], size=IMAGE_SIZE)
        flat = cv2.resize(flat, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
        if self.augment:
            if np.random.rand() < 0.5:
                flat = np.ascontiguousarray(flat[:, ::-1])
                target = np.ascontiguousarray(target[:, ::-1])
            if np.random.rand() < 0.5:
                flat = np.ascontiguousarray(flat[::-1])
                target = np.ascontiguousarray(target[::-1])
        rgb = np.stack([flat, flat, flat], axis=-1)
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(np.transpose(rgb, (2, 0, 1)).astype(np.float32))
        label = torch.from_numpy(target.astype(np.float32)[None])
        return image, label, name
