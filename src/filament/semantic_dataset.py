"""Full-disk semantic pairs. v1 is gray-as-RGB + medoid union. Train JSON only."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .coco_io import AnnotatorView, medoid_view_id, rasterize_polygons
from .paths import IMAGE_SIZE
from .preprocess import read_gray


def gray_as_rgb(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32) / 255.0
    return np.stack([g, g, g], axis=0)


def resize_pair(gray: np.ndarray, mask: np.ndarray, imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    if gray.shape[0] == imgsz:
        return gray, mask
    image = cv2.resize(gray, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    target = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
    return image, target


class FullDiskSemanticDataset(Dataset):
    """One sample per physical JPEG. Target = medoid union mask."""

    def __init__(
        self,
        views: dict[str, AnnotatorView],
        by_file: dict[str, list[str]],
        files: list[str],
        image_dir: Path,
        imgsz: int = 1024,
        augment: bool = False,
        stack: str = "gray_as_rgb",
    ):
        self.views = views
        self.by_file = by_file
        self.files = list(files)
        self.image_dir = Path(image_dir)
        self.imgsz = imgsz
        self.augment = augment
        self.stack = stack

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        file_name = self.files[idx]
        gray = read_gray(self.image_dir / file_name)
        view = self.views[medoid_view_id(self.views, self.by_file[file_name])]
        mask = rasterize_polygons([f.polygon for f in view.filaments], size=IMAGE_SIZE)
        if self.stack == "disk_clahe_sobel":
            from .classical import disk_clahe_sobel_stack

            hwc = disk_clahe_sobel_stack(gray)
            if hwc.shape[0] != self.imgsz:
                hwc = cv2.resize(hwc, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
            if mask.shape[0] != self.imgsz:
                mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
            if self.augment:
                if np.random.rand() < 0.5:
                    hwc = np.ascontiguousarray(hwc[:, ::-1])
                    mask = np.ascontiguousarray(mask[:, ::-1])
                if np.random.rand() < 0.5:
                    hwc = np.ascontiguousarray(hwc[::-1, :])
                    mask = np.ascontiguousarray(mask[::-1, :])
            x = torch.from_numpy(np.transpose(hwc.astype(np.float32) / 255.0, (2, 0, 1)))
        else:
            gray, mask = resize_pair(gray, mask, self.imgsz)
            if self.augment:
                if np.random.rand() < 0.5:
                    gray = np.ascontiguousarray(gray[:, ::-1])
                    mask = np.ascontiguousarray(mask[:, ::-1])
                if np.random.rand() < 0.5:
                    gray = np.ascontiguousarray(gray[::-1, :])
                    mask = np.ascontiguousarray(mask[::-1, :])
            x = torch.from_numpy(gray_as_rgb(gray))
        y = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return x, y, file_name
