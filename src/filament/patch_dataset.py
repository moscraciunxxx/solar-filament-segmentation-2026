"""On-the-fly native-2048 1024-crop pairs for `unet_patch_native_1024_dhebe`.

CLAHE once on the full disk, then crop. Target is the native-2048 medoid
union, cropped — never resize 2048→1024 then crop. Train grid == infer
grid via `patch_boxes()`. Caller passes files (fold != 0 for train).
No YOLO txt export. No test JSON.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .coco_io import AnnotatorView, medoid_view_id, rasterize_polygons
from .patch_grid import PATCH_OVERLAP, PATCH_SIZE, crop_patch, patch_boxes
from .paths import IMAGE_SIZE, TRAIN_IMG_DIR
from .preprocess import apply_clahe, read_gray
from .semantic_dataset import gray_as_rgb


def n_patches(n_files: int, n_boxes: int | None = None) -> int:
    """Regular grid: one crop per `patch_boxes()` entry per JPEG."""
    if n_boxes is None:
        n_boxes = len(patch_boxes())
    return n_files * n_boxes


def geometric_disk_frac(
    box: tuple[int, int, int, int],
    image_size: int = IMAGE_SIZE,
    radius: int = 1000,
) -> float:
    """On-disk fraction from the centered circle. No image I/O."""
    y0, x0, y1, x1 = box
    yy, xx = np.ogrid[y0:y1, x0:x1]
    cy, cx = (image_size - 1) / 2.0, (image_size - 1) / 2.0
    return float((np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) <= radius).mean())


class NativePatchDataset(Dataset):
    """One sample per regular-grid crop.

    Length is `n_patches(len(files))` = `len(files) * 9` for the default
    1024/256 grid on 2048 (origins 0/768/1024). Empty on-disk crops stay
    as background. Optional drop is only all-zero off-limb corners
    (`geometric_disk_frac == 0`; that implies mask==0 on this grid).
    """

    def __init__(
        self,
        views: dict[str, AnnotatorView],
        by_file: dict[str, list[str]],
        files: list[str],
        image_dir: Path | None = None,
        augment: bool = False,
        drop_offlimb_empty: bool = False,
    ):
        self.views = views
        self.by_file = by_file
        self.files = list(files)
        self.image_dir = Path(image_dir) if image_dir is not None else TRAIN_IMG_DIR
        self.augment = augment
        self.boxes = patch_boxes(IMAGE_SIZE, PATCH_SIZE, PATCH_OVERLAP)
        self.items: list[tuple[str, tuple[int, int, int, int]]] = []
        for file_name in self.files:
            for box in self.boxes:
                if drop_offlimb_empty and geometric_disk_frac(box) == 0.0:
                    continue
                self.items.append((file_name, box))
        self._cache_name: str | None = None
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.items)

    def _full_disk(self, file_name: str) -> tuple[np.ndarray, np.ndarray]:
        if self._cache_name == file_name and self._cache is not None:
            return self._cache
        gray = read_gray(self.image_dir / file_name)
        clahe = apply_clahe(gray, clip_limit=2.0, tile=8)
        view = self.views[medoid_view_id(self.views, self.by_file[file_name])]
        mask = rasterize_polygons([f.polygon for f in view.filaments], size=IMAGE_SIZE)
        self._cache_name = file_name
        self._cache = (clahe, mask)
        return self._cache

    def __getitem__(self, idx: int):
        file_name, box = self.items[idx]
        clahe, mask = self._full_disk(file_name)
        gray_crop = np.ascontiguousarray(crop_patch(clahe, box))
        mask_crop = np.ascontiguousarray(crop_patch(mask, box))
        if self.augment:
            if np.random.rand() < 0.5:
                gray_crop = np.ascontiguousarray(gray_crop[:, ::-1])
                mask_crop = np.ascontiguousarray(mask_crop[:, ::-1])
            if np.random.rand() < 0.5:
                gray_crop = np.ascontiguousarray(gray_crop[::-1, :])
                mask_crop = np.ascontiguousarray(mask_crop[::-1, :])
        x = torch.from_numpy(gray_as_rgb(gray_crop))
        y = torch.from_numpy(mask_crop.astype(np.float32)).unsqueeze(0)
        return x, y, file_name, box
