"""Crop U-Net refiner: raise mask IoU of YOLO seeds over the PQ 0.5 gate."""

from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .coco_io import AnnotatorView, medoid_view_id, rasterize_polygons
from .paths import IMAGE_SIZE
from .preprocess import apply_clahe, read_gray


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()

        def block(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
            )

        self.e1 = block(in_ch, base)
        self.e2 = block(base, base * 2)
        self.e3 = block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        y2 = self.d2(torch.cat([self.up2(x3), x2], 1))
        y1 = self.d1(torch.cat([self.up1(y2), x1], 1))
        return self.out(y1)


def crop_window(bbox, pad_ratio=0.8, min_size=96, size=IMAGE_SIZE):
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + pad_ratio)
    side = max(side, min_size)
    x0 = int(np.clip(cx - side / 2.0, 0, size - 1))
    y0 = int(np.clip(cy - side / 2.0, 0, size - 1))
    x1 = int(np.clip(cx + side / 2.0, 1, size))
    y1 = int(np.clip(cy + side / 2.0, 1, size))
    if x1 <= x0:
        x1 = min(size, x0 + 2)
    if y1 <= y0:
        y1 = min(size, y0 + 2)
    return x0, y0, x1, y1


def make_crop_tensor(
    image: np.ndarray,
    seed: np.ndarray,
    box,
    crop_size=256,
    clahe: np.ndarray | None = None,
) -> torch.Tensor:
    x0, y0, x1, y1 = box
    gray = image[y0:y1, x0:x1]
    clahe_img = apply_clahe(image) if clahe is None else clahe
    clahe_c = clahe_img[y0:y1, x0:x1]
    seed_c = seed[y0:y1, x0:x1]
    stack = np.stack([gray, clahe_c, seed_c * 255], axis=0).astype(np.float32) / 255.0
    stacked = np.stack(
        [cv2.resize(ch, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR) for ch in stack]
    )
    return torch.from_numpy(stacked)


class CropRefinerDataset(Dataset):
    def __init__(
        self,
        views: dict[str, AnnotatorView],
        by_file: dict[str, list[str]],
        files: list[str],
        image_dir,
        crop_size=256,
        medoid_only: bool = False,
    ):
        self.samples = []
        self.image_dir = image_dir
        self.crop_size = crop_size
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_max = 48
        for file_name in files:
            image_ids = by_file[file_name]
            if medoid_only:
                image_ids = [medoid_view_id(views, image_ids)]
            for image_id in image_ids:
                for filament in views[image_id].filaments:
                    self.samples.append((file_name, filament))

    def __len__(self):
        return len(self.samples)

    def _image(self, file_name: str) -> tuple[np.ndarray, np.ndarray]:
        hit = self._cache.get(file_name)
        if hit is None:
            image = read_gray(self.image_dir / file_name)
            hit = (image, apply_clahe(image))
            self._cache[file_name] = hit
            if len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(file_name)
        return hit

    def __getitem__(self, idx):
        file_name, filament = self.samples[idx]
        image, clahe = self._image(file_name)
        gt = rasterize_polygons([filament.polygon])
        box = crop_window(filament.bbox)
        k = np.random.choice([0, 1, 3, 5])
        seed = gt.copy()
        if k:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            seed = cv2.erode(seed, kernel) if np.random.rand() < 0.5 else cv2.dilate(seed, kernel)
        x = make_crop_tensor(image, seed, box, self.crop_size, clahe=clahe)
        x0, y0, x1, y1 = box
        y = cv2.resize(gt[y0:y1, x0:x1], (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        return x, torch.from_numpy(y.astype(np.float32)).unsqueeze(0)


def paste_crop(full: np.ndarray, crop: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    resized = cv2.resize(crop.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    out = full.copy().astype(np.float32)
    out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], resized)
    return out


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha=0.3, beta=0.7, eps=1.0) -> torch.Tensor:
    """beta>alpha penalizes FN so near-miss masks grow past the official 0.5 IoU gate."""
    pred = torch.sigmoid(logits)
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    return 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)


@torch.no_grad()
def refine_instances(model, image: np.ndarray, masks: list[np.ndarray], device, thr=0.45) -> list[np.ndarray]:
    model.eval()
    clahe = apply_clahe(image)
    refined = []
    for mask in masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        bbox = (float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
        box = crop_window(bbox)
        tensor = make_crop_tensor(image, mask, box, clahe=clahe).unsqueeze(0).to(device)
        pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        full = np.zeros_like(mask, dtype=np.float32)
        full = paste_crop(full, pred, box)
        binary = (full >= thr).astype(np.uint8)
        if binary.sum() >= 64:
            refined.append(binary)
        else:
            refined.append(mask)
    return refined
