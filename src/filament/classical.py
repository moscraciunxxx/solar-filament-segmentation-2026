"""Physics-shaped classical detector: filaments are locally darker than the disk."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .postprocess import connected_instances, merge_nearby, resolve_overlaps
from .preprocess import apply_clahe, solar_disk_mask


@dataclass
class ClassicalConfig:
    clahe_clip: float = 2.0
    blur_sigma: float = 36.0
    dark_percentile: float = 96.5
    morph_close: int = 7
    min_area: int = 160
    max_area: int = 80_000
    merge_gap: int = 11
    intensity_min: int = 12
    disk_radius: int = 1000


def dark_residual_channel(image: np.ndarray, cfg: ClassicalConfig | None = None) -> np.ndarray:
    """Continuous on-disk dark residual, percentile-scaled to uint8. Not a binary mask."""
    cfg = cfg or ClassicalConfig()
    enhanced = apply_clahe(image, clip_limit=cfg.clahe_clip)
    disk = solar_disk_mask(enhanced, intensity_min=cfg.intensity_min, radius=cfg.disk_radius)
    background = cv2.GaussianBlur(enhanced, (0, 0), cfg.blur_sigma)
    residual = cv2.subtract(background, enhanced) * disk
    on_disk = residual[disk > 0]
    if on_disk.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    lo, hi = np.percentile(on_disk, (1.0, 99.0))
    scale = max(float(hi - lo), 1.0)
    stretched = np.clip((residual.astype(np.float32) - float(lo)) / scale, 0.0, 1.0)
    return (stretched * 255.0).astype(np.uint8)


def phys_stack(image: np.ndarray, cfg: ClassicalConfig | None = None) -> np.ndarray:
    """BGR stack: gray, tile-CLAHE, local dark residual. Train and infer must match."""
    cfg = cfg or ClassicalConfig()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = apply_clahe(gray, clip_limit=cfg.clahe_clip)
    residual = dark_residual_channel(gray, cfg)
    return cv2.merge([gray, clahe, residual])


def disk_median_flatten(image: np.ndarray, cfg: ClassicalConfig | None = None) -> np.ndarray:
    """Quiet-Sun flatten via median at ~R/8 (computed at 512px, lifted). Not CLAHE."""
    cfg = cfg or ClassicalConfig()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    disk = solar_disk_mask(gray, intensity_min=cfg.intensity_min, radius=cfg.disk_radius)
    h, w = gray.shape
    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    med_s = cv2.medianBlur(small, 31)
    med = cv2.resize(med_s, (w, h), interpolation=cv2.INTER_LINEAR)
    med = np.maximum(med, 1)
    ratio = gray.astype(np.float32) / med.astype(np.float32)
    on = ratio[disk > 0]
    mid = float(np.median(on)) if on.size else 1.0
    flat = np.clip(ratio / max(mid, 1e-6) * 128.0, 0.0, 255.0).astype(np.uint8)
    flat[disk == 0] = gray[disk == 0]
    return flat


def flatten_raw_sobel_stack(image: np.ndarray, cfg: ClassicalConfig | None = None) -> np.ndarray:
    """BGR: median flatten, raw gray, Sobel-on-flatten. Method 2. Not phys_stack."""
    cfg = cfg or ClassicalConfig()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat = disk_median_flatten(gray, cfg)
    sx = cv2.Sobel(flat, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(flat, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    disk = solar_disk_mask(gray, intensity_min=cfg.intensity_min, radius=cfg.disk_radius)
    mag = mag * disk.astype(np.float32)
    on = mag[disk > 0]
    if on.size:
        lo, hi = np.percentile(on, (1.0, 99.0))
        mag = np.clip((mag - float(lo)) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    sobel_u8 = (mag * 255.0).astype(np.uint8)
    return cv2.merge([flat, gray, sobel_u8])


def disk_clahe_sobel_stack(image: np.ndarray, cfg: ClassicalConfig | None = None) -> np.ndarray:
    """HWC uint8: dark residual, CLAHE, Sobel magnitude. Not phys_stack / not YOLO phys."""
    cfg = cfg or ClassicalConfig()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = apply_clahe(gray, clip_limit=cfg.clahe_clip)
    residual = dark_residual_channel(gray, cfg)
    sx = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    disk = solar_disk_mask(gray, intensity_min=cfg.intensity_min, radius=cfg.disk_radius)
    mag = mag * disk.astype(np.float32)
    peak = float(mag.max()) if mag.size else 0.0
    if peak > 0:
        mag = mag / peak
    sobel_u8 = (np.clip(mag, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.merge([residual, clahe, sobel_u8])


def disk_clahe_sobel_nchw(image: np.ndarray, cfg: ClassicalConfig | None = None) -> np.ndarray:
    """(3,H,W) float32 in [0,1] for TinyUNet in_ch=3."""
    hwc = disk_clahe_sobel_stack(image, cfg)
    return np.transpose(hwc.astype(np.float32) / 255.0, (2, 0, 1))


def dark_residual_mask(image: np.ndarray, cfg: ClassicalConfig) -> np.ndarray:
    enhanced = apply_clahe(image, clip_limit=cfg.clahe_clip)
    disk = solar_disk_mask(enhanced, intensity_min=cfg.intensity_min, radius=cfg.disk_radius)
    background = cv2.GaussianBlur(enhanced, (0, 0), cfg.blur_sigma)
    residual = cv2.subtract(background, enhanced)
    residual = residual * disk
    on_disk = residual[disk > 0]
    if on_disk.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    thr = float(np.percentile(on_disk, cfg.dark_percentile))
    binary = ((residual >= thr) & (disk > 0)).astype(np.uint8)
    if cfg.morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_close, cfg.morph_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    return binary


def detect_classical(image: np.ndarray, cfg: ClassicalConfig | None = None) -> list[np.ndarray]:
    cfg = cfg or ClassicalConfig()
    binary = dark_residual_mask(image, cfg)
    instances = connected_instances(binary, min_area=cfg.min_area, max_area=cfg.max_area)
    instances = merge_nearby(instances, gap=cfg.merge_gap, max_merged_area=cfg.max_area)
    instances = resolve_overlaps(instances)
    return instances


def detect_merckle(image: np.ndarray, dark_percentile: float = 90.0) -> list[np.ndarray]:
    """Faithful copy of refs/kernels/merckle-046 predict_image (masks only).

    Published public 0.46 is not a local certificate. Score official fold-0 PQ
    before any upload. Does not read test labels or copy champion RLEs.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = gray.astype(np.float32)
    height, width = img.shape
    y, x = np.ogrid[:height, :width]
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    solar_disk = (img > 5) & (((y - center_y) ** 2 + (x - center_x) ** 2) < 1000.0**2)
    background = cv2.GaussianBlur(img, (0, 0), 40.0, borderType=cv2.BORDER_REFLECT)
    dark_residual = background - img
    on_disk = dark_residual[solar_disk]
    if on_disk.size == 0:
        return []
    threshold = max(1.0, float(np.percentile(on_disk, dark_percentile)))
    dark_mask = (dark_residual > threshold) & solar_disk
    kernel = np.ones((3, 3), np.uint8)
    dark_mask = cv2.dilate(dark_mask.astype(np.uint8), kernel, iterations=1)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, 8)
    components: list[tuple[int, int]] = []
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        box_width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        fill = area / max(1, box_width * box_height)
        if area < 80 or area > 150_000:
            continue
        if fill > 0.99:
            continue
        components.append((area, label_id))
    components.sort(reverse=True)
    return [(labels == label_id).astype(np.uint8) for _, label_id in components[:120]]
