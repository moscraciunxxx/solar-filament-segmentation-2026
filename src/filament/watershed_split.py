"""Split one predicted mask on its own distance-transform peaks.

The only cut is ``MIN_PEAK_DISTANCE`` (48 pixels, Euclidean), fixed before
any Panoptic Quality is computed. A peak is one pixel of a 3x3 regional
maximum of the Euclidean distance transform. Lower peaks closer than 48
pixels are the same mode. When at least two modes remain, Meyer watershed
on the inverted distance transform partitions that mask. No image
intensity, ground-truth spine, or polygon is read.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage as ndi

MIN_PEAK_DISTANCE = 48

_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def split_mask(mask: np.ndarray, min_peak_distance: int = MIN_PEAK_DISTANCE) -> list[np.ndarray]:
    """Return the input as one mask, or two or more disjoint covering pieces."""
    pieces = _split_binary(np.asarray(mask) > 0, min_peak_distance)
    if pieces is None:
        return [(np.asarray(mask) > 0).astype(np.uint8)]
    return pieces


def _split_binary(mask: np.ndarray, min_peak_distance: int) -> list[np.ndarray] | None:
    """Return full-frame pieces, or None when the mask is left unchanged."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or int(binary.sum()) < 2:
        return None
    ys, xs = np.nonzero(binary)
    y0 = int(ys.min())
    y1 = int(ys.max())
    x0 = int(xs.min())
    x1 = int(xs.max())
    if (y1 - y0) ** 2 + (x1 - x0) ** 2 < min_peak_distance ** 2:
        return None

    height, width = binary.shape
    y0p = max(y0 - 1, 0)
    x0p = max(x0 - 1, 0)
    y1p = min(y1 + 2, height)
    x1p = min(x1 + 2, width)
    crop = binary[y0p:y1p, x0p:x1p]
    distance = ndi.distance_transform_edt(crop)
    peaks = _peaks(distance, crop, min_peak_distance)
    if len(peaks) < 2:
        return None

    labels = _watershed(-distance, peaks, crop)
    labels = _fill_orphans(labels, crop)
    pieces = _pieces_from_labels(labels, crop)
    if len(pieces) < 2:
        return None

    full: list[np.ndarray] = []
    for piece in pieces:
        canvas = np.zeros((height, width), dtype=np.uint8)
        canvas[y0p:y1p, x0p:x1p] = piece
        full.append(canvas)
    return full


def _peaks(distance: np.ndarray, mask: np.ndarray, min_peak_distance: int) -> list[tuple[int, int]]:
    filtered = ndi.maximum_filter(distance, size=3, mode="constant", cval=0.0)
    regional = mask & (distance > 0) & (distance == filtered)
    labeled, n_regions = ndi.label(regional, structure=np.ones((3, 3), dtype=int))
    if n_regions == 0:
        return []
    candidates: list[tuple[float, int, int]] = []
    for index in range(1, n_regions + 1):
        ys, xs = np.nonzero(labeled == index)
        vals = distance[ys, xs]
        choice = int(np.lexsort((xs, ys, -vals))[0])
        candidates.append((float(vals[choice]), int(ys[choice]), int(xs[choice])))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    kept: list[tuple[int, int]] = []
    min_d2 = min_peak_distance * min_peak_distance
    for _, y, x in candidates:
        if all((y - ky) * (y - ky) + (x - kx) * (x - kx) >= min_d2 for ky, kx in kept):
            kept.append((y, x))
    return kept


def _watershed(topography: np.ndarray, peaks: list[tuple[int, int]], mask: np.ndarray) -> np.ndarray:
    """Marker watershed. ``topography`` is already negated distance (peaks are low)."""
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    queued = np.zeros((height, width), dtype=bool)
    heap: list[tuple[float, int, int, int, int]] = []
    counter = 0
    for lab, (y, x) in enumerate(peaks, start=1):
        labels[y, x] = lab
        queued[y, x] = True
        for ny, nx, nd in _unlabeled_neighbors(y, x, mask, queued):
            queued[ny, nx] = True
            counter += 1
            heapq.heappush(heap, (float(topography[ny, nx]), counter, ny, nx, lab))
    while heap:
        _, _, y, x, lab = heapq.heappop(heap)
        labels[y, x] = lab
        for ny, nx, _ in _unlabeled_neighbors(y, x, mask, queued):
            queued[ny, nx] = True
            counter += 1
            heapq.heappush(heap, (float(topography[ny, nx]), counter, ny, nx, lab))
    return labels


def _unlabeled_neighbors(
    y: int,
    x: int,
    mask: np.ndarray,
    queued: np.ndarray,
) -> list[tuple[int, int, float]]:
    height, width = mask.shape
    found: list[tuple[int, int, float]] = []
    for oy, ox in _OFFSETS:
        ny = y + oy
        nx = x + ox
        if ny < 0 or nx < 0 or ny >= height or nx >= width:
            continue
        if not mask[ny, nx] or queued[ny, nx]:
            continue
        found.append((ny, nx, 0.0))
    return found


def _fill_orphans(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Give foreground the watershed missed to the nearest marker basin.

    A suppressed peak is under 48 pixels from a kept peak, so its component
    stays in that basin instead of becoming a new instance or a hole.
    """
    orphan = mask & (labels == 0)
    if not np.any(orphan):
        return labels
    present = [lab for lab in range(1, int(labels.max()) + 1) if np.any(labels == lab)]
    if not present:
        return labels
    stack = np.stack([ndi.distance_transform_edt(labels != lab) for lab in present], axis=0)
    choice = np.argmin(stack, axis=0)
    filled = labels.copy()
    ys, xs = np.nonzero(orphan)
    for y, x in zip(ys, xs, strict=True):
        filled[y, x] = present[int(choice[y, x])]
    return filled


def _pieces_from_labels(labels: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    pieces: list[np.ndarray] = []
    covered = np.zeros(mask.shape, dtype=bool)
    for lab in range(1, int(labels.max()) + 1):
        piece = labels == lab
        if not np.any(piece):
            continue
        if np.any(covered & piece):
            raise RuntimeError("watershed pieces overlap")
        covered |= piece
        pieces.append(piece.astype(np.uint8))
    if not np.array_equal(covered, mask):
        raise RuntimeError("watershed pieces do not cover the mask")
    return pieces


def check_split_contract() -> None:
    """Geometry checks that do not read labels or Panoptic Quality."""
    disk = np.zeros((80, 80), dtype=np.uint8)
    yy, xx = np.ogrid[:80, :80]
    disk[((yy - 40) ** 2 + (xx - 40) ** 2) <= 18**2] = 1
    if len(split_mask(disk)) != 1:
        raise AssertionError("a single disk must not split")

    bar = np.zeros((40, 320), dtype=np.uint8)
    bar[14:26, 10:310] = 1
    if len(split_mask(bar)) != 1:
        raise AssertionError("a constant-width bar must not split")

    close = np.zeros((120, 160), dtype=np.uint8)
    close[((yy_c := np.ogrid[:120, :160])[0] - 60) ** 2 + (yy_c[1] - 70) ** 2 <= 22**2] = 1
    close[(yy_c[0] - 60) ** 2 + (yy_c[1] - 100) ** 2 <= 22**2] = 1
    if len(split_mask(close)) != 1:
        raise AssertionError("peaks closer than 48 pixels must stay one mask")

    blobs = np.zeros((160, 240), dtype=np.uint8)
    by, bx = np.ogrid[:160, :240]
    blobs[(by - 80) ** 2 + (bx - 70) ** 2 <= 20**2] = 1
    blobs[(by - 80) ** 2 + (bx - 170) ** 2 <= 20**2] = 1
    blobs[78:83, 70:171] = 1
    pieces = split_mask(blobs)
    if len(pieces) != 2:
        raise AssertionError(f"two blobs joined by a neck must split into 2, got {len(pieces)}")
    union = np.zeros(blobs.shape, dtype=np.uint8)
    for piece in pieces:
        union = np.maximum(union, piece)
    if not np.array_equal(union > 0, blobs > 0):
        raise AssertionError("split pieces must cover the parent mask without adding pixels")
    if int(np.logical_and(pieces[0], pieces[1]).sum()) != 0:
        raise AssertionError("split pieces must be disjoint")
    areas = sorted(int(piece.sum()) for piece in pieces)
    if areas[0] < 200:
        raise AssertionError(f"split piece collapsed to area {areas[0]}")
