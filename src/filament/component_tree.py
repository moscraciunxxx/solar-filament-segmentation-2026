"""Component-tree decode of one probability map.

The antichain is chosen from the map alone. Fold-0 labels are not an input.
"""

from __future__ import annotations

import cv2
import numpy as np

LEVELS = np.linspace(0.05, 0.80, 20)
MIN_AREA = 64
BASE_THRESHOLD = 0.5


def fit_log_area_prior(areas: list[float]) -> tuple[float, float]:
    """Median and log-area sigma from training-fold filament areas."""
    clean = np.asarray([a for a in areas if a >= 1.0], dtype=np.float64)
    if clean.size == 0:
        return 1.0, 1.0
    logs = np.log(clean)
    sigma = float(np.std(logs))
    return float(np.median(clean)), max(sigma, 0.35)


def area_prior(area: int, median: float, sigma: float) -> float:
    if area <= 0 or median <= 0 or sigma <= 0:
        return 0.0
    z = (np.log(float(area)) - np.log(float(median))) / float(sigma)
    return float(np.exp(-0.5 * z * z))


def select_antichain(scores: list[float], children: list[list[int]], roots: list[int]) -> list[int]:
    """Max-score set of nodes with no node inside another."""
    best = [0.0] * len(scores)
    take_self = [False] * len(scores)

    def visit(index: int) -> None:
        child_sum = 0.0
        for child in children[index]:
            visit(child)
            child_sum += best[child]
        if scores[index] >= child_sum:
            best[index] = scores[index]
            take_self[index] = True
        else:
            best[index] = child_sum
            take_self[index] = False

    for root in roots:
        visit(root)

    chosen: list[int] = []

    def collect(index: int) -> None:
        if take_self[index]:
            if scores[index] > 0:
                chosen.append(index)
            return
        for child in children[index]:
            collect(child)

    for root in roots:
        collect(root)
    return chosen


def _components(prob: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    mask = (prob >= threshold).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    return labels, int(count)


def _elongation(labels: np.ndarray, label: int) -> float:
    """Long side over short side of the minimum-area rectangle. A square is 1."""
    mask = (labels == label).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        return 1.0
    (_, _), (width, height), _ = cv2.minAreaRect(contour)
    short = max(min(width, height), 1.0)
    return float(max(width, height) / short)


def cap_count(masks: list[np.ndarray], scores: list[float], cap: int) -> list[np.ndarray]:
    """Keep at most `cap` disjoint masks, highest score first."""
    if cap <= 0 or not masks:
        return []
    if len(masks) <= cap:
        return masks
    order = sorted(range(len(masks)), key=lambda index: scores[index], reverse=True)
    return [masks[index] for index in order[:cap]]


def baseline_components(prob: np.ndarray, threshold: float = BASE_THRESHOLD, min_area: int = MIN_AREA) -> list[np.ndarray]:
    """One threshold, then connected components."""
    labels, count = _components(prob, threshold)
    masks = []
    for label in range(1, count):
        mask = labels == label
        if int(mask.sum()) < min_area:
            continue
        masks.append(mask.astype(np.uint8))
    return masks


def build_tree_graph(
    prob: np.ndarray,
    median: float,
    sigma: float,
    levels: np.ndarray = LEVELS,
    min_area_floor: int = MIN_AREA,
) -> dict:
    """Nodes of area at least the floor. Filters are applied later, not here."""
    prob = np.ascontiguousarray(prob, dtype=np.float32)
    flat_prob = prob.ravel()
    label_maps: list[np.ndarray] = []
    level_index: list[list[int | None]] = []
    raw_scores: list[float] = []
    means: list[float] = []
    areas: list[int] = []
    elongs: list[float] = []
    children: list[list[int]] = []
    kept: list[tuple[int, int]] = []

    for level, threshold in enumerate(levels):
        labels, count = _components(prob, float(threshold))
        label_maps.append(labels)
        sums = np.bincount(labels.ravel(), weights=flat_prob, minlength=count)
        counts = np.bincount(labels.ravel(), minlength=count)
        ids: list[int | None] = [None] * count
        for label in range(1, count):
            area = int(counts[label])
            if area < min_area_floor:
                continue
            mean = float(sums[label] / area)
            ids[label] = len(raw_scores)
            raw_scores.append(mean * area_prior(area, median, sigma))
            means.append(mean)
            areas.append(area)
            elongs.append(_elongation(labels, label))
            children.append([])
            kept.append((level, label))
        level_index.append(ids)

    for level in range(1, len(levels)):
        child = label_maps[level].ravel()
        parent = label_maps[level - 1].ravel()
        active = child > 0
        if not np.any(active):
            continue
        child_ids = child[active]
        parent_ids = parent[active]
        order = np.argsort(child_ids, kind="mergesort")
        child_ids = child_ids[order]
        parent_ids = parent_ids[order]
        starts = np.flatnonzero(np.diff(child_ids, prepend=-1))
        for start in starts:
            child_label = int(child_ids[start])
            parent_label = int(parent_ids[start])
            child_node = level_index[level][child_label]
            if child_node is None or parent_label <= 0 or parent_label >= len(level_index[level - 1]):
                continue
            parent_node = level_index[level - 1][parent_label]
            if parent_node is None:
                continue
            children[parent_node].append(child_node)

    return {
        "raw_scores": raw_scores,
        "means": means,
        "areas": areas,
        "elongs": elongs,
        "children": children,
        "kept": kept,
        "label_maps": label_maps,
    }


def masks_from_graph(
    graph: dict,
    *,
    min_area: int,
    min_mean: float,
    min_elongation: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Antichain after the frozen area, probability, and elongation gates."""
    raw_scores = graph["raw_scores"]
    if not raw_scores:
        return [], []
    scores = []
    for index, raw in enumerate(raw_scores):
        keep = (
            graph["areas"][index] >= min_area
            and graph["means"][index] >= min_mean
            and graph["elongs"][index] >= min_elongation
        )
        scores.append(raw if keep else 0.0)
    children = graph["children"]
    pointed = {child for kids in children for child in kids}
    roots = [index for index in range(len(scores)) if index not in pointed]
    selected = select_antichain(scores, children, roots)
    masks = []
    chosen_scores = []
    for index in selected:
        level, label = graph["kept"][index]
        masks.append((graph["label_maps"][level] == label).astype(np.uint8))
        chosen_scores.append(scores[index])
    return masks, chosen_scores


def tree_components(
    prob: np.ndarray,
    median: float,
    sigma: float,
    levels: np.ndarray = LEVELS,
    min_area: int = MIN_AREA,
    min_mean: float = 0.0,
    min_elongation: float = 1.0,
) -> list[np.ndarray]:
    """Nested thresholds, then the max-score disjoint antichain."""
    graph = build_tree_graph(prob, median, sigma, levels=levels, min_area_floor=min_area)
    masks, _scores = masks_from_graph(
        graph,
        min_area=min_area,
        min_mean=min_mean,
        min_elongation=min_elongation,
    )
    return masks
