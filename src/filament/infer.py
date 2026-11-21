from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .classical import ClassicalConfig, detect_classical, flatten_raw_sobel_stack, phys_stack
from .paths import IMAGE_SIZE
from .postprocess import connected_instances, merge_nearby, resolve_overlaps
from .preprocess import apply_clahe, read_gray
from .rle import encode_rle


def pick_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def weights_tag(weights: str | Path) -> str:
    path = Path(weights)
    if path.parent.name == "weights":
        return path.parent.parent.name
    return path.stem


def load_refiner(path: str | Path, device: str):
    import torch

    from .refiner import TinyUNet

    refiner = TinyUNet()
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    refiner.load_state_dict(state)
    refiner.to(device).eval()
    return refiner


def masks_to_rles(masks: list[np.ndarray]) -> list[str]:
    return [encode_rle(m) for m in masks if m.sum() > 0]


def infer_classical_image(path: Path, cfg: ClassicalConfig | None = None) -> list[str]:
    image = read_gray(path)
    return masks_to_rles(detect_classical(image, cfg))


def yolo_masks_to_instances(
    yolo_masks: list[np.ndarray],
    min_area: int = 96,
    max_area: int = 80_000,
    merge_gap: int = 9,
) -> list[np.ndarray]:
    """Keep detector instances intact. CC-splitting a YOLO mask creates PQ fragments."""
    import cv2

    cleaned: list[np.ndarray] = []
    for mask in yolo_masks:
        if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            mask = cv2.resize(mask.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        area = int(mask.sum())
        if area < min_area or area > max_area:
            continue
        cleaned.append((mask > 0).astype(np.uint8))
    cleaned.sort(key=lambda m: int(m.sum()), reverse=True)
    if merge_gap >= 3:
        cleaned = merge_nearby(cleaned, gap=merge_gap, max_merged_area=max_area)
    return resolve_overlaps(cleaned)


def _collect_yolo_masks(result) -> list[np.ndarray]:
    return [mask for mask, _ in _collect_yolo_instances(result)]


def _collect_yolo_instances(result) -> list[tuple[np.ndarray, float]]:
    if result is None or result.masks is None:
        return []
    data = result.masks.data.cpu().numpy()
    if result.boxes is not None and result.boxes.conf is not None:
        confs = [float(c) for c in result.boxes.conf.cpu().numpy().tolist()]
    else:
        confs = []
    while len(confs) < len(data):
        confs.append(0.0)
    return [((layer > 0.5).astype(np.uint8), confs[i]) for i, layer in enumerate(data)]


def _yolo_predict(
    model,
    source,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    augment: bool = False,
    device: str | None = None,
):
    kwargs = dict(
        source=str(source),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        retina_masks=True,
        augment=augment,
        verbose=False,
    )
    if device:
        kwargs["device"] = device
    return model.predict(**kwargs)


def _predict_on_path(
    model,
    path: Path,
    *,
    clahe: bool,
    phys: bool = False,
    residual: bool = False,
    conf: float,
    iou: float,
    imgsz: int,
    augment: bool = False,
    device: str | None = None,
):
    import cv2

    if sum([bool(clahe), bool(phys), bool(residual)]) > 1:
        raise ValueError("Choose at most one of clahe / phys / residual")
    if not clahe and not phys and not residual:
        return _yolo_predict(
            model, path, conf=conf, iou=iou, imgsz=imgsz, augment=augment, device=device
        )
    gray = read_gray(path)
    if residual:
        encoded = flatten_raw_sobel_stack(gray)
        prefix = "filament_residual_"
    elif phys:
        encoded = phys_stack(gray)
        prefix = "filament_phys_"
    else:
        encoded = apply_clahe(gray)
        prefix = "filament_clahe_"
    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
        tmp = Path(tmpdir) / "in.png"
        cv2.imwrite(str(tmp), encoded)
        return _yolo_predict(
            model, tmp, conf=conf, iou=iou, imgsz=imgsz, augment=augment, device=device
        )


def infer_yolo_refined(
    yolo,
    refiner,
    path: Path,
    conf: float = 0.25,
    iou: float = 0.60,
    imgsz: int = 1024,
    device: str = "cpu",
    thr: float = 0.45,
    augment: bool = False,
    merge_gap: int = 0,
    clahe: bool = False,
    phys: bool = False,
    residual: bool = False,
) -> list[str]:
    from .refiner import refine_instances

    image = read_gray(path)
    results = _predict_on_path(
        yolo, path, clahe=clahe, phys=phys, residual=residual, conf=conf, iou=iou, imgsz=imgsz, augment=augment, device=device
    )
    masks = _collect_yolo_masks(results[0] if results else None)
    if not masks:
        return []
    instances = yolo_masks_to_instances(masks, merge_gap=merge_gap)
    refined = refine_instances(refiner, image, instances, device, thr=thr)
    return masks_to_rles(resolve_overlaps(refined))


def infer_yolo_image(
    model,
    path: Path,
    conf: float = 0.20,
    iou: float = 0.60,
    imgsz: int = 1024,
    augment: bool = False,
    merge_gap: int = 0,
    device: str | None = None,
    clahe: bool = False,
    phys: bool = False,
    residual: bool = False,
) -> list[str]:
    results = _predict_on_path(
        model,
        path,
        clahe=clahe,
        phys=phys,
        residual=residual,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        augment=augment,
        device=device,
    )
    masks = _collect_yolo_masks(results[0] if results else None)
    if not masks:
        return []
    instances = yolo_masks_to_instances(masks, merge_gap=merge_gap)
    return masks_to_rles(instances)


def infer_yolo_tta(
    model,
    path: Path,
    conf: float = 0.25,
    iou: float = 0.60,
    imgsz: int = 1280,
    device: str | None = None,
    clahe: bool = False,
    phys: bool = False,
    residual: bool = False,
) -> list[str]:
    """4-flip TTA with IoU clustering. Keeps a mask if ≥2 views agree or area is large."""
    import cv2

    gray = read_gray(path)
    if residual:
        image = flatten_raw_sobel_stack(gray)
    elif phys:
        image = phys_stack(gray)
    elif clahe:
        image = apply_clahe(gray)
    else:
        image = gray
    variants = {
        "orig": image,
        "fl": cv2.flip(image, 1),
        "fu": cv2.flip(image, 0),
        "fud": cv2.flip(image, -1),
    }
    raw: list[np.ndarray] = []
    with tempfile.TemporaryDirectory(prefix="filament_tta_") as tmpdir:
        tmp_root = Path(tmpdir)
        for name, arr in variants.items():
            tmp = tmp_root / f"{name}.png"
            cv2.imwrite(str(tmp), arr)
            results = _yolo_predict(
                model, tmp, conf=conf, iou=iou, imgsz=imgsz, device=device
            )
            masks = _collect_yolo_masks(results[0] if results else None)
            height, width = image.shape[:2]
            for mask in masks:
                if mask.shape != (height, width):
                    mask = cv2.resize(
                        mask, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                if name == "fl":
                    mask = cv2.flip(mask, 1)
                elif name == "fu":
                    mask = cv2.flip(mask, 0)
                elif name == "fud":
                    mask = cv2.flip(mask, -1)
                if mask.sum() >= 64:
                    raw.append((mask > 0).astype(np.uint8))
    clustered = _cluster_masks(raw, iou_thr=0.45, min_votes=2)
    return masks_to_rles(resolve_overlaps(clustered))


def infer_yolo_tiles(
    model,
    path: Path,
    conf: float = 0.35,
    iou: float = 0.60,
    imgsz: int = 640,
    device: str | None = None,
    tile: int = 768,
    overlap: int = 192,
    min_disk_frac: float = 0.15,
) -> list[str]:
    """Native-2048 grid tiles, stitch, then official full-frame instances.

    Trainer tile mAP is not official PQ. merge_gap stays 0. No CC-split.
    """
    import cv2

    from .preprocess import solar_disk_mask
    from .yolo_export import tile_axis_origins

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    disk = solar_disk_mask(gray)
    origins = tile_axis_origins(IMAGE_SIZE, tile, overlap)
    raw: list[tuple[np.ndarray, float]] = []
    with tempfile.TemporaryDirectory(prefix="filament_tile_") as tmpdir:
        tmp_root = Path(tmpdir)
        for y0 in origins:
            for x0 in origins:
                crop = image[y0 : y0 + tile, x0 : x0 + tile]
                if crop.shape[0] != tile or crop.shape[1] != tile:
                    continue
                if float(disk[y0 : y0 + tile, x0 : x0 + tile].mean()) < min_disk_frac:
                    continue
                tmp = tmp_root / f"x{x0:04d}_y{y0:04d}.png"
                cv2.imwrite(str(tmp), crop)
                results = _yolo_predict(
                    model, tmp, conf=conf, iou=iou, imgsz=imgsz, device=device
                )
                for mask, score in _collect_yolo_instances(results[0] if results else None):
                    if mask.shape != (tile, tile):
                        mask = cv2.resize(
                            mask.astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST
                        )
                    full = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                    full[y0 : y0 + tile, x0 : x0 + tile] = (mask > 0).astype(np.uint8)
                    if int(full.sum()) >= 64:
                        raw.append((full, score))
    clustered = _cluster_masks_ranked(raw, iou_thr=0.5)
    return masks_to_rles(yolo_masks_to_instances(clustered, merge_gap=0))


def run_yolo_infer(
    model,
    path: Path,
    *,
    conf: float,
    imgsz: int,
    augment: bool = False,
    merge_gap: int = 0,
    tta: bool = False,
    refiner=None,
    device: str | None = None,
    clahe: bool = False,
    phys: bool = False,
    residual: bool = False,
    tiles: bool = False,
) -> list[str]:
    """One-image dispatch. Refiner wins over TTA if both are set."""
    if sum([bool(clahe), bool(phys), bool(residual)]) > 1:
        raise ValueError("Choose at most one of --clahe / --phys-stack / --residual-stack")
    if tiles:
        if refiner is not None or tta or clahe or phys or residual:
            raise ValueError("tile stitch is native gray/BGR crops only")
        return infer_yolo_tiles(model, path, conf=conf, imgsz=imgsz, device=device)
    if refiner is not None:
        return infer_yolo_refined(
            model,
            refiner,
            path,
            conf=conf,
            imgsz=imgsz,
            device=device or "cpu",
            augment=augment,
            merge_gap=merge_gap,
            clahe=clahe,
            phys=phys,
            residual=residual,
        )
    if tta:
        return infer_yolo_tta(
            model, path, conf=conf, imgsz=imgsz, device=device, clahe=clahe, phys=phys, residual=residual
        )
    return infer_yolo_image(
        model,
        path,
        conf=conf,
        imgsz=imgsz,
        augment=augment,
        merge_gap=merge_gap,
        device=device,
        clahe=clahe,
        phys=phys,
        residual=residual,
    )


def _cluster_masks(masks: list[np.ndarray], iou_thr: float = 0.45, min_votes: int = 2) -> list[np.ndarray]:
    if not masks:
        return []
    used = [False] * len(masks)
    out: list[np.ndarray] = []
    areas = [int(m.sum()) for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    for i in order:
        if used[i]:
            continue
        group = [masks[i]]
        used[i] = True
        for j in order:
            if used[j]:
                continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            if union and (inter / union) >= iou_thr:
                group.append(masks[j])
                used[j] = True
        if len(group) >= min_votes or areas[i] >= 800:
            vote = np.sum(group, axis=0)
            merged = (vote >= max(1, (len(group) + 1) // 2)).astype(np.uint8)
            if merged.sum() >= 64:
                out.append(merged)
    out.sort(key=lambda m: int(m.sum()), reverse=True)
    return out


def _cluster_masks_ranked(
    items: list[tuple[np.ndarray, float]],
    iou_thr: float = 0.5,
) -> list[np.ndarray]:
    """Greedy IoU cluster; majority-vote union. Seed is highest score."""
    if not items:
        return []
    used = [False] * len(items)
    out: list[np.ndarray] = []
    order = sorted(range(len(items)), key=lambda i: items[i][1], reverse=True)
    for i in order:
        if used[i]:
            continue
        seed = items[i][0]
        group = [seed]
        used[i] = True
        for j in order:
            if used[j]:
                continue
            inter = np.logical_and(seed, items[j][0]).sum()
            union = np.logical_or(seed, items[j][0]).sum()
            if union and (inter / union) >= iou_thr:
                group.append(items[j][0])
                used[j] = True
        vote = np.sum(group, axis=0)
        merged = (vote >= max(1, (len(group) + 1) // 2)).astype(np.uint8)
        if merged.sum() >= 64:
            out.append(merged)
    out.sort(key=lambda m: int(m.sum()), reverse=True)
    return out
