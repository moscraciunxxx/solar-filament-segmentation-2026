"""Round a mask through the YOLO prototype grid.

Ultralytics `polygon2mask` downsamples with `cv2.resize` to
`image_size // mask_ratio`. YOLO11's Proto module then lives at stride 4,
so image size 2048 is a 512-pixel grid when mask_ratio is 4. A ratio of 2
is a finer grid that stock Proto does not emit. The upsample matches the
bilinear retina path and keeps every pixel above 0.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch.nn as nn


def roundtrip_mask(mask: np.ndarray, ratio: int) -> np.ndarray:
    """Return the mask after a prototype downsample and a bilinear upsample."""
    if ratio < 1:
        raise ValueError("ratio must be >= 1")
    height, width = mask.shape[:2]
    small_h = height // ratio
    small_w = width // ratio
    if small_h < 1 or small_w < 1:
        raise ValueError("ratio collapses the mask")
    source = (mask > 0).astype(np.uint8)
    small = cv2.resize(source, (small_w, small_h))
    restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return (restored > 0).astype(np.uint8)


def use_stride2_proto(proto) -> None:
    """Add one stride-2 transpose so P3/8 becomes a stride-2 prototype.

    A second call sees the Sequential and leaves the grid alone. Stock Proto
    stays stride 4 until this runs.
    """
    if getattr(proto, "_filament_stride2", False):
        return
    upsample = proto.upsample
    if isinstance(upsample, nn.Sequential):
        proto._filament_stride2 = True
        return
    extra = nn.ConvTranspose2d(upsample.out_channels, upsample.out_channels, 2, 2, 0, bias=True)
    proto.upsample = nn.Sequential(upsample, extra)
    proto._filament_stride2 = True


def _walk(node, seen: set[int]):
    if node is None or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    inner = getattr(node, "model", None)
    if isinstance(inner, (list, tuple)):
        for child in inner:
            yield from _walk(child, seen)
    elif inner is not None:
        yield from _walk(inner, seen)
    children = getattr(node, "children", None)
    if callable(children):
        for child in children():
            yield from _walk(child, seen)


def _find_proto(model):
    for node in _walk(model, set()):
        proto = getattr(node, "proto", None)
        if proto is not None and hasattr(proto, "upsample"):
            return proto
    return None


def patch_model_proto(model) -> None:
    """Find the segment head and give its prototype a stride-2 grid."""
    proto = _find_proto(model)
    if proto is None:
        raise RuntimeError("segment model has no proto")
    use_stride2_proto(proto)


def copy_stride2_upsample(source, target) -> bool:
    """Copy a trained extra upsample onto a freshly patched prototype.

    A resume rebuilds the stock prototype and then adds a new transpose.
    Without this copy that new transpose is random and the saved one is dropped.
    """
    src = _find_proto(source)
    dst = _find_proto(target)
    if src is None or dst is None:
        return False
    src_up = src.upsample
    dst_up = dst.upsample
    if not isinstance(src_up, nn.Sequential) or not isinstance(dst_up, nn.Sequential):
        return False
    if len(list(src_up)) != len(list(dst_up)):
        return False
    dst_up.load_state_dict(src_up.state_dict())
    return True


def validation_pair_hw(
    image_hw: tuple[int, int], proto_hw: tuple[int, int], stride: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ground-truth mask size and the image size process_mask must see.

    Stock validation hardcodes stride 4. A stride-2 prototype is image // 2,
    so ground truth must stay on that grid and the box scale must use the
    real image, which is stride * proto.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    gt_hw = tuple(int(side) // int(stride) for side in image_hw)
    process_shape = tuple(int(stride) * int(side) for side in proto_hw)
    return gt_hw, process_shape


class Stride2SegmentationValidator:
    """Placeholder so the class body below can replace it after the import."""


def _install_stride2_validator() -> None:
    global Stride2SegmentationValidator
    import torch
    import torch.nn.functional as F
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.models.yolo.segment.val import SegmentationValidator
    from ultralytics.utils import ops

    class Stride2SegmentationValidator(SegmentationValidator):
        """Validator whose mask grid follows a stride-2 prototype."""

        mask_stride = 2

        def postprocess(self, preds):
            proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
            detections = DetectionValidator.postprocess(self, preds[0])
            image_hw = tuple(self.mask_stride * int(side) for side in proto.shape[2:])
            proto_hw = tuple(int(side) for side in proto.shape[2:])
            _, process_shape = validation_pair_hw(image_hw, proto_hw, self.mask_stride)
            for index, pred in enumerate(detections):
                coefficient = pred.pop("extra")
                pred["masks"] = self.process(
                    proto[index], coefficient, pred["bboxes"], shape=process_shape
                )
            return detections

        def _prepare_batch(self, si: int, batch: dict):
            prepared = DetectionValidator._prepare_batch(self, si, batch)
            count = prepared["cls"].shape[0]
            if self.args.overlap_mask:
                masks = batch["masks"][si]
                index = torch.arange(1, count + 1, device=masks.device).view(count, 1, 1)
                masks = (masks == index).float()
            else:
                masks = batch["masks"][batch["batch_idx"] == si]
            if count:
                image_hw = tuple(int(side) for side in prepared["imgsz"])
                gt_hw, _ = validation_pair_hw(image_hw, image_hw, self.mask_stride)
                native = self.process is ops.process_mask_native
                mask_size = list(image_hw) if native else list(gt_hw)
                if tuple(masks.shape[-2:]) != tuple(mask_size):
                    masks = F.interpolate(
                        masks[None].float(), mask_size, mode="bilinear", align_corners=False
                    )[0]
                    masks = masks.gt_(0.5)
            prepared["masks"] = masks
            return prepared

    globals()["Stride2SegmentationValidator"] = Stride2SegmentationValidator


_install_stride2_validator()
