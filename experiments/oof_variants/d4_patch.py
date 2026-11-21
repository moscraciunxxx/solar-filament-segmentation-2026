"""D4 transpose augmentation, verbatim from Grok's train_yolo_d4.py (RandomTranspose p=0.5, inserted before Format)."""
import numpy as np

class RandomTranspose:
    """Swap x and y on a square image. No interpolation."""

    calls = 0
    hits = 0

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, labels: dict) -> dict:
        type(self).calls += 1
        if np.random.random() >= self.p:
            return labels
        type(self).hits += 1
        img = labels["img"]
        if img.shape[0] != img.shape[1]:
            raise RuntimeError(f"RandomTranspose requires a square image, got {img.shape}")
        labels["img"] = np.ascontiguousarray(img.transpose(1, 0, 2))
        instances = labels.get("instances")
        if instances is None or len(instances.bboxes) == 0:
            return labels
        instances.convert_bbox("xyxy")
        boxes = instances.bboxes
        boxes[:] = boxes[:, [1, 0, 3, 2]]
        segments = instances.segments
        if segments is not None and segments.size:
            segments[:] = np.ascontiguousarray(segments[..., ::-1])
        return labels


def install_d4_transpose(p: float = 0.5) -> None:
    """Patch YOLODataset.build_transforms so close_mosaic rebuilds keep the transpose."""
    from ultralytics.data.dataset import YOLODataset

    if getattr(YOLODataset.build_transforms, "_d4_transpose", False):
        return
    original = YOLODataset.build_transforms

    def build_transforms(self, hyp=None):
        transforms = original(self, hyp)
        if self.augment:
            transforms.insert(-1, RandomTranspose(p))
        return transforms

    build_transforms._d4_transpose = True
    YOLODataset.build_transforms = build_transforms


