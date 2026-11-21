#!/usr/bin/env python3
"""Champion yolo11s recipe plus a square transpose, so the four transposed TTA views are in distribution.

The only training change is RandomTranspose(p=0.5) inserted before Format.
Validation never sees it. Patience stays high so fold 0 does not select a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.paths import SEED, WEIGHTS_DIR, YOLO_DIR

CHAMPION_NAMES = {
    "yolo11s_seg_erode_gt_2048",
    "yolo11s_seg_erode_gt_2048_s20260911",
    "yolo11s_seg_erode_gt_2048_s20260912",
}


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


def pick_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--name", default="yolo11s_seg")
    parser.add_argument("--data", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--copy-paste", type=float, default=0.15)
    parser.add_argument("--mixup", type=float, default=0.05)
    parser.add_argument("--mosaic", type=float, default=0.8)
    parser.add_argument("--close-mosaic", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--multi-scale", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=int, default=4)
    parser.add_argument("--proto-stride", type=int, default=4, choices=(2, 4))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--qa", action="store_true")
    return parser


def train_kwargs(args: argparse.Namespace, data_yaml: Path, device: str) -> dict:
    return {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "device": device,
        "seed": args.seed,
        "deterministic": True,
        "single_cls": True,
        "workers": 4,
        "cache": False,
        "exist_ok": False,
        "project": str(WEIGHTS_DIR),
        "name": args.name,
        "hsv_h": 0.01,
        "hsv_s": 0.4,
        "hsv_v": 0.3,
        "degrees": 15.0,
        "translate": 0.05,
        "scale": 0.3,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "copy_paste": args.copy_paste,
        "mixup": args.mixup,
        "multi_scale": args.multi_scale,
        "mask_ratio": args.mask_ratio,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "weight_decay": 0.0005,
        "plots": True,
    }


def _epoch_logger():
    state = {"t": time.perf_counter()}

    def on_train_epoch_end(trainer):
        now = time.perf_counter()
        dt = now - state["t"]
        state["t"] = now
        calls = RandomTranspose.calls
        hits = RandomTranspose.hits
        fraction = (hits / calls) if calls else None
        print(
            f"d4_epoch {trainer.epoch + 1} sec {dt:.1f} "
            f"d4_transpose_fraction {fraction} draws {calls}",
            flush=True,
        )
        if trainer.epoch == 0 and fraction is not None and not 0.45 <= fraction <= 0.55:
            print(f"d4_transpose_fraction_outside_band {fraction}", flush=True)
        RandomTranspose.calls = 0
        RandomTranspose.hits = 0

    return on_train_epoch_end


def run_qa(data_yaml: Path) -> None:
    import cv2
    import yaml
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace, colorstr
    from ultralytics.utils.instance import Instances

    install_d4_transpose(0.5)
    img = np.arange(16, dtype=np.uint8).reshape(4, 4, 1).repeat(3, axis=2)
    boxes = np.array([[0.1, 0.2, 0.4, 0.5]], dtype=np.float32)
    segments = np.array([[[0.1, 0.2], [0.4, 0.2], [0.4, 0.5]]], dtype=np.float32)
    labels = {
        "img": img.copy(),
        "instances": Instances(boxes.copy(), segments.copy(), bbox_format="xyxy", normalized=True),
    }
    op = RandomTranspose(p=1.0)
    once = op({
        "img": img.copy(),
        "instances": Instances(boxes.copy(), segments.copy(), bbox_format="xyxy", normalized=True),
    })
    twice = op(once)
    if not np.array_equal(twice["img"], img):
        raise SystemExit("transpose twice did not restore the image")
    if not np.array_equal(twice["instances"].bboxes, boxes):
        raise SystemExit(f"transpose twice did not restore boxes {twice['instances'].bboxes} vs {boxes}")
    if not np.array_equal(twice["instances"].segments, segments):
        raise SystemExit("transpose twice did not restore segments")
    hits = 0
    probe = RandomTranspose(p=0.5)
    for _ in range(400):
        before = img.copy()
        out = probe({"img": before.copy(), "instances": Instances(boxes.copy(), segments.copy(), bbox_format="xyxy", normalized=True)})
        if not np.array_equal(out["img"], before):
            hits += 1
    fraction = hits / 400
    if not 0.45 <= fraction <= 0.55:
        raise SystemExit(f"transpose fraction {fraction} outside 0.45-0.55")

    data = yaml.safe_load(data_yaml.read_text())
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = 2048
    cfg.task = "segment"
    cfg.rect = False
    cfg.cache = False
    cfg.single_cls = True
    cfg.fraction = 1.0
    cfg.mosaic = 0.8
    cfg.mixup = 0.05
    cfg.copy_paste = 0.15
    cfg.degrees = 15.0
    cfg.translate = 0.05
    cfg.scale = 0.3
    cfg.flipud = 0.5
    cfg.fliplr = 0.5
    cfg.hsv_h = 0.01
    cfg.hsv_s = 0.4
    cfg.hsv_v = 0.3
    cfg.close_mosaic = 8
    cfg.mask_ratio = 4
    cfg.overlap_mask = True
    cfg.bgr = 0.0
    cfg.cutmix = 0.0
    cfg.shear = 0.0
    cfg.perspective = 0.0
    cfg.erasing = 0.0
    cfg.crop_fraction = 1.0
    img_path = str(data_yaml.parent / data["train"])
    dataset = build_yolo_dataset(cfg, img_path, 4, data, mode="train", rect=False)
    names = [type(step).__name__ for step in dataset.transforms.transforms]
    if names[-1] != "Format" or type(dataset.transforms.transforms[-2]).__name__ != "RandomTranspose":
        raise SystemExit(f"transpose is not immediately before Format: {names}")
    from copy import copy

    closed = copy(cfg)
    dataset.close_mosaic(closed)
    names_closed = [type(step).__name__ for step in dataset.transforms.transforms]
    if "RandomTranspose" not in names_closed or names_closed[-1] != "Format":
        raise SystemExit(f"close_mosaic dropped transpose: {names_closed}")
    dataset.transforms = dataset.build_transforms(cfg)
    out_dir = ROOT / "artifacts" / "d4" / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for index in range(8):
        label = dataset.get_image_and_label(index)
        for step in dataset.transforms.transforms[:-1]:
            label = step(label)
        image = label["img"]
        if image.shape[0] != image.shape[1]:
            raise SystemExit(f"augmented sample {index} is not square: {image.shape}")
        instances = label["instances"]
        segments = np.array(instances.segments, copy=True)
        height, width = image.shape[:2]
        if instances.normalized and segments.size:
            segments[..., 0] *= width
            segments[..., 1] *= height
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for poly in segments:
            if len(poly) < 3:
                continue
            cv2.polylines(canvas, [np.rint(poly).astype(np.int32).reshape(-1, 1, 2)], True, (40, 60, 220), 2)
        dest = out_dir / f"sample_{index:02d}.png"
        cv2.imwrite(str(dest), canvas)
        saved.append(str(dest))
    val = build_yolo_dataset(cfg, str(data_yaml.parent / data["val"]), 4, data, mode="val", rect=False)
    val_names = [type(step).__name__ for step in val.transforms.transforms]
    if "RandomTranspose" in val_names:
        raise SystemExit(f"validation pipeline contains transpose: {val_names}")
    report = {
        "double_transpose_exact": True,
        "draws": 400,
        "transposed_fraction": fraction,
        "train_pipeline_tail": names[-3:],
        "close_mosaic_has_transpose": True,
        "val_has_transpose": False,
        "overlays": saved,
    }
    (out_dir / "qa.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    del colorstr, IterableSimpleNamespace, YOLODataset


def main() -> None:
    args = build_parser().parse_args()
    data_yaml = Path(args.data) if args.data else (YOLO_DIR / "data.yaml")
    if args.qa:
        if not data_yaml.exists():
            raise SystemExit(f"missing {data_yaml}")
        run_qa(data_yaml)
        return
    if args.name in CHAMPION_NAMES or not args.name.startswith("yolo11s_seg_erode_gt_2048_d4_"):
        raise SystemExit(f"refusing name {args.name}")
    if (WEIGHTS_DIR / args.name).exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite {WEIGHTS_DIR / args.name}")
    install_d4_transpose(0.5)
    from ultralytics import YOLO

    if not data_yaml.exists():
        raise FileNotFoundError(f"missing {data_yaml}")
    device = pick_device()
    if device != "mps":
        raise SystemExit(f"MPS required, got {device}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.resume:
        last = WEIGHTS_DIR / args.name / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"cannot resume, missing {last}")
        model = YOLO(str(last))
        model.add_callback("on_train_epoch_end", _epoch_logger())
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        kwargs = train_kwargs(args, data_yaml, device)
        model.add_callback("on_train_epoch_end", _epoch_logger())
        results = model.train(**kwargs)
    last = WEIGHTS_DIR / args.name / "weights" / "last.pt"
    meta = {
        "device": device,
        "name": args.name,
        "data": str(data_yaml),
        "score_weights": str(last),
        "ignored_best_pt": str(WEIGHTS_DIR / args.name / "weights" / "best.pt"),
        "checkpoint_selection": "none; score last.pt; patience 1000",
        "transpose_p": 0.5,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "patience": args.patience,
        "seed": args.seed,
        "results_dir": str(results.save_dir) if hasattr(results, "save_dir") else None,
    }
    (WEIGHTS_DIR / f"{args.name}_train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
