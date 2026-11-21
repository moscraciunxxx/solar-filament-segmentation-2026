#!/usr/bin/env python3
"""Train a single-class YOLO11s-seg detector on the official train split only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.paths import SEED, WEIGHTS_DIR, YOLO_DIR


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue weights/<name>/weights/last.pt. Does not start a new run.",
    )
    parser.add_argument("--copy-paste", type=float, default=0.15)
    parser.add_argument("--mixup", type=float, default=0.05)
    parser.add_argument("--mosaic", type=float, default=0.8)
    parser.add_argument("--close-mosaic", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--multi-scale", type=float, default=0.0)
    parser.add_argument("--mask-ratio", type=int, default=4)
    parser.add_argument("--proto-stride", type=int, default=4, choices=(2, 4))
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def stride2_trainer():
    """Trainer that inserts one extra prototype upsample before the optimizer exists."""
    from copy import copy

    from ultralytics.models.yolo.segment import SegmentationTrainer

    from filament.proto_grid import (
            Stride2SegmentationValidator,
            copy_stride2_upsample,
            patch_model_proto,
        )

    class Stride2SegmentationTrainer(SegmentationTrainer):
        def setup_model(self):
            checkpoint = super().setup_model()
            patch_model_proto(self.model)
            if isinstance(checkpoint, dict):
                source = checkpoint.get("ema") or checkpoint.get("model")
                if source is not None:
                    copy_stride2_upsample(source, self.model)
            return checkpoint

        def get_validator(self):
            """Score masks on the stride-2 grid instead of the stock stride-4 grid."""
            return Stride2SegmentationValidator(
                self.test_loader,
                save_dir=self.save_dir,
                args=copy(self.args),
                _callbacks=self.callbacks,
            )

    return Stride2SegmentationTrainer


def resume_kwargs(args: argparse.Namespace) -> dict:
    """Resume last.pt. A stride-2 run must keep the aligned validator."""
    kwargs = {"resume": True}
    if args.proto_stride == 2:
        kwargs["trainer"] = stride2_trainer()
    return kwargs


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
        "exist_ok": True,
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


def main() -> None:
    args = build_parser().parse_args()

    from ultralytics import YOLO

    data_yaml = Path(args.data) if args.data else (YOLO_DIR / "data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Run scripts/prepare_yolo.py first: missing {data_yaml}")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    if args.resume:
        last = WEIGHTS_DIR / args.name / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"cannot resume, missing {last}")
        model = YOLO(str(last))
        results = model.train(**resume_kwargs(args))
    else:
        model = YOLO(args.model)
        kwargs = train_kwargs(args, data_yaml, device)
        if args.proto_stride == 2:
            kwargs["trainer"] = stride2_trainer()
        results = model.train(**kwargs)
    best = WEIGHTS_DIR / args.name / "weights" / "best.pt"
    meta = {
        "device": device,
        "name": args.name,
        "data": str(data_yaml),
        "best": str(best),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "lr0": args.lr0,
        "multi_scale": args.multi_scale,
        "mask_ratio": args.mask_ratio,
        "proto_stride": args.proto_stride,
        "seed": args.seed,
        "results_dir": str(results.save_dir) if hasattr(results, "save_dir") else None,
    }
    (WEIGHTS_DIR / f"{args.name}_train_meta.json").write_text(json.dumps(meta, indent=2))
    (WEIGHTS_DIR / "yolo_train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
