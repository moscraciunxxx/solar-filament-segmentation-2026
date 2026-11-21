#!/usr/bin/env python3
"""Train the repaired-label student. Scores last.pt. Does not touch champion weights."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.paths import SEED, WEIGHTS_DIR

PLAN = ROOT / "artifacts" / "phaseB_repair" / "student_plan.json"
NAME = "yolo11s_seg_gong_repair_s20260910"
CHAMPION_NAMES = {
    "yolo11s_seg_erode_gt_2048",
    "yolo11s_seg_erode_gt_2048_s20260911",
    "yolo11s_seg_erode_gt_2048_s20260912",
}


def main() -> None:
    if NAME in CHAMPION_NAMES or NAME.startswith("yolo11s_seg_erode_gt_2048"):
        raise SystemExit("refusing to write a champion weight directory")
    plan = json.loads(PLAN.read_text())
    if plan["weights_name"] != NAME or plan["teacher_pseudo_used"] or plan["fold0_gong_images"]:
        raise SystemExit(f"refusing plan {plan}")
    run_dir = WEIGHTS_DIR / NAME
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite {run_dir}")
    import torch
    from ultralytics import YOLO

    if not torch.backends.mps.is_available():
        raise SystemExit("M5 Max MPS is required")
    model = YOLO("yolo11s-seg.pt")
    model.train(
        data=plan["yaml"],
        epochs=int(plan["epochs"]),
        imgsz=2048,
        batch=4,
        patience=1000,
        device="mps",
        seed=SEED,
        deterministic=True,
        single_cls=True,
        workers=4,
        cache=False,
        exist_ok=False,
        project=str(WEIGHTS_DIR),
        name=NAME,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=15.0,
        translate=0.05,
        scale=0.3,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.8,
        close_mosaic=int(plan["close_mosaic"]),
        copy_paste=0.15,
        mixup=0.05,
        multi_scale=0.0,
        mask_ratio=4,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        plots=True,
    )
    last = run_dir / "weights" / "last.pt"
    if not last.exists():
        raise SystemExit(f"missing {last}")
    score = run_dir / "weights" / "score_last.pt"
    shutil.copyfile(last, score)
    meta = {
        "name": NAME,
        "score_weights": str(score),
        "ignored_best_pt": str(run_dir / "weights" / "best.pt"),
        "val": "dummy_blank",
        "checkpoint_selection": "none; score last.pt",
        "plan": plan,
        "device": "mps",
    }
    (ROOT / "artifacts" / "phaseB_repair" / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
