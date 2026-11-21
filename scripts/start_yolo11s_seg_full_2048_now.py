#!/usr/bin/env python3
"""Train YOLO11s-seg at native 2048 on full medoid labels. Default is dry.

Dest is weights/yolo11s_seg_full_2048 only. Labels are yolo_seg, not the
eroded set. The eroded 2048 parent is left in place. Conf 0.35 is locked
for the later score. Never submits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_CACHE, IMAGE_SIZE, WEIGHTS_DIR, YOLO_DIR
from gate_unet_family import live_mps_owner

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_yolo.py"
RECEIPT = ARTIFACTS / "start_yolo11s_seg_full_2048_now.json"
DEST = WEIGHTS_DIR / "yolo11s_seg_full_2048"
IMGSZ = 2048
CLOSED = {
    "yolo11s_seg",
    "yolo11s_seg_1408",
    "yolo11s_seg_erode_gt_2048",
    "yolo11m_seg_1280",
    "yolo11l_seg_1280",
    "yolo11l_seg_erode_gt",
    "yolo11m_seg_erode_gt",
    "yolo11s_seg_erode_gt",
    "yolo11n_seg_erode_gt",
    "yolo11s_seg_erode_gt_1024",
    "yolo11s_seg_erode_gt_1408",
    "yolo11s_seg_erode_gt_1536",
    "yolo11s_seg_erode_gt_1408_ms",
    "yolo11l_seg_shift",
    "yolo11s_seg_all",
    "yolo11s_seg_phys",
    "yolo11s_seg_shift",
    "yolo11s_seg_tiles",
    "yolo11s_seg_tiles_cm",
    "yolo11s_seg_faint",
    "yolo11s_seg_residual",
    "yolo11s_seg_rater",
    "yolo11s_seg_clahe",
    "yolo11s_seg_pq_any",
}


def _long_side() -> int:
    images = sorted((YOLO_DIR / "images" / "train").glob("*.jpeg"))
    if not images:
        raise SystemExit(f"no train images in {YOLO_DIR}")
    from PIL import Image

    width, height = Image.open(images[0]).size
    return max(width, height)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    if IMGSZ != IMAGE_SIZE:
        raise SystemExit(f"imgsz must be {IMAGE_SIZE}")
    owner = live_mps_owner()
    data = YOLO_DIR / "data.yaml"
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_flag": args.start,
        "started": False,
        "family": DEST.name,
        "labels": "full_medoid_yolo_seg",
        "data": str(data),
        "dest": str(DEST),
        "model": "yolo11s-seg.pt",
        "imgsz": IMGSZ,
        "batch": 4,
        "epochs": 40,
        "patience": 12,
        "locked_conf_a_priori": 0.35,
        "why": (
            "Eroded masks top out near mean IoU 0.68. Full medoid outlines, "
            "used as an oracle, score 0.79 on this fold. No full-medoid model "
            "has been trained at 2048. Thick 1408 scored 0.412."
        ),
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "test_labels_used": False,
        "mps_owner": owner,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if DEST.name in CLOSED:
        out["refused"] = f"dest {DEST.name} is closed"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if (DEST / "weights" / "best.pt").exists():
        out["refused"] = f"already trained: {DEST / 'weights' / 'best.pt'}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if owner:
        out["refused"] = f"{owner} owns MPS"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        raise SystemExit(1)
    if not data.exists():
        out["refused"] = f"missing {data}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    long_side = _long_side()
    out["sample_image_long_side"] = long_side
    if long_side < IMGSZ:
        out["refused"] = f"train images are {long_side}px, refusing to upscale to {IMGSZ}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        raise SystemExit(out["refused"])
    if not args.start:
        out["dry"] = True
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    run_log = ARTIFACTS / "yolo11s_seg_full_2048_family.out"
    run = subprocess.Popen(
        [
            str(PYTHON),
            str(TRAIN),
            "--name",
            DEST.name,
            "--data",
            str(data),
            "--model",
            "yolo11s-seg.pt",
            "--imgsz",
            str(IMGSZ),
            "--batch",
            "4",
            "--epochs",
            "40",
            "--patience",
            "12",
        ],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=run_log.open("a"),
        stderr=subprocess.STDOUT,
    )
    out.update(
        {
            "started": True,
            "dry": False,
            "train_pid": run.pid,
            "train_log": str(run_log.relative_to(ROOT)),
        }
    )
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
