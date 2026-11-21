#!/usr/bin/env python3
"""Start dest-safe YOLO11s@1280 on match-any labels. Default is dry.

Dest is weights/yolo11s_seg_pq_any only. H-alpha JPEGs. Never overwrites HOLD
dests. Never submits. Conf 0.35 locked a priori. Standalone official PQ.
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

from filament.paths import ARTIFACTS, FAST_CACHE, WEIGHTS_DIR, YOLO_PQ_ANY_DIR
from gate_unet_family import live_mps_owner

PYTHON = Path("~/.venvs/filament/bin/python")
TRAIN = ROOT / "scripts" / "train_yolo.py"
WATCH = ROOT / "scripts" / "watch_yolo11s_pq_any_then_score.py"
RECEIPT = ARTIFACTS / "start_yolo11s_pq_any_now.json"
DEST = WEIGHTS_DIR / "yolo11s_seg_pq_any"
CLOSED = {
    "yolo11s_seg",
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
    "yolo11s_seg_1408",
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
    "unet_resnet34",
    "unet_resnet34_1536",
    "unet_semantic",
    "unet_patch_dhebe",
    "unet_3ch_disk",
}
LOCKED_CONF = 0.35
WATCH_NOT_MPS = frozenset(
    {
        "watch_yolo11l_then_score.py",
        "watch_erode_gt_then_score.py",
        "watch_yolo11l_erode_gt_then_score.py",
        "watch_yolo11l_shift_then_score.py",
        "watch_yolo11s_erode_gt_then_score.py",
        "watch_yolo11n_erode_gt_then_score.py",
        "watch_yolo11s_erode_gt_1024_then_score.py",
        "watch_yolo11s_erode_gt_1408_then_score.py",
        "watch_yolo11s_erode_gt_1536_then_score.py",
        "watch_yolo11s_erode_gt_1408_ms_then_score.py",
        "watch_yolo11s_seg_1408_then_score.py",
        "watch_yolo11s_clahe_then_score.py",
        "watch_yolo11s_pq_any_then_score.py",
        "watch_rater_then_score.py",
    }
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    owner = live_mps_owner(ignore=WATCH_NOT_MPS)
    data = YOLO_PQ_ANY_DIR / "data.yaml"
    out = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_flag": args.start,
        "auto_start": False,
        "family": "yolo11s_seg_pq_any",
        "data": str(data),
        "dest": str(DEST),
        "model": "yolo11s-seg.pt",
        "imgsz": 1280,
        "batch": 4,
        "epochs": 40,
        "patience": 12,
        "locked_conf_a_priori": LOCKED_CONF,
        "pixels": "halpha",
        "label_protocol": "match_any_rater_iou_gt_0.5_one_polygon",
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "mps_owner": owner,
        "slot2_stays": 56148889,
        "pair_stays": [56173362, 56148889],
        "parent_bar": 0.4365,
        "extras_or": False,
        "test_labels_used": False,
        "can_certify_0.55": False,
        "why": (
            "Match-any training target: one real rater outline per filament that "
            "already agrees with other raters at IoU>0.5. Not the union dump "
            "(~0.359) and not a HOLD erode/CLAHE/shift dest. Watcher scores "
            "official all-annotator PQ at conf 0.35 and writes HOLD_FOR_PARENT "
            "if pq>0.4365. It never submits."
        ),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if DEST.name in CLOSED:
        out["refused"] = f"dest {DEST.name} is a closed overwrite"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(1)
    if (DEST / "weights" / "best.pt").exists():
        out["refused"] = f"already trained: {DEST / 'weights' / 'best.pt'}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(1)
    if owner:
        out["refused"] = f"Refusing yolo11s_seg_pq_any start: {owner} owns MPS."
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(0 if not args.start else 1)
    if not data.exists():
        out["refused"] = f"missing {data}"
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(out["refused"])
        raise SystemExit(1)
    if not args.start:
        out["dry"] = True
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    env = os.environ.copy()
    env.setdefault("YOLO_CONFIG_DIR", str(FAST_CACHE))
    env.setdefault("MPLCONFIGDIR", str(FAST_CACHE / "mpl"))
    env["PYTHONUNBUFFERED"] = "1"
    run_log = ARTIFACTS / "yolo11s_seg_pq_any_family.out"
    watch_log = ARTIFACTS / "yolo11s_seg_pq_any_watch.out"
    watch = subprocess.Popen(
        [str(PYTHON), str(WATCH)],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdout=watch_log.open("a"),
        stderr=subprocess.STDOUT,
    )
    run = subprocess.Popen(
        [
            str(PYTHON),
            str(TRAIN),
            "--name",
            "yolo11s_seg_pq_any",
            "--data",
            str(data),
            "--model",
            "yolo11s-seg.pt",
            "--imgsz",
            "1280",
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
            "train_pid": run.pid,
            "watch_pid": watch.pid,
            "train_log": str(run_log.relative_to(ROOT)),
            "watch_log": str(watch_log.relative_to(ROOT)),
        }
    )
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
