#!/usr/bin/env python3
"""Build a test CSV from yolo11l_seg_erode_gt only if official PQ beats 0.4365.

Never submits. Locked conf 0.35 / imgsz 1280. Year-shift stays Slot 2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.infer import pick_device, run_yolo_infer
from filament.metric import pred_to_df
from filament.paths import ARTIFACTS, FAST_SUBMISSION_DIR, TEST_IMG_DIR, WEIGHTS_DIR

WEIGHTS = WEIGHTS_DIR / "yolo11l_seg_erode_gt" / "weights" / "best.pt"
OFFICIAL = ARTIFACTS / "yolo11l_seg_erode_gt_fold0_official.json"
ERODE_BAR = 0.4365
LOCKED_CONF = 0.35
LOCKED_IMGSZ = 1280


def main() -> None:
    if not OFFICIAL.exists():
        raise SystemExit(f"missing official {OFFICIAL}")
    payload = json.loads(OFFICIAL.read_text())
    locked = payload.get("locked_row") or {}
    pq = float(locked.get("pq", 0.0))
    if pq <= ERODE_BAR:
        raise SystemExit(f"refuse CSV: official {pq} does not beat erode bar {ERODE_BAR}")
    if not WEIGHTS.exists():
        raise SystemExit(f"missing {WEIGHTS}")
    from ultralytics import YOLO

    device = pick_device("auto")
    model = YOLO(str(WEIGHTS))
    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    if not images:
        raise SystemExit(f"missing test images {TEST_IMG_DIR}")
    frames = [
        pred_to_df(
            path.stem,
            run_yolo_infer(
                model, path, conf=LOCKED_CONF, imgsz=LOCKED_IMGSZ, merge_gap=0, device=device
            ),
        )
        for path in tqdm(images, desc="yolo11l erode-gt test")
    ]
    pred = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["filament_id", "segmentation_rle"]
    )
    FAST_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out = FAST_SUBMISSION_DIR / "submission_yolo11l_erode_gt_conf035.csv"
    pred.to_csv(out, index=False)
    meta = {
        "pq": pq,
        "erode_bar": ERODE_BAR,
        "conf": LOCKED_CONF,
        "imgsz": LOCKED_IMGSZ,
        "n_rows": int(len(pred)),
        "path": str(out),
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "slot2_stays": 56148889,
        "parent_still_does_hidden_exam": True,
    }
    (ARTIFACTS / "make_yolo11l_erode_gt_submission.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
