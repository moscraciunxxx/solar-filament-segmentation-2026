#!/usr/bin/env python3
"""Build a test CSV: frozen Slot 1 NMS-merged with full-disk U-Net instances.

Refuse unless unet_slot1_merge official gate is PROMOTE_LOOK.
Never uploads. Never reads MAGFiLO test labels. Test JPEGs only.
Refuse while train_yolo.py or train_unet.py owns MPS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_unet import decode_instances, infer_prob
from filament.infer import pick_device, run_yolo_infer
from filament.merge import MERGE_IOU, merge_rles
from filament.metric import pred_to_df
from filament.paths import ARTIFACTS, FAST_ARTIFACTS, FAST_SUBMISSION_DIR, TEST_IMG_DIR, WEIGHTS_DIR
from filament.slot1_csv import SLOT1_TEST_CSV, load_slot1_test_rles
from filament.unet import build_fulldisk_unet

SLOT1_WEIGHTS = WEIGHTS_DIR / "yolo11m_seg_1280" / "weights" / "best.pt"
SLOT1_CONF = 0.35
SLOT1_IMGSZ = 1280
UNET_WEIGHTS = WEIGHTS_DIR / "unet_semantic" / "best.pt"
GATE = ARTIFACTS / "unet_slot1_merge_fold0_official.json"


def gpu_train_alive() -> bool:
    for pattern in ("train_yolo.py", "train_unet.py"):
        proc = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if proc.returncode not in (0, 1):
            continue
        if any(int(x) != os.getpid() for x in proc.stdout.split() if x.strip().isdigit()):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(UNET_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thr", type=float, default=0.45)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--gate-json", default=str(GATE))
    parser.add_argument(
        "--research-csv",
        action="store_true",
        help="Write a do-not-upload CSV even if gate is not PROMOTE_LOOK.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if gpu_train_alive():
        raise SystemExit("Refusing test infer while train_yolo.py or train_unet.py owns MPS.")

    gate_path = Path(args.gate_json)
    decision = None
    if gate_path.exists():
        payload = json.loads(gate_path.read_text())
        decision = payload["gate"]["decision"]
        locked = payload.get("locked")
        if locked is not None:
            args.thr = float(locked)
    if decision != "PROMOTE_LOOK" and not args.research_csv:
        raise SystemExit(
            f"Gate is {decision!r} (need PROMOTE_LOOK). Pass --research-csv only for a local do-not-upload file."
        )

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError(TEST_IMG_DIR)

    slot1_by_stem = None
    slot1_source = "yolo_infer"
    if SLOT1_TEST_CSV.exists():
        slot1_by_stem = load_slot1_test_rles(images)
        slot1_source = str(SLOT1_TEST_CSV)
    elif not SLOT1_WEIGHTS.exists():
        raise FileNotFoundError(SLOT1_WEIGHTS)

    device = pick_device(args.device)
    slot1 = None
    if slot1_by_stem is None:
        from ultralytics import YOLO

        slot1 = YOLO(str(SLOT1_WEIGHTS))
    unet = build_fulldisk_unet(base=args.base)
    unet.load_state_dict(torch.load(weights, map_location=device))
    unet.to(device)
    unet.eval()

    frames = []
    n_added = 0
    for path in tqdm(images, desc="test merge Slot1+UNet"):
        if slot1_by_stem is not None:
            slot_rles = slot1_by_stem[path.stem]
        else:
            slot_rles = run_yolo_infer(
                slot1, path, conf=SLOT1_CONF, imgsz=SLOT1_IMGSZ, merge_gap=0, device=device
            )
        extra = decode_instances(infer_prob(unet, path, args.imgsz, device), args.thr)
        merged, added = merge_rles(slot_rles, extra, iou_thr=MERGE_IOU)
        n_added += added
        if merged:
            frames.append(pred_to_df(path.stem, merged))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["filament_id", "segmentation_rle"]
    )
    FAST_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = "PROMOTE" if decision == "PROMOTE_LOOK" else "DONOTUPLOAD"
    out = Path(args.out) if args.out else FAST_SUBMISSION_DIR / f"submission_slot1_unet_merged_{tag}.csv"
    df.to_csv(out, index=False)
    meta = {
        "do_not_upload": decision != "PROMOTE_LOOK",
        "gate": decision,
        "gate_json": str(gate_path),
        "slot1_weights": str(SLOT1_WEIGHTS),
        "slot1_source": slot1_source,
        "slot1_re_inferred_yolo": slot1_by_stem is None,
        "unet_weights": str(weights.resolve()),
        "thr": args.thr,
        "n_images": len(images),
        "n_rows": int(len(df)),
        "n_unet_added": n_added,
        "merge_iou": MERGE_IOU,
        "path": str(out),
        "leakage": "test JPEGs only; no MAGFiLO public-test labels",
        "never_submits": True,
    }
    (FAST_ARTIFACTS / f"{out.stem}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
