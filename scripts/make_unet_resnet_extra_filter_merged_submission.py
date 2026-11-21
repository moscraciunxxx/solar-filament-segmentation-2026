#!/usr/bin/env python3
"""Build a test CSV only after extra-filter PROMOTE_LOOK. Never submits.

Default --dry-refuse. Hose floor min_area>=642. Test JPEGs + Slot1 CSV only.
No MAGFiLO test labels. Filter dest is resnet_extra_filter, not a U-Net overwrite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import FAST_ARTIFACTS, FAST_SUBMISSION_DIR, TEST_IMG_DIR, WEIGHTS_DIR
from filament.slot1_csv import SLOT1_TEST_CSV, load_slot1_test_rles
from gate_unet_family import (
    EXTRA_FILTER_GATE,
    HOSE_FLOOR,
    PROMOTE,
    live_mps_owner,
    read_gate,
    refuse_extra_filter_csv,
)
from resnet_extra_filter import FILTER_DEST

UNET_WEIGHTS = WEIGHTS_DIR / "unet_resnet34" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-refuse", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gate-json", default=str(EXTRA_FILTER_GATE))
    parser.add_argument("--weights", default=str(UNET_WEIGHTS))
    parser.add_argument("--filter-dest", default=str(FILTER_DEST))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thr", type=float, default=0.50)
    parser.add_argument("--min-area", type=int, default=1034)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    reason = refuse_extra_filter_csv(Path(args.gate_json), min_area=int(args.min_area))
    if args.dry_refuse or not args.run:
        print(reason)
        raise SystemExit(0)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(2)

    decision, payload = read_gate(Path(args.gate_json))
    locked_row = payload.get("locked_row") if isinstance(payload, dict) else None
    if isinstance(locked_row, dict):
        if locked_row.get("thr") is not None:
            args.thr = float(locked_row["thr"])
        if locked_row.get("min_area") is not None:
            args.min_area = int(locked_row["min_area"])
    if int(args.min_area) < HOSE_FLOOR:
        raise SystemExit(f"Refusing extra-filter CSV: min_area {args.min_area} < {HOSE_FLOOR}.")

    owner = live_mps_owner()
    if owner:
        raise SystemExit(f"Refusing extra-filter test CSV: {owner} owns MPS.")

    import pandas as pd
    import torch
    from tqdm import tqdm

    from eval_unet import decode_instances, infer_prob
    from filament.classical import ClassicalConfig, dark_residual_channel
    from filament.infer import pick_device
    from filament.merge import MERGE_IOU, extra_rles, merge_rles
    from filament.metric import pred_to_df
    from filament.preprocess import read_gray
    from filament.unet_resnet import build_resnet34_unet
    from resnet_extra_filter import (
        assert_safe_filter_dest,
        features_for_rles,
        keep_extra_rles,
        load_rejector,
    )

    dest = assert_safe_filter_dest(Path(args.filter_dest))
    clf, meta = load_rejector(dest)
    keep_thr = float(meta.get("threshold", 0.5))
    weights = Path(args.weights)
    if "unet_resnet34" not in Path(weights).parts:
        raise SystemExit(f"Refusing extra-filter CSV: decoder must be unet_resnet34 (got {weights}).")
    if not weights.exists():
        raise FileNotFoundError(weights)
    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError(TEST_IMG_DIR)
    if not SLOT1_TEST_CSV.exists():
        raise SystemExit(f"missing Slot1 test CSV {SLOT1_TEST_CSV}")

    device = pick_device(args.device)
    slot1_by_stem = load_slot1_test_rles(images)
    unet = build_resnet34_unet(in_ch=3, pretrained=False)
    unet.load_state_dict(torch.load(weights, map_location=device))
    unet.to(device)
    unet.eval()
    cfg = ClassicalConfig()

    frames = []
    n_added = 0
    for path in tqdm(images, desc="test merge Slot1+filtered-ResNet extras"):
        slot_rles = slot1_by_stem[path.stem]
        extra = decode_instances(
            infer_prob(unet, path, args.imgsz, device),
            args.thr,
            min_area=int(args.min_area),
        )
        extras = extra_rles(slot_rles, extra, iou_thr=MERGE_IOU)
        image = read_gray(path)
        residual = dark_residual_channel(image, cfg)
        feats = features_for_rles(extras, image, residual)
        kept = keep_extra_rles(extras, feats, clf, keep_thr)
        merged, added = merge_rles(slot_rles, kept, iou_thr=MERGE_IOU)
        n_added += added
        if merged:
            frames.append(pred_to_df(path.stem, merged))

    df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    )
    FAST_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = "PROMOTE" if decision == PROMOTE else "DONOTUPLOAD"
    out = Path(args.out) if args.out else FAST_SUBMISSION_DIR / f"submission_slot1_unet_resnet_extra_filter_{tag}.csv"
    df.to_csv(out, index=False)
    rec = {
        "do_not_upload": decision != PROMOTE,
        "gate": decision,
        "filter_dest": str(dest.resolve()),
        "unet_weights": str(weights.resolve()),
        "thr": args.thr,
        "min_area": int(args.min_area),
        "hose_floor": HOSE_FLOOR,
        "n_images": len(images),
        "n_rows": int(len(df)),
        "n_unet_added": n_added,
        "path": str(out),
        "leakage": "test JPEGs only; no MAGFiLO public-test labels",
        "never_submits": True,
        "kaggle_submit_called": False,
        "do_not_submit": True,
    }
    (FAST_ARTIFACTS / f"{out.stem}_meta.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
