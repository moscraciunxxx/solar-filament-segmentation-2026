#!/usr/bin/env python3
"""Build a test CSV: frozen Slot 1 NMS-merged with 3ch U-Net instances.

Legal only after the 3ch merge gate is PROMOTE_LOOK. Never uploads.
Never reads MAGFiLO test labels. Weights: unet_3ch_disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_ARTIFACTS, FAST_SUBMISSION_DIR, TEST_IMG_DIR, WEIGHTS_DIR
from filament.slot1_csv import SLOT1_TEST_CSV, load_slot1_test_rles
from gate_unet_family import CH3_GATE, PROMOTE, live_mps_owner, read_gate

UNET_WEIGHTS = WEIGHTS_DIR / "unet_3ch_disk" / "best.pt"
WEIGHTS_NAME = "unet_3ch_disk"
MERGE_IOU = 0.5


def refuse(gate_path: Path, *, research_csv: bool = False) -> str | None:
    owner = live_mps_owner()
    if owner:
        return f"Refusing 3ch merge test CSV: {owner} owns MPS."
    decision, _ = read_gate(gate_path)
    if decision != PROMOTE and not research_csv:
        if decision is None:
            return f"Gate is None (need PROMOTE_LOOK). missing {gate_path}."
        return f"Gate is {decision!r} (need PROMOTE_LOOK)."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-refuse", action="store_true")
    parser.add_argument("--gate-json", default=str(CH3_GATE))
    parser.add_argument("--weights", default=str(UNET_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thr", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=1034)
    parser.add_argument("--research-csv", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="")
    parser.add_argument("--base", type=int, default=32)
    args = parser.parse_args()

    gate_path = Path(args.gate_json)
    reason = refuse(gate_path, research_csv=args.research_csv)
    if args.dry_refuse:
        if reason:
            print(reason)
        raise SystemExit(0)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(1)

    decision, payload = read_gate(gate_path)
    locked_row = payload.get("locked_row") if isinstance(payload, dict) else None
    if isinstance(locked_row, dict):
        if locked_row.get("thr") is not None:
            args.thr = float(locked_row["thr"])
        if locked_row.get("min_area") is not None:
            args.min_area = int(locked_row["min_area"])
    if int(args.min_area) < 642:
        raise SystemExit(
            f"Refusing 3ch CSV: min_area {args.min_area} < 642 (hose floor)."
        )

    weights = Path(args.weights)
    if WEIGHTS_NAME not in weights.parts or "unet_semantic" in weights.parts:
        raise SystemExit("Refusing: weights must be unet_3ch_disk.")
    if not weights.exists():
        raise FileNotFoundError(weights)

    import pandas as pd
    import torch
    from tqdm import tqdm

    from eval_unet import decode_instances
    from eval_unet_3ch import infer_prob_3ch
    from filament.infer import pick_device
    from filament.merge import merge_rles
    from filament.metric import pred_to_df
    from filament.unet import build_fulldisk_unet

    images = sorted(TEST_IMG_DIR.glob("*.jpeg"))
    if not images:
        raise FileNotFoundError(TEST_IMG_DIR)
    slot1_by_stem = load_slot1_test_rles(images) if SLOT1_TEST_CSV.exists() else None
    if slot1_by_stem is None:
        raise SystemExit(f"missing Slot1 test CSV {SLOT1_TEST_CSV}")
    device = pick_device(args.device)
    model = build_fulldisk_unet(in_ch=3, base=args.base)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    frames = []
    n_added = 0
    for path in tqdm(images, desc="test merge Slot1+3ch"):
        extra = decode_instances(infer_prob_3ch(model, path, args.imgsz, device), args.thr, min_area=int(args.min_area))
        merged, added = merge_rles(slot1_by_stem[path.stem], extra, iou_thr=MERGE_IOU)
        n_added += added
        if merged:
            frames.append(pred_to_df(path.stem, merged))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    FAST_SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    FAST_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tag = "PROMOTE" if decision == PROMOTE else "DONOTUPLOAD"
    out = Path(args.out) if args.out else FAST_SUBMISSION_DIR / f"submission_slot1_unet_3ch_merged_{tag}.csv"
    df.to_csv(out, index=False)
    meta = {
        "do_not_upload": decision != PROMOTE,
        "gate": decision,
        "thr": args.thr,
        "min_area": int(args.min_area),
        "n_rows": int(len(df)),
        "n_unet_added": n_added,
        "path": str(out),
        "never_submits": True,
        "not_yolo_phys": True,
    }
    (FAST_ARTIFACTS / f"{out.stem}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
