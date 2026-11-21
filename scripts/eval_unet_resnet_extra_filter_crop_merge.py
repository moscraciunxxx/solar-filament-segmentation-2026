#!/usr/bin/env python3
"""Fold-0 Slot1 merge after fold!=0 tiny crop-CNN extras rejector.

Same mined ResNet extras (folds 1-4). Same precision-first recall>=0.4 lock.
Dest is resnet_extra_filter_crop — does not overwrite logistic / HGB / ctx.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from gate_unet_family import (
    EXTRA_FILTER_CROP_GATE,
    FILTER_CROP_WEIGHTS_NAME,
    FILTER_CTX_WEIGHTS_NAME,
    FILTER_HGB_WEIGHTS_NAME,
    FILTER_WEIGHTS_NAME,
    HOSE_FLOOR,
    RESNET_GATE,
    refuse_extra_filter,
)
from resnet_extra_crop import fit_extra_rejector_crop
from resnet_extra_filter import parse_min_areas, run_fold0_merge

CROP_DEST = WEIGHTS_DIR / FILTER_CROP_WEIGHTS_NAME
DEFAULT_WEIGHTS = WEIGHTS_DIR / "unet_resnet34" / "best.pt"
BLOCKED = {FILTER_WEIGHTS_NAME, FILTER_HGB_WEIGHTS_NAME, FILTER_CTX_WEIGHTS_NAME}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-refuse", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gate-json", default=str(RESNET_GATE))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--filter-dest", default=str(CROP_DEST))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thrs", default="0.45,0.50,0.55,0.60")
    parser.add_argument("--min-area", type=int, default=HOSE_FLOOR)
    parser.add_argument("--min-areas", default="642,1034")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    dest = Path(args.filter_dest)
    if dest.name in BLOCKED or any(name in dest.parts for name in BLOCKED):
        print("Refusing crop extra-filter dest: must not overwrite logistic/HGB/ctx.", file=sys.stderr)
        raise SystemExit(2)
    reason = refuse_extra_filter(
        Path(args.gate_json),
        min_area=int(args.min_area),
        filter_dest=dest,
        resnet_weights=Path(args.weights),
    )
    if args.dry_refuse or not args.run:
        print(reason)
        raise SystemExit(0)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(2)

    min_areas = parse_min_areas(args.min_areas, args.min_area)
    out = run_fold0_merge(
        weights=Path(args.weights),
        filter_dest=dest,
        imgsz=int(args.imgsz),
        thrs=[float(x) for x in args.thrs.split(",") if x.strip()],
        min_areas=min_areas,
        device=args.device,
        fold=int(args.fold),
        seed=SEED,
        train_img_dir=TRAIN_IMG_DIR,
        fit_fn=fit_extra_rejector_crop,
        official_dest=EXTRA_FILTER_CROP_GATE,
        family="unet_resnet_extra_filter_crop_slot1_merge",
        builder="build_resnet34_unet + tiny crop-CNN extras rejector",
        decode="ResNet CC extras, fold!=0-trained crop rejector, then Slot1 NMS-merge",
        use_context=True,
        use_crop=True,
    )
    print(
        json.dumps(
            {
                "path": str(EXTRA_FILTER_CROP_GATE),
                "locked": out.get("locked_row"),
                "gate": out.get("gate"),
                "do_not_submit": True,
                "scored": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
