#!/usr/bin/env python3
"""Fold-0 Slot1 merge after fold!=0 HGB extras rejector. Never submits.

Reuses mined ResNet extras (folds 1-4). Same precision-first recall>=0.4 lock
as logistic — not a fold-0 PQ invert of the 0.414 HOLD. Dest is
resnet_extra_filter_hgb (does not overwrite the logistic rejector).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from gate_unet_family import (
    EXTRA_FILTER_HGB_GATE,
    FILTER_HGB_WEIGHTS_NAME,
    HOSE_FLOOR,
    RESNET_GATE,
    refuse_extra_filter,
)
from resnet_extra_filter import (
    FILTER_DEST,
    fit_extra_rejector_hgb,
    parse_min_areas,
    run_fold0_merge,
)

HGB_DEST = WEIGHTS_DIR / FILTER_HGB_WEIGHTS_NAME
HGB_GATE = EXTRA_FILTER_HGB_GATE
DEFAULT_WEIGHTS = WEIGHTS_DIR / "unet_resnet34" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-refuse", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gate-json", default=str(RESNET_GATE))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--filter-dest", default=str(HGB_DEST))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thrs", default="0.45,0.50,0.55,0.60")
    parser.add_argument("--min-area", type=int, default=HOSE_FLOOR)
    parser.add_argument("--min-areas", default="642,1034")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    dest = Path(args.filter_dest)
    if dest.resolve() == Path(FILTER_DEST).resolve():
        print("Refusing HGB extra-filter dest: must not overwrite logistic resnet_extra_filter.", file=sys.stderr)
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
        fit_fn=fit_extra_rejector_hgb,
        official_dest=HGB_GATE,
        family="unet_resnet_extra_filter_hgb_slot1_merge",
        builder="build_resnet34_unet + hgb_morphometrics extras rejector",
        decode="ResNet CC extras, fold!=0-trained HistGB rejector, then Slot1 NMS-merge",
    )
    print(
        json.dumps(
            {
                "path": str(HGB_GATE),
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
