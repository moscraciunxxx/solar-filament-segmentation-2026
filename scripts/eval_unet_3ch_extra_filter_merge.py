#!/usr/bin/env python3
"""Fold-0 Slot1 merge after a fold!=0-trained 3ch extras rejector. Never submits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ch3_extra_filter import FILTER_DEST, parse_min_areas, run_fold0_merge
from filament.paths import SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
from gate_unet_family import CH3_GATE, HOSE_FLOOR, refuse_3ch_extra_filter

DEFAULT_WEIGHTS = WEIGHTS_DIR / "unet_3ch_disk" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-refuse", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gate-json", default=str(CH3_GATE))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--filter-dest", default=str(FILTER_DEST))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thrs", default="0.45,0.50,0.55,0.60")
    parser.add_argument("--min-area", type=int, default=HOSE_FLOOR)
    parser.add_argument("--min-areas", default="642,1034")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    reason = refuse_3ch_extra_filter(
        Path(args.gate_json),
        min_area=int(args.min_area),
        filter_dest=Path(args.filter_dest),
        ch3_weights=Path(args.weights),
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
        filter_dest=Path(args.filter_dest),
        imgsz=int(args.imgsz),
        thrs=[float(x) for x in args.thrs.split(",") if x.strip()],
        min_areas=min_areas,
        device=args.device,
        fold=int(args.fold),
        seed=SEED,
        train_img_dir=TRAIN_IMG_DIR,
    )
    print(
        json.dumps(
            {
                "path": str(ROOT / "artifacts" / "unet_3ch_extra_filter_slot1_merge_fold0_official.json"),
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
