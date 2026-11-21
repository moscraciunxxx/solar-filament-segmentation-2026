#!/usr/bin/env python3
"""Mine fold!=0 3ch extras + GT labels. Never fold-0. Never submits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ch3_extra_filter import FILTER_DEST, parse_min_areas, run_mine
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
    parser.add_argument("--exclude-fold", type=int, default=0)
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
    meta = run_mine(
        weights=Path(args.weights),
        filter_dest=Path(args.filter_dest),
        imgsz=int(args.imgsz),
        thrs=[float(x) for x in args.thrs.split(",") if x.strip()],
        min_areas=min_areas,
        device=args.device,
        exclude_fold=int(args.exclude_fold),
        seed=SEED,
        train_img_dir=TRAIN_IMG_DIR,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
