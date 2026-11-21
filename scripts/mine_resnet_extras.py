#!/usr/bin/env python3
"""Mine fold!=0 ResNet extras + GT labels. Never fold-0. Never submits.

Writes FAST_ARTIFACTS/resnet_extras_foldne0.jsonl. ResNet weights are read-only
(unet_resnet34). Filter dest is WEIGHTS_DIR/resnet_extra_filter.
Default --dry-refuse. Does not start MPS on --dry-refuse. No MAGFiLO test labels.
Not classical new seeds.
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
from gate_unet_family import HOSE_FLOOR, RESNET_GATE, refuse_extra_filter
from resnet_extra_filter import FILTER_DEST, parse_min_areas

DEFAULT_WEIGHTS = WEIGHTS_DIR / "unet_resnet34" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-refuse", action="store_true")
    parser.add_argument("--run", action="store_true", help="Mine fold!=0 extras. Off by default.")
    parser.add_argument("--gate-json", default=str(RESNET_GATE))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--filter-dest", default=str(FILTER_DEST))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--thrs", default="0.45,0.50,0.55,0.60")
    parser.add_argument("--min-area", type=int, default=HOSE_FLOOR)
    parser.add_argument("--min-areas", default="642,1034")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--exclude-fold", type=int, default=0)
    args = parser.parse_args()

    reason = refuse_extra_filter(
        Path(args.gate_json),
        min_area=int(args.min_area),
        filter_dest=Path(args.filter_dest),
        resnet_weights=Path(args.weights),
    )
    if args.dry_refuse or not args.run:
        print(reason)
        raise SystemExit(0)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(2)

    min_areas = parse_min_areas(args.min_areas, args.min_area)
    from resnet_extra_filter import run_mine

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
