#!/usr/bin/env python3
"""Build a test CSV only after HGB extra-filter PROMOTE_LOOK. Never submits.

Dest is resnet_extra_filter_hgb. Refuses the logistic dest. Default --dry-refuse.
On --run after a scored PROMOTE_LOOK, reuses the extra-filter CSV infer with HGB dest.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import WEIGHTS_DIR
from gate_unet_family import (
    EXTRA_FILTER_HGB_GATE,
    FILTER_HGB_WEIGHTS_NAME,
    HOSE_FLOOR,
    refuse_extra_filter_csv,
)
from resnet_extra_filter import FILTER_DEST

PYTHON = Path("~/.venvs/filament/bin/python")
LOGISTIC_MAKE = ROOT / "scripts" / "make_unet_resnet_extra_filter_merged_submission.py"
HGB_DEST = WEIGHTS_DIR / FILTER_HGB_WEIGHTS_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-refuse", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gate-json", default=str(EXTRA_FILTER_HGB_GATE))
    parser.add_argument("--filter-dest", default=str(HGB_DEST))
    parser.add_argument("--min-area", type=int, default=1034)
    args = parser.parse_args()

    dest = Path(args.filter_dest)
    if dest.resolve() == Path(FILTER_DEST).resolve():
        print("Refusing HGB CSV dest: must not overwrite logistic resnet_extra_filter.", file=sys.stderr)
        raise SystemExit(2)
    reason = refuse_extra_filter_csv(Path(args.gate_json), min_area=int(args.min_area))
    if args.dry_refuse or not args.run:
        print(reason)
        raise SystemExit(0)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(2)
    os.execv(
        str(PYTHON),
        [
            str(PYTHON),
            str(LOGISTIC_MAKE),
            "--run",
            "--gate-json",
            str(args.gate_json),
            "--filter-dest",
            str(dest),
            "--min-area",
            str(int(args.min_area)),
        ],
    )


if __name__ == "__main__":
    main()
