#!/usr/bin/env python3
"""Build a test CSV only after context extra-filter PROMOTE_LOOK. Never submits.

Dest is resnet_extra_filter_ctx. Refuses logistic and HGB dests. Default --dry-refuse.
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
    EXTRA_FILTER_CTX_GATE,
    FILTER_CTX_WEIGHTS_NAME,
    FILTER_HGB_WEIGHTS_NAME,
    FILTER_WEIGHTS_NAME,
    refuse_extra_filter_csv,
)

PYTHON = Path("~/.venvs/filament/bin/python")
LOGISTIC_MAKE = ROOT / "scripts" / "make_unet_resnet_extra_filter_merged_submission.py"
CTX_DEST = WEIGHTS_DIR / FILTER_CTX_WEIGHTS_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-refuse", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gate-json", default=str(EXTRA_FILTER_CTX_GATE))
    parser.add_argument("--filter-dest", default=str(CTX_DEST))
    parser.add_argument("--min-area", type=int, default=1034)
    args = parser.parse_args()

    dest = Path(args.filter_dest)
    blocked = {FILTER_WEIGHTS_NAME, FILTER_HGB_WEIGHTS_NAME}
    if dest.name in blocked or any(name in dest.parts for name in blocked):
        print("Refusing ctx CSV dest: must not overwrite logistic or HGB.", file=sys.stderr)
        raise SystemExit(2)
    reason = refuse_extra_filter_csv(Path(args.gate_json), min_area=int(args.min_area))
    if args.dry_refuse or not args.run:
        print(reason)
        raise SystemExit(0)
    if reason:
        print(reason, file=sys.stderr)
        raise SystemExit(2)
    print(
        "Context extra-filter CSV infer is not wired until locked_row feature dim is stored; "
        "parent must not upload without a context-aware make path.",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
