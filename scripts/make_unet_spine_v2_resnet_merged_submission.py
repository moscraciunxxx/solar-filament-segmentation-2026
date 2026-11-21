#!/usr/bin/env python3
"""Refuse-closed test-CSV stub for ResNet spine-v2 merge. Never submits.

A public test CSV cannot ridge-split: test JPEGs have no train-JSON spines,
and MAGFiLO test labels are forbidden. This file exists so nobody reuses
make_unet_resnet_merged_submission.py (no ridge, min_area 64 hose risk).

Legal only after artifacts/unet_spine_v2_resnet_slot1_merge_fold0_official.json
is PROMOTE_LOOK — and even then it still refuses to write a test CSV.
Hose floor min_area>=642 is enforced if a later patch adds a writer.
Default --dry-refuse. Never uploads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_unet_spine_v2 import HOSE_FLOOR
from gate_unet_family import PROMOTE, SPINE_V2_RESNET_GATE, live_mps_owner, read_gate


def refuse(gate_path: Path, min_area: int) -> str | None:
    owner = live_mps_owner()
    if owner:
        return f"Refusing spine-v2 ResNet test CSV: {owner} owns MPS."
    if int(min_area) < HOSE_FLOOR:
        return f"Refusing spine-v2 ResNet test CSV: min_area {min_area} < {HOSE_FLOOR} (hose floor)."
    decision, _ = read_gate(gate_path)
    if decision != PROMOTE:
        if decision is None:
            return f"Gate is None (need PROMOTE_LOOK). missing {gate_path}."
        return f"Gate is {decision!r} (need PROMOTE_LOOK)."
    return (
        "Refusing spine-v2 ResNet test CSV: test JPEGs have no train-JSON spines "
        "(leak-free). Do not invent MAGFiLO test ridges. Fold-0 merge only."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-refuse",
        action="store_true",
        help="Print refuse reason. Does not infer or write a CSV.",
    )
    parser.add_argument("--gate-json", default=str(SPINE_V2_RESNET_GATE))
    parser.add_argument("--min-area", type=int, default=1034)
    args = parser.parse_args()

    reason = refuse(Path(args.gate_json), int(args.min_area))
    if args.dry_refuse:
        print(reason)
        raise SystemExit(0)
    print(reason, file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
