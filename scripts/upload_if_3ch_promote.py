#!/usr/bin/env python3
"""Upload Slot1+3ch merged CSV only after 3ch PROMOTE_LOOK. Default is dry."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_SUBMISSION_DIR
from gate_unet_family import CH3_GATE, PROMOTE, read_gate

KAGGLE = Path("~/.local/bin/kaggle")
COMP = "filament-segmentation-2026"
CSV = FAST_SUBMISSION_DIR / "submission_slot1_unet_3ch_merged_PROMOTE.csv"
BANNED = ("faint", "merckle", "phys", "tiles", "tinyunet", "unet_semantic", "resnet34", "unet_patch", "slot1_unet_merged_PROMOTE")
REMAINING = ARTIFACTS / "kaggle_quota_remaining.json"
RECEIPT = ARTIFACTS / "upload_if_3ch_promote.json"


def refuse() -> str | None:
    decision, payload = read_gate(CH3_GATE)
    if decision != PROMOTE:
        return f"Refusing 3ch upload: gate is {decision!r} (need PROMOTE_LOOK)."
    if not CSV.exists():
        return f"Refusing 3ch upload: missing {CSV}."
    if any(b.lower() in CSV.name.lower() for b in BANNED):
        return f"Refusing 3ch upload: banned filename {CSV.name}."
    locked_row = payload.get("locked_row") if isinstance(payload, dict) else None
    if not isinstance(locked_row, dict) or locked_row.get("min_area") is None:
        return "Refusing 3ch upload: locked_row.min_area missing (hose floor require)."
    if int(locked_row["min_area"]) < 642:
        return f"Refusing 3ch upload: locked min_area {locked_row['min_area']} < 642."
    if REMAINING.exists():
        rem = json.loads(REMAINING.read_text()).get("remaining")
        if rem is not None and int(rem) < 1:
            return "Refusing 3ch upload: remaining looks is 0."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    reason = refuse()
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "csv": str(CSV),
        "gate": str(CH3_GATE),
        "upload_flag": args.upload,
        "kaggle_submit_called": False,
        "slot2_stays": 56148889,
        "not_yolo_phys": True,
        "certified_top10": False,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if reason:
        out["refused"] = reason
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(reason)
        raise SystemExit(0 if not args.upload else 1)
    if not args.upload:
        out["dry"] = True
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    proc = subprocess.run(
        [str(KAGGLE), "competitions", "submit", "-c", COMP, "-f", str(CSV), "-m",
         "Slot1+3ch U-Net merge PROMOTE_LOOK; year-shift 56148889 stays Slot2; climb look not leak pack"],
        capture_output=True, text=True, check=False,
    )
    out["kaggle_submit_called"] = True
    out["returncode"] = proc.returncode
    out["stdout_tail"] = (proc.stdout or "")[-2000:]
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"returncode": proc.returncode, "csv": str(CSV), "kaggle_submit_called": True}, indent=2))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
