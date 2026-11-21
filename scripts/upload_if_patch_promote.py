#!/usr/bin/env python3
"""Upload Slot1+patch merged CSV only after patch PROMOTE_LOOK. Default is dry.

Never uses the v1 TinyUNet or ResNet34 CSV. Never uploads unless all of:
patch merge gate == PROMOTE_LOOK, remaining looks >= 1, patch PROMOTE CSV
exists, and --upload is passed. Year-shift 56148889 stays Slot 2.
"""

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
from gate_unet_family import PATCH_GATE, PROMOTE, read_gate

KAGGLE = Path("~/.local/bin/kaggle")
COMP = "filament-segmentation-2026"
CSV = FAST_SUBMISSION_DIR / "submission_slot1_unet_patch_merged_PROMOTE.csv"
BANNED = (
    "faint",
    "merckle",
    "conf025",
    "phys",
    "tiles",
    "tinyunet",
    "unet_semantic",
    "resnet34",
    "slot1_unet_merged_PROMOTE",
)
REMAINING = ARTIFACTS / "kaggle_quota_remaining.json"
RECEIPT = ARTIFACTS / "upload_if_patch_promote.json"


def refuse() -> str | None:
    decision, payload = read_gate(PATCH_GATE)
    if decision != PROMOTE:
        return f"Refusing patch upload: gate is {decision!r} (need PROMOTE_LOOK)."
    if not CSV.exists():
        return f"Refusing patch upload: missing {CSV}. Run make_unet_patch_merged_submission.py first."
    name = CSV.name.lower()
    if any(b.lower() in name for b in BANNED):
        return f"Refusing patch upload: banned filename {CSV.name}."
    locked_row = payload.get("locked_row") if isinstance(payload, dict) else None
    if not isinstance(locked_row, dict) or locked_row.get("min_area") is None:
        return "Refusing patch upload: locked_row.min_area missing (hose floor require)."
    if int(locked_row["min_area"]) < 642:
        return f"Refusing patch upload: locked min_area {locked_row['min_area']} < 642 (hose floor)."
    if REMAINING.exists():
        rem = json.loads(REMAINING.read_text()).get("remaining")
        if rem is not None and int(rem) < 1:
            return "Refusing patch upload: remaining looks is 0."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Actually call kaggle competitions submit. Default is dry.",
    )
    args = parser.parse_args()
    reason = refuse()
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "csv": str(CSV),
        "gate": str(PATCH_GATE),
        "upload_flag": args.upload,
        "kaggle_submit_called": False,
        "do_not_upload_faint_or_merckle": True,
        "do_not_upload_v1_tinyunet": True,
        "do_not_upload_resnet_csv": True,
        "slot2_stays": 56148889,
        "hidden_exam": {
            "public_best": 56148686,
            "hedge": 56148889,
            "certified_top10": False,
            "do_not_chase_0.55": True,
        },
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if reason:
        out["refused"] = reason
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(reason)
        raise SystemExit(0 if not args.upload else 1)
    if not args.upload:
        out["dry"] = True
        out["would_submit"] = str(CSV)
        RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
        return
    proc = subprocess.run(
        [
            str(KAGGLE),
            "competitions",
            "submit",
            "-c",
            COMP,
            "-f",
            str(CSV),
            "-m",
            "Slot1+patch-UNet merge PROMOTE_LOOK; year-shift 56148889 stays Slot2; climb look not leak pack",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    out["kaggle_submit_called"] = True
    out["returncode"] = proc.returncode
    out["stdout_tail"] = (proc.stdout or "")[-2000:]
    out["stderr_tail"] = (proc.stderr or "")[-1000:]
    RECEIPT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("returncode", "csv", "kaggle_submit_called")}, indent=2))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
