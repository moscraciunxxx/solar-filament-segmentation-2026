#!/usr/bin/env python3
"""Upload Slot1+crop extra-filter CSV only after a copied crop PROMOTE_LOOK.

Default is dry. --upload required. Year-shift 56148889 stays Slot 2.
Hidden-exam: climb look, not certified top-10, not a leak-pack chase.
Never uploads ctx 0.4155 / logistic 0.414 / HGB HOLDs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filament.paths import ARTIFACTS, FAST_SUBMISSION_DIR
from gate_unet_family import EXTRA_FILTER_CROP_GATE, HOSE_FLOOR, PROMOTE, read_gate

KAGGLE = Path("~/.local/bin/kaggle")
COMP = "filament-segmentation-2026"
CSV = FAST_SUBMISSION_DIR / "submission_slot1_unet_resnet_extra_filter_crop_PROMOTE.csv"
REMAINING = ARTIFACTS / "kaggle_quota_remaining.json"
RECEIPT = ARTIFACTS / "upload_if_crop_promote.json"
BANNED = (
    "faint",
    "merckle",
    "ctx",
    "hgb",
    "tinyunet",
    "unet_semantic",
    "extra_filter_PROMOTE",
    "extra_filter_DONOTUPLOAD",
)


def refuse() -> str | None:
    decision, payload = read_gate(EXTRA_FILTER_CROP_GATE)
    if decision != PROMOTE:
        return f"Refusing crop upload: gate is {decision!r} (need PROMOTE_LOOK)."
    if not CSV.exists():
        return f"Refusing crop upload: missing {CSV}. Run make_unet_resnet_extra_filter_crop_merged_submission.py --run first."
    name = CSV.name.lower()
    if any(b.lower() in name for b in BANNED if b != "ctx"):
        return f"Refusing crop upload: banned filename {CSV.name}."
    if "crop" not in name:
        return f"Refusing crop upload: filename must be the crop PROMOTE CSV ({CSV.name})."
    locked_row = payload.get("locked_row") if isinstance(payload, dict) else None
    if isinstance(locked_row, dict) and locked_row.get("min_area") is not None:
        if int(locked_row["min_area"]) < HOSE_FLOOR:
            return f"Refusing crop upload: locked min_area {locked_row['min_area']} < {HOSE_FLOOR}."
    if REMAINING.exists():
        rem = json.loads(REMAINING.read_text()).get("remaining")
        if rem is not None and int(rem) < 1:
            return "Refusing crop upload: remaining looks is 0."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    reason = refuse()
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "csv": str(CSV),
        "gate": str(EXTRA_FILTER_CROP_GATE),
        "upload_flag": args.upload,
        "kaggle_submit_called": False,
        "do_not_upload_ctx_or_logistic_hold": True,
        "slot2_stays": 56148889,
        "hidden_exam": {
            "public_best": 56148686,
            "hedge": 56148889,
            "this_file_if_uploaded": "climb look: Slot1 boxes + crop-CNN extras; year-shift stays Slot 2",
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
    import subprocess

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
            "Slot1+crop extra-filter PROMOTE_LOOK; year-shift 56148889 stays Slot2; climb look not leak pack",
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
