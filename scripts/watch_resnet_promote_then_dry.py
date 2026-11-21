#!/usr/bin/env python3
"""After ResNet promote CSV is built, dry-run the ResNet upload. Never --upload."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("~/.venvs/filament/bin/python")
PROMOTE_READY = ROOT / "artifacts" / "promote_resnet_look_if_ready.json"
UPLOAD = ROOT / "scripts" / "upload_if_resnet_promote.py"
OUT = ROOT / "artifacts" / "resnet_promote_dry_upload.json"


def main() -> None:
    print(f"waiting for {PROMOTE_READY}", flush=True)
    while not PROMOTE_READY.exists():
        time.sleep(15)
    ready = json.loads(PROMOTE_READY.read_text())
    out = {
        "written": datetime.now(timezone.utc).isoformat(),
        "decision": ready.get("decision"),
        "built_csv": ready.get("built_csv"),
        "do_not_submit": True,
        "kaggle_submit_called": False,
        "upload_flag": False,
        "note": "Dry upload only. Overwatch must pass --upload on upload_if_resnet_promote.py.",
    }
    if ready.get("built_csv") is not True:
        out["skipped"] = ready.get("reason") or "CSV not built"
        OUT.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2), flush=True)
        return
    proc = subprocess.run([str(PYTHON), str(UPLOAD)], check=False)
    out["dry_returncode"] = proc.returncode
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    if proc.returncode != 0:
        raise SystemExit(f"upload_if_resnet_promote dry failed: {proc.returncode}")


if __name__ == "__main__":
    main()
