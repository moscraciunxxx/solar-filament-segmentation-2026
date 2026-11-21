#!/usr/bin/env python3
"""After wait_quota_reset writes its snapshot, parse remaining looks. Never submits."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "kaggle_quota_after_reset.json"
PARSE = ROOT / "scripts" / "parse_kaggle_quota.py"
PYTHON = Path("~/.venvs/filament/bin/python")


def main() -> None:
    print(f"waiting for {RAW}", flush=True)
    while not RAW.exists():
        time.sleep(10)
    subprocess.run([str(PYTHON), str(PARSE)], check=False)


if __name__ == "__main__":
    main()
