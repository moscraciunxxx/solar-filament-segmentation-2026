#!/usr/bin/env python3
"""Sanity-check a test CSV. Does not score and does not read test labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.paths import TEST_IMG_DIR


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_submission_ids.py CSV")
    path = Path(sys.argv[1])
    text = path.read_text().splitlines()
    header = text[0] if text else ""
    stems = {p.stem for p in TEST_IMG_DIR.glob("*.jpeg")}
    unknown = 0
    nulls = 0
    ids = 0
    seen = set()
    for line in text[1:]:
        if not line.strip():
            continue
        fid, _, rle = line.partition(",")
        ids += 1
        if rle == "" or rle.lower() == "nan":
            nulls += 1
        stem = fid.rsplit("_", 1)[0]
        seen.add(stem)
        if stem not in stems:
            unknown += 1
    report = {
        "path": str(path),
        "header": header,
        "header_ok": header == "filament_id,segmentation_rle",
        "n_rows": ids,
        "n_images": len(seen),
        "n_test_jpegs": len(stems),
        "unknown_ids": unknown,
        "null_encodings": nulls,
    }
    print(json.dumps(report, indent=2))
    if not report["header_ok"] or unknown or nulls:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
