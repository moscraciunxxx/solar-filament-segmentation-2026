#!/usr/bin/env python3
"""Same submission checks as the U-Net make_csv.py: header filament_id,segmentation_rle; unique ids; ids only for the 180 test stems;
integer index; non-empty RLE; every RLE decodes to 2048x2048; no overlapping masks within an image; no mask < 32 px. Exit 4 on any error.
usage: csv_check.py SUB.csv TEST_IMGDIR"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from pycocotools import mask as mu
OUT, TD = Path(sys.argv[1]), Path(sys.argv[2]); stems = sorted(p.stem for p in TD.glob("*.jpeg"))
chk = pd.read_csv(OUT); errs = []
if list(chk.columns) != ["filament_id", "segmentation_rle"]: errs.append(f"columns {list(chk.columns)}")
if chk["filament_id"].duplicated().any(): errs.append("duplicate filament_id")
if chk["segmentation_rle"].isna().any(): errs.append("empty RLE")
st = chk["filament_id"].astype(str).str.rsplit("_", n=1).str[0]
if not set(st) <= set(stems): errs.append("ids for non-test stems")
if not chk["filament_id"].astype(str).str.rsplit("_", n=1).str[1].str.isdigit().all(): errs.append("non-integer index")
for s, g in chk.groupby(st):
    ms = mu.decode([{"size": [2048, 2048], "counts": r} for r in g["segmentation_rle"]])
    if ms.shape[:2] != (2048, 2048): errs.append(f"{s} shape"); continue
    if (ms.sum(2) > 1).any(): errs.append(f"{s} overlapping masks")
    if (ms.reshape(-1, ms.shape[2]).sum(0) < 32).any(): errs.append(f"{s} tiny mask")
rep = {"rows": len(chk), "test_images": len(stems), "images_with_rows": int(st.nunique()), "errors": errs[:20], "FORMAT_OK": not errs}
json.dump(rep, open(OUT.with_suffix(".check.json"), "w"), indent=1); print("CSV_CHECK", json.dumps(rep), flush=True)
if errs: sys.exit(4)
