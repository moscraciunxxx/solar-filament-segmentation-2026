#!/usr/bin/env python3
"""P6/P7 frozen decode of uint8 U-Net maps -> submission CSV + format checks.
Rule (EffNetV2-S family, frozen in P6): components of u8 > 0.35*255 (8-connected) kept if max > 0.995*255, mean >= 0.85*255 (area >= 0);
then area-descending exclusive resolve, remnant < 32 px dropped. CSV: filament_id = <stem>_<i> (1-based), segmentation_rle = COCO compressed counts.
Format reference: the champion's submitted CSV (header filament_id,segmentation_rle); the competition ships no sample_submission file.
usage: make_csv.py MAPDIR TEST_IMGDIR OUT.csv"""
import sys, json
from pathlib import Path
import numpy as np, cv2, pandas as pd
from pycocotools import mask as mu
MD, TD, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
def comps(u):
    n, lab, st, _ = cv2.connectedComponentsWithStats((u > 0.35 * 255).astype(np.uint8), connectivity=8); out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]; sub = lab[y:y + h, x:x + w] == i; v = u[y:y + h, x:x + w][sub]
        if v.max() > 0.995 * 255 and v.mean() >= 0.85 * 255:
            m = np.zeros((2048, 2048), bool); m[y:y + h, x:x + w] = sub; out.append((m, int(a)))
    out.sort(key=lambda t: -t[1]); cl = np.zeros((2048, 2048), bool); res = []
    for m, a in out:
        e = m & ~cl
        if e.sum() >= 32: cl |= e; res.append(e)
    return res
stems = sorted(p.stem for p in TD.glob("*.jpeg")); rows = []; per = {}
for s in stems:
    u = cv2.imread(str(MD / f"{s}.png"), cv2.IMREAD_GRAYSCALE); assert u is not None and u.shape == (2048, 2048), f"missing map {s}"
    ms = comps(u); per[s] = len(ms)
    for i, m in enumerate(ms, 1):
        rows.append((f"{s}_{i}", mu.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")))
df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]); df.to_csv(OUT, index=False)
# ---- format checks ----
chk = pd.read_csv(OUT); errs = []
if list(chk.columns) != ["filament_id", "segmentation_rle"]: errs.append(f"columns {list(chk.columns)}")
if chk["filament_id"].duplicated().any(): errs.append("duplicate filament_id")
if chk["segmentation_rle"].isna().any(): errs.append("empty RLE")
st = chk["filament_id"].str.rsplit("_", n=1).str[0]
if not set(st) <= set(stems): errs.append("ids for non-test stems")
if not chk["filament_id"].str.rsplit("_", n=1).str[1].str.isdigit().all(): errs.append("non-integer index")
for s, g in chk.groupby(st):
    ms = mu.decode([{"size": [2048, 2048], "counts": r} for r in g["segmentation_rle"]])
    if ms.shape[:2] != (2048, 2048): errs.append(f"{s} shape"); continue
    if (ms.sum(2) > 1).any(): errs.append(f"{s} overlapping masks")
    if (ms.reshape(-1, ms.shape[2]).sum(0) < 32).any(): errs.append(f"{s} tiny mask")
rep = {"rows": len(chk), "test_images": len(stems), "images_with_rows": int((pd.Series(per) > 0).sum()), "mean_per_img": float(np.mean(list(per.values()))), "errors": errs[:20], "FORMAT_OK": not errs}
json.dump(rep, open(OUT.with_suffix(".check.json"), "w"), indent=1); print("CSV_CHECK", json.dumps(rep), flush=True)
if errs: sys.exit(4)
