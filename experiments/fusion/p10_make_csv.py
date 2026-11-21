#!/usr/bin/env python3
"""P10 frozen dense fusion -> test CSV. usage: p10_make_csv.py OUT.csv UNET_MAPDIR CACHE_S10 CACHE_S11 CACHE_S12
F = 0.4 * (fraction of the 72 champion view masks) + 0.6 * U-Net prob; 8-conn comps of F > 0.30 kept if max > 0.60, mean >= 0.60, area >= 200."""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2, pandas as pd
sys.path.insert(0, "/workspace/champ/ckit")
from filament.rle import decode_rle, encode_rle
RULE = dict(a=0.4, t_lo=0.3, t_hi=0.6, m=0.6, amin=200)
OUT, UD, CS = Path(sys.argv[1]), Path(sys.argv[2]), [Path(c) for c in sys.argv[3:6]]
def one(stem):
    Y = np.zeros((2048, 2048), np.float32); n = 0
    for c in CS:
        d = json.loads((c / f"{stem}.json").read_text()); assert len(d) == 24
        for k in sorted(d):
            for r in d[k]: Y += decode_rle(r)
            n += 1
    assert n == 72; Y /= n
    u = cv2.imread(str(UD / f"{stem}.png"), cv2.IMREAD_GRAYSCALE); assert u is not None and u.shape == (2048, 2048)
    F = RULE["a"] * Y + (1 - RULE["a"]) * (u.astype(np.float32) / 255.0)
    nc, lab, st, _ = cv2.connectedComponentsWithStats((F > RULE["t_lo"]).astype(np.uint8), connectivity=8); rles = []
    for i in range(1, nc):
        x, y, w, h, ar = st[i]
        if ar < max(32, RULE["amin"]): continue
        sub = lab[y:y + h, x:x + w] == i; vals = F[y:y + h, x:x + w][sub]
        if vals.max() > RULE["t_hi"] and vals.mean() >= RULE["m"]:
            m = np.zeros((2048, 2048), np.uint8); m[y:y + h, x:x + w] = sub; rles.append(encode_rle(m))
    return stem, rles
if __name__ == "__main__":
    stems = sorted(p.stem for p in UD.glob("*.png")); assert len(stems) == 180
    with Pool(3) as p: res = dict(p.imap(one, stems))
    rows = [{"filament_id": f"{s}_{i + 1}", "segmentation_rle": r} for s in stems for i, r in enumerate(res[s])]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(OUT, index=False)
    print("rows", len(rows), "images_with_rows", sum(1 for s in stems if res[s]))
