#!/usr/bin/env python3
"""Quick pooled PQ (all annotators, pycocotools GT, IoU>0.5, area-desc exclusive resolve, drop <32) for RF-DETR outputs.
usage: rfd_quickpq.py PREDDIR STEMS_FILE [SCORE_THR]"""
import sys, json
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
P, stems = Path(sys.argv[1]), Path(sys.argv[2]).read_text().split(); thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
ann = next(Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json")); coco = json.loads(ann.read_text())
vid = {}; [vid.setdefault(Path(im["file_name"]).stem, []).append(im["id"]) for im in coco["images"]]
segs = {}; [segs.setdefault(a["image_id"], []).append(a["segmentation"]) for a in coco["annotations"]]
tot = np.zeros(4); ninst = 0
for s in stems:
    rec = [r for r in json.loads((P / f"{s}.json").read_text()) if r["score"] >= thr]
    ms = sorted([mu.decode({"size": [2048, 2048], "counts": r["rle"]}).astype(bool) for r in rec], key=lambda m: -m.sum())
    cl = np.zeros((2048, 2048), bool); pr = []
    for m in ms:
        e = m & ~cl
        if e.sum() >= 32: cl |= e; pr.append(e)
    ninst += len(pr)
    for v in vid[s]:
        gl = []
        for sg in segs.get(v, []):
            pol = sg if isinstance(sg[0], list) else [sg]; gl.append(mu.decode(mu.merge(mu.frPyObjects([list(map(float, q)) for q in pol if len(q) >= 6], 2048, 2048))).astype(bool))
        hp = np.zeros(len(pr), bool); hg = np.zeros(len(gl), bool)
        for i, m in enumerate(pr):
            for j, g in enumerate(gl):
                it = (m & g).sum()
                if it and it / (m.sum() + g.sum() - it) > 0.5: tot[0] += it / (m.sum() + g.sum() - it); tot[1] += 1; hp[i] = hg[j] = True
        tot[2] += (~hp).sum(); tot[3] += (~hg).sum()
pq = tot[0] / max(1e-9, tot[1] + 0.5 * tot[2] + 0.5 * tot[3])
print(json.dumps({"quick_pq": pq, "TP_FP_FN": tot[1:].astype(int).tolist(), "inst_per_img": ninst / len(stems), "thr": thr}), flush=True)
