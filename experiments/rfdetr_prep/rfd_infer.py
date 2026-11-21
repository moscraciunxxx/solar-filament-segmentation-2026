#!/usr/bin/env python3
"""Tiled RF-DETR-Seg inference -> per-stem instance list (pycocotools RLE + score) for our scorer.
usage: rfd_infer.py CKPT OUTDIR IMGDIR [STEMS_FILE]   (9 tiles 768/stride 640; masks pasted to 2048; greedy merge by score:
drop if mask IoU > 0.5 with a kept mask or >= 70% of its area inside a kept mask). Score threshold 0.2 (decode tuned only on OOF later)."""
import sys, json
from pathlib import Path
import numpy as np, cv2
from pycocotools import mask as mu
from rfdetr import RFDETRSegMedium
ck, OUT, IMGDIR = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
stems = Path(sys.argv[4]).read_text().split() if len(sys.argv) > 4 else sorted(p.stem for p in IMGDIR.glob("*.jpeg"))
model = RFDETRSegMedium(resolution=768) if ck == "pretrained" else RFDETRSegMedium(pretrain_weights=ck, resolution=768)
T, POS = 768, [0, 640, 1280]
for s in stems:
    img = cv2.imread(str(IMGDIR / f"{s}.jpeg")); cand = []
    for y0 in POS:
        for x0 in POS:
            det = model.predict(img[y0:y0 + T, x0:x0 + T, ::-1].copy(), threshold=0.2)
            if det.mask is None: continue
            for m, sc in zip(det.mask, det.confidence):
                if m.sum() < 16: continue
                M = np.zeros((2048, 2048), bool); M[y0:y0 + T, x0:x0 + T] = m; cand.append((float(sc), M))
    cand.sort(key=lambda t: -t[0]); kept = []
    for sc, M in cand:
        a = M.sum(); ok = True
        for _, K in kept:
            i = np.logical_and(M, K).sum()
            if i and (i / (a + K.sum() - i) > 0.5 or i >= 0.7 * a): ok = False; break
        if ok: kept.append((sc, M))
    rec = [{"score": sc, "rle": mu.encode(np.asfortranarray(M.astype(np.uint8)))["counts"].decode()} for sc, M in kept]
    (OUT / f"{s}.json").write_text(json.dumps(rec))
print("INFER_DONE", len(stems), flush=True)
