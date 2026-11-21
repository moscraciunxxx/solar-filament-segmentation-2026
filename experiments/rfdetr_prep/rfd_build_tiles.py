#!/usr/bin/env python3
"""Build RF-DETR-Seg COCO tiles from OUR eroded-medoid labels (filament-d4-erode-gt: yolo_seg_erode_gt, YOLO-seg polygons).
usage: rfd_build_tiles.py OUTDIR TRAIN_FOLDS VALID_FOLDS     e.g. "2,3,4" "1"   or  "0,1,2,3,4" "0"  (full refit; valid = monitor only)
Tiles: 768 px at native scale, stride 640 -> 3x3 = 9 tiles per 2048 image (resolution 768 = multiple of 24, 1:1 scale, no thin-filament downsampling).
Instance masks are rasterised from the polygons, cut per tile, and re-polygonised (cv2.findContours, parts >= 16 px)."""
import sys, json, csv, random
from pathlib import Path
import numpy as np, cv2
OUT = Path(sys.argv[1]); TR = {int(x) for x in sys.argv[2].split(",")}; VA = {int(x) for x in sys.argv[3].split(",")}
import os
INP = Path(os.environ.get("RFD_INP", "/kaggle/input")); KIT = Path(__file__).parent; random.seed(20261113)
d4 = next(INP.rglob("yolo_seg_erode_gt/data.yaml")).parent
folds = {r["stem"]: int(r["fold"]) for r in csv.DictReader(open(KIT / "folds_ours.csv"))}
imgs = {p.stem: p for s in ("train", "val") for p in (d4 / "images" / s).iterdir()}
labs = {p.stem: p for s in ("train", "val") for p in (d4 / "labels" / s).iterdir()}
assert len(imgs) == 707 or os.environ.get("RFD_INP"), len(imgs)
folds = {s: f for s, f in folds.items() if s in imgs}
T, S = 768, 640; POS = [0, 640, 1280]
def inst_masks(stem):
    ms = []
    if stem not in labs: return ms
    for line in labs[stem].read_text().splitlines():
        v = line.split()
        if len(v) < 7: continue
        pts = (np.array(v[1:], np.float32).reshape(-1, 2) * 2048).round().astype(np.int32)
        m = np.zeros((2048, 2048), np.uint8); cv2.fillPoly(m, [pts], 1); ms.append(m)
    return ms
VLIM = int(sys.argv[4]) if len(sys.argv) > 4 else 0
def build(split, fset, keep_empty, limit=0):
    d = OUT / split; d.mkdir(parents=True, exist_ok=True); coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "filament", "supercategory": "filament"}]}
    aid = 1; n_img = 0
    sel = sorted(s for s, f in folds.items() if f in fset)
    for stem in (sel[:limit] if limit else sel):
        img = cv2.imread(str(imgs[stem])); assert img is not None and img.shape[:2] == (2048, 2048), stem
        ms = inst_masks(stem)
        for y0 in POS:
            for x0 in POS:
                anns = []
                for m in ms:
                    c = m[y0:y0 + T, x0:x0 + T]
                    if c.sum() < 16: continue
                    cs, _ = cv2.findContours(c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    segs = [cc.reshape(-1).astype(float).tolist() for cc in cs if cv2.contourArea(cc) >= 16 and len(cc) >= 3]
                    if not segs: continue
                    ys, xs = np.nonzero(c); anns.append({"segmentation": segs, "bbox": [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)], "area": float(c.sum())})
                if not anns and random.random() > keep_empty: continue
                n_img += 1; fn = f"{stem}_{y0}_{x0}.jpg"; cv2.imwrite(str(d / fn), img[y0:y0 + T, x0:x0 + T], [cv2.IMWRITE_JPEG_QUALITY, 95])
                coco["images"].append({"id": n_img, "file_name": fn, "width": T, "height": T})
                for a in anns: a.update(id=aid, image_id=n_img, category_id=1, iscrowd=0); coco["annotations"].append(a); aid += 1
    (d / "_annotations.coco.json").write_text(json.dumps(coco)); print(split, "tiles", n_img, "instances", aid - 1, flush=True)
build("train", TR, 0.3); build("valid", VA, 1.0, VLIM)
import shutil; shutil.copytree(OUT / "valid", OUT / "test", dirs_exist_ok=True)
