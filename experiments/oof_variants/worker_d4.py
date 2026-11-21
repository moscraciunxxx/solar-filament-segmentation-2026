#!/usr/bin/env python3
"""One held-out fold: train champion yolo11s erode-gt recipe on folds {1,2,3,4}-{k}, then 8x3 TTA on fold k.
OOM auto batch-halving is disabled (raise). Checkpoint = last.pt (no selection on fold k)."""
import os, sys, json, time, csv, traceback
from pathlib import Path
k = int(sys.argv[1]); KIT = Path(sys.argv[2]); WORK = Path("/kaggle/working"); IMGS = WORK / "imgs"
sys.path.insert(0, str(KIT))
OUT = WORK / f"fold{k}"; OUT.mkdir(exist_ok=True)
def log(*a):
    msg = " ".join(str(x) for x in a); print(f"[fold{k}] {msg}", flush=True)
    with open(OUT / "log.txt", "a") as h: h.write(msg + "\n")
import numpy as np, cv2, torch
folds = {r["file_name"]: int(r["fold"]) for r in csv.DictReader(open(KIT / "folds_ours.csv"))}
train = sorted(f for f, v in folds.items() if v in (1, 2, 3, 4) and v != k)
held = sorted(f for f, v in folds.items() if v == k)
log("train", len(train), "held", len(held), "gpu", torch.cuda.get_device_name(0))
D = WORK / f"data{k}"
for split, files in (("train", train), ("val", held)):
    (D / "images" / split).mkdir(parents=True, exist_ok=True); (D / "labels" / split).mkdir(parents=True, exist_ok=True)
    for f in files:
        s = f.rsplit(".", 1)[0]
        dst = D / "images" / split / f
        if not dst.exists(): os.symlink(IMGS / f, dst)
        lab = WORK / "labs" / (s + ".txt")
        (D / "labels" / split / (s + ".txt")).write_text(lab.read_text() if lab.exists() else "")
(D / "data.yaml").write_text(f"path: {D}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\nnc: 1\n")
res = {"fold": k, "n_train": len(train), "n_held": len(held), "train_images_from_folds": sorted({folds[f] for f in train})}
from ultralytics import YOLO
# Memory-only patch v3 (identical math, verified loss/grad diff 0.0): per-image gt_mask + logits + BCE built inside a
# checkpointed function so none of the (n_fg, 512, 512) tensors are kept for backward. v2 (logits only) still OOMed folds 2, 3.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segloss_patch as _SP
from ultralytics.utils import loss as _L
_L.v8SegmentationLoss.calculate_segmentation_loss = _SP.calculate_segmentation_loss
log("seg-loss checkpoint patch v3 active")
import d4_patch as _D4
_D4.install_d4_transpose(0.5)
log("D4 RandomTranspose p=0.5 active")
model = YOLO(str(KIT / "yolo11s-seg.pt")) if (KIT / "yolo11s-seg.pt").exists() else YOLO("yolo11s-seg.pt")
def no_halving(tr):
    tr._oom_retries = 3  # ultralytics then re-raises OOM instead of halving
def check_batch(tr):
    if tr.batch_size != 4: raise SystemExit(f"BATCH_CHANGED {tr.batch_size}")
t_ep = {"t": time.time()}
def ep_end(tr):
    now = time.time(); log("epoch", tr.epoch + 1, "sec", round(now - t_ep["t"], 1), "gpu_peak_gib", round(torch.cuda.max_memory_reserved() / 2**30, 2)); t_ep["t"] = now
model.add_callback("on_train_epoch_start", no_halving)
model.add_callback("on_train_batch_end", check_batch)
model.add_callback("on_train_epoch_end", ep_end)
t0 = time.time()
kw = dict(data=str(D / "data.yaml"), epochs=40, imgsz=2048, batch=4, patience=1000, device=0, seed=20260910,
          deterministic=True, single_cls=True, workers=4, cache=False, exist_ok=True, project=str(WORK / "runs"),
          name=f"yolo11s_d4b4_2048_holdout_f{k}", hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, degrees=15.0, translate=0.05,
          scale=0.3, flipud=0.5, fliplr=0.5, mosaic=0.8, close_mosaic=8, copy_paste=0.15, mixup=0.05, multi_scale=0.0,
          mask_ratio=4, optimizer="AdamW", lr0=0.001, weight_decay=0.0005, amp=True, val=False, plots=False)
try:
    model.train(**kw)
except BaseException as e:
    res["train_error"] = f"{type(e).__name__}: {str(e)[:500]}"; log("TRAIN_FAIL", res["train_error"])
    (OUT / "result.json").write_text(json.dumps(res, indent=2)); raise
res["train_hours"] = (time.time() - t0) / 3600
last = WORK / "runs" / f"yolo11s_d4b4_2048_holdout_f{k}" / "weights" / "last.pt"
os.system(f"cp {last} {OUT}/last_f{k}.pt")
log("TRAIN_OK hours", round(res["train_hours"], 2)); (OUT / "result.json").write_text(json.dumps(res, indent=2))
# ---- 8 views x 3 scales TTA, identical to eval_parent_tta._predict_views ----
from filament.infer import _collect_yolo_instances
from filament.rle import encode_rle
r90, r270, r180 = cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180
VIEWS = [("id", lambda im: im, lambda m: m), ("r90", lambda im: cv2.rotate(im, r90), lambda m: cv2.rotate(m, r270)),
         ("r180", lambda im: cv2.rotate(im, r180), lambda m: cv2.rotate(m, r180)), ("r270", lambda im: cv2.rotate(im, r270), lambda m: cv2.rotate(m, r90)),
         ("h", lambda im: cv2.flip(im, 1), lambda m: cv2.flip(m, 1)), ("v", lambda im: cv2.flip(im, 0), lambda m: cv2.flip(m, 0)),
         ("h_r90", lambda im: cv2.rotate(cv2.flip(im, 1), r90), lambda m: cv2.flip(cv2.rotate(m, r270), 1)),
         ("v_r90", lambda im: cv2.rotate(cv2.flip(im, 0), r90), lambda m: cv2.flip(cv2.rotate(m, r270), 0))]
m = YOLO(str(last)); t1 = time.time()
f0 = sorted(f for f, v in folds.items() if v == 0)
for C, files in ((OUT / "tta", held), (OUT / "tta_f0", f0)):
  C.mkdir(exist_ok=True)
  for i, f in enumerate(files):
      image = cv2.imread(str(IMGS / f), cv2.IMREAD_COLOR); assert image.shape[:2] == (2048, 2048)
      out = {}
      for scale in (1792, 2048, 2304):
          for name, fw, inv in VIEWS:
              view = fw(image)
              r = m.predict(source=view, conf=0.35, iou=0.60, imgsz=scale, retina_masks=True, augment=False, verbose=False, device=0)
              rles = []
              for mask, _s in _collect_yolo_instances(r[0] if r else None):
                  if mask.shape != view.shape[:2]:
                      mask = cv2.resize(mask.astype(np.uint8), (view.shape[1], view.shape[0]), interpolation=cv2.INTER_NEAREST)
                  rest = inv((mask > 0).astype(np.uint8))
                  if int(rest.sum()) >= 96: rles.append(encode_rle(rest))
              out[f"{scale}_{name}"] = rles
      (C / (f.rsplit(".", 1)[0] + ".json")).write_text(json.dumps(out))
      if i % 20 == 0: log("tta", C.name, i, len(files), round(time.time() - t1, 1))
res["tta_hours"] = (time.time() - t1) / 3600; res["status"] = "DONE"; res["d4_draws_main"] = [_D4.RandomTranspose.calls, _D4.RandomTranspose.hits]
(OUT / "result.json").write_text(json.dumps(res, indent=2)); log("FOLD_DONE", json.dumps(res))
