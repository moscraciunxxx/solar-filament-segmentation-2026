#!/usr/bin/env python3
"""Champion recipe refit on ALL 707 images (no validation fold). usage: champ_refit_worker.py train SEED | tta SEED WEIGHTS CACHE_OUT SHARD NSHARD
Recipe = champion args.yaml (yolo11s-seg, eroded-medoid labels, 40 epochs, imgsz 2048, batch 4, AdamW lr0 0.001, lrf 0.01, wd 0.0005,
warmup 3, hsv 0.01/0.4/0.3, degrees 15, translate 0.05, scale 0.3, flipud/fliplr 0.5, mosaic 0.8, close_mosaic 8, copy_paste 0.15 flip,
mixup 0.05, randaugment, erasing 0.4, mask_ratio 4, overlap_mask, single_cls, deterministic, amp) + v3 seg-loss checkpoint patch
(identical math), OOM auto-halving disabled (_oom_retries=3 -> raises), batch == 4 asserted every batch.
Checkpoint = last.pt after epoch 40: the champion used best.pt selected on fold 0, which was epoch 40 = last for seeds 20260910/11 and
epoch 39 for 20260912; fold 0 is now training data, so no selection is possible -> last.pt (40 epochs) for all three.
Early alarms: epoch-1 time > MAX_EP1_SEC (default 330 s => > ~3.6 h run) aborts; any non-finite epoch loss aborts; epoch-3 mean total loss
must be < epoch-1 mean total loss (collapse alarm). Ultralytics' forced final-epoch validation runs on 4 TRAIN images (no held-out data exists)."""
import os, sys, json, time, glob, hashlib, math
from pathlib import Path
MODE = sys.argv[1]; SEED = int(sys.argv[2]); KIT = Path(__file__).resolve().parent; sys.path.insert(0, str(KIT))
W = Path("/kaggle/working"); INP = Path("/kaggle/input"); OUT = W / f"s{SEED}"; OUT.mkdir(parents=True, exist_ok=True)
def log(*a):
    m = " ".join(str(x) for x in a); print(f"[s{SEED} {MODE}] {m}", flush=True)
    with open(OUT / f"log_{MODE}.txt", "a") as h: h.write(m + "\n")
import numpy as np, cv2, torch
if MODE == "train":
    d4 = next(INP.rglob("yolo_seg_erode_gt/data.yaml")).parent
    labs = sorted(glob.glob(str(d4 / "labels" / "*" / "*.txt")))
    md5 = hashlib.md5(b"".join(Path(p).read_bytes() for p in labs)).hexdigest()
    log("labels", len(labs), "md5", md5); assert len(labs) == 707 and md5 == "b8d377fd1aff503ef3a3394d1a64d1f9", "label set differs from the champion's yolo_seg_erode_gt"
    D = W / f"data_s{SEED}"
    for sp in ("train", "val"): (D / "images" / sp).mkdir(parents=True, exist_ok=True); (D / "labels" / sp).mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for sp in ("train", "val") for p in (d4 / "images" / sp).iterdir()); assert len(imgs) == 707, len(imgs)
    for p in imgs:
        lab = d4 / "labels" / p.parent.name / (p.stem + ".txt")
        os.symlink(p, D / "images" / "train" / p.name); (D / "labels" / "train" / (p.stem + ".txt")).write_text(lab.read_text() if lab.exists() else "")
    for p in imgs[:4]:   # ultralytics always validates the final epoch; point it at 4 train images (no held-out data in a full refit)
        os.symlink(p, D / "images" / "val" / p.name); (D / "labels" / "val" / (p.stem + ".txt")).write_text((D / "labels" / "train" / (p.stem + ".txt")).read_text())
    (D / "data.yaml").write_text(f"path: {D}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\nnc: 1\n")
    from ultralytics import YOLO
    import segloss_patch as _SP
    from ultralytics.utils import loss as _L
    _L.v8SegmentationLoss.calculate_segmentation_loss = _SP.calculate_segmentation_loss; log("seg-loss checkpoint patch v3 active")
    model = YOLO(str(KIT / "yolo11s-seg.pt"))
    st = {"t": time.time(), "loss": {}}; MAX_EP1 = float(os.environ.get("MAX_EP1_SEC", "330"))
    def no_halving(tr): tr._oom_retries = 3
    def check_batch(tr):
        if tr.batch_size != 4: raise SystemExit(f"BATCH_CHANGED {tr.batch_size}")
    def ep_end(tr):
        now = time.time(); e = tr.epoch + 1; sec = now - st["t"]; st["t"] = now
        tl = [float(x) for x in tr.tloss.detach().cpu().numpy().ravel()] if tr.tloss is not None else []
        st["loss"][e] = sum(tl); log("epoch", e, "sec", round(sec, 1), "loss", [round(x, 4) for x in tl], "gpu_peak_gib", round(torch.cuda.max_memory_reserved() / 2**30, 2))
        if not tl or not all(math.isfinite(x) for x in tl): raise SystemExit(f"COLLAPSE_ALARM non-finite loss at epoch {e}")
        if e == 1 and sec > MAX_EP1: raise SystemExit(f"SPEED_ALARM epoch 1 took {sec:.0f}s > {MAX_EP1:.0f}s")
        if e == 3 and not (st["loss"][3] < st["loss"][1]): raise SystemExit(f"COLLAPSE_ALARM epoch-3 loss {st['loss'][3]:.4f} >= epoch-1 {st['loss'][1]:.4f}")
        if e == 3: log("EARLY_CHECK_OK", json.dumps({k: round(v, 4) for k, v in st["loss"].items()}))
    model.add_callback("on_train_epoch_start", no_halving); model.add_callback("on_train_batch_end", check_batch); model.add_callback("on_train_epoch_end", ep_end)
    t0 = time.time(); name = f"yolo11s_erode_gt_2048_full707_s{SEED}"
    kw = dict(data=str(D / "data.yaml"), epochs=40, imgsz=2048, batch=4, patience=1000, device=0, seed=SEED, deterministic=True, single_cls=True,
              workers=4, cache=False, exist_ok=True, project=str(W / "runs"), name=name, optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.937,
              weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3, degrees=15.0,
              translate=0.05, scale=0.3, flipud=0.5, fliplr=0.5, mosaic=0.8, close_mosaic=8, copy_paste=0.15, copy_paste_mode="flip", mixup=0.05,
              auto_augment="randaugment", erasing=0.4, multi_scale=0.0, overlap_mask=True, mask_ratio=4, cos_lr=False, rect=False, amp=True,
              val=False, plots=False)
    res = {"seed": SEED, "n_train": 707}
    try:
        model.train(**kw)
    except BaseException as e:
        res["train_error"] = f"{type(e).__name__}: {str(e)[:500]}"; log("TRAIN_FAIL", res["train_error"]); (OUT / "result.json").write_text(json.dumps(res, indent=2)); raise
    last = W / "runs" / name / "weights" / "last.pt"
    import shutil; shutil.copy(last, OUT / f"last_s{SEED}.pt"); shutil.rmtree(W / "runs" / name, ignore_errors=True); shutil.rmtree(D, ignore_errors=True)
    res.update(train_hours=(time.time() - t0) / 3600, epoch_loss=st["loss"], status="TRAIN_DONE"); (OUT / "result.json").write_text(json.dumps(res, indent=2)); log("TRAIN_OK", json.dumps(res))
else:  # tta: 8 views x 3 scales, IDENTICAL to scripts/eval_parent_tta._predict_views (conf 0.35, iou 0.60, retina masks, >= 96 px)
    wts, cache, shard, nshard = Path(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]); cache.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    from filament.infer import _collect_yolo_instances
    from filament.rle import encode_rle
    ted = next(p for p in INP.rglob("test_images") if p.is_dir()); images = sorted(ted.glob("*.jpeg")); assert len(images) == 180
    r90, r270, r180 = cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180
    VIEWS = [("id", lambda im: im, lambda m: m), ("r90", lambda im: cv2.rotate(im, r90), lambda m: cv2.rotate(m, r270)),
             ("r180", lambda im: cv2.rotate(im, r180), lambda m: cv2.rotate(m, r180)), ("r270", lambda im: cv2.rotate(im, r270), lambda m: cv2.rotate(m, r90)),
             ("h", lambda im: cv2.flip(im, 1), lambda m: cv2.flip(m, 1)), ("v", lambda im: cv2.flip(im, 0), lambda m: cv2.flip(m, 0)),
             ("h_r90", lambda im: cv2.rotate(cv2.flip(im, 1), r90), lambda m: cv2.flip(cv2.rotate(m, r270), 1)),
             ("v_r90", lambda im: cv2.rotate(cv2.flip(im, 0), r90), lambda m: cv2.flip(cv2.rotate(m, r270), 0))]
    m = YOLO(str(wts)); t1 = time.time(); mine = images[shard::nshard]
    for i, p in enumerate(mine):
        image = cv2.imread(str(p), cv2.IMREAD_COLOR); assert image is not None and image.shape[:2] == (2048, 2048), p
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
        (cache / f"{p.stem}.json").write_text(json.dumps(out))
        if i % 20 == 0: log("tta", i, len(mine), round(time.time() - t1, 1))
    log("TTA_DONE", len(mine), round((time.time() - t1) / 3600, 3), "h")
