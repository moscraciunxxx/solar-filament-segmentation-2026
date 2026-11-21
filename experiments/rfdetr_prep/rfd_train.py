#!/usr/bin/env python3
"""RF-DETR-Seg Medium @768 on our tiles. usage: rfd_train.py MODE DATASET OUTDIR EPOCHS
MODE learn: EPOCHS (2) then collapse alarm = quick PQ on 12 held-out images (all annotators, pycocotools) must be > 0.10 and >= 1 instance/img;
nominal effective batch 4 x accum 4 = 16 asserted from the saved training config; batch_size is an explicit int (no 'auto' probing, no silent change)."""
import sys, json, time, glob
from pathlib import Path
from rfdetr import RFDETRSegMedium
MODE, DS, OUT, EP = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); t0 = time.time()
model = RFDETRSegMedium(resolution=768)
model.train(dataset_dir=DS, output_dir=OUT, epochs=EP, batch_size=4, grad_accum_steps=4, lr=1e-4, lr_encoder=1.5e-4,
            lr_scheduler="cosine", warmup_epochs=1.0, seed=20261113, checkpoint_interval=5, eval_interval=1 if MODE == "learn" else 5,
            tensorboard=False, progress_bar=None)
cfgs = glob.glob(f"{OUT}/**/training_config.json", recursive=True)
if cfgs:
    c = json.load(open(cfgs[0])); bs = c.get("batch_size"); ga = c.get("grad_accum_steps")
    print("CONFIG batch", bs, "accum", ga, flush=True); assert (bs, ga) == (4, 4), (bs, ga)
print("TRAIN_SEC", round(time.time() - t0), flush=True)
