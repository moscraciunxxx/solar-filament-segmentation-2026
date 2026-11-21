# Learning test (~30 min cap): our own retrain of the dariushafshar U-Net recipe on OUR folds (fold 1 held out, fold 0 never used).
# GPU0: EfficientNetV2-S, GPU1: ResNet34, 700 real-schedule steps each, collapse alarm at step 500, batch-16 assert. No submission.
import os, sys, subprocess, json, time
from pathlib import Path
INP = Path("/kaggle/input"); W = Path("/kaggle/working")
whl = next(INP.rglob("segmentation_models_pytorch-0.5.0-py3-none-any.whl"))
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(whl)])
import torch, timm, segmentation_models_pytorch as smp
print("ENV torch", torch.__version__, "timm", timm.__version__, "smp", smp.__version__, "gpus", torch.cuda.device_count(), flush=True)
wk = next(INP.rglob("worker_unet.py"))
LANES = [("1", "tu-tf_efficientnetv2_s"), ("1", "resnet34")]
procs = [subprocess.Popen(["bash", "-c", f"CUDA_VISIBLE_DEVICES={i} {sys.executable} {wk} {k} {enc} learn {W}/out 0"]) for i, (k, enc) in enumerate(LANES)]
codes = [p.wait() for p in procs]
print("WORKER_EXIT", codes, flush=True)
for r in sorted((W / "out").rglob("result.json")): print("RESULT", r.parent.name, r.read_text(), flush=True)
print("ALL_DONE", flush=True)
