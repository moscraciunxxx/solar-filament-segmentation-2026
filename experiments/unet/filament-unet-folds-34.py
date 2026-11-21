# Our own retrain of the dariushafshar U-Net recipe, held-out folds 3 and 4 of OUR split (fold 0 never trained on). No submission.
# GPU0 -> fold 3, GPU1 -> fold 4. Steps fitted to TRAIN_SEC at step 200 (cap 10500). Outputs: weights + uint8 8-view TTA maps (held fold, fold 0, test).
import os, sys, subprocess, time
from pathlib import Path
ENC = "tu-tf_efficientnetv2_s"; TRAIN_SEC = 7000
INP = Path("/kaggle/input"); W = Path("/kaggle/working")
whl = next(INP.rglob("segmentation_models_pytorch-0.5.0-py3-none-any.whl"))
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(whl)])
import torch
print("gpus", torch.cuda.device_count(), flush=True)
wk = next(INP.rglob("worker_unet.py"))
procs = [subprocess.Popen(["bash", "-c", f"CUDA_VISIBLE_DEVICES={i} {sys.executable} {wk} {k} {ENC} full {W}/out {TRAIN_SEC}"]) for i, k in enumerate([3, 4])]
codes = [p.wait() for p in procs]
print("WORKER_EXIT", codes, flush=True)
for d in sorted((W / "out").iterdir()):
    print("RESULT", d.name, (d / "result.json").read_text() if (d / "result.json").exists() else None, flush=True)
    subprocess.call(["tar", "czf", str(W / f"{d.name}_maps.tgz"), "-C", str(d)] + [x for x in ("held", "f0", "test") if (d / x).exists()])
    for s in ("held", "f0", "test"): subprocess.call(["rm", "-rf", str(d / s)])
print("ALL_DONE", flush=True)
