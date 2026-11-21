import os, sys, subprocess, shutil, time, json
from pathlib import Path
FOLDS = [1, 2]
LANES = [[1], [2]]
os.environ["YOLO_CONFIG_DIR"] = "/tmp/yolo"; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "ultralytics==8.4.146"])
import torch
INP = Path("/kaggle/input"); W = Path("/kaggle/working")
kit = next(INP.rglob("worker.py")).parent; lsw = next(INP.rglob("worker_ls.py")); d4 = next(INP.rglob("yolo_seg_erode_gt/data.yaml")).parent
(W / "imgs").mkdir(exist_ok=True); (W / "labs").mkdir(exist_ok=True)
t = time.time()
for s in ("train", "val"):
    for p in (d4 / "images" / s).iterdir(): shutil.copy(p, W / "imgs" / p.name)
    for p in (d4 / "labels" / s).iterdir(): shutil.copy(p, W / "labs" / p.name)
n_img = len(list((W / "imgs").iterdir())); n_lab = len(list((W / "labs").iterdir()))
ng = torch.cuda.device_count()
print("COPY", n_img, n_lab, round(time.time() - t, 1), "s; gpus", ng, [torch.cuda.get_device_name(i) for i in range(ng)], flush=True)
assert n_img == 707, n_img
if ng >= 2:
    lane = lambda i, ks: " ; ".join(f"CUDA_VISIBLE_DEVICES={i} {sys.executable} {lsw} {k} {kit}" for k in ks)
    procs = [subprocess.Popen(["bash", "-c", lane(i, ks)], env=dict(os.environ)) for i, ks in enumerate(LANES)]
    codes = [p.wait() for p in procs]
else:
    codes = [subprocess.call([sys.executable, str(lsw), str(k), str(kit)]) for k in FOLDS]
print("WORKER_EXIT", codes, flush=True)
for k in FOLDS:
    shutil.rmtree(W / f"data{k}", ignore_errors=True)
shutil.rmtree(W / "imgs", ignore_errors=True); shutil.rmtree(W / "labs", ignore_errors=True)
shutil.rmtree(W / "runs", ignore_errors=True)
for k in FOLDS:
    subprocess.call(["tar", "czf", str(W / f"fold{k}_tta.tgz"), "-C", str(W / f"fold{k}"), "tta", "tta_f0"]); shutil.rmtree(W / f"fold{k}" / "tta", ignore_errors=True); shutil.rmtree(W / f"fold{k}" / "tta_f0", ignore_errors=True)
print("ALL_DONE", flush=True)
