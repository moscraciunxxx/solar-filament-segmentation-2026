#!/usr/bin/env python3
"""Out-of-fold Mask R-CNN instances for one held-out fold.

The fold-4 checkpoint was trained on folds 1-3. Checkpoints ho1, ho2, and ho3
are each trained with that fold held out. Fold 0 is predicted only by the
fold-4 checkpoint, which never trained on fold 0. Does not submit.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filament.coco_io import grouped_folds, load_coco
from filament.paths import FAST_ARTIFACTS, IMAGE_SIZE, SEED, TRAIN_IMG_DIR
from filament.rle import encode_rle

spec = importlib.util.spec_from_file_location(
    "train_maskrcnn_convnext", ROOT / "scripts" / "train_maskrcnn_convnext.py"
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-fold", type=int, required=True)
    parser.add_argument("--also-fold0", action="store_true")
    args = parser.parse_args()
    if args.holdout_fold not in (1, 2, 3, 4):
        raise SystemExit("holdout fold must be 1, 2, 3, or 4")
    weights = trainer.WEIGHTS_DIR / trainer.family_for_holdout(args.holdout_fold) / "best.pt"
    if not weights.is_file():
        raise SystemExit(f"missing {weights}")
    out_dir = FAST_ARTIFACTS / "maskrcnn_oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = trainer.build_model()
    state = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    files = sorted(fn for fn, fold in fold_of.items() if fold == args.holdout_fold)
    if args.also_fold0:
        if args.holdout_fold != 4:
            raise SystemExit("fold 0 is predicted only by the fold-4 checkpoint")
        files = sorted(fn for fn, fold in fold_of.items() if fold in (0, 4))
    tag = f"ho{args.holdout_fold}"
    out_path = out_dir / f"{tag}.json"
    if out_path.exists():
        raise SystemExit(f"already written: {out_path}")
    rows = []
    for index, file_name in enumerate(files, start=1):
        image = cv2.imread(str(TRAIN_IMG_DIR / file_name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(file_name)
        if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
            image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            pred = model([tensor.to(device)])[0]
        instances = []
        scores = pred["scores"].detach().cpu()
        masks = pred["masks"].detach().cpu()
        for score, mask in zip(scores, masks):
            prob = mask[0].numpy()
            binary = (prob > 0.5).astype(np.uint8)
            if int(binary.sum()) < 96:
                continue
            inside = prob[binary > 0]
            instances.append(
                {
                    "score": float(score),
                    "mean_prob": float(inside.mean()) if inside.size else 0.0,
                    "rle": encode_rle(binary),
                }
            )
        rows.append({"file_name": file_name, "fold": int(fold_of[file_name]), "instances": instances})
        if index % 20 == 0:
            print(f"infer {tag} {index}/{len(files)}", flush=True)
    payload = {
        "holdout_fold": args.holdout_fold,
        "weights": str(weights),
        "checkpoint_epoch": state.get("epoch"),
        "n_images": len(rows),
        "n_instances": int(sum(len(row["instances"]) for row in rows)),
        "rows": rows,
        "test_labels_used": False,
        "kaggle_submit_called": False,
    }
    out_path.write_text(json.dumps(payload) + "\n")
    print("DONE maskrcnn_oof", tag, payload["n_images"], payload["n_instances"], flush=True)


if __name__ == "__main__":
    main()
