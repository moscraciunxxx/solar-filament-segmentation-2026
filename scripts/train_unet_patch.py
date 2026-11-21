#!/usr/bin/env python3
"""Native-2048 patch U-Net. Gated. Start only on KILL_NO_NEW_TP.

Dataset: src/filament/patch_dataset.py NativePatchDataset (fold != 0 train,
fold == 0 val). Weights: WEIGHTS_DIR/unet_patch_dhebe — never unet_semantic.
TinyUNet via build_fulldisk_unet(in_ch=3), BCE + tversky (same as v1).

refuse_patch() is the first train-path check: missing/wrong gate or a live
MPS owner (train_unet.py / train_yolo.py / eval_unet) exits 1. --dry-refuse
prints that reason and exits 0 without importing torch. Never submits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gate_unet_family import GATE, refuse_patch, resolve_start_gate

FAMILY = "unet_patch_native_1024_dhebe"
WEIGHTS_NAME = "unet_patch_dhebe"
CURVE_NAME = "unet_patch_dhebe_train_curve.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-json", default=str(GATE))
    parser.add_argument(
        "--dry-refuse",
        action="store_true",
        help="Print refuse_patch() and exit 0. Does not import torch.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--base", type=int, default=32)
    return parser.parse_args()


def run_train_loop(args: argparse.Namespace) -> None:
    """BCE+Tversky TinyUNet on native 1024 crops. First line re-checks the gate."""
    gate = resolve_start_gate(Path(args.gate_json))
    reason = refuse_patch(gate)
    if reason:
        raise SystemExit(reason)

    import json

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from filament.coco_io import grouped_folds, load_coco
    from filament.patch_dataset import NativePatchDataset
    from filament.paths import ARTIFACTS, SEED, TRAIN_IMG_DIR, WEIGHTS_DIR
    from filament.refiner import tversky_loss
    from filament.unet import build_fulldisk_unet

    def pick_device(requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    device = pick_device(args.device)
    views, by_file = load_coco()
    fold_of = grouped_folds(by_file, views, seed=SEED)
    train_files = sorted(fn for fn, fold in fold_of.items() if fold != 0)
    val_files = sorted(fn for fn, fold in fold_of.items() if fold == 0)
    if any(fold_of[fn] == 0 for fn in train_files):
        raise SystemExit("Refusing: NativePatchDataset train files include fold-0.")
    if set(train_files) & set(val_files):
        raise SystemExit("Refusing: train/val file overlap.")
    if not train_files or not val_files:
        raise SystemExit("Refusing: empty fold!=0 train or fold==0 val split.")

    train_ds = NativePatchDataset(
        views, by_file, train_files, TRAIN_IMG_DIR, augment=True
    )
    val_ds = NativePatchDataset(
        views, by_file, val_files, TRAIN_IMG_DIR, augment=False
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_fulldisk_unet(in_ch=3, base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()

    out_dir = WEIGHTS_DIR / WEIGHTS_NAME
    if out_dir.name != WEIGHTS_NAME or "unet_semantic" in out_dir.parts:
        raise SystemExit("Refusing: weights dir must be unet_patch_dhebe, never unet_semantic.")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    history: list[dict] = []
    best_val = float("inf")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    curve_path = ARTIFACTS / CURVE_NAME

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y, *_ in tqdm(train_loader, desc=f"unet_patch {epoch}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = bce(logits, y) + tversky_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)
            n += x.size(0)
        train_loss = running / max(1, n)

        model.eval()
        val_running = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y, *_ in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = bce(logits, y) + tversky_loss(logits, y)
                val_running += float(loss.item()) * x.size(0)
                val_n += x.size(0)
        val_loss = val_running / max(1, val_n)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(model.state_dict(), out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
        curve = {
            "family": FAMILY,
            "best_val_loss": best_val,
            "best_epoch": int(min(history, key=lambda r: r["val_loss"])["epoch"]),
            "history": history,
            "val_loss_is_not_official_pq": True,
            "overwrote_unet_semantic": False,
            "weights_dir": str(out_dir),
            "n_train_patches": len(train_ds),
            "n_val_patches": len(val_ds),
            "n_train_files": len(train_files),
            "n_val_files": len(val_files),
        }
        (out_dir / "train_curve.json").write_text(json.dumps(curve, indent=2))
        curve_path.write_text(json.dumps(curve, indent=2))

    print(
        json.dumps(
            {
                "best": str(best_path),
                "device": device,
                "weights_dir": str(out_dir),
                "do_not_submit": True,
                "overwrote_unet_semantic": False,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.dry_refuse:
        print(refuse_patch(resolve_start_gate(Path(args.gate_json))))
        raise SystemExit(0)

    gate = resolve_start_gate(Path(args.gate_json))
    reason = refuse_patch(gate)
    if reason:
        raise SystemExit(reason)
    run_train_loop(args)


if __name__ == "__main__":
    main()
