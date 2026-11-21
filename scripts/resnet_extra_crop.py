"""Fold!=0 extra-crop appearance rejector. Never submits. Never reads test labels."""

from __future__ import annotations

import numpy as np

from filament.rle import decode_rle
from filament.paths import SEED
from resnet_extra_filter import (
    REJECTOR_THR,
    context_feature_names,
    context_features,
    extra_feature_names,
    features_for_rles_context,
    jpeg_and_residual,
    lock_rejector_thr,
)

CROP_SIDE = 32
CROP_PAD = 24
CROP_DIM = CROP_SIDE * CROP_SIDE


def crop_pixel_names() -> tuple[str, ...]:
    return tuple(f"crop_{i}" for i in range(CROP_DIM))


def extra_crop_vector(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    import cv2

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.zeros(CROP_DIM, dtype=np.float32)
    h, w = image.shape[:2]
    y0 = max(int(ys.min()) - CROP_PAD, 0)
    y1 = min(int(ys.max()) + CROP_PAD + 1, h)
    x0 = max(int(xs.min()) - CROP_PAD, 0)
    x1 = min(int(xs.max()) + CROP_PAD + 1, w)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return np.zeros(CROP_DIM, dtype=np.float32)
    patch = cv2.resize(patch, (CROP_SIDE, CROP_SIDE), interpolation=cv2.INTER_AREA)
    return (patch.astype(np.float32) / 255.0).reshape(-1)


def features_for_rles_crop(
    rles: list[str],
    image: np.ndarray,
    residual: np.ndarray,
    slot_masks: list[np.ndarray],
) -> np.ndarray:
    base = features_for_rles_context(rles, image, residual, slot_masks)
    if not rles:
        return np.zeros((0, base.shape[1] + CROP_DIM), dtype=np.float32)
    crops = np.stack([extra_crop_vector(decode_rle(rle), image) for rle in rles], axis=0)
    return np.concatenate([base, crops], axis=1)


def rows_to_xy_crop(rows: list[dict], slot1_by_file: dict) -> tuple[np.ndarray, np.ndarray]:
    names = extra_feature_names() + context_feature_names() + crop_pixel_names()
    if not rows:
        return np.zeros((0, len(names)), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    mask_cache: dict[str, list[np.ndarray]] = {}
    jpeg_cache: dict[str, np.ndarray] = {}
    xs = []
    ys = []
    for row in rows:
        file_name = str(row["file_name"])
        if file_name not in jpeg_cache:
            jpeg_cache[file_name], _ = jpeg_and_residual(file_name)
        if file_name not in mask_cache:
            frame = slot1_by_file.get(file_name)
            rles = []
            if frame is not None and len(frame):
                rles = frame["segmentation_rle"].tolist()
            mask_cache[file_name] = [decode_rle(rle) for rle in rles]
        mask = decode_rle(row["rle"])
        morph = np.asarray(row["features"], dtype=np.float32).reshape(-1)
        ctx = context_features(mask, mask_cache[file_name])
        crop = extra_crop_vector(mask, jpeg_cache[file_name])
        xs.append(np.concatenate([morph, ctx, crop], axis=0))
        ys.append(int(row["label"]))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int32)


class TinyCropNet:
    """Built at call time so this module imports without torch at collect time."""

    @staticmethod
    def build(morph_dim: int):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 8, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(8, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.fc = nn.Sequential(
                    nn.Linear(16 * 8 * 8 + int(morph_dim), 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 1),
                )

            def forward(self, morph, crop):
                import torch

                h = self.conv(crop).reshape(crop.size(0), -1)
                return self.fc(torch.cat([h, morph], dim=1)).squeeze(1)

        return _Net()


class CropRejector:
    """sklearn-like wrapper so keep_extra_rles can call predict_proba."""

    def __init__(self, state: dict, morph_dim: int):
        self.state = {k: np.asarray(v) for k, v in state.items()}
        self.morph_dim = int(morph_dim)
        self._net = None

    def _net_mod(self):
        if self._net is None:
            import torch

            net = TinyCropNet.build(self.morph_dim)
            net.load_state_dict({k: torch.from_numpy(v) for k, v in self.state.items()})
            net.eval()
            self._net = net
        return self._net

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        import torch

        x = np.asarray(feats, dtype=np.float32)
        if x.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        morph = torch.from_numpy(x[:, : self.morph_dim])
        crop = torch.from_numpy(x[:, self.morph_dim :]).reshape(-1, 1, CROP_SIDE, CROP_SIDE)
        with torch.no_grad():
            logit = self._net_mod()(morph, crop).cpu().numpy().astype(np.float32)
        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        return np.stack([1.0 - p, p], axis=1)

    def predict(self, feats: np.ndarray) -> np.ndarray:
        return (self.predict_proba(feats)[:, 1] >= 0.5).astype(np.int32)


def fit_extra_rejector_crop(features: np.ndarray, labels: np.ndarray):
    """Tiny crop CNN + morph/ctx vector. Same fold!=0 precision-first lock."""
    import torch
    import torch.nn as nn

    y = np.asarray(labels, dtype=np.int32)
    x = np.asarray(features, dtype=np.float32)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    morph_dim = int(x.shape[1] - CROP_DIM) if x.ndim == 2 and x.shape[1] > CROP_DIM else 18
    meta = {
        "status": "fallback_morph",
        "type": "tiny_crop_cnn_plus_morph_ctx",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_rows": int(len(y)),
        "feature_names": list(extra_feature_names()) + list(context_feature_names()) + list(crop_pixel_names()),
        "threshold": REJECTOR_THR,
        "fit_folds": "fold!=0 only",
        "never_fit_fold0": True,
        "classical_new_seeds": False,
        "seed": SEED,
        "not_logistic_cut_retune": True,
        "not_hgb_morph_only": True,
        "crop_side": CROP_SIDE,
        "morph_dim": morph_dim,
    }
    if n_pos < 8 or n_neg < 8 or len(y) < 16 or x.shape[1] < CROP_DIM + 8:
        return None, meta

    torch.manual_seed(int(SEED))
    net = TinyCropNet.build(morph_dim)
    net.train()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y.astype(np.float32))
    morph = xt[:, :morph_dim]
    crop = xt[:, morph_dim:].reshape(-1, 1, CROP_SIDE, CROP_SIDE)
    order = np.arange(len(y))
    rng = np.random.default_rng(int(SEED))
    batch = 64
    for _epoch in range(25):
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            opt.zero_grad()
            logit = net(morph[idx], crop[idx])
            loss = crit(logit, yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        logit = net(morph, crop).cpu().numpy().astype(np.float32)
    proba = 1.0 / (1.0 + np.exp(-logit))
    thr, lock = lock_rejector_thr(y, proba)
    state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    clf = CropRejector(state, morph_dim)
    pred = (proba >= 0.5).astype(np.int32)
    meta.update(
        {
            "status": "fitted",
            "train_acc": float((pred == y).mean()),
            "threshold": thr,
            "foldne0_lock": lock,
            "fallback_morph_used": False,
        }
    )
    return clf, meta
