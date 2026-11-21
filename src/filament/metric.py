"""Official-style Panoptic Quality from the host self-evaluation notebook.

Matching rule (authoritative):
  TP  = every GT/pred pair with IoU > 0.5 (not a unique Hungarian match)
  FP  = predictions whose column never exceeds 0.5
  FN  = GT filaments whose row never exceeds 0.5
  PQ  = sum(IoU of TP pairs) / (|TP| + 0.5|FP| + 0.5|FN|)

GT filament_id prefix before '_' is '{annotator}-{image_stem}'.
Pred filament_id prefix before '_' is '{image_stem}'.
Each annotator-image is scored independently against the same predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .rle import rles_to_layers


def overlap_matrices(
    gt_layers: torch.Tensor,
    pred_layers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]
    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)
    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection
    zero = torch.tensor(0.0, device=gt_layers.device)
    iou = torch.where(union == 0, zero, intersection / union)
    dice = torch.where(union == 0, zero, 2 * intersection / (gt_areas + pred_areas))
    return iou, dice


@dataclass
class PQBreakdown:
    pq: float
    tp: int
    fp: int
    fn: int
    mean_tp_iou: float
    n_annotator_images: int


def score_pq(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    iou_threshold: float = 0.5,
) -> PQBreakdown:
    gt_keys = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_keys = pred_df["filament_id"].str.split("_", n=1).str[0]
    tp_ious: list[float] = []
    fp_count = 0
    fn_count = 0

    for annotator_image in gt_keys.unique():
        image_id = annotator_image.split("-", maxsplit=1)[1]
        gt_rles = gt_df.loc[gt_keys == annotator_image, "segmentation_rle"].tolist()
        pred_rles = pred_df.loc[pred_keys == image_id, "segmentation_rle"].tolist()
        n_gt, n_pred = len(gt_rles), len(pred_rles)
        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue
        gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
        pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
        iou_matrix, _ = overlap_matrices(gt_layers, pred_layers)
        hit = iou_matrix > iou_threshold
        tp_ious.extend(iou_matrix[hit].tolist())
        fp_count += int((hit.sum(dim=0) == 0).sum().item())
        fn_count += int((hit.sum(dim=1) == 0).sum().item())

    tp_count = len(tp_ious)
    denom = tp_count + 0.5 * fp_count + 0.5 * fn_count
    pq = (sum(tp_ious) / denom) if denom > 0 else 0.0
    return PQBreakdown(
        pq=float(pq),
        tp=tp_count,
        fp=fp_count,
        fn=fn_count,
        mean_tp_iou=float(np.mean(tp_ious)) if tp_ious else 0.0,
        n_annotator_images=int(gt_keys.nunique()),
    )


def gt_annotations_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert rasterized GT records to the official two-column CSV schema.

    Each record: {annotator_image, index, rle}.
    filament_id = f"{annotator_image}_{index}".
    """
    rows = [
        {"filament_id": f"{rec['annotator_image']}_{rec['index']}", "segmentation_rle": rec["rle"]}
        for rec in records
    ]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def pred_to_df(image_stem: str, rles: list[str]) -> pd.DataFrame:
    rows = [
        {"filament_id": f"{image_stem}_{i + 1}", "segmentation_rle": rle}
        for i, rle in enumerate(rles)
    ]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
