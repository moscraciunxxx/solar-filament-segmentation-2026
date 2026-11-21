from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from .paths import IMAGE_SIZE


def encode_rle(mask: np.ndarray) -> str:
    """Encode a binary (H, W) mask as a COCO compressed RLE counts string."""
    mask_u8 = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask_u8)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return counts


def decode_rle(counts: str, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> np.ndarray:
    rle = {"size": [height, width], "counts": counts}
    return mask_utils.decode(rle).astype(np.uint8)


def rles_to_layers(rles: list[str], height: int = IMAGE_SIZE, width: int = IMAGE_SIZE) -> np.ndarray:
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    masks = mask_utils.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)
