"""Full-disk TinyUNet wrapper. v1 reuses the crop refiner body at 1024."""

from __future__ import annotations

from .refiner import TinyUNet


def build_fulldisk_unet(in_ch: int = 3, base: int = 32) -> TinyUNet:
    return TinyUNet(in_ch=in_ch, base=base)
