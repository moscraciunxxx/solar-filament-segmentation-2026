"""U-Net with a ConvNeXt-Tiny encoder. Logits, one channel."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _match(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    if up.shape[-2:] != skip.shape[-2:]:
        up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    return up


class ConvNeXtUNet(nn.Module):
    """Skips at H/4, H/8, H/16. Bottleneck at H/32."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = convnext_tiny(weights=weights)
        features = net.features
        self.stem = features[0]
        self.stage1 = features[1]
        self.down2 = features[2]
        self.stage2 = features[3]
        self.down3 = features[4]
        self.stage3 = features[5]
        self.down4 = features[6]
        self.stage4 = features[7]
        self.up3 = nn.ConvTranspose2d(768, 384, 2, stride=2)
        self.dec3 = _block(768, 384)
        self.up2 = nn.ConvTranspose2d(384, 192, 2, stride=2)
        self.dec2 = _block(384, 192)
        self.up1 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.dec1 = _block(192, 96)
        self.up0 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.dec0 = _block(48, 48)
        self.up_full = nn.ConvTranspose2d(48, 32, 2, stride=2)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        skip1 = self.stage1(x)
        skip2 = self.stage2(self.down2(skip1))
        skip3 = self.stage3(self.down3(skip2))
        bottleneck = self.stage4(self.down4(skip3))
        d3 = self.dec3(torch.cat((_match(self.up3(bottleneck), skip3), skip3), dim=1))
        d2 = self.dec2(torch.cat((_match(self.up2(d3), skip2), skip2), dim=1))
        d1 = self.dec1(torch.cat((_match(self.up1(d2), skip1), skip1), dim=1))
        d0 = self.dec0(self.up0(d1))
        full = self.up_full(d0)
        if full.shape[-2:] != images.shape[-2:]:
            full = F.interpolate(full, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(full)


def soft_bce_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE plus soft Dice. Target is a probability in [0, 1], not a hard mask."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return bce + dice.mean()
