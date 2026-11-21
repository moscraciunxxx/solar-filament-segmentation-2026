"""ResNet34 full-disk U-Net. NOT WIRED into train/eval.

Intended API (same logit contract as TinyUNet / build_fulldisk_unet):

    build_resnet34_unet(in_ch=3) -> nn.Module
    forward(x[B,in_ch,H,W]) -> logits[B,1,H,W]

Do not import this from scripts/train_unet.py, scripts/eval_unet.py, or
scripts/eval_unet_merge.py while v1 TinyUNet is live on MPS. Start only after
eval_unet_merge.py writes RESEARCH_HOLD or KILL_WORSE_OR_DESERT (also
RESEARCH_HOLD_SPINE_V2). Not the answer to KILL_NO_NEW_TP.

Encoder channels are torchvision ResNet-34 BasicBlock (expansion=1), not guessed
weight tensors. Default is random init — no ImageNet download on import.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models import resnet34
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    resnet34 = None
    _IMPORT_ERROR: BaseException | None = exc
else:
    _IMPORT_ERROR = None

_Base = nn.Module if nn is not None else object


def _require_stack() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError("not wired; start only after RESEARCH_HOLD") from _IMPORT_ERROR


def _conv_block(in_ch: int, out_ch: int):
    """Same Conv-BN-ReLU pair as TinyUNet decoder stages."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _match_hw(up, skip):
    if up.shape[-2:] != skip.shape[-2:]:
        up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    return up


class ResNet34UNet(_Base):
    """ResNet-34 encoder + U-Net decoder. 1-channel logits at input HxW.

    Skip map (torchvision resnet34, BasicBlock expansion=1):

        conv1+bn+relu  64ch  H/2
        layer1         64ch  H/4
        layer2        128ch  H/8
        layer3        256ch  H/16
        layer4        512ch  H/32
    """

    def __init__(self, in_ch: int = 3, pretrained: bool = False):
        _require_stack()
        super().__init__()
        if resnet34 is None:
            raise RuntimeError("not wired; start only after RESEARCH_HOLD")
        if pretrained:
            raise RuntimeError(
                "ImageNet weights are opt-in later; default scaffold uses random init. "
                "not wired; start only after RESEARCH_HOLD"
            )

        backbone = resnet34(weights=None)
        if in_ch != 3:
            backbone.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Decoder: /32 -> /16 -> /8 -> /4 -> /2 -> 1:1. Last stage base=32 matches TinyUNet.
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d4 = _conv_block(256 + 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3 = _conv_block(128 + 128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2 = _conv_block(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.d1 = _conv_block(64 + 64, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d0 = _conv_block(32, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        s0 = self.relu(self.bn1(self.conv1(x)))  # /2, 64
        s1 = self.layer1(self.maxpool(s0))  # /4, 64
        s2 = self.layer2(s1)  # /8, 128
        s3 = self.layer3(s2)  # /16, 256
        z = self.layer4(s3)  # /32, 512

        y = self.d4(torch.cat([_match_hw(self.up4(z), s3), s3], 1))
        y = self.d3(torch.cat([_match_hw(self.up3(y), s2), s2], 1))
        y = self.d2(torch.cat([_match_hw(self.up2(y), s1), s1], 1))
        y = self.d1(torch.cat([_match_hw(self.up1(y), s0), s0], 1))
        y = self.up0(y)
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return self.out(self.d0(y))


def build_resnet34_unet(in_ch: int = 3, pretrained: bool = False) -> ResNet34UNet:
    """Not wired. Swap in after the v1 merge gate; do not call from train_unet.py now."""
    _require_stack()
    return ResNet34UNet(in_ch=in_ch, pretrained=pretrained)
