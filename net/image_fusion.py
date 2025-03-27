import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractor import BasicEncoder, ResidualBlock
from .fusion import BAttentionFeatureFusion

from .utils import InputPadder
from .utils import HSVRGB, RGBHSV, GuidedFilter


class HSVNet(torch.nn.Module):

    def __init__(self):
        super(HSVNet, self).__init__()
        self.encoder = BasicEncoder(downsample=2, output_dim=256)
        self.fusion = BAttentionFeatureFusion(in_channels=256, reduction=4)
        self.channel_reduction = nn.Sequential(
            ResidualBlock(256, 128),
            ResidualBlock(128, 64),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.ReLU(),
        )

        self.hsv2rgb = HSVRGB()
        self.rgb2hsv = RGBHSV()
        self.gf = GuidedFilter()
        self.padder = None

    def forward(
        self,
        v: list[torch.Tensor],
        n: list[torch.Tensor],
    ):

        islist = isinstance(v, list)
        if islist:
            v = torch.cat(v, dim=0)
            n = torch.cat(n, dim=0)
        B, _, H, W = v.size()

        if self.padder is None:
            self.padder = InputPadder(v.size(), divis_by=32)

        v, n = self.padder.pad(v, n)

        hsv = self.rgb2hsv(v)

        hsv_input = (v / 255 * 2 - 1).contiguous()
        n_input = (n / 255 * 2 - 1).contiguous()

        hsv_fmap = self.encoder(hsv_input)
        nir_fmap = self.encoder(n_input.repeat(1, 3, 1, 1))
        fusion_fmap = self.fusion(hsv_fmap, nir_fmap)

        w = F.sigmoid(self.channel_reduction(fusion_fmap))
        w = F.interpolate(w, scale_factor=4, mode="bilinear", align_corners=False)

        rgb = self.hsv2rgb(
            torch.concat(
                [hsv[:, :1], hsv[:, 1:2], w[:, :1] * hsv[:, 2:3] + w[:, 1:2] * n],
                dim=1,
            )
        )
        rgb = self.gf(n / 255, rgb / 255, radius=5) * 255
        rgb = torch.clip(rgb[..., :H, :W], 0, 255)

        return rgb

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(False)
                m.eval()
