"""UNet-based Bayer denoiser model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ds_conv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-separable conv: 3×3 depthwise then 1×1 pointwise."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch),
        nn.Conv2d(in_ch, out_ch, 1),
    )


def bayer_split(x: torch.Tensor) -> torch.Tensor:
    """(B, H, W) → (B, 4, H/2, W/2)."""
    return torch.stack([
        x[:, 0::2, 0::2],
        x[:, 0::2, 1::2],
        x[:, 1::2, 0::2],
        x[:, 1::2, 1::2],
    ], dim=1)


def bayer_merge(x: torch.Tensor) -> torch.Tensor:
    """(B, 4, H/2, W/2) → (B, H, W)."""
    B, _, Hh, Wh = x.shape
    out = torch.zeros(B, Hh * 2, Wh * 2, device=x.device, dtype=x.dtype)
    out[:, 0::2, 0::2] = x[:, 0]
    out[:, 0::2, 1::2] = x[:, 1]
    out[:, 1::2, 0::2] = x[:, 2]
    out[:, 1::2, 1::2] = x[:, 3]
    return out


class DenoiseNet(nn.Module):
    """UNet denoiser operating on Bayer-split pixel values.

    Input to forward(): (B, 5, H/2, W/2) — 4 Bayer channels + 1 noise-std
    channel, normalised to [0, 1] by white_level.

    Output of forward(): (B, 4, H/2, W/2) — denoised Bayer channels
    (residual added to the 4 input pixel channels).
    """

    _SCALING_FACTORS = [1, 2, 1]
    _BASE_CH = 16

    def __init__(self, ch: int | None = None, scaling_factors: list[int] | None = None) -> None:
        super().__init__()
        nf = ch if ch is not None else self._BASE_CH
        sf = scaling_factors if scaling_factors is not None else list(self._SCALING_FACTORS)
        self.ch = nf
        self.scaling_factors = sf
        in_ch = [5] + [nf] * (len(sf) - 1)

        self.enc_skip = nn.ModuleList()
        self.enc_c1   = nn.ModuleList()
        for i, stride in enumerate(sf):
            self.enc_skip.append(nn.Conv2d(in_ch[i], nf, 3, stride=stride, padding=1))
            #self.enc_skip.append(_ds_conv(in_ch[i], nf, stride=stride))
            self.enc_c1.append(_ds_conv(nf, nf))

        self.dec_skip = nn.ModuleList()
        self.dec_c1   = nn.ModuleList()
        for _ in range(len(sf) - 1):
            #self.dec_skip.append(nn.Conv2d(nf * 2, nf, 3, padding=1))
            self.dec_skip.append(_ds_conv(nf * 2, nf))
            self.dec_c1.append(_ds_conv(nf, nf))

        self.out_c1 = _ds_conv(nf, nf, stride=sf[0])
        self.out_c2 = nn.Conv2d(nf, 4, 3, padding=1)
        #self.out_c2 = _ds_conv(nf, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        feat = x

        for i in range(len(self.scaling_factors)):
            x_skip = F.relu(self.enc_skip[i](feat))
            out    = F.relu(self.enc_c1[i](x_skip))
            feat   = x_skip + out
            skips  = [feat] + skips

        feat  = skips[0]
        skips = skips[1:]

        for i in range(len(self.dec_skip)):
            up    = F.interpolate(feat, size=skips[0].shape[-2:],
                                  mode='bilinear', align_corners=False)
            cat   = torch.cat([skips[0], up], dim=1)
            skips = skips[1:]
            x_skip = F.relu(self.dec_skip[i](cat))
            out    = F.relu(self.dec_c1[i](x_skip))
            feat   = x_skip + out

        out = F.relu(self.out_c1(feat))
        out = self.out_c2(out)
        return x[:, :4] + out  # residual
