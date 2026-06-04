"""NAFNet-based Bayer denoiser model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class NAFBlock(nn.Module):
    """NAFNet block: LayerNorm + SimpleGate + SCA, no nonlinear activations."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.conv1 = nn.Conv2d(c, 2 * c, 1)
        self.dw    = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.sca   = nn.Sequential(nn.AvgPool2d(7, stride=1, padding=3), nn.Conv2d(c, c, 1, bias=False))
        self.conv2 = nn.Conv2d(c, c, 1)
        self.norm2 = nn.LayerNorm(c)
        self.conv3 = nn.Conv2d(c, 2 * c, 1)
        self.conv4 = nn.Conv2d(c, c, 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dw(self.conv1(x))
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2                    # SimpleGate
        x = x * self.sca(x)            # Simplified Channel Attention
        x = self.conv2(x) + inp

        y = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = self.conv3(y)
        y1, y2 = y.chunk(2, dim=1)
        return self.conv4(y1 * y2) + x  # SimpleGate + residual


class DenoiseNet(nn.Module):
    """NAFNet denoiser operating on Bayer-split pixel values.

    Input to forward(): (B, 5, H/2, W/2) — 4 Bayer channels + 1 noise-std
    channel, normalised to [0, 1] by white_level.

    Output of forward(): (B, 4, H/2, W/2) — denoised Bayer channels
    (residual added to the 4 input pixel channels).

    scaling_factors controls depth and subsampling: each entry is the stride
    for that encoder level (1 = same resolution, 2 = downsample ×2).
    """

    _SCALING_FACTORS = [1, 2, 1]
    _BASE_CH = 16
    _NB = 1

    def __init__(
        self,
        ch: int | None = None,
        scaling_factors: list[int] | None = None,
        nb: int | None = None,
    ) -> None:
        super().__init__()
        nf = ch if ch is not None else self._BASE_CH
        sf = scaling_factors if scaling_factors is not None else list(self._SCALING_FACTORS)
        nb = nb if nb is not None else self._NB
        self.ch = nf
        self.scaling_factors = sf
        self.nb = nb

        self.inp = nn.Conv2d(5, nf, 3, padding=1)

        self.enc_blocks = nn.ModuleList(
            [NAFBlock(nf) for _ in sf[:-1]]
            + [nn.Sequential(*[NAFBlock(nf) for _ in range(nb)])]
        )
        self.downs = nn.ModuleList([
            nn.Conv2d(nf, nf, 2, stride=2) if s > 1 else nn.Identity()
            for s in sf
        ])

        n_dec = len(sf) - 1
        self.fuse       = nn.ModuleList([nn.Conv2d(2 * nf, nf, 1) for _ in range(n_dec)])
        self.dec_blocks = nn.ModuleList([NAFBlock(nf) for _ in range(n_dec)])

        self.out = nn.Conv2d(nf, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.inp(x)

        skips = []
        for enc, down in zip(self.enc_blocks, self.downs):
            feat  = enc(feat)
            skips = [feat] + skips   # LIFO: deepest first
            feat  = down(feat)

        feat  = skips[0]
        skips = skips[1:]

        for fuse, dec in zip(self.fuse, self.dec_blocks):
            up    = F.interpolate(feat, size=skips[0].shape[-2:],
                                  mode='bilinear', align_corners=False)
            feat  = dec(fuse(torch.cat([skips[0], up], dim=1)))
            skips = skips[1:]

        return x[:, :4] + self.out(feat)
