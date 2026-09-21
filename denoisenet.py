# Copyright (c) 2026 Raspberry Pi

"""Simple UNET-style Bayer denoiser model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoiseBlock(nn.Module):
    """Encoder/decoder block with simple channel attention mechanism.
    """

    def __init__(self, c: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(c, c, 1)
        self.dw    = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.conv2 = nn.Conv2d(c, c, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c),
            nn.Conv2d(c, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dw(self.conv1(inp)))
        x = self.conv2(x + inp)
        return self.gate(x) * x


class DenoiseNet(nn.Module):
    """Unet-style denoiser operating on Bayer-split pixel values.

    Input to forward(): (B, 4, H/2, W/2) — 4 Bayer channels,
    normalised to [0, 1] by white_level.

    Output of forward(): (B, 4, H/2, W/2) — denoised Bayer channels
    (residual added to the 4 input pixel channels).

    scaling_factors controls subsampling between encoder levels: each entry
    is the stride between one encoder level and the next (N = downsample ×N
    via a non-overlapping NxN conv; any N works, not just 2). There is
    always one more encoder level than there are scaling factors -- the
    last encoder level (the bottleneck) has no downsample after it.
    """

    _SCALING_FACTORS = [2]
    _BASE_CH = 32
    _NUM_ENCODERS = [1, 1]

    def __init__(
        self,
        ch: int | None = None,
        scaling_factors: list[int] | None = None,
        num_encoders: list[int] | None = None,
    ) -> None:
        super().__init__()
        nf = ch if ch is not None else self._BASE_CH
        sf = scaling_factors if scaling_factors is not None else list(self._SCALING_FACTORS)
        num_encoders = num_encoders if num_encoders is not None else list(self._NUM_ENCODERS)
        assert len(num_encoders) == len(sf) + 1, \
            "num_encoders must have one more entry than scaling_factors (the bottleneck level)"
        self.ch = nf
        self.scaling_factors = sf
        self.num_encoders = num_encoders

        self.inp = nn.Conv2d(4, nf, 2, stride=2)

        self.enc_blocks = nn.ModuleList([
            nn.Sequential(*[DenoiseBlock(nf) for _ in range(n)]) for n in num_encoders
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(nf, nf, s, stride=s) if s > 1 else nn.Identity()
            #nn.Sequential(
            #    nn.Conv2d(nf, nf, 2, stride=2, groups=nf),
            #    nn.Conv2d(nf, nf, 1),
            #) if s > 1 else nn.Identity()
            for s in sf
        ])

        n_dec = len(sf)
        self.dec_blocks = nn.ModuleList([DenoiseBlock(nf) for _ in range(n_dec)])

        self.out = nn.Conv2d(nf, 16, 3, padding=1)
        #self.out = nn.Sequential(
        #    nn.Conv2d(nf, nf, 3, padding=1, groups=nf),
        #    nn.Conv2d(nf, 16, 1),
        #)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.inp(x)

        skips = []
        for i, enc in enumerate(self.enc_blocks):
            feat  = enc(feat)
            skips = [feat] + skips
            if i < len(self.downs):  # bottleneck level has no down() to run
                feat = self.downs[i](feat)

        feat = skips.pop(0)

        for dec in self.dec_blocks:
            skip = skips.pop(0)
            up   = F.interpolate(feat, size=skip.shape[-2:],
                                 mode='bilinear', align_corners=False)
            feat = dec(skip + up)

        return x[:, :4] + self.shuffle(self.out(feat))
