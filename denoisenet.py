# Copyright (c) 2026 Raspberry Pi

"""Simple UNET-style Bayer denoiser model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _replicate_pad1(x: torch.Tensor) -> torch.Tensor:
    """Amount-1 replicate padding via slice + concat.

    Matches F.pad(x, (1,1,1,1), mode="replicate"), but is support better
    on "brainstorm".
    """
    x = torch.cat([x[:, :, :1, :], x, x[:, :, -1:, :]], dim=2)
    x = torch.cat([x[:, :, :, :1], x, x[:, :, :, -1:]], dim=3)
    return x


class FixedBilinearUpsample(nn.Module):
    """Depthwise conv + PixelShuffle that exactly reproduces
    F.interpolate(x, scale_factor=r, mode='bilinear', align_corners=False)
    for an integer r, as a fixed (non-learned) op.

    Used instead of F.interpolate because the deployment target beyond tflite
    (the "brainstorm") doesn't support dynamic-size resize very well.
    """

    def __init__(self, channels: int, scale_factor: int) -> None:
        super().__init__()
        r = scale_factor
        self.r = r
        self.channels = channels

        # Per-phase (0..r-1) delta/frac: each output phase blends the input
        # sample at relative offset `delta` and `delta+1` (delta in {-1,0}),
        # with weights (1-frac, frac) -- same bilinear taps as the
        # ConvTranspose2d version, just laid out per-phase instead of as one
        # shared tent kernel.
        taps = []
        for p in range(r):
            src_offset = (p + 0.5) / r - 0.5
            delta = -1 if src_offset < 0 else 0
            frac = src_offset - delta
            taps.append((delta, frac))

        kernel = torch.zeros(r * r, 1, 3, 3)
        for py in range(r):
            dy, fy = taps[py]
            wy = {dy: 1 - fy, dy + 1: fy}
            for px in range(r):
                dx, fx = taps[px]
                wx = {dx: 1 - fx, dx + 1: fx}
                idx = py * r + px  # matches PixelShuffle's channel grouping
                for oy, wyv in wy.items():
                    for ox, wxv in wx.items():
                        kernel[idx, 0, oy + 1, ox + 1] = wyv * wxv

        weight = kernel.repeat(channels, 1, 1, 1)  # depthwise: (C*r*r, 1, 3, 3)
        self.register_buffer("weight", weight, persistent=False)
        self.shuffle = nn.PixelShuffle(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = _replicate_pad1(x)
        out = F.conv2d(x_pad, self.weight.to(x.dtype), groups=self.channels)
        return self.shuffle(out)


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
        # For when the target platform needs to avoid F.interpolate.
        self.ups = nn.ModuleList([FixedBilinearUpsample(nf, s) for s in reversed(sf)])

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

        for j, dec in enumerate(self.dec_blocks):
            skip = skips.pop(0)
            # up = F.interpolate(feat, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            up   = self.ups[j](feat)
            feat = dec(skip + up)

        return x[:, :4] + self.shuffle(self.out(feat))
