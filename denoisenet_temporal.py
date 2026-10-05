# Copyright (c) 2026 Raspberry Pi

"""Recurrent UNET-style Bayer denoiser, doing joint temporal + spatial denoising.

Parallel copy of denoisenet.py (see temporal_plan.md): the only difference is
that self.inp takes 32 input channels instead of 16 -- the current frame's 16
packed channels concatenated with the previous frame's 16-channel output (or
zeros for the first frame of a sequence). Everything else, including the
residual skip connection (which uses only the current-frame half of the
input), is unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def space_to_depth(x: torch.Tensor, r: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, C*r*r, H/r, W/r) via pixel-unshuffle.

    Channel index c*r*r + i*r + j holds original channel c's (i, j)
    sub-position within each rxr block -- i.e. output channel k's original
    channel is k // (r*r), so per-original-channel data (e.g. white-balance
    gains) expands to the packed channels via repeat_interleave(r*r, dim=1).
    Verified against PyTorch's own convention, not just assumed.
    """
    return F.pixel_unshuffle(x, r)


def depth_to_space(x: torch.Tensor, r: int) -> torch.Tensor:
    """Inverse of space_to_depth: (B, C*r*r, H, W) -> (B, C, H*r, W*r)."""
    return F.pixel_shuffle(x, r)


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
    """Recurrent Unet-style denoiser operating on space-to-depth-packed Bayer
    values, with the previous frame's output fed back as extra input channels.

    Input to forward(): (B, 32, H/4, W/4) -- the current frame's 4 Bayer
    channels, further packed 2x2 via space_to_depth(2) and normalised to
    [0, 1] by white_level (channels 0:16), concatenated with the previous
    frame's 16-channel output in the same packed/normalised space (channels
    16:32), or zeros for the first frame of a sequence. The Bayer split,
    packing, white_level normalisation, and previous-output concatenation
    are all done by the caller -- none of it is part of this model.

    Output of forward(): (B, 16, H/4, W/4) -- denoised, still packed
    (residual added directly in packed space, using only the current-frame
    half of the input). The caller unpacks the result back to (B, 4, H/2, W/2)
    with depth_to_space(2), and feeds it back as this same frame's
    contribution to the *next* frame's input.

    scaling_factors controls subsampling between encoder levels: each entry
    is the stride between one encoder level and the next (N = downsample ×N
    via a non-overlapping NxN conv; any N works, not just 2). There is
    always one more encoder level than there are scaling factors -- the
    last encoder level (the bottleneck) has no downsample after it.
    """

    _SCALING_FACTORS = [2]
    _BASE_CH = 64
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

        self.inp = nn.Conv2d(32, nf, 1)

        self.enc_blocks = nn.ModuleList([
            nn.Sequential(*[DenoiseBlock(nf) for _ in range(n)]) for n in num_encoders
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(nf, nf, s, stride=s) if s > 1 else nn.Identity()
            for s in sf
        ])

        n_dec = len(sf)
        self.dec_blocks = nn.ModuleList([DenoiseBlock(nf) for _ in range(n_dec)])

        self.out = nn.Conv2d(nf, 16, 3, padding=1)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        # EXPERIMENT: discard the previous-frame half of the input, to see how
        # the network trains without it. Zeroed rather than sliced off so that
        # self.inp (and checkpoints) keep their 32-channel shape.
        #x = torch.cat([x[:, :16], torch.zeros_like(x[:, 16:])], dim=1)
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
            up = F.interpolate(feat, size=skip.shape[-2:], mode='nearest')
            feat = dec(skip + up)

        return self.out(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :16] + self.residual(x)
