"""NAFNet-based Bayer denoiser model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoiseBlock(nn.Module):
    """NAFNet-style block: mean-|x| norm + ReLU + MFA.
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
        x = nn.gelu(self.dw(self.conv1(inp)))
        x = self.conv2(x + inp)
        return self.gate(x) * x


class DenoiseNet(nn.Module):
    """NAFNet denoiser operating on Bayer-split pixel values.

    Input to forward(): (B, 4, H/2, W/2) — 4 Bayer channels,
    normalised to [0, 1] by white_level.

    Output of forward(): (B, 4, H/2, W/2) — denoised Bayer channels
    (residual added to the 4 input pixel channels).

    scaling_factors controls depth and subsampling: each entry is the stride
    for that encoder level (1 = same resolution, 2 = downsample ×2).
    """

    _SCALING_FACTORS = [2, 2]
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
        assert len(num_encoders) == len(sf), "num_encoders must have one entry per scaling level"
        self.ch = nf
        self.scaling_factors = sf
        self.num_encoders = num_encoders

        self.inp = nn.Conv2d(4, nf, 2, stride=2)

        self.enc_blocks = nn.ModuleList([
            nn.Sequential(*[DenoiseBlock(nf) for _ in range(n)]) for n in num_encoders
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(nf, nf, 2, stride=2) if s > 1 else nn.Identity()
            #nn.Sequential(
            #    nn.Conv2d(nf, nf, 2, stride=2, groups=nf),
            #    nn.Conv2d(nf, nf, 1),
            #) if s > 1 else nn.Identity()
            for s in sf
        ])

        n_dec = len(sf) - 1
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
        n = len(self.enc_blocks)
        for i, (enc, down) in enumerate(zip(self.enc_blocks, self.downs)):
            feat  = enc(feat)
            skips = [feat] + skips
            if i < n - 1:             # last level's down() output is never used below
                feat = down(feat)

        feat  = skips[0]
        skips = skips[1:]

        for dec in self.dec_blocks:
            up    = F.interpolate(feat, size=skips[0].shape[-2:],
                                  mode='bilinear', align_corners=False)
            feat = dec(skips[0] + up)
            skips = skips[1:]

        return x[:, :4] + self.shuffle(self.out(feat))
