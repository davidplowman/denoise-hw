"""NAFNet-based Bayer denoiser model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mfdnet import MFA, MFDB


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
    """NAFNet-style block: mean-|x| norm + ReLU + MFA.

    SimpleGate (a GLU-style multiplicative gate) needed two same-width
    branches -- conv1a/conv1b+dw_a/dw_b, conv3a/conv3b -- purely to have two
    things to multiply together. A plain activation doesn't need that: one
    branch + ReLU replaces each gated pair, which removes both the gate's
    BinaryOp multiply (real measured cost) and the second branch's convs
    entirely, not just the multiply.

    Normalisation divides by the per-pixel mean of |x| across channels
    rather than RMS (sqrt(mean(x^2))) -- same channel-mean machinery (a
    fixed 1x1 conv, not .mean(dim=1), for the groups-generalises-cleanly
    reason noted below), but one fewer UnaryOp per block since there's no
    sqrt. A learned gain can absorb the resulting constant scale difference
    from RMS, though it's a different statistic (L1 vs L2), not just a
    cheaper way to compute the same one -- worth checking training responds
    fine to it, same as the ReLU swap. Not mathematically equivalent to what
    this replaces -- needs retraining.

    "SCA" (simplified channel attention, in the original NAFNet) used to be
    a bare nn.Parameter(1, c, 1, 1) here -- a static, per-channel learned
    gain with zero dependence on the input's content, not real attention at
    all. Replaced with mfdnet.py's MFA gate (Haar-downsample -> small conv
    stack -> sigmoid -> bilinear-upsample -> multiply): a genuine
    input-dependent, spatially-varying, frequency-aware gate. Substantially
    more expensive (roughly 1,700x the params of the sca it replaces, per
    block) -- an experiment in trading that cost for real per-pixel
    attention, not a free upgrade.
    """

    def __init__(self, c: int) -> None:
        super().__init__()
        # Per-pixel channel mean, expressed as a fixed 1x1 conv: every output
        # channel = mean of the c input channels. A plain (non-learned)
        # buffer, not a Parameter -- there's nothing to train.
        #self.register_buffer("mean_kernel", torch.full((c, c, 1, 1), 1.0 / c))
        #self.norm1_gain = nn.Parameter(torch.ones(1, c, 1, 1))
        #self.norm1_eps  = 1e-6

        self.conv1 = nn.Conv2d(c, c, 1)
        self.dw    = nn.Conv2d(c, c, 3, padding=1, groups=c)
        #self.sca = nn.Parameter(torch.ones(1, c, 1, 1))
        self.mfa = MFA(c)
        self.conv3 = nn.Conv2d(c, c, 1)
        #self.conv4 = nn.Conv2d(c, c, 1)

    def forward_old(self, inp: torch.Tensor) -> torch.Tensor:
        #mabs = F.conv2d(inp.abs(), self.mean_kernel)   # mean of |inp| over channels, per pixel
        #scale = mabs.add(self.norm1_eps)               # no sqrt -- one fewer UnaryOp than RMS
        #x = inp / scale * self.norm1_gain
        x = inp

        x = F.gelu(self.dw(self.conv1(x)))
        #x = x * self.sca
        x = self.mfa(x)

        y = F.relu(self.conv3(x + inp))
        return self.conv4(y) + x  # residual

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.dw(self.conv1(inp)))

        x = self.conv3(x + inp)
        return self.mfa(x)


class DenoiseNet(nn.Module):
    """NAFNet denoiser operating on Bayer-split pixel values.

    Input to forward(): (B, 4, H/2, W/2) — 4 Bayer channels,
    normalised to [0, 1] by white_level.

    Output of forward(): (B, 4, H/2, W/2) — denoised Bayer channels
    (residual added to the 4 input pixel channels).

    forward()'s external shape contract above is unchanged, but internally
    the network works in a 16-channel domain at half the resolution again --
    one more space-to-depth step of the same kind bayer_split() does to go
    from a 1-channel Bayer image to 4 channels. Going in, that fold is
    learned: self.inp is a stride-2 conv straight over the 4 input channels,
    so the "fold 2x2 block into channels" and "mix across channels" steps
    happen in one op rather than a fixed nn.PixelUnshuffle(2) followed by a
    separate conv. Coming back out, self.out + nn.PixelShuffle(2) undoes it
    the other way (fixed channel-to-space, not learned) right before the
    last conv.

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

        # Folds each 2x2 block of the 4 Bayer channels into nf channels at
        # half the resolution again, in one learned stride-2 conv -- see
        # the class docstring. self.out + self.shuffle (declared below)
        # undo this the other way on the way back out.
        self.inp = nn.Conv2d(4, nf, 2, stride=2)

        self.enc_blocks = nn.ModuleList([
            nn.Sequential(*[NAFBlock(nf) for _ in range(n)]) for n in num_encoders
        ])
        # Experiment (reverted): NAFBlock -> MFDB(1, nf) (mfdnet.py's block, just 1 RepConv per MFDB).
        #self.enc_blocks = nn.ModuleList([
        #    nn.Sequential(*[MFDB(1, nf) for _ in range(n)]) for n in num_encoders
        #])
        self.downs = nn.ModuleList([
            nn.Conv2d(nf, nf, 2, stride=2) if s > 1 else nn.Identity()
            #nn.Sequential(
            #    nn.Conv2d(nf, nf, 2, stride=2, groups=nf),
            #    nn.Conv2d(nf, nf, 1),
            #) if s > 1 else nn.Identity()
            for s in sf
        ])

        n_dec = len(sf) - 1
        #self.fuse       = nn.ModuleList([nn.Conv2d(2 * nf, nf, 1) for _ in range(n_dec)])
        self.skip_gain = nn.ParameterList([nn.Parameter(torch.ones(1, nf, 1, 1)) for _ in range(n_dec)])
        self.up_gain   = nn.ParameterList([nn.Parameter(torch.ones(1, nf, 1, 1)) for _ in range(n_dec)])
        self.dec_blocks = nn.ModuleList([NAFBlock(nf) for _ in range(n_dec)])
        # Experiment (reverted): NAFBlock -> MFDB(1, nf), same as enc_blocks above.
        #self.dec_blocks = nn.ModuleList([MFDB(1, nf) for _ in range(n_dec)])

        self.out = nn.Conv2d(nf, 16, 3, padding=1)
        #self.out = nn.Sequential(
        #    nn.Conv2d(nf, nf, 3, padding=1, groups=nf),
        #    nn.Conv2d(nf, 16, 1),
        #)
        # Unfolds self.out's 16 channels back to 4, at double the
        # resolution -- the inverse of what self.inp folds together above,
        # but fixed (unlearned) rather than learned.
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.inp(x)

        skips = []
        n = len(self.enc_blocks)
        for i, (enc, down) in enumerate(zip(self.enc_blocks, self.downs)):
            feat  = enc(feat)
            skips = [feat] + skips   # LIFO: deepest first
            if i < n - 1:             # last level's down() output is never used below
                feat = down(feat)

        feat  = skips[0]
        skips = skips[1:]

        #for fuse, dec in zip(self.fuse, self.dec_blocks):
        for skip_gain, up_gain, dec in zip(self.skip_gain, self.up_gain, self.dec_blocks):
            up    = F.interpolate(feat, size=skips[0].shape[-2:],
                                  mode='bilinear', align_corners=False)
            #feat  = dec(fuse(torch.cat([skips[0], up], dim=1)))
            #feat = dec(skips[0] * skip_gain + up * up_gain)
            feat = dec(skips[0] + up)
            skips = skips[1:]

        return x[:, :4] + self.shuffle(self.out(feat))
