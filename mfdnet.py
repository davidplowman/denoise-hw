"""MFDNet-based Bayer denoiser model.

Reference: Jia et al., MFDNet -- a multi-frequency-domain image denoiser.
https://github.com/JL-DY/MFDNet/tree/master/model

MFDNet applies two cascaded single-level Haar wavelet transforms to the raw
input (each: 2x spatial downsample, 4x channel multiply -- an *analysis*
transform, not just a channel-stacking reshuffle: each output sub-band is a
genuine linear combination -- average/horizontal-detail/vertical-detail/
diagonal-detail -- of the 4 pixels in each 2x2 block, unlike a bare
pixel-unshuffle), runs the result through a cascade of MFDB ("multi-
frequency denoising block") stages, then reconstructs via one more conv +
pixel-shuffle back up, with a whole-network residual connection.

As with ffdnet.py, our 4 Bayer planes already are one level of that
downsampling (the CFA split); this applies only ONE Haar level on top
(not MFDNet's own two), landing at the same total depth-from-sensor as
upstream MFDNet, and reconstructs with a matching single PixelShuffle(2)
(not PixelShuffle(4)).

Each MFDB's body is a stack of RepConv blocks (a 3-conv c->2c->2c->c
expand/project residual block, no activations -- exactly the reference's
RepConv, not the "Plain" variant's single plain conv (model/Plaindn.py,
MFDB_Plain) tried first for speed; RepConv costs roughly 4-5x more per rep
but is the paper's actual default, presumably for a reason).

The reference's Haar transform (model/Haar_trans.py, dwt_init) is written
as strided slicing (x[:, :, 0::2, 0::2] etc.) -- which traces fine in
PyTorch but pnnx's ncnn exporter explicitly rejects for ncnn conversion
("slice with step %d is not supported", found when converting ffdnet.py).
HaarTransform below is a from-scratch reformulation as a single fixed
(non-learned), strided, per-channel-grouped 2x2 conv -- numerically the
same transform per input channel (verified against dwt_init's arithmetic
directly), just expressed as a standard, ncnn-safe strided conv instead of
strided slicing. Output channel order ends up grouped by original input
channel rather than upstream's grouped-by-sub-band; every consumer here is
a full (non-grouped) conv, so -- as with nafnet.py's PixelUnshuffle -- the
ordering is functionally irrelevant to what the network can learn.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarTransform(nn.Module):
    """Single-level 2D Haar wavelet transform, fixed (non-learned): (B, c,
    H, W) -> (B, 4c, H/2, W/2), 4 sub-bands (avg/horiz/vert/diag detail)
    per input channel. See module docstring for why this is a strided
    grouped conv rather than the reference's strided slicing."""

    def __init__(self, c: int) -> None:
        super().__init__()
        basis = torch.tensor([
            [[ 0.5,  0.5], [ 0.5,  0.5]],  # LL: average
            [[-0.5,  0.5], [-0.5,  0.5]],  # HL: horizontal detail
            [[-0.5, -0.5], [ 0.5,  0.5]],  # LH: vertical detail
            [[ 0.5, -0.5], [-0.5,  0.5]],  # HH: diagonal detail
        ])
        self.register_buffer("kernel", basis.repeat(c, 1, 1).unsqueeze(1))  # (4c, 1, 2, 2)
        self.groups = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, stride=2, groups=self.groups)


class MFA(nn.Module):
    """Multi-frequency attention gate: a strided-conv downsample, a small
    conv stack down to a sigmoid gate, bilinear-upsample back, then scale
    the input by that gate.

    The reference decomposes this first downsample as an explicit
    HaarTransform followed by a learned Conv2d -- but with no nonlinearity
    between them, that composition is just some other (larger-kernel)
    conv: HaarTransform is invertible per 2x2 block (a lossless change of
    basis, not a bottleneck), so anything the pair could compute, a single
    directly-parameterized strided conv can compute too, at the same
    param/MAC cost. Collapsed into one Conv2d(c, c, 4, stride=2, padding=1)
    here accordingly -- simpler code, same expressivity and cost, at the
    loss of the explicit frequency-decomposition inductive bias (average/
    horizontal/vertical/diagonal detail) the reference's factored form
    provides, which may or may not matter in practice."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            #nn.Conv2d(c, c, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            #nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True),
            #nn.Conv2d(c, c, 1, padding=1), nn.Sigmoid(),

            #nn.Conv2d(c, c, 3, stride=2, padding=1), nn.Sigmoid(),
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c), nn.Conv2d(c, c, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # size=, not the reference's bare scale_factor=2 -- matches
        # nafnet.py's own F.interpolate(..., size=skips[0].shape[-2:], ...)
        # convention; not a behaviour change in our always-power-of-2 case.
        gate = F.interpolate(self.gate(x), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x * gate


class RepConv(nn.Module):
    """Expand-project residual conv: 1x1 (c->2c) -> 3x3 (2c->2c) -> 1x1
    (2c->c), then a residual add. No activation functions at all -- matches
    the reference exactly (RepConv in model/MFDB.py); the nonlinearity in
    an MFDB comes entirely from its trailing MFA gate's sigmoid, not from
    its RepConv stack."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.repconv = nn.Sequential(
            #nn.Conv2d(c, 2 * c, 1),
            #nn.Conv2d(2 * c, 2 * c, 3, padding=1),
            #nn.Conv2d(2 * c, c, 1),
            nn.Conv2d(c, c, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.repconv(x) + x


class MFDB(nn.Module):
    """Multi-frequency denoising block: k RepConvs, then an MFA gate,
    wrapped in a residual connection."""

    def __init__(self, k: int, c: int) -> None:
        super().__init__()
        layers = [RepConv(c) for _ in range(k)]
        layers += [MFA(c)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


class DenoiseNet(nn.Module):
    """MFDNet-style denoiser (single Haar level) operating on Bayer-split
    pixel values.

    Input to forward(): (B, 4, H/2, W/2) -- 4 Bayer channels, normalised to
    [0, 1] by white_level.

    Output of forward(): (B, 4, H/2, W/2) -- denoised Bayer channels
    (whole-network residual, as in the reference and the rest of this
    codebase).

    scaling_factors/num_encoders don't apply (no U-Net encoder levels). ch
    doesn't apply either: unlike nafnet/ffdnet, there's no free-standing
    "channel width" knob here -- the Haar transform fixes the channel count
    flowing through the MFDB stack at 4x the input channels (16 for our
    4-channel input), by construction, not as a tunable choice. Model size
    is controlled by _M (number of MFDB blocks) and _K (RepConvs per block)
    instead.
    """

    _M = 2  # number of MFDB blocks
    _K = 4  # RepConvs per MFDB block

    def __init__(
        self,
        ch: int | None = None,
        scaling_factors: list[int] | None = None,
        num_encoders: list[int] | None = None,
    ) -> None:
        super().__init__()
        #assert ch is None and scaling_factors is None and num_encoders is None, \
        #    "MFDNet's channel width is fixed by the Haar transform, not a free " \
        #    "parameter -- adjust _M/_K instead"
        c = 4 * 4  # one Haar level on 4 Bayer channels -> 16 channels
        self.ch = c
        self.scaling_factors = None
        self.num_encoders = None

        self.haar = HaarTransform(4)
        self.blocks = nn.Sequential(*[MFDB(self._K, c) for _ in range(self._M)])
        self.recon = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        haar = self.haar(x)
        feat = self.blocks(haar) + haar
        return self.recon(feat) + x
