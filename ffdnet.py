"""FFDNet-based Bayer denoiser model.

Reference: Zhang, Zuo, Zhang, "FFDNet: Toward a Fast and Flexible Solution
for CNN-Based Image Denoising", TIP 2018.
https://github.com/cszn/KAIR/blob/master/models/network_ffdnet.py

FFDNet ordinarily pixel-unshuffles a 1-channel image ×2 into 4 sub-images
(plus a tiled noise-level map) before a flat stack of conv+ReLU layers, then
pixel-shuffles the result back up. Our 4 Bayer planes already *are* that
4-sub-image representation -- one level of Bayer-splitting further down --
and the caller (denoise_network.py) already reassembles the output back to
full resolution, same as for nafnet.DenoiseNet. So this is just FFDNet's
conv stack itself, operating directly on the 4 Bayer channels in and out:
no PixelUnshuffle/PixelShuffle, no noise-level map (may be added later as
an extra input channel), no batch norm (matches FFDNet's default
act_mode='R', i.e. plain conv+ReLU, not 'BR').
"""

import torch
import torch.nn as nn


class DenoiseNet(nn.Module):
    """FFDNet-style flat conv stack operating on Bayer-split pixel values.

    Input to forward(): (B, 4, H/2, W/2) -- 4 Bayer channels, normalised to
    [0, 1] by white_level.

    Output of forward(): (B, 4, H/2, W/2) -- denoised Bayer channels.
    Unlike upstream FFDNet's forward() (which returns the stack's raw
    output), a residual connection back to the input is added here, as in
    nafnet.DenoiseNet -- consistent with the rest of this codebase, and
    generally easier to train than direct regression.

    scaling_factors/num_encoders are U-Net encoder-level concepts that
    don't apply to this flat, single-resolution architecture -- must be
    left as None. ch sets the (single) hidden channel count.

    Each conv's dilation cycles through _DILATIONS rather than a single
    fixed rate. A *fixed* dilation (tried first, and rolled back) means
    every layer only ever samples pixels sharing the output pixel's own
    (row, col) parity -- no layer ever mixes the 4 parity sub-lattices
    together, so the network runs as 4 mutually-unaware sub-networks over
    4 different 2x2-offset grids. That's "gridding", and it's exactly what
    produced visible 2x2-pixel artifacts in practice. Cycling through
    [1, 2, 3] (repeated) is the standard fix (Hybrid Dilated Convolution --
    see Wang et al., "Understanding Convolution for Semantic Segmentation",
    2018): no two rates in the group share a common factor, so every
    parity gets mixed in by some layer, and there are no sampling gaps for
    a 3x3 kernel.

    The 64->64 body convs (dense 3x3) hold ~98% of all params/MACs, so
    those -- and only those -- are depthwise-separable here: a per-channel
    3x3 (spatial filtering, groups=nc) conv followed by a 1x1 (cross-channel
    mixing) conv, each with its own ReLU, instead of one dense 3x3 conv --
    ~8x fewer params/MACs per body layer, at the cost of restricting each
    layer to "filter, then mix" rather than arbitrary joint spatial+channel
    mixing in one step. The 1x1 conv doesn't add to the receptive field, so
    this doesn't change the RF math at all, just cost. Head (4->64) and
    tail (64->4) stay dense convs -- they're under 1% of params each, and
    are exactly the two points where full cross-channel mixing (fusing the
    4 Bayer planes in; projecting features back down to 4 planes out)
    matters most, so there's little to gain and real quality to lose by
    restricting them.
    """

    _BASE_CH = 64
    _DILATIONS = [1, 2, 3] * 3  # HDC cycle; also sets the total conv count

    def __init__(
        self,
        ch: int | None = None,
        scaling_factors: list[int] | None = None,
        num_encoders: list[int] | None = None,
    ) -> None:
        super().__init__()
        assert scaling_factors is None and num_encoders is None, \
            "FFDNet has no encoder levels -- scaling_factors/num_encoders don't apply"
        nc = ch if ch is not None else self._BASE_CH
        self.ch = nc
        self.scaling_factors = None
        self.num_encoders = None

        d = self._DILATIONS
        layers = [nn.Conv2d(4, nc, 3, padding=d[0], dilation=d[0]), nn.ReLU(inplace=True)]
        for i in range(1, len(d) - 1):
            # Depthwise-separable body layer -- see class docstring.
            layers += [
                nn.Conv2d(nc, nc, 3, padding=d[i], dilation=d[i], groups=nc),
                nn.ReLU(inplace=True),
                nn.Conv2d(nc, nc, 1),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(nc, 4, 3, padding=d[-1], dilation=d[-1])]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)
