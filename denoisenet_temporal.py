# Copyright (c) 2026 Raspberry Pi

"""Recurrent UNET-style Bayer denoiser, doing joint temporal + spatial denoising.

Parallel copy of denoisenet.py (see temporal_plan.md). By default the only
difference is that self.inp takes 32 input channels instead of 16 -- the
current frame's 16 packed channels concatenated with the previous frame's
16-channel output (or zeros for the first frame of a sequence). Everything
else, including the residual skip connection (which uses only the current
frame), is unchanged.

Optional extras (see DenoiseNet): blending the output with the previous one
through a learned per-pixel factor, learned hidden state channels carried
from frame to frame (optionally with a gated update), and not feeding the
previous output back in at all.

The network works on square-rooted pixel values (see to_net): forward()
takes and returns linear images, but square-roots its inputs first, does
everything (network, residual add, clamp, blend) in that space and squares
the result. This spends int8's levels more evenly in perceptual terms --
finer in the darks, coarser in the highlights. On the device the host does
the same: a sqrt lookup on the input, keeping prev in sqrt space, and
squaring only the output it passes on.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Recorded in every checkpoint (see check_checkpoint), so models trained on
# linear pixel values, before the sqrt transform, can't be loaded by mistake.
INPUT_TRANSFORM = "sqrt"

_SQRT_EPS = 1e-6


def to_net(x: torch.Tensor) -> torch.Tensor:
    """Linear [0, 1] pixel values -> the network's (square-rooted) space,
    also [0, 1].

    Below _SQRT_EPS (under 0.07 of a 16-bit level) it's a straight line to
    (0, 0) instead, to keep the gradient finite there; so it matches a plain
    sqrt on every raw pixel value, 0 included, and from_net(0) round-trips
    exactly.
    """
    root = torch.sqrt(torch.clamp(x, min=_SQRT_EPS))
    return torch.where(x >= _SQRT_EPS, root, torch.clamp(x, min=0.0) / math.sqrt(_SQRT_EPS))


def from_net(y: torch.Tensor) -> torch.Tensor:
    """Inverse of to_net: the network's space -> linear pixel values."""
    return y * y


def check_checkpoint(ckpt, path: str = "checkpoint") -> None:
    """Raise if ckpt (a loaded checkpoint) wasn't trained with this
    module's INPUT_TRANSFORM."""
    found = ckpt.get("input_transform") if isinstance(ckpt, dict) else None
    if found != INPUT_TRANSFORM:
        raise ValueError(
            f"{path} was trained with input transform {found or 'none (linear)'}, but "
            f"denoisenet_temporal.py now uses {INPUT_TRANSFORM!r}: retrain it")


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

    def __init__(self, c: int, gate_clamp: float | None = None) -> None:
        super().__init__()

        # gate_clamp limits the gate's pre-sigmoid values to +-gate_clamp
        # (where the sigmoid is saturated anyway), so int8 can give them a
        # fixed, tight range. The quantised exports leave the clamp out of
        # the graph (clamp_in_graph False): their int8 range is fixed to
        # +-gate_clamp, so int8 saturation does the same clamp.
        #self.gate_clamp = gate_clamp
        #self.clamp_in_graph = True
        self.conv1 = nn.Conv2d(c, c, 1)
        self.dw    = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.conv2 = nn.Conv2d(c, c, 1)
        #self.gate = nn.Sequential(
        #    nn.Conv2d(c, c, 3, padding=1, groups=c),
        #    nn.Conv2d(c, c, 1),
        #    nn.Sigmoid(),
        #)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dw(self.conv1(inp)))
        x = self.conv2(x) + inp
        return x
        #g = self.gate[1](self.gate[0](x))
        #if self.gate_clamp and self.clamp_in_graph:
        #    g = torch.clamp(g, -self.gate_clamp, self.gate_clamp)
        #return self.gate[2](g) * x


class DenoiseNet(nn.Module):
    """Recurrent Unet-style denoiser operating on space-to-depth-packed Bayer
    values, with the previous frame's output (and optionally a learned hidden
    state) fed back as extra input channels.

    forward(x, prev, h) takes:
      x:    (B, 16, H/4, W/4) -- the current frame's 4 Bayer channels, further
            packed 2x2 via space_to_depth(2) and normalised to [0, 1] by
            white_level.
      prev: the previous frame's output in the same packed/normalised space,
            or None on the first frame of a sequence.
      h:    the previous frame's hidden state, (B, state_ch, H/4, W/4) in
            [0, 1], or None on the first frame.
    The Bayer split, packing and white_level normalisation are done by the
    caller -- none of it is part of this model. x, prev and out are linear;
    forward() maps x and prev to_net (sqrt) first, and everything below --
    the network, s and the blend -- happens in that space, out being mapped
    back with from_net. The network's input is the concatenation [x, prev
    (if feed_image), h (if state_ch)], with zeros standing in for prev and h
    on the first frame.

    It returns (out, h_new):
      out:   (B, 16, H/4, W/4) -- denoised, still packed. The candidate
             s = clamp(x + residual, 0, 1) (residual added directly in packed space). With
             blend_output, out = prev + alpha * (s - prev), alpha being a
             learned per-pixel factor in [0, 1] shared across the 16
             channels; on the first frame out = s. The caller unpacks out
             with depth_to_space(2) and passes it back as the next frame's
             prev.
      h_new: the new hidden state, or None when state_ch is 0. Ungated,
             h_new = sigmoid(c); gated (state_gated), h_new = z * h + (1 - z)
             * c with c = sigmoid(...) and z = sigmoid(...), so it stays in
             [0, 1]. c, z and alpha all come from 1x1 heads on the final
             decoder features; the blends happen outside the deep path.

    The defaults (state_ch=0, feed_image=True, state_gated=False,
    blend_output=False) reproduce the original 32-input-channel model, so
    its checkpoints still load.

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
        state_ch: int = 0,
        feed_image: bool = True,
        state_gated: bool = False,
        blend_output: bool = False,
        gate_clamp: float | None = None,
    ) -> None:
        super().__init__()
        nf = ch if ch is not None else self._BASE_CH
        sf = scaling_factors if scaling_factors is not None else list(self._SCALING_FACTORS)
        num_encoders = num_encoders if num_encoders is not None else list(self._NUM_ENCODERS)
        assert len(num_encoders) == len(sf) + 1, \
            "num_encoders must have one more entry than scaling_factors (the bottleneck level)"
        assert state_ch >= 0, "state_ch must be non-negative"
        assert not state_gated or state_ch > 0, "state_gated needs state_ch > 0"
        self.ch = nf
        self.scaling_factors = sf
        self.num_encoders = num_encoders
        self.state_ch = state_ch
        self.feed_image = feed_image
        self.state_gated = state_gated
        self.blend_output = blend_output
        self.gate_clamp = gate_clamp
        self.in_ch = 16 + (16 if feed_image else 0) + state_ch

        self.inp = nn.Conv2d(self.in_ch, nf, 1)

        self.enc_blocks = nn.ModuleList([
            nn.Sequential(*[DenoiseBlock(nf, gate_clamp) for _ in range(n)]) for n in num_encoders
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(nf, nf, s, stride=s) if s > 1 else nn.Identity()
            for s in sf
        ])

        n_dec = len(sf)
        self.dec_blocks = nn.ModuleList([DenoiseBlock(nf, gate_clamp) for _ in range(n_dec)])

        self.out = nn.Conv2d(nf, 16, 3, padding=1)

        if blend_output:
            self.alpha = nn.Conv2d(nf, 1, 1)
            nn.init.zeros_(self.alpha.bias)  # alpha starts around 0.5
        if state_ch:
            self.state_c = nn.Conv2d(nf, state_ch, 1)
        if state_gated:
            self.state_z = nn.Conv2d(nf, state_ch, 1)
            nn.init.constant_(self.state_z.bias, 2.0)  # z ~ 0.88: start biased to remembering

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The deep path: (B, in_ch, H/4, W/4) input -> final decoder features."""
        # EXPERIMENT: discard the previous-frame half of the input, to see how
        # the network trains without it. Zeroed rather than sliced off so that
        # self.inp (and checkpoints) keep their 32-channel shape.
        #x = torch.cat([x[:, :16], torch.zeros_like(x[:, 16:])], dim=1)
        return self._deep(self.inp(x))

    def _deep(self, feat: torch.Tensor) -> torch.Tensor:
        """features() from self.inp's output on."""
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

        return feat

    def forward(
        self,
        x: torch.Tensor,
        prev: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """See the class docstring. With return_aux, also returns a dict of
        the intermediate maps ("alpha", "z") the model computed, for display,
        and (with blend_output) the unblended "candidate" s (linear, as out),
        for its loss."""
        if self.state_ch and h is None:
            h = x.new_zeros(x.shape[0], self.state_ch, *x.shape[2:])
        x = to_net(x)
        prev = to_net(prev) if prev is not None else None
        parts = [x]
        if self.feed_image:
            parts.append(prev if prev is not None else torch.zeros_like(x))
        if self.state_ch:
            parts.append(h)
        feat = self.features(torch.cat(parts, dim=1) if len(parts) > 1 else x)

        aux = {}
        # clamp the candidate to the valid range; the blend below is then a
        # convex combination of in-range values, so the output stays in range
        out = torch.clamp(x + self.out(feat), 0.0, 1.0)
        if self.blend_output:
            aux["candidate"] = from_net(out)
        if self.blend_output and prev is not None:
            alpha = torch.sigmoid(self.alpha(feat))
            out = prev + alpha * (out - prev)
            aux["alpha"] = alpha
        out = from_net(out)

        h_new = None
        if self.state_ch:
            h_new = torch.sigmoid(self.state_c(feat))
            if self.state_gated:
                z = torch.sigmoid(self.state_z(feat))
                h_new = z * h + (1 - z) * h_new
                aux["z"] = z

        if return_aux:
            return out, h_new, aux
        return out, h_new

    def core(self, x: torch.Tensor, prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The exportable part of a feed_image + blend_output model (no
        hidden state): returns (residual, alpha_logit), (B, 16, H/4, W/4) and
        (B, 1, H/4, W/4). x and prev are in the network's space (to_net of
        the linear images; prev is zeros on the first frame). The caller
        completes forward() itself, in float and in that same space: s =
        clamp(x + residual, 0, 1), then on every frame but the first (where
        out = s), out = prev + sigmoid(alpha_logit) * (s - prev), keeping out
        as the next prev and passing on from_net(out)."""
        assert self.feed_image and self.blend_output and not self.state_ch, \
            "core() needs a feed_image + blend_output model with no hidden state"
        feat = self.features(torch.cat([x, prev], dim=1))
        return self.out(feat), self.alpha(feat)


class TemporalCoreNet(nn.Module):
    """Wraps a DenoiseNet to export just its core (see DenoiseNet.core):
    (x, prev) -> (residual, alpha_logit), with the skip add, clamp and blend
    left to the caller, in float. x and prev are in the network's space (see
    to_net), as for DenoiseNet.core.

    Its inputs and outputs are channels-last (NHWC), as the exports' are:
    x and prev are concatenated on the channel axis before a single
    permute to NCHW, which LiteRT folds into its own layout conversion -- a
    channel concat of NCHW tensors instead gets wrapped in mid-graph
    transposes. It's one concat and one self.inp conv, so in int8 a single
    quantisation point, with x's and prev's weights sharing their scales.
    """

    def __init__(self, net: DenoiseNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor, prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        net = self.net
        feat = net._deep(net.inp(torch.cat([x, prev], dim=3).permute(0, 3, 1, 2)))
        return net.out(feat).permute(0, 2, 3, 1), net.alpha(feat).permute(0, 2, 3, 1)


def set_clamp_in_graph(net: DenoiseNet, on: bool) -> None:
    """Include (on) or leave out the gate clamp ops when tracing net (see
    DenoiseBlock)."""
    for m in net.modules():
        if isinstance(m, DenoiseBlock):
            m.clamp_in_graph = on
