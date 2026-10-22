"""Quantisation-aware training (QAT) support for the temporal DenoiseNet.

Only feed_image + blend_output models without hidden state are supported --
the ones convert_temporal.py exports. Their core (DenoiseNet.core: (x, prev)
-> (residual, alpha_logit)) is run with simulated int8 (PT2E fake-quantise
modules, set up exactly as for the int8 export), and the rest of each frame
-- the sqrt transform, s = clamp(x + residual, 0, 1), the blend with prev
and squaring the result -- is done in float, as the device will.
QATTemporalNet wraps this behind DenoiseNet's own forward() signature, so
train_temporal.py's training and validation code runs it unchanged.

A QAT checkpoint holds both the core's full QAT state ("qat_state_dict":
weights plus every fake-quantise module's ranges) and the plain float
weights ("state_dict"), so the float tools can still load it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.export import Dim

from convert import (
    _INPUT_QUANT_SCALE, _INPUT_QUANT_ZERO_POINT, _make_fixed_input_quantizer,
    _patch_add_annotator,
)
from denoisenet_temporal import (
    DenoiseNet, TemporalCoreNet, from_net, set_clamp_in_graph, to_net,
)


def check_model(model: DenoiseNet) -> None:
    """Raise if model isn't one QAT (and convert_temporal.py) supports."""
    if model.state_ch or not model.feed_image or not model.blend_output:
        raise ValueError(
            "QAT needs a feed_image + blend_output model without hidden state "
            f"(this one: feed_image={model.feed_image}, blend_output={model.blend_output}, "
            f"state_ch={model.state_ch})")


def _fixed_spec(scale: float, zero_point: int):
    from torchao.quantization.pt2e.quantizer import FixedQParamsQuantizationSpec

    return FixedQParamsQuantizationSpec(
        dtype=torch.int8, scale=scale, zero_point=zero_point,
        quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine,
    )


def _fix_annotations(model: torch.fx.GraphModule, gate_clamp: float | None) -> None:
    """Adjust the quantizer's annotations of a TemporalCoreNet graph:

    - The input concat (which the quantizer leaves unannotated, its cat
      annotator failing to match, like convert.py's add) gets the fixed
      [0, 1] input scale on both inputs and its output, and self.inp's conv
      shares that quantisation (through the permute between them), so x and
      prev are quantised exactly once, at the fixed scale.
    - With gate_clamp, each gate's pre-sigmoid conv output gets a fixed
      +-gate_clamp range: int8 saturation then does the clamp the float model
      does (the clamp ops themselves are left out of the traced graph).
    """
    from torchao.quantization.pt2e.quantizer import (
        QuantizationAnnotation, SharedQuantizationSpec,
    )

    input_spec = _fixed_spec(_INPUT_QUANT_SCALE, _INPUT_QUANT_ZERO_POINT)
    for node in model.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is torch.ops.aten.cat.default and \
                all(a.op == "placeholder" for a in node.args[0]):
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map={a: input_spec for a in node.args[0]},
                output_qspec=input_spec, _annotated=True,
            )
            # through the permute to self.inp's conv
            for permute in node.users:
                for conv in permute.users:
                    ann = conv.meta.get("quantization_annotation")
                    if ann and permute in ann.input_qspec_map:
                        ann.input_qspec_map[permute] = SharedQuantizationSpec(node)
        elif gate_clamp and node.target is torch.ops.aten.sigmoid.default:
            ann = node.args[0].meta.get("quantization_annotation")
            if ann is None:
                raise RuntimeError(f"gate logits {node.args[0].name} aren't annotated")
            ann.output_qspec = _fixed_spec(gate_clamp / 127, 0)


def make_quantizer(is_qat: bool, gate_clamp: float | None = None):
    """The int8 export's PT2E quantizer (per-channel weights, fixed [0, 1]
    input scales, convert.py's add-annotation fix), for QAT or calibration,
    with the TemporalCoreNet-specific annotations of _fix_annotations.

    For QAT, the quantizer's conv + batch-norm fusion patterns are skipped:
    they fail to trace on this torch version, and the model has no batch
    norm for them to match anyway.
    """
    from litert_torch.quantize import pt2e_quantizer

    _patch_add_annotator()
    quantizer = _make_fixed_input_quantizer(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True, is_qat=is_qat)
    )
    if is_qat:
        quantizer.STATIC_QAT_ONLY_OPS = []
    annotate = quantizer.annotate

    def _annotate(model):
        model = annotate(model)
        _fix_annotations(model, gate_clamp)
        return model

    quantizer.annotate = _annotate
    return quantizer


def export_core(model: DenoiseNet, example: tuple, dynamic_shapes=None) -> torch.fx.GraphModule:
    """torch.export model's TemporalCoreNet for quantisation, leaving the
    gate clamp ops out of the graph (see DenoiseBlock)."""
    set_clamp_in_graph(model, False)
    try:
        return torch.export.export(TemporalCoreNet(model), example,
                                   dynamic_shapes=dynamic_shapes).module()
    finally:
        set_clamp_in_graph(model, True)


def make_qat_core(model: DenoiseNet, size: tuple[int, int] | None = None) -> torch.fx.GraphModule:
    """Export model's core and prepare it for QAT. The result shares model's
    weights (training it trains them).

    size: a fixed (packed_h, packed_w), batch 1, as the export needs; or None
    for any batch size and any packed size that's a multiple of the model's
    total downscaling, for training and testing. Like TemporalCoreNet, the
    result is channels-last.
    """
    check_model(model)
    device = next(model.parameters()).device
    if size is None:
        align = math.prod(model.scaling_factors or [1])
        example = tuple(torch.rand(2, 8 * align, 8 * align, 16, device=device) for _ in range(2))
        batch, h, w = Dim("B", min=1, max=1024), Dim("H", min=1, max=4096 // align), \
            Dim("W", min=1, max=4096 // align)
        dims = {0: batch, 1: align * h, 2: align * w}
        dynamic_shapes = {"x": dims, "prev": dims}
    else:
        example = tuple(torch.rand(1, *size, 16, device=device) for _ in range(2))
        dynamic_shapes = None

    from torchao.quantization.pt2e import allow_exported_model_train_eval, quantize_pt2e

    exported = export_core(model, example, dynamic_shapes)
    prepared = quantize_pt2e.prepare_qat_pt2e(
        exported, make_quantizer(is_qat=True, gate_clamp=model.gate_clamp))
    allow_exported_model_train_eval(prepared)  # so .train()/.eval() work on it
    return prepared


def freeze_ranges(module: nn.Module) -> None:
    """Stop the fake-quantise modules updating their ranges (fake
    quantisation itself stays on)."""
    from torchao.quantization.pt2e import disable_observer

    module.apply(disable_observer)


class QATTemporalNet(nn.Module):
    """A QAT core behind DenoiseNet's forward() signature: (x, prev, h,
    return_aux) -> (out, None[, aux]), aux holding "candidate" and (but on
    the first frame, prev None) "alpha". As DenoiseNet.forward, x, prev, out
    and the candidate are linear, the core working on their to_net (sqrt)
    versions. Everything outside the core is float, as on the device (see
    convert_temporal.host_step). It also carries the float model's config
    attributes, for checkpointing."""

    _CONFIG = ("ch", "scaling_factors", "num_encoders", "blend_output",
               "state_ch", "state_gated", "feed_image", "gate_clamp", "wide_low")

    def __init__(self, core: torch.fx.GraphModule, model: DenoiseNet) -> None:
        super().__init__()
        self.core = core
        for name in self._CONFIG:
            setattr(self, name, getattr(model, name))

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None = None,
                h: torch.Tensor | None = None, return_aux: bool = False):
        x = to_net(x)
        prev = to_net(prev) if prev is not None else None
        r, a = self.core_nchw(x, prev if prev is not None else torch.zeros_like(x))
        s = torch.clamp(x + r, 0.0, 1.0)
        aux = {"candidate": from_net(s)}
        out = s
        if prev is not None:
            alpha = torch.sigmoid(a)
            out = prev + alpha * (s - prev)
            aux["alpha"] = alpha
        out = from_net(out)
        if return_aux:
            return out, None, aux
        return out, None

    def core_nchw(self, x: torch.Tensor, prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The (channels-last) core, on and returning NCHW tensors (x and
        prev in the network's space, as DenoiseNet.core)."""
        r, a = self.core(x.permute(0, 2, 3, 1), prev.permute(0, 2, 3, 1))
        return r.permute(0, 3, 1, 2), a.permute(0, 3, 1, 2)

    def float_state_dict(self) -> dict:
        """The core's weights, as a DenoiseNet state_dict."""
        prefix = "net."
        return {k[len(prefix):]: v for k, v in self.core.state_dict().items()
                if k.startswith(prefix)}

    def checkpoint_entries(self) -> dict:
        """What a QAT checkpoint holds beyond the config keys."""
        return {"state_dict": self.float_state_dict(),
                "qat_state_dict": self.core.state_dict(), "qat": True}


def load_qat_net(model: DenoiseNet, ckpt: dict, size: tuple[int, int] | None = None,
                 frozen: bool = True) -> QATTemporalNet:
    """Rebuild a QATTemporalNet for model (already holding the checkpoint's
    float weights) from a QAT checkpoint, at the given size (see
    make_qat_core). frozen stops the ranges updating, for testing and
    export."""
    core = make_qat_core(model, size)
    core.load_state_dict(ckpt["qat_state_dict"])
    if frozen:
        freeze_ranges(core)
    return QATTemporalNet(core, model).eval()
