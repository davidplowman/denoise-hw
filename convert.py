#!/usr/bin/env python3
"""Convert a DenoiseNet .pth checkpoint to NCNN, LiteRT (TFLite), ONNX, or
OpenVINO model files.

Only the network itself (network.DenoiseNet) is exported. The Bayer split/
merge, the further 2x2 space-to-depth/depth-to-space packing, and
white_level normalisation done around it in denoise_network.py are NOT part
of the exported graph and must be reproduced by whatever code calls the
converted model.

Example:
    python convert.py checkpoints/imx662_32_2221e1221_07908.pth --verify
    python convert.py checkpoints/imx662_32_2221e1221_07908.pth --format lite --verify
    python convert.py checkpoints/imx662_32_2221e1221_07908.pth --format onnx --verify
    python convert.py checkpoints/imx662_32_2221e1221_07908.pth --format openvino --verify
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

from network import DenoiseNet, ResidualDenoiseNet, space_to_depth


def round_up(value: int, align: int) -> int:
    """Round value up to the next multiple of align."""
    return ((value + align - 1) // align) * align


def compute_bayer_size(model: DenoiseNet, width: int, height: int) -> tuple[int, int, int, int]:
    """Round width/height up to the model's required alignment and return the
    resulting (width, height, bayer_w, bayer_h) -- bayer_w/bayer_h being the
    intermediate Bayer-split size, before the caller's further 2x2 packing
    (see pack_size below).

    scaling_factors holds the stride between each encoder level and the next
    (denoisenet.DenoiseNet); there's always one more encoder level than there are
    scaling factors (the bottleneck level has no downsample after it), and
    every entry actually runs. So the real alignment needed is the Bayer
    split's own halving, times the caller's further fixed 2x2 packing before
    self.inp (self.inp is a plain pointwise conv with no downscale of its
    own -- the 2x2 factor here is always present, unlike self.inp's own
    downscale in earlier variants, kept as a defensive fallback for those),
    times the product of every scaling factor.
    """
    inp_stride = getattr(getattr(model, "inp", None), "stride", (1,))
    align = 2 * 2 * inp_stride[0]
    if model.scaling_factors:
        for s in model.scaling_factors:
            align *= s

    new_width = round_up(width, align)
    new_height = round_up(height, align)
    if new_width != width:
        print(f"  Rounding width up:  {width} -> {new_width} (must be a multiple of {align} px)")
    if new_height != height:
        print(f"  Rounding height up: {height} -> {new_height} (must be a multiple of {align} px)")

    bayer_w, bayer_h = new_width // 2, new_height // 2
    return new_width, new_height, bayer_w, bayer_h


def pack_size(bayer_w: int, bayer_h: int) -> tuple[int, int]:
    """Bayer-split size -> the network's actual (packed_w, packed_h) after
    the caller's further 2x2 space-to-depth packing."""
    return bayer_w // 2, bayer_h // 2


def load_model(pth_path: Path) -> DenoiseNet:
    """Build a DenoiseNet from a checkpoint, mirroring DenoiseNetwork.__init__."""
    ckpt = torch.load(pth_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ch = ckpt.get("ch")
        scaling_factors = ckpt.get("scaling_factors")
        num_encoders = ckpt.get("num_encoders")
        state_dict = ckpt["state_dict"]
    else:
        ch = scaling_factors = num_encoders = None
        state_dict = ckpt

    model = DenoiseNet(ch=ch, scaling_factors=scaling_factors, num_encoders=num_encoders)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a DenoiseNet .pth checkpoint to NCNN, LiteRT (TFLite), or ONNX model files.",
    )
    parser.add_argument("pth_file", type=Path, help="Trained checkpoint to convert (.pth).")
    parser.add_argument(
        "--format", choices=["ncnn", "lite", "onnx", "openvino"], default="ncnn",
        help="Target format (default: %(default)s). 'ncnn' writes a .param/.bin pair "
             "(via pnnx); 'lite' writes a single .tflite file for ai-edge-litert "
             "(via litert_torch); 'onnx' writes a single .onnx file (via "
             "torch.onnx.export, cleaned up with onnxsim); 'openvino' writes an "
             "IR .xml/.bin pair (via openvino.convert_model).",
    )
    parser.add_argument(
        "--width", type=int, default=1936,
        help="Sensor output width in pixels, before Bayer-splitting (default: %(default)s).",
    )
    parser.add_argument(
        "--height", type=int, default=1096,
        help="Sensor output height in pixels, before Bayer-splitting (default: %(default)s).",
    )
    parser.add_argument(
        "--quantize", choices=["fp16", "fp32", "int8"], default="fp16",
        help="Precision of the exported weights (default: %(default)s). "
             "'int8' (--format lite only) statically quantizes both weights "
             "and activations to genuine native int8 ops, and exports only "
             "DenoiseNet's residual branch (see ResidualDenoiseNet) -- the "
             "caller must add the skip connection back itself, in float. "
             "The model's input/output tensors are themselves int8 (scale/ "
             "zero_point recorded in the .tflite); the caller must quantize "
             "its float input and dequantize the int8 output itself.",
    )
    parser.add_argument(
        "--calib-dataset", default="/opt/datasets/PI_RAW", metavar="ROOT",
        help="Root directory of real captured folders (same convention as "
             "dataset.py build --dataset) used to calibrate --quantize int8's "
             "activation ranges (default: %(default)s). Ignored otherwise.",
    )
    parser.add_argument(
        "--calib-count", type=int, default=8, metavar="N",
        help="Number of real images to calibrate --quantize int8 with "
             "(default: %(default)s). Ignored otherwise.",
    )
    parser.add_argument(
        "--output-scale", type=float, default=None, metavar="SCALE",
        help="--quantize int8 only: overwrite the output tensor's "
             "quantization scale with this value (e.g. 0.002), leaving its "
             "zero_point as calibrated. Default: unchanged, i.e. use "
             "whatever scale calibration derives.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run the exported model on a random input and compare it to PyTorch.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for the output file(s), created if it doesn't exist "
             "(default: a directory named after --format, e.g. 'ncnn' or 'lite').",
    )
    parser.add_argument(
        "--output-name", default=None,
        help="Base filename, without extension, for the output files "
             "(default: '<output stem>_<tensor CxHxW>_<quantize>').",
    )
    parser.add_argument(
        "--output-stem", default=None,
        help="Replaces <checkpoint stem> in the default --output-name "
             "(default: the checkpoint's filename without .pth). Ignored if --output-name is given.",
    )
    parser.add_argument(
        "--keep-intermediate", action="store_true",
        help="Keep the intermediate TorchScript/pnnx files pnnx produces along the way "
             "(--format ncnn only; default: delete them, keeping only the .param/.bin).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for the tracing/--verify input tensor (default: %(default)s).",
    )
    return parser.parse_args(argv)


def export_ncnn(
    model: DenoiseNet, example: torch.Tensor, out_prefix: Path,
    quantize: str, keep_intermediate: bool, _args: argparse.Namespace,
) -> list[Path]:
    """Export to NCNN .param/.bin via pnnx. Returns [param_path, bin_path]."""
    import pnnx  # imported lazily so --help works without it installed

    param_path = out_prefix.with_suffix(".param")
    bin_path = out_prefix.with_suffix(".bin")

    intermediate_pt = str(out_prefix) + ".pt"
    pnnx_pnnxparam = str(out_prefix) + ".pnnx.param"
    pnnx_pnnxbin = str(out_prefix) + ".pnnx.bin"
    pnnx_pnnxpy = str(out_prefix) + "_pnnx.py"
    pnnx_pnnxonnx = str(out_prefix) + ".pnnx.onnx"
    pnnx_ncnnpy = str(out_prefix) + "_ncnn.py"

    pnnx.export(
        model,
        intermediate_pt,
        inputs=example,
        ncnnparam=str(param_path),
        ncnnbin=str(bin_path),
        ncnnpy=pnnx_ncnnpy,
        pnnxparam=pnnx_pnnxparam,
        pnnxbin=pnnx_pnnxbin,
        pnnxpy=pnnx_pnnxpy,
        pnnxonnx=pnnx_pnnxonnx,
        fp16=(quantize == "fp16"),
    )

    intermediate_files = [
        Path(intermediate_pt), Path(pnnx_pnnxparam), Path(pnnx_pnnxbin),
        Path(pnnx_pnnxpy), Path(pnnx_pnnxonnx), Path(pnnx_ncnnpy),
    ]
    if keep_intermediate:
        print("Kept intermediate pnnx files alongside the output.")
    else:
        for f in intermediate_files:
            f.unlink(missing_ok=True)

    if not (param_path.exists() and bin_path.exists()):
        sys.exit("error: conversion did not produce the expected .param/.bin files")
    return [param_path, bin_path]


def verify_ncnn(
    model: DenoiseNet, example: torch.Tensor, outputs: list[Path], quantize: str,
    _args: argparse.Namespace,
) -> None:
    import ncnn

    param_path, bin_path = outputs
    with torch.no_grad():
        ref = model(example)[0].numpy()  # (16, packed_h, packed_w)

    net = ncnn.Net()
    net.load_param(str(param_path))
    net.load_model(str(bin_path))
    ex = net.create_extractor()
    ex.input(net.input_names()[0], ncnn.Mat(example[0].numpy()))
    ret, mat_out = ex.extract(net.output_names()[0])
    if ret != 0:
        sys.exit(f"error: ncnn extraction failed (ret={ret})")
    out = mat_out.numpy()
    _report_diff(out, ref, example, quantize, engine="ncnn")


def _pack_bayer(bayer, white_level: float) -> torch.Tensor:
    """Normalise + pack a raw Bayer crop into the network's actual packed
    input format: Bayer-split into 4 channels, then the caller's further 2x2
    space_to_depth, then divide by white_level -- same transform train.py's
    _prepare_input applies, duplicated here rather than imported, matching
    this file's existing convention (see module docstring) of keeping that
    packing logic per-caller rather than shared.
    """
    t = torch.from_numpy(bayer.astype("float32"))[None]  # (1, H, W)
    split = torch.stack([
        t[:, 0::2, 0::2], t[:, 0::2, 1::2],
        t[:, 1::2, 0::2], t[:, 1::2, 1::2],
    ], dim=1)  # (1, 4, H/2, W/2)
    return space_to_depth(split / white_level, 2)


def _build_calibration_examples(
    new_width: int, new_height: int, calib_dataset: str, calib_count: int,
) -> list[torch.Tensor]:
    """Load a handful of real captured noisy DNGs, cropped to the exported
    model's exact (pre-packing) sensor resolution, normalised and packed the
    same way the network's callers do. Used to calibrate --quantize int8's
    activation ranges -- a random tensor isn't representative enough for
    that (see the int8 planning notes).
    """
    from dng import DNG
    from folders import find_folders

    found = find_folders(calib_dataset)
    if not found:
        sys.exit(
            f"error: no captured folders with noisy.dng/gt.dng found under "
            f"{calib_dataset} (needed to calibrate --quantize int8; pass "
            "--calib-dataset to point elsewhere)"
        )

    examples = []
    for folder in found[:calib_count]:
        img = DNG(str(Path(folder) / "noisy.dng"))
        h, w = img.bayer.shape
        if h < new_height or w < new_width:
            continue
        crop = img.bayer[:new_height, :new_width]
        examples.append(_pack_bayer(crop, img.white_level))

    if not examples:
        sys.exit(
            f"error: none of the {len(found)} captured folders under "
            f"{calib_dataset} are large enough for a {new_width}x{new_height} "
            "calibration crop"
        )
    return examples


def _patch_add_annotator() -> None:
    """Work around a bug in litert_torch's vendored PT2E quantizer utils:
    _annotate_add's get_source_partitions() call only matches the operator.add
    /torch.add/operator.iadd *callables*, never the string "add" -- unlike
    every sibling annotator (_annotate_mul passes ["mul", "mul_", ...], with
    the string forms first). On this torch/torchao version, add/mul nodes
    exported via torch.export carry no source_fn_stack (empty regardless),
    so get_source_partitions can only match them via the string-keyed
    torch_fn fallback -- meaning _annotate_add can never match anything, and
    every torch.add site in the model (not just the ones after
    F.interpolate) silently gets skipped by PT2E's own annotation pass.
    A couple of those adds still end up native int8 in the exported .tflite
    purely by accident (when both their inputs happen to already be int8
    from neighbouring quantized convs with no intervening float-only op),
    but the ones downstream of F.interpolate don't get that lucky, since
    upsample_nearest2d isn't in PT2E's shared-quantization-spec propagation
    list either -- confirmed by inspecting exported op tensor dtypes
    directly (ADD showed float32 in/out right after RESIZE_NEAREST_NEIGHBOR,
    while ADD elsewhere showed int8).

    There's no public quantizer config for this (PT2EQuantizer.set_operator_type
    stores its config but nothing ever reads it in this version -- also
    confirmed by inspection). Patching in a corrected annotator -- identical
    to upstream's _annotate_add but with the string forms added, matching
    _annotate_mul's own pattern -- is the only fix available; it's applied
    once, idempotently, into the same OP_TO_ANNOTATOR registry PT2EQuantizer
    itself reads from.
    """
    import itertools
    import operator

    from litert_torch.quantize import pt2e_quantizer_utils as u
    from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
    from torchao.quantization.pt2e.quantizer import QuantizationAnnotation

    if getattr(u.OP_TO_ANNOTATOR["add"], "_patched_for_string_match", False):
        return  # already patched (e.g. a previous call in this process)

    def _annotate_add_fixed(gm, quantization_config, filter_fn=None):
        add_partitions = get_source_partitions(
            gm.graph, ["add", "add_", operator.add, torch.add, operator.iadd],
            filter_fn,
        )
        add_partitions = list(itertools.chain(*add_partitions.values()))
        annotated_partitions = []
        for add_partition in add_partitions:
            annotated_partitions.append(add_partition.nodes)
            add_node = add_partition.output_nodes[0]
            if u._is_annotated([add_node]):
                continue
            input_act_qspec = u.get_input_act_qspec(quantization_config)
            output_act_qspec = u.get_output_act_qspec(quantization_config)
            input_qspec_map = {}
            for arg in (add_node.args[0], add_node.args[1]):
                if isinstance(arg, torch.fx.Node):
                    input_qspec_map[arg] = input_act_qspec
            add_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map, output_qspec=output_act_qspec,
                _annotated=True,
            )
        return annotated_partitions

    _annotate_add_fixed._patched_for_string_match = True
    u.OP_TO_ANNOTATOR["add"] = _annotate_add_fixed


# The network's input is always normalised to [0, 1] (by white_level -- see
# denoisenet.py's docstring), so unlike every other activation it needs no
# calibration: fix its quantization to the exact affine map [0, 1] -> [-128,
# 127], i.e. scale=1/255, zero_point=-128, rather than letting a
# HistogramObserver derive a scale from calibration data that only
# approximately covers [0, 1].
_INPUT_QUANT_SCALE = 1 / 255
_INPUT_QUANT_ZERO_POINT = -128


def _make_fixed_input_quantizer(global_config) -> "torch.nn.Module":
    """Build a PT2EQuantizer that behaves exactly like PT2EQuantizer(), except
    the model's own input activation is always quantized with the fixed
    (scale, zero_point) above, instead of a calibrated HistogramObserver spec.

    There's no working public config for this: PT2EQuantizer.set_operator_type
    stores its config but nothing ever reads it back (dead code in this
    version), and set_module_name's nn_module_stack name-mangling mis-parses
    a nested attribute path like "net.inp" here (it assumes exactly the
    "L__self___<flat_name>" format and off-by-one truncates names that
    contain a dot, confirmed by inspecting the actual nn_module_stack keys
    torch.export produces for this model) -- so it can't target this one
    module either. Overriding annotate() to fix up just the placeholder's
    outgoing edge after the normal annotation pass is the reliable way to do
    this at the one point (the graph input) where there's no producer node
    with a competing annotation to fight with.
    """
    from litert_torch.quantize import pt2e_quantizer
    from torchao.quantization.pt2e.quantizer import FixedQParamsQuantizationSpec

    fixed_input_spec = FixedQParamsQuantizationSpec(
        dtype=torch.int8, scale=_INPUT_QUANT_SCALE, zero_point=_INPUT_QUANT_ZERO_POINT,
        quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine,
    )

    class _FixedInputQuantizer(pt2e_quantizer.PT2EQuantizer):
        def annotate(self, model):
            model = super().annotate(model)
            # every graph input (just one for DenoiseNet; x and prev for
            # convert_temporal.py's core) is a [0, 1] image
            for placeholder in (n for n in model.graph.nodes if n.op == "placeholder"):
                for user in list(placeholder.users):
                    ann = user.meta.get("quantization_annotation")
                    if ann and placeholder in ann.input_qspec_map:
                        ann.input_qspec_map[placeholder] = fixed_input_spec
            return model

    return _FixedInputQuantizer().set_global(global_config)


def _override_output_scale(prepared, output_scale: float, output_index: int | None = None) -> None:
    """Force the calibrated output observer to report output_scale from
    calculate_qparams(), leaving its calibrated zero_point untouched.

    Must run after prepare_pt2e (the observer submodule must exist) and
    after calibration (so the zero_point being kept is the calibrated one,
    not an unconverged one), but before convert_pt2e, which is what actually
    reads calculate_qparams() to bake the scale/zero_point into the graph as
    constants -- there's no public option for this, so the observer instance
    feeding the graph's output is patched directly.

    output_index picks which output, for a graph with several (default: the
    graph must have exactly one).
    """
    output_node = next(n for n in prepared.graph.nodes if n.op == "output")
    if output_index is None:
        (ret_node,) = output_node.args[0]
    else:
        ret_node = output_node.args[0][output_index]
    # look through a final layout permute (convert_temporal.py's core)
    while ret_node.op == "call_function" and ret_node.target is torch.ops.aten.permute.default:
        ret_node = ret_node.args[0]
    if ret_node.op != "call_module":
        sys.exit("error: --output-scale expected an observer directly at the "
                  "model output and found something else -- the exported "
                  "graph's structure may have changed.")

    observer = getattr(prepared, ret_node.target)
    orig_calculate_qparams = observer.calculate_qparams

    def _patched_calculate_qparams():
        scale, zero_point = orig_calculate_qparams()
        return torch.full_like(scale, output_scale), zero_point

    observer.calculate_qparams = _patched_calculate_qparams


def _quantize_int8(
    model: DenoiseNet, example: torch.Tensor, calib_dataset: str, calib_count: int,
    output_scale: float | None = None, calib_examples: list[torch.Tensor] | None = None,
):
    """Run the PT2E static-quantization flow on ResidualDenoiseNet(model),
    calibrated with real captured data (or with calib_examples, packed
    tensors of example's shape, if given -- calib_dataset and calib_count
    are then ignored).

    Returns (quantized, quantizer): the quantized GraphModule -- which
    litert_torch.convert accepts just like any other nn.Module -- and the
    PT2EQuantizer used to produce it. The quantizer must also be passed back
    into litert_torch.convert(..., quant_config=...): without it, the
    converter leaves the PT2E graph's quantize/dequantize ops un-fused,
    wrapping every op in dequantize-to-float/requantize instead of lowering
    to native int8 kernels (confirmed empirically by inspecting the exported
    op list -- this isn't optional).

    output_scale, if given, overwrites the output tensor's calibrated
    quantization scale (its zero_point is left as calibrated) -- see
    _override_output_scale.
    """
    from litert_torch.quantize import pt2e_quantizer
    from torchao.quantization.pt2e import quantize_pt2e

    _patch_add_annotator()

    residual_model = ResidualDenoiseNet(model).eval()
    exported = torch.export.export(residual_model, (example,)).module()

    quantizer = _make_fixed_input_quantizer(
        pt2e_quantizer.get_symmetric_quantization_config(is_per_channel=True)
    )
    prepared = quantize_pt2e.prepare_pt2e(exported, quantizer)

    # example is (1, 16, packed_h, packed_w); undo compute_bayer_size's two
    # halvings (sensor -> Bayer-split -> caller's further 2x2 packing) to
    # get back the pre-packing crop size calibration data must match.
    if calib_examples is None:
        packed_h, packed_w = example.shape[-2:]
        new_width, new_height = packed_w * 4, packed_h * 4
        print(f"  Calibrating int8 ranges from real data under {calib_dataset} "
              f"(up to {calib_count} images)...")
        calib_examples = _build_calibration_examples(
            new_width, new_height, calib_dataset, calib_count,
        )
    with torch.no_grad():
        for calib in calib_examples:
            prepared(calib)
    print(f"  Calibrated with {len(calib_examples)} image(s).")

    if output_scale is not None:
        _override_output_scale(prepared, output_scale)
        print(f"  Overriding output quantization scale with {output_scale:g} "
              "(zero_point left as calibrated).")

    quantized = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    # Exported/quantized GraphModules don't support plain .train()/.eval()
    # (raises NotImplementedError) -- export_lite wraps this in
    # to_channel_last_io(...).eval(), which recurses into it, so make that
    # safe rather than special-casing the wrap for int8.
    from torchao.quantization.pt2e import allow_exported_model_train_eval
    allow_exported_model_train_eval(quantized)
    return quantized, quantizer


def export_lite(
    model: DenoiseNet, example: torch.Tensor, out_prefix: Path,
    quantize: str, _keep_intermediate: bool, args: argparse.Namespace,
) -> list[Path]:
    """Export to a LiteRT (.tflite) flatbuffer via litert_torch. Returns [tflite_path].

    Per how-to-convert.txt (matching this repo's working NPU export for the m9
    model): fp16/fp32 always export plain fp32 -- fp16 weight-only
    quantization introduces DEQUANTIZE ops that break NPU compilation. And
    wrap the model with to_channel_last_io so the NCHW<->NHWC conversion
    happens for free at the graph boundary; otherwise litert_torch inserts it
    as mid-graph Transpose ops, which were the only unsupported nodes left
    and fragmented compilation into 8 partitions. The traced example is
    permuted to NHWC to match.

    int8 takes a different path (see _quantize_int8): full static
    quantization of just DenoiseNet's residual branch, calibrated with real
    data, lowered to genuine native int8 ops (not a quantize/dequantize-
    wrapped float graph -- that requires passing quant_config to
    litert_torch.convert, see _quantize_int8's docstring). As a consequence
    the exported model's input/output tensors are themselves int8 (with a
    scale/zero_point recorded in the .tflite file), not float32 -- whatever
    loads this model must quantize its float input and dequantize the int8
    output itself, the same way verify_lite does below.

    litert_torch produces no intermediate files to clean up, so _keep_intermediate
    is accepted (for a uniform export-function signature) but unused.
    """
    import litert_torch  # imported lazily so --help works without it installed

    tflite_path = out_prefix.with_suffix(".tflite")
    quant_config = None

    if quantize == "int8":
        from litert_torch.quantize import quant_config as qcfg

        exported_model, quantizer = _quantize_int8(
            model, example, args.calib_dataset, args.calib_count,
            output_scale=args.output_scale,
        )
        quant_config = qcfg.QuantConfig(pt2e_quantizer=quantizer)
    else:
        if quantize == "fp16":
            print("  Note: --quantize fp16 is ignored for --format lite (breaks NPU "
                  "compilation -- see how-to-convert.txt); exporting fp32 instead.")
        exported_model = model

    # .eval() on the wrapper itself, not just model -- ChannelLastIOWrapper is
    # its own nn.Module and defaults to training mode regardless of the inner
    # model's mode.
    wrapper = litert_torch.to_channel_last_io(exported_model, args=[0], outputs=[0]).eval()
    example_nhwc = example.permute(0, 2, 3, 1).contiguous()
    edge_model = litert_torch.convert(wrapper, (example_nhwc,), quant_config=quant_config)
    edge_model.export(str(tflite_path))

    if not tflite_path.exists():
        sys.exit("error: conversion did not produce the expected .tflite file")
    return [tflite_path]


def _quantize_for_tensor(x, detail: dict):
    """Quantize a float array to a TFLite tensor's int dtype using its stored
    (scale, zero_point) -- scale is 0 for a tensor that isn't quantized (e.g.
    plain float32 I/O), in which case this is just a dtype cast."""
    import numpy as np

    scale, zero_point = detail["quantization"]
    if scale == 0:
        return x.astype(detail["dtype"])
    info = np.iinfo(detail["dtype"])
    q = np.round(x / scale + zero_point)
    return np.clip(q, info.min, info.max).astype(detail["dtype"])


def _dequantize_from_tensor(x, detail: dict):
    """Inverse of _quantize_for_tensor: int tensor -> float32."""
    scale, zero_point = detail["quantization"]
    if scale == 0:
        return x.astype("float32")
    return (x.astype("float32") - zero_point) * scale


def verify_lite(
    model: DenoiseNet, example: torch.Tensor, outputs: list[Path], quantize: str,
    args: argparse.Namespace,
) -> None:
    from ai_edge_litert.interpreter import Interpreter

    (tflite_path,) = outputs
    ref_model = ResidualDenoiseNet(model) if quantize == "int8" else model

    if quantize == "int8":
        # The residual branch's useful output range on real images is tiny
        # (observed ~+-0.13) compared to what it produces on the random
        # tracing example (observed ~+-0.78, since DenoiseNet was never
        # trained on uniform noise) -- int8's activation ranges are
        # calibrated from real data, so verifying against random noise wildly
        # overstates the quantization error. Swap in one real calibration
        # image instead, purely for this comparison.
        packed_h, packed_w = example.shape[-2:]
        example = _build_calibration_examples(
            packed_w * 4, packed_h * 4, args.calib_dataset, 1,
        )[0]

    with torch.no_grad():
        ref = ref_model(example)[0].numpy()  # (16, packed_h, packed_w)

    example_nhwc = example.permute(0, 2, 3, 1).contiguous().numpy()

    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    # int8 export has int8 input/output tensors (see export_lite's
    # docstring) -- quantize the float example going in and dequantize the
    # int8 result coming out using each tensor's own stored scale/zero_point,
    # rather than a naive dtype cast (which would just truncate).
    interp.set_tensor(in_det["index"], _quantize_for_tensor(example_nhwc, in_det))
    interp.invoke()
    out = interp.get_tensor(out_det["index"])[0]  # (packed_h, packed_w, 16), NHWC
    out = _dequantize_from_tensor(out, out_det)
    out = out.transpose(2, 0, 1)  # -> (16, packed_h, packed_w) to match ref
    _report_diff(out, ref, example, quantize, engine="LiteRT")


def export_onnx(
    model: DenoiseNet, example: torch.Tensor, out_prefix: Path,
    quantize: str, _keep_intermediate: bool, _args: argparse.Namespace,
) -> list[Path]:
    """Export to ONNX via torch.onnx.export, cleaned up with onnxsim. Returns [onnx_path].

    onnxsim folds constants and removes dead nodes left behind by the export
    (pnnx does the equivalent automatically for the ncnn path); no intermediate
    files are produced, so _keep_intermediate is unused, as for --format lite.
    """
    import onnx
    from onnxsim import simplify

    onnx_path = out_prefix.with_suffix(".onnx")

    export_model, export_example = model, example
    if quantize == "fp16":
        # .half() mutates a module in place -- copy first so the fp32 `model`
        # passed in (and reused for --verify's reference pass) is untouched.
        export_model = copy.deepcopy(model).half()
        export_example = example.half()

    torch.onnx.export(
        export_model, (export_example,), str(onnx_path),
        input_names=["input"], output_names=["output"],
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    onnx_model, ok = simplify(onnx_model)
    if not ok:
        print("  WARNING: onnxsim could not verify the simplified graph; keeping it anyway.")
    onnx.save(onnx_model, str(onnx_path))

    if not onnx_path.exists():
        sys.exit("error: conversion did not produce the expected .onnx file")

    onnx_outputs = [onnx_path]
    data_path = Path(str(onnx_path) + ".data")  # external-weights companion file, if any
    if data_path.exists():
        onnx_outputs.append(data_path)
    return onnx_outputs


def verify_onnx(
    model: DenoiseNet, example: torch.Tensor, outputs: list[Path], quantize: str,
    _args: argparse.Namespace,
) -> None:
    import numpy as np
    import onnxruntime as ort

    onnx_path = outputs[0]
    with torch.no_grad():
        ref = model(example)[0].numpy()  # full-fp32 PyTorch reference, as for the other formats

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_dtype = np.float16 if quantize == "fp16" else np.float32
    out = sess.run(None, {in_name: example.numpy().astype(in_dtype)})[0][0]
    _report_diff(out.astype(np.float32), ref, example, quantize, engine="ONNX Runtime")


def export_openvino(
    model: DenoiseNet, example: torch.Tensor, out_prefix: Path,
    quantize: str, _keep_intermediate: bool, _args: argparse.Namespace,
) -> list[Path]:
    """Export to OpenVINO IR (.xml/.bin) via openvino.convert_model. Returns [xml_path, bin_path].

    convert_model always produces fp32 weights; save_model's compress_to_fp16
    does the fp16 packing as a separate step (OpenVINO's equivalent of pnnx's
    fp16=True for ncnn). No intermediate files are produced, so
    _keep_intermediate is unused, as for --format lite/onnx.
    """
    import openvino as ov  # imported lazily so --help works without it installed

    xml_path = out_prefix.with_suffix(".xml")
    bin_path = out_prefix.with_suffix(".bin")

    ov_model = ov.convert_model(model, example_input=example)
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=(quantize == "fp16"))

    if not (xml_path.exists() and bin_path.exists()):
        sys.exit("error: conversion did not produce the expected .xml/.bin files")
    return [xml_path, bin_path]


def verify_openvino(
    model: DenoiseNet, example: torch.Tensor, outputs: list[Path], quantize: str,
    _args: argparse.Namespace,
) -> None:
    import openvino as ov

    xml_path = outputs[0]
    with torch.no_grad():
        ref = model(example)[0].numpy()  # (16, packed_h, packed_w)

    core = ov.Core()
    compiled = core.compile_model(str(xml_path), "CPU")
    req = compiled.create_infer_request()
    req.infer({0: example.numpy()})
    out = req.get_output_tensor(0).data[0]
    _report_diff(out, ref, example, quantize, engine="OpenVINO")


def _report_diff(out, ref, example: torch.Tensor, quantize: str, engine: str) -> None:
    import numpy as np

    diff = np.abs(out - ref)
    rel = diff / (np.abs(ref) + 1e-6)
    print(f"  input shape:   {tuple(example.shape)}")
    print(f"  max abs diff:  {diff.max():.6g}")
    print(f"  mean abs diff: {diff.mean():.6g}")
    print(f"  mean rel diff: {rel.mean():.6g}")

    if quantize == "int8":
        # int8 has no fixed absolute tolerance that makes sense across
        # differently-scaled outputs (ref here is DenoiseNet's residual
        # branch alone, not the full [0, 1]-ish output) -- use a tolerance
        # relative to the reference's own peak magnitude instead.
        tol = 0.05 * (np.abs(ref).max() + 1e-6)
    else:
        tol = 5e-3 if quantize == "fp16" else 1e-4
    if diff.max() > tol:
        print(f"  WARNING: max diff exceeds the expected tolerance ({tol:g}) for {quantize}")
    else:
        print(f"  OK: {engine} output matches PyTorch within tolerance ({tol:g}).")


FORMATS = {
    "ncnn": (export_ncnn, verify_ncnn),
    "lite": (export_lite, verify_lite),
    "onnx": (export_onnx, verify_onnx),
    "openvino": (export_openvino, verify_openvino),
}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t_start = time.time()

    if not args.pth_file.exists():
        sys.exit(f"error: checkpoint not found: {args.pth_file}")

    print(f"Checkpoint:    {args.pth_file}")
    print(f"Target format: {args.format}")
    print(f"Quantize:      {args.quantize}")
    print(f"Requested sensor size (WxH): {args.width} x {args.height}")

    model = load_model(args.pth_file)
    print(
        f"Model config:  ch={model.ch}  scaling_factors={model.scaling_factors}  "
        f"num_encoders={model.num_encoders}"
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    _, _, bayer_w, bayer_h = compute_bayer_size(model, args.width, args.height)
    packed_w, packed_h = pack_size(bayer_w, bayer_h)
    print(f"Network input tensor shape: (1, 16, {packed_h}, {packed_w})")
    # Tag outputs with the actual network input tensor shape (post-padding),
    # not the requested/nominal image size -- so the filename always
    # reflects what the network really ran on.
    tensor_shape = f"16x{packed_h}x{packed_w}"

    output_dir = args.output_dir or Path(args.format)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or args.pth_file.stem
    base_name = args.output_name or f"{output_stem}_{tensor_shape}_{args.quantize}"
    out_prefix = output_dir / base_name

    torch.manual_seed(args.seed)
    example = torch.rand(1, 16, packed_h, packed_w)

    export_fn, verify_fn = FORMATS[args.format]

    print(f"Tracing model and exporting to {args.format} ...")
    t_export = time.time()
    outputs = export_fn(model, example, out_prefix, args.quantize, args.keep_intermediate, args)
    print(f"Export finished in {time.time() - t_export:.1f}s")

    for path in outputs:
        size = path.stat().st_size
        size_str = f"{size:,} bytes" if size < 1e6 else f"{size / 1e6:.2f} MB"
        print(f"Wrote {path} ({size_str})")

    if args.verify:
        print("Verifying exported model against PyTorch ...")
        verify_fn(model, example, outputs, args.quantize, args)

    print(f"Done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
