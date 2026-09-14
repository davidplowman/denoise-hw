#!/usr/bin/env python3
"""Convert a DenoiseNet .pth checkpoint to NCNN, LiteRT (TFLite), or ONNX model files.

Only the network itself (network.DenoiseNet) is exported. The Bayer split/
merge and white_level normalisation done around it in denoise_network.py are
NOT part of the exported graph and must be reproduced by whatever code calls
the converted model.

Example:
    python convert.py checkpoints/imx662_32_2221e1221_07908.pth --verify
    python convert.py checkpoints/imx662_32_2221e1221_07908.pth --format lite --verify
    python convert.py checkpoints/imx662_32_2221e1221_07908.pth --format onnx --verify
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

from network import DenoiseNet


def round_up(value: int, align: int) -> int:
    """Round value up to the next multiple of align."""
    return ((value + align - 1) // align) * align


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
        "--format", choices=["ncnn", "lite", "onnx"], default="ncnn",
        help="Target format (default: %(default)s). 'ncnn' writes a .param/.bin pair "
             "(via pnnx); 'lite' writes a single .tflite file for ai-edge-litert "
             "(via litert_torch); 'onnx' writes a single .onnx file (via "
             "torch.onnx.export, cleaned up with onnxsim).",
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
        "--quantize", choices=["fp16", "fp32"], default="fp16",
        help="Precision of the exported weights (default: %(default)s).",
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
             "(default: '<output stem>_<width>x<height>_<quantize>').",
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
    quantize: str, keep_intermediate: bool,
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


def verify_ncnn(model: DenoiseNet, example: torch.Tensor, outputs: list[Path], quantize: str) -> None:
    import ncnn

    param_path, bin_path = outputs
    with torch.no_grad():
        ref = model(example)[0].numpy()  # (4, bayer_h, bayer_w)

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


def _lite_quant_config(quantize: str):
    """Build the litert_torch QuantConfig for the requested precision."""
    if quantize != "fp16":
        return None
    # Weight-only fp16 casting (activations/compute stay fp32) - the LiteRT
    # equivalent of pnnx's fp16=True for ncnn.
    from litert_torch.generative.quantize.quant_recipe import GenerativeQuantRecipe
    from litert_torch.generative.quantize.quant_recipe_utils import create_layer_quant_fp16
    from litert_torch.quantize.quant_config import QuantConfig

    return QuantConfig(generative_recipe=GenerativeQuantRecipe(default=create_layer_quant_fp16()))


def export_lite(
    model: DenoiseNet, example: torch.Tensor, out_prefix: Path,
    quantize: str, _keep_intermediate: bool,
) -> list[Path]:
    """Export to a LiteRT (.tflite) flatbuffer via litert_torch. Returns [tflite_path].

    litert_torch produces no intermediate files to clean up, so _keep_intermediate
    is accepted (for a uniform export-function signature) but unused.
    """
    import litert_torch  # imported lazily so --help works without it installed

    tflite_path = out_prefix.with_suffix(".tflite")
    edge_model = litert_torch.convert(model, (example,), quant_config=_lite_quant_config(quantize))
    edge_model.export(str(tflite_path))

    if not tflite_path.exists():
        sys.exit("error: conversion did not produce the expected .tflite file")
    return [tflite_path]


def verify_lite(model: DenoiseNet, example: torch.Tensor, outputs: list[Path], quantize: str) -> None:
    from ai_edge_litert.interpreter import Interpreter

    (tflite_path,) = outputs
    with torch.no_grad():
        ref = model(example)[0].numpy()  # (4, bayer_h, bayer_w)

    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    interp.set_tensor(in_det["index"], example.numpy().astype(in_det["dtype"]))
    interp.invoke()
    out = interp.get_tensor(out_det["index"])[0]
    _report_diff(out, ref, example, quantize, engine="LiteRT")


def export_onnx(
    model: DenoiseNet, example: torch.Tensor, out_prefix: Path,
    quantize: str, _keep_intermediate: bool,
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


def verify_onnx(model: DenoiseNet, example: torch.Tensor, outputs: list[Path], quantize: str) -> None:
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


def _report_diff(out, ref, example: torch.Tensor, quantize: str, engine: str) -> None:
    import numpy as np

    diff = np.abs(out - ref)
    rel = diff / (np.abs(ref) + 1e-6)
    print(f"  input shape:   {tuple(example.shape)}")
    print(f"  max abs diff:  {diff.max():.6g}")
    print(f"  mean abs diff: {diff.mean():.6g}")
    print(f"  mean rel diff: {rel.mean():.6g}")

    tol = 5e-3 if quantize == "fp16" else 1e-4
    if diff.max() > tol:
        print(f"  WARNING: max diff exceeds the expected tolerance ({tol:g}) for {quantize}")
    else:
        print(f"  OK: {engine} output matches PyTorch within tolerance ({tol:g}).")


FORMATS = {
    "ncnn": (export_ncnn, verify_ncnn),
    "lite": (export_lite, verify_lite),
    "onnx": (export_onnx, verify_onnx),
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

    # Each encoder level with a scaling factor > 1 halves the spatial size
    # (network.py always uses a stride-2 Conv2d there). The Bayer split itself
    # halves width/height once more before that. Sensor width/height must
    # therefore be a multiple of 2 * (2 ** num_downsampling_levels).
    num_downsamples = sum(1 for s in model.scaling_factors if s > 1)
    align = 2 * (2 ** num_downsamples)

    width = round_up(args.width, align)
    height = round_up(args.height, align)
    if width != args.width:
        print(f"  Rounding width up:  {args.width} -> {width} (must be a multiple of {align} px)")
    if height != args.height:
        print(f"  Rounding height up: {args.height} -> {height} (must be a multiple of {align} px)")

    bayer_w, bayer_h = width // 2, height // 2
    print(f"Network input tensor shape: (1, 4, {bayer_h}, {bayer_w})")

    output_dir = args.output_dir or Path(args.format)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or args.pth_file.stem
    base_name = args.output_name or f"{output_stem}_{width}x{height}_{args.quantize}"
    out_prefix = output_dir / base_name

    torch.manual_seed(args.seed)
    example = torch.rand(1, 4, bayer_h, bayer_w)

    export_fn, verify_fn = FORMATS[args.format]

    print(f"Tracing model and exporting to {args.format} ...")
    t_export = time.time()
    outputs = export_fn(model, example, out_prefix, args.quantize, args.keep_intermediate)
    print(f"Export finished in {time.time() - t_export:.1f}s")

    for path in outputs:
        size = path.stat().st_size
        size_str = f"{size:,} bytes" if size < 1e6 else f"{size / 1e6:.2f} MB"
        print(f"Wrote {path} ({size_str})")

    if args.verify:
        print("Verifying exported model against PyTorch ...")
        verify_fn(model, example, outputs, args.quantize)

    print(f"Done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
