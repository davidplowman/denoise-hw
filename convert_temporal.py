#!/usr/bin/env python3
"""Convert a temporal (recurrent) DenoiseNet checkpoint to LiteRT (TFLite).

Only feed_image + blend_output models without hidden state are supported.
Just the network's core is exported (see DenoiseNet.core), for both fp32 and
int8, so the two share one interface:

    inputs:  x     (1, H/4, W/4, 16)  sqrt of the noisy frame, packed and normalised
             prev  (1, H/4, W/4, 16)  the previous out (zeros on frame 0)
    outputs: r     (1, H/4, W/4, 16)  the residual
             a     (1, H/4, W/4, 1)   the alpha logit

all NHWC, and all in the network's square-rooted space (see
denoisenet_temporal.to_net). The caller square-roots each noisy frame (a
lookup on the raw values) and completes the frame in float (see host_step):

    s   = clamp(x + r, 0, 1)
    out = s                                  on the first frame
    out = prev + sigmoid(a) * (s - prev)     on every later frame

keeping out, in float and still square-rooted, as the next frame's prev,
and passing on out * out, the linear denoised frame. Keeping the blend and the
fed-back image out of the network matters most for int8: in static areas
alpha is small and the output is mostly prev, so an int8 output would get
stuck on 8-bit levels, with rounding errors compounding frame after frame.

For int8, x and prev are quantised with the fixed scale 1/255, zero_point
-128 (they're [0, 1] images), and r and a have their own calibrated scales;
the caller quantises the inputs and dequantises the outputs itself. The
activation ranges are calibrated by running the float model closed-loop over
real clips with synthetic noise, so prev is the model's own output, as it
will be on the device.

As for convert.py, the Bayer split, 2x2 packing and white_level
normalisation are left to the caller.

Example:
    python convert_temporal.py checkpoints/run_32_2e11_g32-128_d50_t10_blend_c0.25_1.pth --verify
    python convert_temporal.py checkpoints/run_32_2e11_g32-128_d50_t10_blend_c0.25_1.pth --quantize int8 --verify
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import synthetic_noise
from convert import (
    _dequantize_from_tensor, _override_output_scale, _quantize_for_tensor,
    compute_bayer_size, pack_size,
)
import qat_temporal
from dataset_temporal import frame_paths
from denoisenet_temporal import DenoiseNet, TemporalCoreNet, from_net, to_net
from dng import DNG
from test_temporal import _load_model, _pack

_SENSOR_NAME = "imx662"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a temporal DenoiseNet checkpoint to LiteRT (TFLite).",
    )
    parser.add_argument("pth_file", type=Path, help="Trained checkpoint to convert (.pth).")
    parser.add_argument("--quantize", choices=["fp32", "int8"], default="fp32",
                        help="Precision of the export (default: %(default)s).")
    parser.add_argument("--width", type=int, default=1936,
                        help="Sensor output width in pixels (default: %(default)s).")
    parser.add_argument("--height", type=int, default=1096,
                        help="Sensor output height in pixels (default: %(default)s).")
    parser.add_argument("--dataset", default="/opt/datasets/IMX662_temporal", metavar="ROOT",
                        help="Root of the temporal clips used to calibrate and verify "
                             "(default: %(default)s).")
    parser.add_argument("--calib-clips", nargs="+",
                        default=["office_static_0", "study_global_1", "hand_local_0"],
                        help="Clips to calibrate --quantize int8 with, ideally a mix of "
                             "static and moving content (default: %(default)s).")
    parser.add_argument("--calib-frames", type=int, default=8,
                        help="Frames of each clip to calibrate with, from frame 0 "
                             "(default: %(default)s).")
    parser.add_argument("--calib-gains", nargs="+", type=float, default=[32.0, 64.0, 128.0],
                        help="Synthetic noise gains to calibrate with (default: %(default)s).")
    parser.add_argument("--output-scale", type=float, default=None, metavar="SCALE",
                        help="--quantize int8 only: overwrite the residual output's "
                             "quantization scale with this value, leaving its zero_point "
                             "as calibrated (default: as calibrated).")
    parser.add_argument("--verify", action="store_true",
                        help="Run the export closed-loop over --verify-clip, next to "
                             "PyTorch, and compare them.")
    parser.add_argument("--verify-clip", default="cabinet_D65_100_local_0",
                        help="Clip for --verify (default: %(default)s).")
    parser.add_argument("--verify-gain", type=float, default=64.0,
                        help="Synthetic noise gain for --verify (default: %(default)s).")
    parser.add_argument("--summary-json", type=Path, default=None, metavar="PATH",
                        help="with --verify, also write its closed-loop mean PSNRs (linear, "
                             "packed space) to this JSON file")
    parser.add_argument("--verify-frames", type=int, default=30,
                        help="Frames to run for --verify (default: %(default)s).")
    parser.add_argument("--output-dir", type=Path, default=Path("lite"),
                        help="Directory for the .tflite file (default: %(default)s).")
    parser.add_argument("--output-name", default=None,
                        help="Base filename, without extension (default: "
                             "'<output stem>_16x<h>x<w>_<quantize>').")
    parser.add_argument("--output-stem", default=None,
                        help="Replaces the checkpoint stem in the default --output-name.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the synthetic noise and defect maps (default: %(default)s).")
    return parser.parse_args(argv)


def host_step(x: np.ndarray, r: np.ndarray, alpha_logit: np.ndarray,
              prev: np.ndarray | None) -> np.ndarray:
    """Complete one frame from the core's outputs, in float -- the reference
    for what the device's own code must do. x, r: (16, h, w); alpha_logit:
    (1, h, w); prev: the previous frame's result, or None on the first frame.
    x, prev and the result are all in the network's (sqrt) space; the linear
    output is the result squared.
    """
    s = np.clip(x + r, 0.0, 1.0)
    if prev is None:
        return s
    alpha = 1.0 / (1.0 + np.exp(-alpha_logit))
    return prev + alpha * (s - prev)


def _clip_frames(clip_dir: Path, gain: float, count: int, seed: int,
                 width: int, height: int):
    """Yield (x, gt) for the first count frames of a clip, both (1, 16, h, w)
    packed and normalised, with synthetic noise at gain made as
    test_temporal.py does. gt is on the same black level as x (the target
    gain's), as the network's output is."""
    paths = frame_paths(clip_dir)[:count]
    if not paths:
        sys.exit(f"error: no frames found in {clip_dir}")
    rng = np.random.default_rng(seed)
    target_black = synthetic_noise.black_level(gain)
    defects = None
    for dp in paths:
        d = DNG(str(dp), sensor_name=_SENSOR_NAME)
        if d.bayer.shape[0] < height or d.bayer.shape[1] < width:
            sys.exit(f"error: {dp} is {d.bayer.shape[1]}x{d.bayer.shape[0]}, "
                     f"smaller than the {width}x{height} export")
        bayer = d.bayer[:height, :width]
        if defects is None:
            defects = synthetic_noise.make_defect_map(shape=bayer.shape, seed=seed)
        pixel = bayer.astype(np.float32) - d.black_level + target_black
        noisy = synthetic_noise.add_noise(pixel, gain, rng, defects)
        gt = np.clip(pixel, 0, d.white_level)
        yield _pack(noisy, d.white_level), _pack(gt, d.white_level)


def _calibration_pairs(model: DenoiseNet, args: argparse.Namespace,
                       width: int, height: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(x, prev) pairs from running the float model closed-loop over the
    calibration clips, so prev is the model's own output; frame 0's prev is
    zeros, as on the device. Both are in the network's (sqrt) space, as the
    core sees them."""
    pairs = []
    for clip in args.calib_clips:
        for gain in args.calib_gains:
            prev = None
            for x, _ in _clip_frames(Path(args.dataset) / clip, gain, args.calib_frames,
                                     args.seed, width, height):
                pairs.append((to_net(x), to_net(prev) if prev is not None
                              else torch.zeros_like(x)))
                with torch.no_grad():
                    prev, _ = model(x, prev)
    return pairs


def _nhwc(t: torch.Tensor) -> torch.Tensor:
    return t.permute(0, 2, 3, 1).contiguous()


def _quantize_int8(core: TemporalCoreNet, pairs: list, output_scale: float | None):
    """PT2E static quantisation of the core, calibrated on pairs. As
    convert.py's _quantize_int8; returns (quantized, quantizer), the
    quantizer being needed again by litert_torch.convert."""
    from torchao.quantization.pt2e import allow_exported_model_train_eval, quantize_pt2e

    exported = qat_temporal.export_core(core.net, tuple(_nhwc(t) for t in pairs[0]))
    quantizer = qat_temporal.make_quantizer(is_qat=False, gate_clamp=core.net.gate_clamp)
    prepared = quantize_pt2e.prepare_pt2e(exported, quantizer)
    with torch.no_grad():
        for x, prev in pairs:
            prepared(_nhwc(x), _nhwc(prev))
    print(f"  Calibrated with {len(pairs)} (x, prev) pairs.")

    if output_scale is not None:
        _override_output_scale(prepared, output_scale, output_index=0)
        print(f"  Overriding the residual output's quantization scale with {output_scale:g}.")

    quantized = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    allow_exported_model_train_eval(quantized)  # see convert.py's _quantize_int8
    return quantized, quantizer


def _qat_int8(model: DenoiseNet, ckpt: dict, size: tuple[int, int],
              output_scale: float | None):
    """Convert a QAT checkpoint's core at the export size, with its learned
    int8 ranges (no calibration). Returns (quantized, quantizer), as
    _quantize_int8."""
    from torchao.quantization.pt2e import allow_exported_model_train_eval, quantize_pt2e

    prepared = qat_temporal.load_qat_net(copy.deepcopy(model), ckpt, size=size).core
    if output_scale is not None:
        _override_output_scale(prepared, output_scale, output_index=0)
        print(f"  Overriding the residual output's quantization scale with {output_scale:g}.")
    quantized = quantize_pt2e.convert_pt2e(prepared, fold_quantize=False)
    allow_exported_model_train_eval(quantized)
    return quantized, qat_temporal.make_quantizer(is_qat=True, gate_clamp=model.gate_clamp)


def export_lite(core: TemporalCoreNet, example: tuple[torch.Tensor, torch.Tensor],
                tflite_path: Path, quantize: str, pairs: list | None,
                output_scale: float | None, qat_ckpt: dict | None = None) -> None:
    """Export the core to .tflite; it's channels-last already (see
    TemporalCoreNet), as is example. int8 uses qat_ckpt's learned ranges if
    given, else calibrates on pairs."""
    import litert_torch

    model, quant_config = core, None
    if quantize == "int8":
        from litert_torch.quantize import quant_config as qcfg

        if qat_ckpt is not None:
            model, quantizer = _qat_int8(core.net, qat_ckpt, tuple(example[0].shape[1:3]),
                                         output_scale)
        else:
            model, quantizer = _quantize_int8(core, pairs, output_scale)
        quant_config = qcfg.QuantConfig(pt2e_quantizer=quantizer)

    model.eval()  # (returns None for the quantised GraphModules, so not chained)
    edge_model = litert_torch.convert(model, example, quant_config=quant_config)
    edge_model.export(str(tflite_path))
    if not tflite_path.exists():
        sys.exit("error: conversion did not produce the expected .tflite file")


class LiteCore:
    """Runs an exported core: float NCHW (x, prev) in, float NCHW (r, a) out,
    handling the NHWC layout, int8 (de)quantisation and I/O tensor order.

    The outputs are told apart by their channel counts (16 and 1). The two
    inputs can't be, so they're taken in the order the interpreter lists
    them; swap_inputs reverses that, and --verify checks which is right.
    """

    def __init__(self, tflite_path: Path, swap_inputs: bool = False) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.interp = Interpreter(model_path=str(tflite_path),
                                  num_threads=max(1, (os.cpu_count() or 2) // 2))
        self.interp.allocate_tensors()
        self.inputs = self.interp.get_input_details()
        if swap_inputs:
            self.inputs = self.inputs[::-1]
        outs = self.interp.get_output_details()
        self.out_r = next(o for o in outs if o["shape"][-1] == 16)
        self.out_a = next(o for o in outs if o["shape"][-1] == 1)

    def __call__(self, x: np.ndarray, prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        for det, t in zip(self.inputs, (x, prev)):
            nhwc = np.ascontiguousarray(t.transpose(1, 2, 0)[None])
            self.interp.set_tensor(det["index"], _quantize_for_tensor(nhwc, det))
        self.interp.invoke()
        r, a = (_dequantize_from_tensor(self.interp.get_tensor(det["index"])[0], det)
                .transpose(2, 0, 1) for det in (self.out_r, self.out_a))
        return r, a


def describe(tflite_path: Path, quantize: str) -> None:
    """Print the I/O tensors and op counts (of the graph itself, without the
    CPU delegate), and for int8 flag any op with a float tensor -- everything
    should be int8 or int32."""
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    interp = Interpreter(
        model_path=str(tflite_path),
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interp.allocate_tensors()
    for kind, dets in (("input", interp.get_input_details()),
                       ("output", interp.get_output_details())):
        for d in dets:
            scale, zp = d["quantization"]
            q = f"  scale {scale:.6g} zero_point {zp}" if scale else ""
            print(f"  {kind:6s} {d['name']:40s} {tuple(d['shape'])}  {np.dtype(d['dtype']).name}{q}")

    dtypes = {t["index"]: np.dtype(t["dtype"]).name for t in interp.get_tensor_details()}
    ops = interp._get_ops_details()
    counts = Counter(op["op_name"] for op in ops)
    print("  ops: " + ", ".join(f"{name} x{n}" for name, n in sorted(counts.items())))
    float_ops = Counter(
        op["op_name"] for op in ops
        if any(dtypes.get(i) == "float32" for i in list(op["inputs"]) + list(op["outputs"]) if i >= 0)
    )
    if quantize == "int8" and float_ops:
        print("  ops with float32 tensors: "
              + ", ".join(f"{name} x{n}" for name, n in sorted(float_ops.items())))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))


def verify(model: DenoiseNet | qat_temporal.QATTemporalNet, tflite_path: Path, quantize: str,
           args: argparse.Namespace, width: int, height: int) -> None:
    """Run --verify-clip closed-loop both through PyTorch and through the
    export plus host_step, each feeding back its own output, and compare.
    model is the PyTorch reference: the float model, or for a QAT
    checkpoint's int8 export, its simulated-int8 QATTemporalNet.

    Every frame also gets a single-step check: the export and the PyTorch
    core are given the same (x, prev) -- PyTorch's own -- and their r and a
    compared directly. The export side works in the network's (sqrt) space
    throughout, as the device will; PSNRs are of the linear outputs.
    """
    lite = LiteCore(tflite_path)
    core = model.core_nchw if isinstance(model, qat_temporal.QATTemporalNet) else model.core
    tol_rel = 0.05 if quantize == "int8" else 1e-4  # as convert.py's _report_diff

    prev_pt = prev_lt = None
    max_dr = max_da = peak_r = peak_a = 0.0
    psnr_pt, psnr_lt, gaps = [], [], []
    print(f"  {'frame':>5s}  {'PSNR torch':>10s}  {'PSNR lite':>10s}  {'mean |diff|':>11s}")
    frames = _clip_frames(Path(args.dataset) / args.verify_clip, args.verify_gain,
                          args.verify_frames, args.seed, width, height)
    for t, (x, gt) in enumerate(frames):
        x_np, gt_np = to_net(x)[0].numpy(), gt[0].numpy()
        prev_in = to_net(prev_pt) if prev_pt is not None else torch.zeros_like(x)

        # single step, on PyTorch's own (x, prev)
        with torch.no_grad():
            r_ref, a_ref = (v[0].numpy() for v in core(to_net(x), prev_in))
        r_lt, a_lt = lite(x_np, prev_in[0].numpy())
        max_dr = max(max_dr, float(np.abs(r_lt - r_ref).max()))
        max_da = max(max_da, float(np.abs(a_lt - a_ref).max()))
        peak_r = max(peak_r, float(np.abs(r_ref).max()))
        peak_a = max(peak_a, float(np.abs(a_ref).max()))

        if t == 1:
            # x and prev have the same shape, so check they're in the right order
            r_sw, a_sw = LiteCore(tflite_path, swap_inputs=True)(x_np, prev_in[0].numpy())
            if np.abs(r_sw - r_ref).mean() < np.abs(r_lt - r_ref).mean():
                sys.exit("error: the export's inputs appear to be in the order (prev, x), "
                         "not (x, prev)")

        # closed loop: each side feeds back its own output
        with torch.no_grad():
            prev_pt, _ = model(x, prev_pt)
        if t > 0:
            r_lt, a_lt = lite(x_np, prev_lt)
        prev_lt = host_step(x_np, r_lt, a_lt, prev_lt)

        out_pt, out_lt = prev_pt[0].numpy(), from_net(prev_lt)
        psnr_pt.append(_psnr(out_pt, gt_np))
        psnr_lt.append(_psnr(out_lt, gt_np))
        gaps.append(float(np.abs(out_lt - out_pt).mean()))
        print(f"  {t:5d}  {psnr_pt[-1]:10.3f}  {psnr_lt[-1]:10.3f}  {gaps[-1]:11.3g}")

    print(f"  mean PSNR: torch {np.mean(psnr_pt):.3f} dB, lite {np.mean(psnr_lt):.3f} dB "
          f"({np.mean(psnr_lt) - np.mean(psnr_pt):+.3f} dB)")
    n = max(1, len(gaps) // 3)
    print(f"  mean |diff|: first third {np.mean(gaps[:n]):.3g}, last third {np.mean(gaps[-n:]):.3g} "
          "(growing would mean drift)")
    ok = True
    for name, diff, peak in (("residual", max_dr, peak_r), ("alpha logit", max_da, peak_a)):
        tol = tol_rel * max(peak, 1.0) if quantize == "fp32" else tol_rel * peak
        status = "OK" if diff <= tol else "WARNING: exceeds"
        ok &= diff <= tol
        print(f"  single step {name}: max |diff| {diff:.3g} (peak {peak:.3g})  "
              f"{status} tolerance {tol:.3g}")
    if ok:
        print("  OK: single-step outputs match PyTorch within tolerance.")
    if args.summary_json is not None:
        args.summary_json.write_text(json.dumps({
            "tflite": str(tflite_path), "quantize": quantize, "clip": args.verify_clip,
            "gain": args.verify_gain, "frames": len(psnr_pt),
            "psnr_torch": float(np.mean(psnr_pt)), "psnr_lite": float(np.mean(psnr_lt)),
            "single_step_ok": bool(ok)}, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t_start = time.time()
    if not args.pth_file.exists():
        sys.exit(f"error: checkpoint not found: {args.pth_file}")

    model = _load_model(str(args.pth_file))
    if model.state_ch or not model.feed_image or not model.blend_output:
        sys.exit("error: only feed_image + blend_output models without hidden state are "
                 f"supported (this one: feed_image={model.feed_image}, "
                 f"blend_output={model.blend_output}, state_ch={model.state_ch})")
    print(f"Checkpoint: {args.pth_file}")
    print(f"Quantize:   {args.quantize}")
    print(f"Model:      ch={model.ch}  scaling_factors={model.scaling_factors}  "
          f"num_encoders={model.num_encoders}  "
          f"parameters={sum(p.numel() for p in model.parameters()):,}")

    width, height, bayer_w, bayer_h = compute_bayer_size(model, args.width, args.height)
    packed_w, packed_h = pack_size(bayer_w, bayer_h)
    print(f"Network input tensors: x, prev (1, 16, {packed_h}, {packed_w})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem or args.pth_file.stem
    name = args.output_name or f"{stem}_16x{packed_h}x{packed_w}_{args.quantize}"
    tflite_path = args.output_dir / f"{name}.tflite"

    ckpt = torch.load(args.pth_file, map_location="cpu")
    qat_ckpt = ckpt if isinstance(ckpt, dict) and ckpt.get("qat") else None
    if qat_ckpt is not None:
        print("QAT checkpoint: " + ("using its learned int8 ranges (no calibration)"
                                    if args.quantize == "int8" else "exporting its float weights"))

    core = TemporalCoreNet(model).eval()
    pairs = None
    if args.quantize == "int8" and qat_ckpt is None:
        print(f"Calibrating from {len(args.calib_clips)} clip(s) x {len(args.calib_gains)} "
              f"gain(s) x {args.calib_frames} frame(s) ...")
        pairs = _calibration_pairs(model, args, width, height)
        example = tuple(_nhwc(t) for t in pairs[0])
    else:
        torch.manual_seed(args.seed)
        example = (torch.rand(1, packed_h, packed_w, 16), torch.rand(1, packed_h, packed_w, 16))

    print("Exporting ...")
    t_export = time.time()
    export_lite(core, example, tflite_path, args.quantize, pairs, args.output_scale,
                qat_ckpt if args.quantize == "int8" else None)
    print(f"Export finished in {time.time() - t_export:.1f}s")
    print(f"Wrote {tflite_path} ({tflite_path.stat().st_size:,} bytes)")
    describe(tflite_path, args.quantize)

    if args.verify:
        print(f"Verifying closed-loop on {args.verify_clip} at gain {args.verify_gain:g} ...")
        ref = model
        if qat_ckpt is not None and args.quantize == "int8":
            ref = qat_temporal.load_qat_net(copy.deepcopy(model), qat_ckpt)
            print("  (reference: the simulated-int8 QAT model)")
        verify(ref, tflite_path, args.quantize, args, width, height)

    print(f"Done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
