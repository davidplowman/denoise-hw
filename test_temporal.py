"""Human-viewable evaluation of the recurrent (temporal) DenoiseNet on one clip.

Runs a trained checkpoint frame-by-frame over a real clip from the temporal
dataset, synthesizing noise at a chosen gain and threading the previous
frame's output forward exactly as in training. Writes 16-bit gt/noisy/pred
PNGs per frame and prints PSNR/SSIM against ground truth.

With --still-checkpoint, also runs the single-image (non-temporal) DenoiseNet
on exactly the same noisy frames, writing <N>_still.png and reporting its
PSNR/SSIM alongside, for comparison. With --still-int8 it is also run in
simulated int8 (convert.py's int8 export: post-training quantisation of its
residual branch, the skip add in float), as the "still_int8" output, writing
<N>_still_int8.png -- the fair comparison for a QAT checkpoint. Its ranges
are calibrated on a few of the clip's own frames, with noise synthesised at
the test gain from an independent RNG stream (so not the test frames' noise),
or with --still-calib-dataset on real captures, as convert.py does.

With --change-threshold, also reports PSNR over only the pixels where the GT
changed from the previous frame by more than the threshold (measured in the
training loss's white-balanced, gamma-corrected space, max over the packed
channels) -- this is where ghosting from the previous output shows up.

From the second frame on, also reports a flicker measure for each output,
mean |(pred_t - pred_{t-1}) - (gt_t - gt_{t-1})| in the same space (the
training script's --temporal-weight term), and in the summary the raw
frame-to-frame differences mean |pred_t - pred_{t-1}| and |gt_t - gt_{t-1}|,
so the GT's own flicker can be compared with the output's on static clips.
All three are computed after averaging the differences over
--flicker-scale x --flicker-scale raw pixels (same-colour samples), which
suppresses pixel-level noise and measures the more visible blobby shimmer.

For models with a blended output or hidden state, --save-alpha and
--save-state write the per-frame alpha and state maps as greyscale PNGs
(0..1 -> 0..255, nearest-upscaled to full resolution), and print their means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import synthetic_noise
from dataset_temporal import frame_paths
from denoisenet_temporal import DenoiseNet, depth_to_space, space_to_depth
from dng import DNG
from network import DenoiseNet as StillDenoiseNet
from train_temporal import _smooth, _to_wb

_SENSOR_NAME = "imx662"


def _pack(bayer: np.ndarray, white_level: float) -> torch.Tensor:
    """(H, W) raw Bayer -> (1, 16, H/4, W/4) packed, normalised tensor."""
    img = bayer.astype(np.float32)
    ch = np.stack([img[0::2, 0::2], img[0::2, 1::2],
                   img[1::2, 0::2], img[1::2, 1::2]], axis=0)
    x_bayer = torch.from_numpy(ch / white_level).unsqueeze(0).float()
    return space_to_depth(x_bayer, 2)


def _unpack(x: torch.Tensor, white_level: float, shape: tuple[int, int]) -> np.ndarray:
    """(1, 16, H/4, W/4) packed tensor -> uint16 (H, W) raw Bayer, clipped to [0, white_level]."""
    ch_out = depth_to_space(x, 2).squeeze(0).detach().numpy() * white_level
    result = np.empty(shape, dtype=np.float32)
    result[0::2, 0::2] = ch_out[0]
    result[0::2, 1::2] = ch_out[1]
    result[1::2, 0::2] = ch_out[2]
    result[1::2, 1::2] = ch_out[3]
    return np.clip(result, 0, white_level).astype(np.uint16)


def _save_png(rgb: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), rgb[..., ::-1])  # RGB -> BGR for cv2
    print(f"  {path}")


def _load_model(checkpoint: str, model_cls: type = DenoiseNet) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        extra = {}
        if model_cls is DenoiseNet:
            extra = {k: ckpt[k] for k in ("state_ch", "feed_image", "state_gated", "blend_output",
                                          "gate_clamp")
                     if k in ckpt}
        model = model_cls(ch=ckpt.get("ch"), scaling_factors=ckpt.get("scaling_factors"),
                           num_encoders=ckpt.get("num_encoders"), **extra)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = model_cls()
        model.load_state_dict(ckpt)
    model.eval()
    return model


def _quantize_still(still_model: torch.nn.Module, x: torch.Tensor, dng_paths: list[Path],
                    args: argparse.Namespace, defects: np.ndarray) -> torch.nn.Module:
    """still_model's residual branch in simulated int8, exactly as convert.py's
    int8 export quantises it, for inputs of x's shape (see the module
    docstring for the calibration data)."""
    import convert  # imported lazily: it pulls in the conversion tools

    calib = None
    if args.still_calib_dataset is None:
        # independent of the test noise's RNG, so the test frames are untouched
        rng = np.random.default_rng(None if args.seed is None else [args.seed, 1])
        n = min(args.still_calib_count, len(dng_paths))
        picks = [dng_paths[i * len(dng_paths) // n] for i in range(n)]
        calib = []
        for dp in picks:
            dng = DNG(str(dp), sensor_name=_SENSOR_NAME)
            pixel = (dng.bayer.astype(np.float32) - dng.black_level
                     + synthetic_noise.black_level(args.gain))
            noisy = synthetic_noise.add_noise(pixel, args.gain, rng, defects).astype(np.uint16)
            calib.append(_pack(noisy, dng.white_level))
        print(f"Still model int8: calibrating on {n} of the clip's frames "
              f"(fresh noise at gain {args.gain:g})")
    quantized, _ = convert._quantize_int8(
        still_model, x, args.still_calib_dataset, args.still_calib_count, calib_examples=calib)
    return quantized


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the temporal DenoiseNet on one clip.")
    parser.add_argument("clip", help="Name of the clip folder (e.g. office_0)")
    parser.add_argument("--checkpoint", required=True, metavar="PTH")
    parser.add_argument("--still-checkpoint", default=None, metavar="PTH",
                         help="also denoise each frame with this single-image "
                              "(non-temporal) DenoiseNet checkpoint, for comparison")
    parser.add_argument("--still-int8", action="store_true",
                         help="also run the still model in simulated int8, as convert.py's "
                              "int8 export would (needs --still-checkpoint)")
    parser.add_argument("--still-calib-dataset", default=None, metavar="ROOT",
                         help="with --still-int8, calibrate on real captures under ROOT (as "
                              "convert.py --calib-dataset) instead of the clip's own frames")
    parser.add_argument("--still-calib-count", type=int, default=4, metavar="N",
                         help="with --still-int8, number of images to calibrate on "
                              "(default: %(default)s)")
    parser.add_argument("--output", default="output_temporal", metavar="DIR")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--dataset", default="/opt/datasets/IMX662_temporal", metavar="ROOT")
    parser.add_argument("--seed", type=int, default=None,
                         help="seed for the noise/defect-map RNG (default: unseeded)")
    parser.add_argument("--change-threshold", type=float, default=None, metavar="T",
                         help="also report PSNR over pixels where |gt_t - gt_{t-1}| > T "
                              "(WB/gamma space, as in the training loss)")
    parser.add_argument("--flicker-scale", type=int, default=4, metavar="K",
                         help="average frame-to-frame differences over K x K raw pixels "
                              "before measuring flicker (1 or even; 1 = no smoothing)")
    parser.add_argument("--save-masks", action="store_true",
                         help="with --change-threshold, write <N>_mask.png for each frame")
    parser.add_argument("--save-alpha", action="store_true",
                         help="with a blended-output model, write <N>_alpha.png for each frame")
    parser.add_argument("--summary-json", type=Path, default=None, metavar="PATH",
                         help="also write the end-of-run means (per output: PSNR, SSIM, "
                              "changed-region PSNR, flicker) to this JSON file")
    parser.add_argument("--save-state", action="store_true",
                         help="with a hidden-state model, write <N>_state<C>.png for each "
                              "frame and state channel (and print the gate's mean if gated)")
    args = parser.parse_args()
    if args.flicker_scale != 1 and (args.flicker_scale < 2 or args.flicker_scale % 2):
        parser.error("--flicker-scale must be 1 or an even number")
    if args.still_int8 and not args.still_checkpoint:
        parser.error("--still-int8 needs --still-checkpoint")
    k = args.flicker_scale

    clip_dir = Path(args.dataset) / args.clip
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    dng_paths = frame_paths(clip_dir)[args.offset:]
    if not dng_paths:
        parser.error(f"no frames found in {clip_dir} from offset {args.offset}")

    print(f"Clip:   {clip_dir}  ({len(dng_paths)} frames from offset {args.offset})")
    print(f"Output: {out}")

    model = _load_model(args.checkpoint)
    print(f"Model: ch={model.ch}  scaling={model.scaling_factors}  encoders={model.num_encoders}  "
          f"blend={model.blend_output}  feed_image={model.feed_image}  "
          f"state_ch={model.state_ch}{' (gated)' if model.state_gated else ''}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and ckpt.get("qat"):
        import qat_temporal  # imported lazily: it pulls in the conversion tools

        model = qat_temporal.load_qat_net(model, ckpt)
        print("QAT checkpoint: running with simulated int8, as exported")

    still_model = None
    if args.still_checkpoint:
        still_model = _load_model(args.still_checkpoint, StillDenoiseNet)
        print(f"Still model: ch={still_model.ch}  scaling={still_model.scaling_factors}  "
              f"encoders={still_model.num_encoders}")

    rng = np.random.default_rng(args.seed)
    defects = None  # one map, shared across the whole processed sequence
    still_int8 = None  # built on the first frame, once its size is known

    # scores[(output, reference)] = list of per-frame (PSNR, SSIM)
    scores: dict[tuple[str, str], list[tuple[float, float]]] = {}
    # changed[(output, reference)] = [squared error sum, sample count], pooled
    # over the changed-region pixels of all frames
    changed: dict[tuple[str, str], list[float]] = {}
    changed_pixels = [0, 0]  # [masked, total] over frames with a previous frame
    # flicker[output] = per-frame (consistency term, mean |pred_t - pred_{t-1}|)
    flicker: dict[str, list[tuple[float, float]]] = {}
    gt_deltas: list[float] = []  # per-frame mean |gt_t - gt_{t-1}|

    def score(name: str, ref_name: str, ref_rgb: np.ndarray, rgb: np.ndarray,
              mask: np.ndarray | None, flick: float | None) -> None:
        p = psnr(ref_rgb, rgb, data_range=65535)
        s = ssim(ref_rgb, rgb, data_range=65535, channel_axis=2)
        scores.setdefault((name, ref_name), []).append((p, s))
        extra = ""
        if mask is not None:
            acc = changed.setdefault((name, ref_name), [0.0, 0])
            n = int(mask.sum())
            if n:
                err = (ref_rgb[mask].astype(np.float64) - rgb[mask]) ** 2
                acc[0] += err.sum()
                acc[1] += err.size
                extra = f"  changed PSNR {10 * np.log10(65535 ** 2 / err.mean()):8.3f}"
            else:
                extra = "  changed PSNR        —"
            extra += f" ({100 * n / mask.size:5.2f}%)"
        if flick is not None:
            extra += f"  flicker@{k} {flick:.5f}"
        print(f"  frame {num}  PSNR {p:8.3f}  SSIM {s:8.4f}{extra}  ({name} vs {ref_name})")

    channel_gains = None  # (1, 16), from the first frame, as in training
    prev_gt_wb = None
    prev_pred_wb: dict[str, torch.Tensor] = {}
    prev_output = state = None

    def save_map(m: torch.Tensor, path: Path) -> None:
        """(H/4, W/4) map in [0, 1] -> full-resolution 8-bit greyscale PNG."""
        img = np.repeat(np.repeat(m.numpy(), 4, axis=0), 4, axis=1)
        cv2.imwrite(str(path), np.round(np.clip(img, 0, 1) * 255).astype(np.uint8))
        print(f"  {path}")

    for dp in dng_paths:
        num = dp.stem  # "00", "01", ...
        gt_dng = DNG(str(dp), sensor_name=_SENSOR_NAME)

        if defects is None:
            # seeded too, so runs with the same --seed see identical noise
            defects = synthetic_noise.make_defect_map(shape=gt_dng.bayer.shape, seed=args.seed)

        signal = gt_dng.bayer.astype(np.float32) - gt_dng.black_level
        target_black = synthetic_noise.black_level(args.gain)
        pixel = signal + target_black
        noisy_bayer = synthetic_noise.add_noise(pixel, args.gain, rng, defects).astype(np.uint16)

        x = _pack(noisy_bayer, gt_dng.white_level)
        if args.still_int8 and still_int8 is None:
            still_int8 = _quantize_still(still_model, x, dng_paths, args, defects)
        with torch.no_grad():
            pred, state, aux = model(x, prev_output, state, return_aux=True)
        prev_output = pred

        if "alpha" in aux:
            print(f"  frame {num}  mean alpha {aux['alpha'].mean().item():.4f}")
            if args.save_alpha:
                save_map(aux["alpha"][0, 0], out / f"{num}_alpha.png")
        if "z" in aux:
            print(f"  frame {num}  mean z {aux['z'].mean().item():.4f}  "
                  f"(per channel: {' '.join(f'{v:.3f}' for v in aux['z'][0].mean(dim=(1, 2)).tolist())})")
        if state is not None and args.save_state:
            for c in range(state.shape[1]):
                save_map(state[0, c], out / f"{num}_state{c}.png")

        pred_bayer = _unpack(pred, gt_dng.white_level, gt_dng.bayer.shape)

        # noisy_bayer is anchored to the *target gain's* black level, not
        # gt_dng's own captured black level -- subtract it ourselves and
        # tell to_rgb not to subtract anything else (same pattern test.py
        # uses for its own post-processed outputs).
        noisy_signal = np.clip(noisy_bayer.astype(np.float32) - target_black, 0, None).astype(np.uint16)

        gt_rgb    = gt_dng.to_rgb(output_bps=16)
        noisy_rgb = gt_dng.to_rgb(bayer=noisy_signal, subtract_black_level=False, output_bps=16)
        pred_rgb  = gt_dng.to_rgb(bayer=pred_bayer, output_bps=16)

        # GT in the training loss's WB/gamma packed space, for the
        # changed-region mask and the flicker measure
        if channel_gains is None:
            cidx = {'R': 0, 'G': 1, 'B': 2}
            g4 = torch.tensor([gt_dng.wb_gains[cidx[gt_dng.bayer_pattern[k]]] for k in range(4)],
                              dtype=torch.float32)
            channel_gains = g4.repeat_interleave(4).unsqueeze(0)
        gt_wb = _to_wb(_pack(gt_dng.bayer, gt_dng.white_level), channel_gains)
        gt_delta = None if prev_gt_wb is None else gt_wb - prev_gt_wb
        if gt_delta is not None:
            gt_deltas.append(_smooth(gt_delta, k).abs().mean().item())

        mask = None
        if args.change_threshold is not None:
            if gt_delta is not None:
                diff = gt_delta.abs().amax(dim=1)[0].numpy()  # (H/4, W/4)
                mask = np.repeat(np.repeat(diff > args.change_threshold, 4, axis=0), 4, axis=1)
                assert mask.shape == gt_rgb.shape[:2], \
                    f"mask {mask.shape} doesn't match RGB {gt_rgb.shape[:2]}"
                changed_pixels[0] += int(mask.sum())
                changed_pixels[1] += mask.size
                if args.save_masks:
                    path = out / f"{num}_mask.png"
                    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
                    print(f"  {path}")
        prev_gt_wb = gt_wb

        _save_png(gt_rgb,    out / f"{num}_gt.png")
        _save_png(noisy_rgb, out / f"{num}_noisy.png")
        _save_png(pred_rgb,  out / f"{num}_pred.png")

        outputs = [("temporal", pred_rgb, pred)]
        if still_model is not None:
            with torch.no_grad():
                still_pred = still_model(x)
            still_bayer = _unpack(still_pred, gt_dng.white_level, gt_dng.bayer.shape)
            still_rgb = gt_dng.to_rgb(bayer=still_bayer, output_bps=16)
            _save_png(still_rgb, out / f"{num}_still.png")
            outputs.append(("still", still_rgb, still_pred))
        if still_int8 is not None:
            with torch.no_grad():
                still_pred = x + still_int8(x)  # skip add in float, as on the device
            still_bayer = _unpack(still_pred, gt_dng.white_level, gt_dng.bayer.shape)
            still_rgb = gt_dng.to_rgb(bayer=still_bayer, output_bps=16)
            _save_png(still_rgb, out / f"{num}_still_int8.png")
            outputs.append(("still_int8", still_rgb, still_pred))

        for name, rgb, packed in outputs:
            pred_wb = _to_wb(packed, channel_gains)
            flick = None
            if gt_delta is not None:
                pred_delta = _smooth(pred_wb - prev_pred_wb[name], k)
                flick = (pred_delta - _smooth(gt_delta, k)).abs().mean().item()
                flicker.setdefault(name, []).append((flick, pred_delta.abs().mean().item()))
            prev_pred_wb[name] = pred_wb
            score(name, "gt", gt_rgb, rgb, mask, flick)

    # summary[output] = the means below, for --summary-json
    summary: dict[str, dict] = {}
    print(f"\nMean over {len(dng_paths)} frames:")
    for (name, ref_name), vals in scores.items():
        p, s = np.mean(vals, axis=0)
        print(f"  PSNR {p:8.3f}  SSIM {s:8.4f}  ({name} vs {ref_name})")
        summary.setdefault(name, {}).update(psnr=float(p), ssim=float(s))

    if args.change_threshold is not None and changed_pixels[1]:
        print(f"\nChanged regions (threshold {args.change_threshold}, "
              f"{100 * changed_pixels[0] / changed_pixels[1]:.2f}% of pixels, "
              f"pooled from the second frame onwards):")
        for (name, ref_name), (sq_err, count) in changed.items():
            p = f"{10 * np.log10(65535 ** 2 * count / sq_err):8.3f}" if count else "       —"
            print(f"  PSNR {p}  ({name} vs {ref_name})")
            if count:
                summary.setdefault(name, {})["changed_psnr"] = float(p)

    if gt_deltas:
        print(f"\nFrame-to-frame differences (WB/gamma space, averaged over {k}x{k} raw pixels, "
              f"mean from the second frame onwards):")
        print(f"  |gt_t - gt_t-1|     {np.mean(gt_deltas):.5f}")
        for name, vals in flicker.items():
            f, d = np.mean(vals, axis=0)
            print(f"  |pred_t - pred_t-1| {d:.5f}  flicker@{k} {f:.5f}  ({name})")
            summary.setdefault(name, {})["flicker"] = float(f)

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(
            {"clip": args.clip, "checkpoint": args.checkpoint, "gain": args.gain,
             "frames": len(dng_paths), "outputs": summary}, indent=2))

if __name__ == "__main__":
    main()
