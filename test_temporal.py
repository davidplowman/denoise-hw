"""Human-viewable evaluation of the recurrent (temporal) DenoiseNet on one clip.

Runs a trained checkpoint frame-by-frame over a real clip from the temporal
dataset, synthesizing noise at a chosen gain and threading the previous
frame's output forward exactly as in training. Writes 16-bit gt/noisy/pred
PNGs per frame and prints PSNR/SSIM against ground truth.

With --still-checkpoint, also runs the single-image (non-temporal) DenoiseNet
on exactly the same noisy frames, writing <N>_still.png and reporting its
PSNR/SSIM alongside, for comparison.

With --change-threshold, also reports PSNR over only the pixels where the GT
changed from the previous frame by more than the threshold (measured in the
training loss's white-balanced, gamma-corrected space, max over the packed
channels) -- this is where ghosting from the previous output shows up.
"""

from __future__ import annotations

import argparse
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
from train_temporal import _to_wb

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
        model = model_cls(ch=ckpt.get("ch"), scaling_factors=ckpt.get("scaling_factors"),
                           num_encoders=ckpt.get("num_encoders"))
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = model_cls()
        model.load_state_dict(ckpt)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the temporal DenoiseNet on one clip.")
    parser.add_argument("clip", help="Name of the clip folder (e.g. office_0)")
    parser.add_argument("--checkpoint", required=True, metavar="PTH")
    parser.add_argument("--still-checkpoint", default=None, metavar="PTH",
                         help="also denoise each frame with this single-image "
                              "(non-temporal) DenoiseNet checkpoint, for comparison")
    parser.add_argument("--output", default="output_temporal", metavar="DIR")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--dataset", default="/opt/datasets/IMX662_temporal", metavar="ROOT")
    parser.add_argument("--seed", type=int, default=None,
                         help="seed for the noise/defect-map RNG (default: unseeded)")
    parser.add_argument("--change-threshold", type=float, default=None, metavar="T",
                         help="also report PSNR over pixels where |gt_t - gt_{t-1}| > T "
                              "(WB/gamma space, as in the training loss)")
    parser.add_argument("--save-masks", action="store_true",
                         help="with --change-threshold, write <N>_mask.png for each frame")
    args = parser.parse_args()

    clip_dir = Path(args.dataset) / args.clip
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    dng_paths = frame_paths(clip_dir)[args.offset:]
    if not dng_paths:
        parser.error(f"no frames found in {clip_dir} from offset {args.offset}")

    print(f"Clip:   {clip_dir}  ({len(dng_paths)} frames from offset {args.offset})")
    print(f"Output: {out}")

    model = _load_model(args.checkpoint)
    print(f"Model: ch={model.ch}  scaling={model.scaling_factors}  encoders={model.num_encoders}")

    still_model = None
    if args.still_checkpoint:
        still_model = _load_model(args.still_checkpoint, StillDenoiseNet)
        print(f"Still model: ch={still_model.ch}  scaling={still_model.scaling_factors}  "
              f"encoders={still_model.num_encoders}")

    rng = np.random.default_rng(args.seed)
    defects = None  # one map, shared across the whole processed sequence

    # scores[(output, reference)] = list of per-frame (PSNR, SSIM)
    scores: dict[tuple[str, str], list[tuple[float, float]]] = {}
    # changed[(output, reference)] = [squared error sum, sample count], pooled
    # over the changed-region pixels of all frames
    changed: dict[tuple[str, str], list[float]] = {}
    changed_pixels = [0, 0]  # [masked, total] over frames with a previous frame

    def score(name: str, ref_name: str, ref_rgb: np.ndarray, rgb: np.ndarray,
              mask: np.ndarray | None) -> None:
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
        print(f"  frame {num}  PSNR {p:8.3f}  SSIM {s:8.4f}{extra}  ({name} vs {ref_name})")

    channel_gains = None  # (1, 16), from the first frame, as in training
    prev_gt_wb = None
    prev_output = None

    for dp in dng_paths:
        num = dp.stem  # "00", "01", ...
        gt_dng = DNG(str(dp), sensor_name=_SENSOR_NAME)

        if defects is None:
            defects = synthetic_noise.make_defect_map(shape=gt_dng.bayer.shape)

        signal = gt_dng.bayer.astype(np.float32) - gt_dng.black_level
        target_black = synthetic_noise.black_level(args.gain)
        pixel = signal + target_black
        noisy_bayer = synthetic_noise.add_noise(pixel, args.gain, rng, defects).astype(np.uint16)

        x = _pack(noisy_bayer, gt_dng.white_level)
        if prev_output is None:
            prev_output = torch.zeros_like(x)
        with torch.no_grad():
            pred = model(torch.cat([x, prev_output], dim=1))
        prev_output = pred

        pred_bayer = _unpack(pred, gt_dng.white_level, gt_dng.bayer.shape)

        # noisy_bayer is anchored to the *target gain's* black level, not
        # gt_dng's own captured black level -- subtract it ourselves and
        # tell to_rgb not to subtract anything else (same pattern test.py
        # uses for its own post-processed outputs).
        noisy_signal = np.clip(noisy_bayer.astype(np.float32) - target_black, 0, None).astype(np.uint16)

        gt_rgb    = gt_dng.to_rgb(output_bps=16)
        noisy_rgb = gt_dng.to_rgb(bayer=noisy_signal, subtract_black_level=False, output_bps=16)
        pred_rgb  = gt_dng.to_rgb(bayer=pred_bayer, output_bps=16)

        mask = None
        if args.change_threshold is not None:
            if channel_gains is None:
                cidx = {'R': 0, 'G': 1, 'B': 2}
                g4 = torch.tensor([gt_dng.wb_gains[cidx[gt_dng.bayer_pattern[k]]] for k in range(4)],
                                  dtype=torch.float32)
                channel_gains = g4.repeat_interleave(4).unsqueeze(0)
            gt_wb = _to_wb(_pack(gt_dng.bayer, gt_dng.white_level), channel_gains)
            if prev_gt_wb is not None:
                diff = (gt_wb - prev_gt_wb).abs().amax(dim=1)[0].numpy()  # (H/4, W/4)
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

        outputs = [("temporal", pred_rgb)]
        if still_model is not None:
            with torch.no_grad():
                still_pred = still_model(x)
            still_bayer = _unpack(still_pred, gt_dng.white_level, gt_dng.bayer.shape)
            still_rgb = gt_dng.to_rgb(bayer=still_bayer, output_bps=16)
            _save_png(still_rgb, out / f"{num}_still.png")
            outputs.append(("still", still_rgb))

        for name, rgb in outputs:
            score(name, "gt", gt_rgb, rgb, mask)

    print(f"\nMean over {len(dng_paths)} frames:")
    for (name, ref_name), vals in scores.items():
        p, s = np.mean(vals, axis=0)
        print(f"  PSNR {p:8.3f}  SSIM {s:8.4f}  ({name} vs {ref_name})")

    if args.change_threshold is not None and changed_pixels[1]:
        print(f"\nChanged regions (threshold {args.change_threshold}, "
              f"{100 * changed_pixels[0] / changed_pixels[1]:.2f}% of pixels, "
              f"pooled from the second frame onwards):")
        for (name, ref_name), (sq_err, count) in changed.items():
            p = f"{10 * np.log10(65535 ** 2 * count / sq_err):8.3f}" if count else "       —"
            print(f"  PSNR {p}  ({name} vs {ref_name})")

if __name__ == "__main__":
    main()
