"""Run the trained DenoiseNetwork on one dataset folder and save PNG outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import ctypes
import subprocess

from denoise_hw import DenoiseHW
from denoise_network import DenoiseNetwork
from dng import DNG

import math
import numpy as np

_here = Path(__file__).parent
_cdn_so = _here / "cdn_bayer.so"
if not _cdn_so.exists():
    print("Building cdn_bayer.so...")
    subprocess.run(
        ["g++", "-O2", "-shared", "-fPIC", "-o", str(_cdn_so), str(_here / "cdn_bayer.cpp")],
        check=True,
    )
_cdn_lib = ctypes.CDLL(_cdn_so)
_cdn_lib.run_cdn.restype = None
_cdn_lib.run_cdn.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


def _save(rgb, path: Path) -> None:
    Image.fromarray(rgb).save(path)
    print(f"  {path}")


def do_cdn(denoised, dng):
    tuning = dng.tuning
    cdn = tuning.denoise_mode("normal").cdn
    noise_model = tuning.noise_model
    cdn_thresh = int(cdn.deviation * math.sqrt(dng.gain) * noise_model.slope / 4)
    cdn_iir_strength = int(cdn.strength * 256 / 10)
    cdn_g_adjust = 0
    cdn_bayer_order = {"RGGB": 0, "GBRG": 1, "BGGR": 2, "GRBG": 3}[dng.bayer_pattern]

    arr = np.ascontiguousarray(denoised, dtype=np.uint16)
    height, width = arr.shape[:2]
    stride_bytes = arr.strides[0]
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    _cdn_lib.run_cdn(ptr, width, height, stride_bytes,
                     cdn_thresh, cdn_iir_strength, cdn_g_adjust, cdn_bayer_order)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Test DenoiseNetwork on one folder.")
    parser.add_argument("folder", help="Path relative to PI_RAW/Data (e.g. cabinet_H_2/imx219_ag1_test)")
    parser.add_argument("--weights", required=True, metavar="PTH")
    parser.add_argument("--output",  default="output", metavar="DIR")
    parser.add_argument("--dataset", default="/home/davidp/denoise/datasets/PI_RAW", metavar="ROOT")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    folder = Path(args.dataset) / "Data" / args.folder
    out    = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {folder}")
    print(f"Output: {out}")

    noisy_dng = DNG(str(folder / "noisy.dng"))
    gt_dng    = DNG(str(folder / "gt.dng"))

    dn       = DenoiseNetwork(noisy_dng, factor=1.0, weights_file=args.weights)
    print(f"Model: ch={dn.model.ch}  scaling={dn.model.scaling_factors}")
    denoised = dn.process(noisy_dng.bayer)

    denoised = denoised.astype(np.float32) - noisy_dng.black_level
    channel_gains = {'R': noisy_dng.wb_gains[0], 'G': noisy_dng.wb_gains[1], 'B': noisy_dng.wb_gains[2]}
    for i, colour in enumerate(noisy_dng.bayer_pattern):
        denoised[i // 2::2, i % 2::2] *= channel_gains[colour]
    denoised = np.clip(denoised, 0, noisy_dng.white_level).astype(np.uint16)

    denoised = do_cdn(denoised, noisy_dng)

    net_rgb  = noisy_dng.to_rgb(bayer=denoised, subtract_black_level=False, apply_colour_gains=False)
    _save(net_rgb, out / "net.png")

    gt_rgb = gt_dng.to_rgb()

    if not args.quick:
        _save(noisy_dng.to_rgb(),                              out / "noisy.png")
        _save(gt_rgb,                                          out / "gt.png")
        hw_denoised = DenoiseHW(noisy_dng, factor=1.0).process(noisy_dng.bayer)
        _save(noisy_dng.to_rgb(bayer=hw_denoised),             out / "hw.png")

        hw_cdn = hw_denoised.astype(np.float32) - noisy_dng.black_level
        for i, colour in enumerate(noisy_dng.bayer_pattern):
            hw_cdn[i // 2::2, i % 2::2] *= channel_gains[colour]
        hw_cdn = np.clip(hw_cdn, 0, noisy_dng.white_level).astype(np.uint16)
        hw_cdn = do_cdn(hw_cdn, noisy_dng)
        _save(noisy_dng.to_rgb(bayer=hw_cdn, subtract_black_level=False, apply_colour_gains=False),
              out / "hw_cdn.png")

    print(f"\n{'':12s}  {'PSNR':>8s}  {'SSIM':>8s}")
    if args.quick:
        comparisons = [("net", net_rgb)]
    else:
        noisy_rgb  = noisy_dng.to_rgb()
        hw_rgb     = noisy_dng.to_rgb(bayer=hw_denoised)
        hw_cdn_rgb = noisy_dng.to_rgb(bayer=hw_cdn, subtract_black_level=False, apply_colour_gains=False)
        comparisons = [("noisy", noisy_rgb), ("hw", hw_rgb), ("hw_cdn", hw_cdn_rgb), ("net", net_rgb)]
    for label, rgb in comparisons:
        p = psnr(gt_rgb, rgb, data_range=255)
        s = ssim(gt_rgb, rgb, data_range=255, channel_axis=2)
        print(f"  {label:10s}  {p:8.3f}  {s:8.4f}")


if __name__ == "__main__":
    main()
