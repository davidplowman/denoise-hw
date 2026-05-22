"""Run the trained DenoiseNetwork on one dataset folder and save PNG outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from denoise_hw import DenoiseHW
from denoise_network import DenoiseNetwork
from dng import DNG


def _save(rgb, path: Path) -> None:
    Image.fromarray(rgb).save(path)
    print(f"  {path}")


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
    denoised = dn.process(noisy_dng.bayer)
    net_rgb  = noisy_dng.to_rgb(bayer=denoised)
    _save(net_rgb, out / "net.png")

    gt_rgb = gt_dng.to_rgb()

    if not args.quick:
        _save(noisy_dng.to_rgb(),                              out / "noisy.png")
        _save(gt_rgb,                                          out / "gt.png")
        hw_denoised = DenoiseHW(noisy_dng, factor=1.0).process(noisy_dng.bayer)
        _save(noisy_dng.to_rgb(bayer=hw_denoised),             out / "hw.png")

    print(f"\n{'':12s}  {'PSNR':>8s}  {'SSIM':>8s}")
    if args.quick:
        comparisons = [("net", net_rgb)]
    else:
        noisy_rgb = noisy_dng.to_rgb()
        hw_rgb    = noisy_dng.to_rgb(bayer=hw_denoised)
        comparisons = [("noisy", noisy_rgb), ("hw", hw_rgb), ("net", net_rgb)]
    for label, rgb in comparisons:
        p = psnr(gt_rgb, rgb, data_range=255)
        s = ssim(gt_rgb, rgb, data_range=255, channel_axis=2)
        print(f"  {label:10s}  {p:8.3f}  {s:8.4f}")


if __name__ == "__main__":
    main()
