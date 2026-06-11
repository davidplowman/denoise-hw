"""Dataset cache builder and loader for Bayer image denoising training."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve

from dng import DNG
from folders import find_folders

_LAP = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)


@dataclass
class ImageRecord:
    path: str
    top_left: tuple[int, int]  # (y, x) in Bayer coords
    detail: float


def _patch_detail(patch: np.ndarray) -> float:
    scores = []
    for r in range(2):
        for c in range(2):
            ch = patch[r::2, c::2].astype(np.float32)
            mean_sq = ch.mean() ** 2
            scores.append(float(convolve(ch, _LAP).var() / (mean_sq + 1e-6)))
    return 20 * float(np.mean(scores))


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def _cmd_build(args: argparse.Namespace) -> None:
    cache_dir = Path(args.name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ps = args.patch_size

    folders = find_folders(args.dataset, filter=args.filter)
    records: list[ImageRecord] = []

    for folder_str in folders:
        folder = Path(folder_str)
        print(f"Processing {folder_str}...")
        dng = DNG(str(folder / "noisy.dng"))

        H, W = dng.bayer.shape
        for r0 in range(0, H - ps + 1, ps):
            for c0 in range(0, W - ps + 1, ps):
                patch = dng.bayer[r0:r0 + ps, c0:c0 + ps]
                records.append(ImageRecord(
                    path=folder_str,
                    top_left=(r0, c0),
                    detail=_patch_detail(patch),
                ))

    np.save(str(cache_dir / "patch_size.npy"), np.array(ps, dtype=np.int32))
    np.save(str(cache_dir / "patch_list.npy"), np.array(records, dtype=object))
    print(f"Built cache: {len(records)} patches from {len(folders)} folders.")


def _cmd_list(args: argparse.Namespace) -> None:
    cache_dir = Path(args.name)
    patch_size = int(np.load(str(cache_dir / "patch_size.npy")))
    records: np.ndarray = np.load(str(cache_dir / "patch_list.npy"), allow_pickle=True)

    print(f"Patch size: {patch_size}")

    groups: dict[str, list] = defaultdict(list)
    for rec in records:
        groups[rec.path].append(rec)

    for path, recs in sorted(groups.items()):
        print(f"  {path}: {len(recs)} patches")

    print(f"\nTotal: {len(records)} patches from {len(groups)} images.")


def _cmd_show(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw, ImageFont

    key_path = Path(args.key)
    query = str(key_path.parent if key_path.is_file() else key_path)

    cache_dir = Path(args.cache)
    patch_size = int(np.load(str(cache_dir / "patch_size.npy")))
    all_records: np.ndarray = np.load(str(cache_dir / "patch_list.npy"), allow_pickle=True)
    folder_records = [r for r in all_records if r.path == query or r.path.endswith("/" + query)]

    if not folder_records:
        print(f"No patches found for {query}")
        return

    folder = folder_records[0].path

    dng = DNG(str(Path(folder) / "noisy.dng"))
    rgb = dng.to_rgb()
    sy = rgb.shape[0] / dng.bayer.shape[0]
    sx = rgb.shape[1] / dng.bayer.shape[1]

    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except OSError:
        font = ImageFont.load_default()

    for rec in folder_records:
        r0, c0 = rec.top_left
        x0, y0 = int(c0 * sx), int(r0 * sy)
        x1 = int((c0 + patch_size) * sx) - 1
        y1 = int((r0 + patch_size) * sy) - 1
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=2)
        label = f"{rec.detail:.1f}"
        tx, ty = (x0 + x1) // 2, (y0 + y1) // 2
        bbox = draw.textbbox((tx, ty), label, font=font, anchor="mm")
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=(255, 255, 0), font=font, anchor="mm")

    img.save(args.name, quality=90)
    print(f"Saved {args.name}")


# ---------------------------------------------------------------------------
# Dataset class and load_datasets
# ---------------------------------------------------------------------------

class PatchDataset:
    """Iterable patch dataset backed by pre-loaded full images."""

    def __init__(
        self,
        records: list[ImageRecord],
        images: dict[str, dict[str, np.ndarray]],
        patch_size: int,
    ) -> None:
        self._records = records
        self._images = images
        self._patch_size = patch_size

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        rec = self._records[idx]
        r0, c0 = rec.top_left
        p = self._patch_size
        ph, pw, ph2 = r0 // 2, c0 // 2, p // 2
        imgs = self._images[rec.path]
        return {
            "noisy":         imgs["noisy"]   [r0:r0+p,    c0:c0+p   ].astype(np.float32),
            "gt":            imgs["gt"]      [r0:r0+p,    c0:c0+p   ].astype(np.float32),
            "std":           imgs["std_map"] [ph:ph+ph2,  pw:pw+ph2 ],
            "channel_gains": imgs["channel_gains"],
            "white_level":   imgs["white_level"],
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def load_datasets(
    cache_dir: str = "cache",
    filters: list[str] | None = None,
    detail_top: float = 1.0,
    val_frac: float = 0.1,
) -> tuple[PatchDataset, PatchDataset]:
    """Load train and validation patch datasets from the cache.

    Args:
        cache_dir: Path to the cache directory built by the ``build`` command.
        filters: Substrings that must all appear in a patch's folder path.
        detail_top: Fraction of patches to retain (highest detail first).
        val_frac: Fraction of retained patches used for validation.

    Returns:
        ``(train_ds, val_ds)`` as :class:`PatchDataset` instances.
    """
    import sys
    sys.modules["__main__"].__dict__.setdefault("ImageRecord", ImageRecord)

    cache = Path(cache_dir)
    patch_size = int(np.load(str(cache / "patch_size.npy")))
    all_records: list[ImageRecord] = list(
        np.load(str(cache / "patch_list.npy"), allow_pickle=True)
    )

    if filters:
        all_records = [r for r in all_records if all(f in r.path for f in filters)]

    all_records.sort(key=lambda r: r.detail, reverse=True)
    if detail_top < 1.0:
        all_records = all_records[:max(1, int(len(all_records) * detail_top))]

    rng = np.random.default_rng(seed=0)
    rng.shuffle(all_records)

    images: dict[str, dict[str, np.ndarray]] = {}
    for rec in all_records:
        if rec.path not in images:
            noisy_dng = DNG(str(Path(rec.path) / "noisy.dng"))
            gt_dng = DNG(str(Path(rec.path) / "gt.dng"))

            img_f = noisy_dng.bayer.astype(np.float32)
            avg = (img_f[0::2, 0::2] + img_f[0::2, 1::2] +
                   img_f[1::2, 0::2] + img_f[1::2, 1::2]) / 4
            noise_cal = noisy_dng.tuning.get_noise2_calibration(noisy_dng.gain)
            avg_bl = np.maximum(avg - noisy_dng.black_level, 0.0)
            std_map = np.maximum(
                noise_cal[0] + noise_cal[1] * np.sqrt(avg_bl) + noise_cal[2] * avg_bl,
                1.0,
            ).astype(np.float32)

            _cidx = {'R': 0, 'G': 1, 'B': 2}
            channel_gains = np.array(
                [noisy_dng.wb_gains[_cidx[noisy_dng.bayer_pattern[i]]] for i in range(4)],
                dtype=np.float32,
            )

            images[rec.path] = {
                "noisy":         noisy_dng.bayer,
                "gt":            gt_dng.bayer,
                "std_map":       std_map,
                "channel_gains": channel_gains,
                "white_level":   np.int32(noisy_dng.white_level),
            }

    n_val = max(1, int(len(all_records) * val_frac))
    train_records = all_records[:-n_val]
    val_records = all_records[-n_val:]

    return PatchDataset(train_records, images, patch_size), PatchDataset(val_records, images, patch_size)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bayer denoising dataset cache tool.")
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build", help="Build the patch cache.")
    build_p.add_argument("--name", default="cache", metavar="DIR")
    build_p.add_argument("--patch-size", type=int, default=256, metavar="N")
    build_p.add_argument(
        "--dataset",
        default="/home/davidp/denoise/datasets/PI_RAW",
        metavar="ROOT",
    )
    build_p.add_argument("--filter", nargs="*", default=None, metavar="KEYWORD")

    list_p = sub.add_parser("list", help="List cached patches.")
    list_p.add_argument("--name", default="cache", metavar="DIR")

    show_p = sub.add_parser("show", help="Visualise patches on an RGB image.")
    show_p.add_argument("--key", required=True, metavar="PATH")
    show_p.add_argument("--name", default="show.jpg", metavar="FILE")
    show_p.add_argument("--cache", default="cache", metavar="DIR")

    args = parser.parse_args()
    {"build": _cmd_build, "list": _cmd_list, "show": _cmd_show}[args.command](args)


if __name__ == "__main__":
    main()
