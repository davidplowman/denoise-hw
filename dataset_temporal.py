"""Sequence dataset for temporal (recurrent) Bayer denoising training.

Unlike dataset.py, there is no cache: clips are loaded directly from
/opt/datasets/IMX662_temporal (or a similar root), and synthetic noise is
added on the fly per sequence draw (see synthetic_noise.py), at a fresh
gain/defect map/time offset every time a sequence is sampled. See
tunings/temporal_plan.md for the overall design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import synthetic_noise
from dng import DNG

_SENSOR_NAME = "imx662"


def frame_paths(clip_dir: str | Path) -> list[Path]:
    """A clip's frame DNGs in time order."""
    return sorted(Path(clip_dir).glob("*.dng"))


def _token_matches(token: str, path: str) -> bool:
    """A "!token" matches when "token" is NOT in path; otherwise plain substring match."""
    if token.startswith("!"):
        return token[1:] not in path
    return token in path


def find_clips(root: str, filter: list[str] | None = None) -> list[str]:
    """Find clip folders (subdirectories of root containing *.dng files).

    Args:
        root: Dataset root directory (e.g. /opt/datasets/IMX662_temporal).
        filter: Optional list of substrings that must all match the clip's
                path (a "!token" entry matches when token is absent instead).

    Returns:
        Sorted list of absolute clip folder paths.
    """
    clips: list[str] = []
    for p in sorted(Path(root).iterdir()):
        if not p.is_dir() or not frame_paths(p):
            continue
        folder = str(p.resolve())
        if filter is None or all(_token_matches(token, folder) for token in filter):
            clips.append(folder)
    return clips


@dataclass
class TemporalRecord:
    clip_path: str
    top_left: tuple[int, int]  # (y, x) in Bayer coords


@dataclass
class ClipData:
    frames: list[np.ndarray]     # per-frame uint16 Bayer arrays, time order
    black_levels: list[float]    # per-frame black level
    white_level: np.int32
    channel_gains: np.ndarray    # (4,) float32, from the clip's first frame


class TemporalPatchDataset:
    """Sequence dataset backed by pre-loaded clip data.

    Each item is a fresh draw for its (clip, tile) record: a random time
    offset, a log-uniform noise gain in [gain_min, gain_max], a fresh defect
    map, and (train only) the existing brightness augmentation -- all
    redrawn on every call, so every epoch sees different sequences even for
    the same record.

    With a non-zero window_stride, each record instead yields one item per
    window starting at 0, stride, 2*stride, ... (as long as the window fits
    in its clip). E.g. for a 30-frame clip and seq_len 10, a stride of 10
    gives frames 0-9, 10-19, 20-29, and a stride of 5 gives windows starting
    at 0, 5, 10, 15, 20. Any leftover frames (when the stride doesn't divide
    num_frames - seq_len) are used by shifting each window by a random
    amount (up to the leftover count) on each draw.

    With fixed_seed set (used for validation), every item is instead the
    same on every call and in every DataLoader worker: its randomness comes
    from an RNG seeded by (fixed_seed, item index), and its gain is taken
    from a log-spaced grid over [gain_min, gain_max] (one grid point per
    item, assigned in a seeded random order), so the set covers the gain
    range evenly and is identical across epochs and runs.
    """

    def __init__(
        self,
        records: list[TemporalRecord],
        clips: dict[str, ClipData],
        patch_size: int,
        seq_len: int,
        gain_aug_prob: float = 0.0,
        gain_aug_max: float = 1.0,
        gain_min: float = 4.0,
        gain_max: float = 400.0,
        window_stride: int = 0,
        fixed_seed: int | None = None,
    ) -> None:
        self._records = records
        self._clips = clips
        self._patch_size = patch_size
        self._seq_len = seq_len
        self._gain_aug_prob = gain_aug_prob
        self._gain_aug_max = gain_aug_max
        self._log_gain_min = math.log10(gain_min)
        self._log_gain_max = math.log10(gain_max)
        self._rng = np.random.default_rng()

        self._window_stride = window_stride

        # (record index, window start or None for a random offset) per item
        self._items: list[tuple[int, int | None]] = []
        for i, rec in enumerate(records):
            if window_stride > 0:
                last = len(clips[rec.clip_path].frames) - seq_len
                self._items.extend((i, w) for w in range(0, last + 1, window_stride))
            else:
                self._items.append((i, None))

        self._fixed_seed = fixed_seed
        self._fixed_gains = None
        if fixed_seed is not None:
            n = len(self._items)
            order = np.random.default_rng([fixed_seed, n]).permutation(n)
            log_g = self._log_gain_min + (order + 0.5) / n * (self._log_gain_max - self._log_gain_min)
            self._fixed_gains = 10 ** log_g

    def reseed(self, seed: int) -> None:
        """Re-seed the sampling RNG (call from a DataLoader worker_init_fn)."""
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> dict:
        rec_idx, start = self._items[idx]
        rec = self._records[rec_idx]
        clip = self._clips[rec.clip_path]
        r0, c0 = rec.top_left
        p = self._patch_size
        T = self._seq_len
        num_frames = len(clip.frames)

        if self._fixed_seed is None:
            rng = self._rng
        else:
            rng = np.random.default_rng([self._fixed_seed, idx])

        if start is None:
            offset = int(rng.integers(0, num_frames - T + 1))
        else:
            slack = (num_frames - T) % self._window_stride
            offset = start + int(rng.integers(0, slack + 1))
        if self._fixed_gains is None:
            gain = float(10 ** rng.uniform(self._log_gain_min, self._log_gain_max))
            defects = synthetic_noise.make_defect_map(shape=(p, p))
        else:
            gain = float(self._fixed_gains[idx])
            defects = synthetic_noise.make_defect_map(shape=(p, p), seed=int(rng.integers(2**32)))

        brighten = (self._gain_aug_prob > 0 and rng.random() < self._gain_aug_prob)
        brighten_g = np.float32(rng.uniform(1.0, self._gain_aug_max)) if brighten else None

        wl = clip.white_level
        gt_frames = []
        noisy_frames = []
        for t in range(T):
            frame = clip.frames[offset + t]
            bl = clip.black_levels[offset + t]

            gt_tile = frame[r0:r0 + p, c0:c0 + p].astype(np.float32)
            if brighten:
                gt_tile = np.clip((gt_tile - bl) * brighten_g + bl, 0.0, wl).astype(np.float32)

            signal = gt_tile - bl
            pixel = signal + synthetic_noise.black_level(gain)
            noisy_tile = synthetic_noise.add_noise(pixel, gain, rng, defects).astype(np.float32)

            gt_frames.append(gt_tile)
            noisy_frames.append(noisy_tile)

        return {
            "noisy":         np.stack(noisy_frames, axis=0),  # (T, p, p)
            "gt":            np.stack(gt_frames, axis=0),     # (T, p, p)
            "channel_gains": clip.channel_gains,
            "white_level":   clip.white_level,
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def worker_init_fn(worker_id: int) -> None:
    """Re-seed a TemporalPatchDataset's sampling RNG in each DataLoader worker."""
    import torch

    info = torch.utils.data.get_worker_info()
    info.dataset.reseed(info.seed)


def load_datasets(
    dataset_root: str = "/opt/datasets/IMX662_temporal",
    filters: list[str] | None = None,
    val_frac: float = 0.1,
    patch_size: int = 256,
    seq_len: int = 10,
    gain_aug_prob: float = 0.5,
    gain_aug_max: float = 2.0,
    gain_min: float = 4.0,
    gain_max: float = 400.0,
    window_stride: int = 0,
    val_seed: int | None = 0,
) -> tuple[TemporalPatchDataset, TemporalPatchDataset]:
    """Load train and validation sequence datasets from a clip directory.

    Args:
        dataset_root: Root directory of clip folders.
        filters: Substrings that must all match a clip's path ("!token"
            excludes instead); see find_clips.
        val_frac: Fraction of (clip, tile) records used for validation.
        patch_size: Spatial tile size.
        seq_len: Number of consecutive frames per training sequence.
        gain_aug_prob/gain_aug_max: Brightness augmentation, applied to the
            training split only (matches dataset.py's PatchDataset).
        gain_min/gain_max: Synthetic noise gain range, sampled log-uniformly
            per sequence -- applies to both splits (this is the task itself,
            not an optional augmentation).
        window_stride: If non-zero, give every window of each tile starting
            at a multiple of this stride its own item, rather than one
            randomly placed window per tile (see TemporalPatchDataset).
            Applies to both splits, after the split itself, so a tile's
            windows never straddle train and val.
        val_seed: If not None, the validation set is fixed (identical every
            epoch, in every worker and every run) with evenly log-spaced
            gains; see TemporalPatchDataset's fixed_seed. None draws fresh
            random validation sequences on every call, like training.

    Returns:
        (train_ds, val_ds) as TemporalPatchDataset instances.
    """
    clip_paths = find_clips(dataset_root, filter=filters)
    if not clip_paths:
        raise ValueError(
            f"No clips found under {dataset_root}"
            + (f" matching filter {filters}" if filters else "")
        )

    clips: dict[str, ClipData] = {}
    records: list[TemporalRecord] = []

    for clip_path in clip_paths:
        dng_paths = frame_paths(clip_path)
        if len(dng_paths) < seq_len:
            raise ValueError(
                f"Clip {clip_path} has only {len(dng_paths)} frames, "
                f"fewer than --seq-len {seq_len}"
            )

        frames: list[np.ndarray] = []
        black_levels: list[float] = []
        white_level = None
        channel_gains = None
        for i, dp in enumerate(dng_paths):
            d = DNG(str(dp), sensor_name=_SENSOR_NAME)
            frames.append(d.bayer)
            black_levels.append(d.black_level)
            if i == 0:
                white_level = np.int32(d.white_level)
                _cidx = {'R': 0, 'G': 1, 'B': 2}
                channel_gains = np.array(
                    [d.wb_gains[_cidx[d.bayer_pattern[k]]] for k in range(4)],
                    dtype=np.float32,
                )

        clips[clip_path] = ClipData(
            frames=frames, black_levels=black_levels,
            white_level=white_level, channel_gains=channel_gains,
        )

        H, W = frames[0].shape
        ps = patch_size
        for r0 in range(0, H - ps + 1, ps):
            for c0 in range(0, W - ps + 1, ps):
                records.append(TemporalRecord(clip_path=clip_path, top_left=(r0, c0)))

    rng = np.random.default_rng(seed=0)
    rng.shuffle(records)

    n_val = max(1, int(len(records) * val_frac))
    train_records = records[:-n_val]
    val_records = records[-n_val:]

    train_ds = TemporalPatchDataset(
        train_records, clips, patch_size, seq_len,
        gain_aug_prob=gain_aug_prob, gain_aug_max=gain_aug_max,
        gain_min=gain_min, gain_max=gain_max, window_stride=window_stride,
    )
    val_ds = TemporalPatchDataset(
        val_records, clips, patch_size, seq_len,
        gain_min=gain_min, gain_max=gain_max, window_stride=window_stride,
        fixed_seed=val_seed,
    )
    return train_ds, val_ds
