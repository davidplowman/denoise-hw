"""Load DNG files and expose Bayer data, EXIF metadata, and RGB conversion."""

from __future__ import annotations

from pathlib import Path

import exifread
import numpy as np
import rawpy

from tuning import load_tuning

_TUNINGS_DIR = Path(__file__).parent / "tunings"


def _sensor_from_folder(folder: Path) -> str:
    """Derive the sensor name from a dataset folder name.

    The sensor name is a prefix of the folder name that matches one of the
    available tuning files.  Trailing ``_token`` components are stripped one
    at a time until a match is found.

    Args:
        folder: Path to the folder containing noisy.dng / gt.dng.

    Returns:
        Sensor name string (e.g. "imx219").

    Raises:
        ValueError: If no tuning file matches any prefix of the folder name.
    """
    available = {p.stem for p in _TUNINGS_DIR.glob("*.json")}
    parts = folder.name.split("_")
    for i in range(len(parts), 0, -1):
        candidate = "_".join(parts[:i])
        if candidate in available:
            return candidate
    raise ValueError(
        f"Could not determine sensor name from folder '{folder.name}'. "
        f"Available sensors: {sorted(available)}"
    )


def _apply_geq(bayer: np.ndarray, bayer_pattern: str, geq, white_level: int) -> np.ndarray:
    """Apply Green Equalisation to both green channels of a Bayer image.

    The raw array is reflect-padded by one pixel on every side.  For each
    green pixel the threshold is ``geq.offset + geq.slope * pixel``.  Its
    four diagonal raw-array neighbours (which are always the other green
    channel in any Bayer pattern) are read from the padded array, averaged,
    and half the difference from the central pixel is clamped to ±threshold
    and added back.  Only the green pixel positions are written to the result.

    Args:
        bayer: uint16 raw Bayer array.
        bayer_pattern: 2×2 pattern string, e.g. ``"RGGB"``.
        geq: GEQ offset and slope from the tuning file.
        white_level: Sensor saturation level used to clip the output.

    Returns:
        A new uint16 array with GEQ applied to both green channels.
    """
    g_pos = [(i // 2, i % 2) for i, c in enumerate(bayer_pattern) if c == 'G']
    padded = np.pad(bayer, 1, mode='reflect').astype(np.float32)
    result = bayer.copy()

    for ra, ca in g_pos:
        G_A = bayer[ra::2, ca::2].astype(np.float32)
        Hg, Wg = G_A.shape
        # After padding by 1, bayer[ra + 2i, ca + 2j] sits at padded[ra+1+2i, ca+1+2j].
        # Its four diagonal raw-array neighbours are at offsets (±1, ±1), which in
        # padded coordinates are at rows ra+2i and ra+2i+2, cols ca+2j and ca+2j+2.
        avg_other = (
            padded[ra    : ra + 2*Hg    : 2, ca    : ca + 2*Wg    : 2] +
            padded[ra    : ra + 2*Hg    : 2, ca + 2: ca + 2*Wg + 2: 2] +
            padded[ra + 2: ra + 2*Hg + 2: 2, ca    : ca + 2*Wg    : 2] +
            padded[ra + 2: ra + 2*Hg + 2: 2, ca + 2: ca + 2*Wg + 2: 2]
        ) / 4.0
        threshold  = geq.offset + geq.slope * G_A
        correction = np.clip((avg_other - G_A) / 2.0, -threshold, threshold)
        result[ra::2, ca::2] = np.clip(G_A + correction, 0, white_level).astype(np.uint16)

    return result


def _bayer_pattern(raw: rawpy.RawPy) -> str:
    """Return the 2×2 Bayer pattern as a string, e.g. 'RGGB'."""
    desc = raw.color_desc.decode()
    pat = raw.raw_pattern
    return "".join(desc[pat[r, c]] for r in range(2) for c in range(2))


def _mean_black_level(raw: rawpy.RawPy) -> float:
    """Return the mean black level across all channels."""
    blc = raw.black_level_per_channel
    return float(np.mean(blc)) if blc is not None else 0.0


class DNG:
    """A loaded DNG raw image with Bayer data and EXIF metadata.

    Attributes:
        path: Absolute path to the DNG file.
        bayer: Raw Bayer array (uint16, full sensor size).
        bayer_pattern: 2×2 Bayer colour pattern, e.g. ``"RGGB"``.
        black_level: Mean black level across channels.
        white_level: Sensor white (saturation) level.
        exposure_time: Shutter speed in seconds.
        iso: ISO speed rating.
        gain: Analogue gain relative to ISO 100 (``iso / 100``).
        sensor_name: Sensor identifier derived from the parent folder name.
        tuning: Loaded tuning object for the sensor.
    """

    def __init__(self, path: str) -> None:
        """Load a DNG file and extract Bayer data and EXIF metadata.

        Args:
            path: Path to the DNG file to load.
        """
        self.path = str(path)
        folder = Path(path).parent
        self.sensor_name: str = _sensor_from_folder(folder)
        self.tuning = load_tuning(self.sensor_name)

        with rawpy.imread(self.path) as raw:
            self.bayer: np.ndarray = raw.raw_image_visible.copy()
            self.bayer_pattern: str = _bayer_pattern(raw)
            self.black_level: float = _mean_black_level(raw)
            self.white_level: int = int(raw.white_level)
            wb = raw.camera_whitebalance
            self.wb_gains: tuple[float, float, float] = (wb[0] / wb[1], 1.0, wb[2] / wb[1])

        self.exposure_time, self.iso = self._read_exif(self.path)
        self.gain: float = self.iso / 100.0

        if self.tuning.geq is not None:
            self.bayer = _apply_geq(self.bayer, self.bayer_pattern, self.tuning.geq, self.white_level)


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_rgb(self, bayer: np.ndarray | None = None) -> np.ndarray:
        """Convert (optionally pre-processed) Bayer data to an RGB image.

        Args:
            bayer: Optional uint16 Bayer array to use instead of the
                   original raw data.  Must have the same shape as
                   :attr:`bayer`.

        Returns:
            uint8 RGB image array of shape (H, W, 3).
        """
        if bayer is None:
            bayer = self.bayer
        with rawpy.imread(self.path) as raw:
            if bayer is not None:
                raw.raw_image_visible[:] = bayer
            return raw.postprocess(use_camera_wb=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_exif(path: str) -> tuple[float, int]:
        """Read exposure time and ISO from EXIF tags.

        Args:
            path: Path to the DNG file.

        Returns:
            Tuple of (exposure_time_seconds, iso).
        """
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh)

        exposure_time = 0.0
        iso = 100

        # PiDNG stores tags under the "Image" namespace rather than "EXIF".
        # Try both so standard DNG files also work.
        et_tag = tags.get("EXIF ExposureTime") or tags.get("Image ExposureTime")
        if et_tag is not None:
            val = et_tag.values[0]
            exposure_time = float(val.num) / float(val.den) if val.den else float(val)

        iso_tag = tags.get("EXIF ISOSpeedRatings") or tags.get("Image ISOSpeedRatings")
        if iso_tag is not None:
            iso = int(iso_tag.values[0])

        return exposure_time, iso
