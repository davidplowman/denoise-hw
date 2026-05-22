"""Hardware-style spatial Bayer denoiser using sensor tuning parameters."""

from __future__ import annotations

import math

import numpy as np

from dng import DNG

class DenoiseHW:
    """Spatial denoiser tuned to a specific image's sensor and gain.

    Parameters are derived from the sensor tuning file and the image gain
    (ISO) so that the denoising strength scales with the expected noise level.

    Attributes:
        noise_constant: Constant term of the per-pixel noise std model.
        noise_slope: Slope term of the per-pixel noise std model (multiplied
            by sqrt(pixel - black_level) to give std).
        strength: Blend weight between denoised and original (0–1).
        black_level: Sensor black level used for variance estimation.
        white_level: Sensor white (saturation) level.
    """

    # Neighbour displacements — always multiples of 2 so neighbours share the
    # same Bayer channel as the central pixel.
    X_DISPS = [-14, -10, -8, -4, -2, 0, 2, 4, 8, 10, 12, 14]
    Y_DISPS = [-6, -4, -2, 0, 2, 4, 6, 8]

    # 3×3 patch offsets used to compare neighbourhoods.
    PATCH_OFFSETS = [(py, px) for py in (-1, 0, 1) for px in (-1, 0, 1)]

    # Padding required so every shift (including ±1 patch offsets) stays within
    # the padded image.
    PAD_Y = max(abs(d) for d in Y_DISPS) + 1
    PAD_X = max(abs(d) for d in X_DISPS) + 1

    def __init__(self, dng: DNG, factor = None) -> None:
        """Initialise the denoiser for a particular image.

        Args:
            dng: Loaded DNG image whose sensor name and gain are used to
                 look up tuning parameters.
        """
        tuning = dng.tuning
        sdn = tuning.denoise_mode("normal").sdn
        ag = dng.gain
        deviation = sdn.deviation_no_tdn
        self.strength: float = sdn.strength_no_tdn
        self.black_level: float = dng.black_level
        self.white_level: int = dng.white_level

        self.deviation = 4.5 / 3.2 * deviation
        self.noise_calibration = tuning.get_noise2_calibration(ag)
        if self.noise_calibration:
            self.noise_calibration = [c * factor for c in self.noise_calibration]

        self.strength = 1.0

    def _shift(self, padded: np.ndarray, H: int, W: int, dy: int, dx: int) -> np.ndarray:
        """Return an H×W view of padded shifted by (dy, dx)."""
        return padded[self.PAD_Y + dy : self.PAD_Y + dy + H,
                      self.PAD_X + dx : self.PAD_X + dx + W]

    def compute_gaussian_weights(self, bayer: np.ndarray) -> np.ndarray:
        """Compute per-pixel Gaussian blend weights for all 63 spatial neighbours.

        Args:
            bayer: uint16 Bayer array of shape (H, W).

        Returns:
            float32 array of shape (63, H, W) — one weight map per neighbour.
        """
        img = bayer.astype(np.float32)
        H, W = img.shape
        padded = np.pad(img, ((self.PAD_Y, self.PAD_Y), (self.PAD_X, self.PAD_X)), mode="reflect")
        std = self._compute_std(padded) * self.deviation
        denom = std * std * 4
        gaussian_weights = []
        for dy in self.Y_DISPS:
            for dx in self.X_DISPS:
                patch_diff = np.zeros_like(img)
                for py, px in self.PATCH_OFFSETS:
                    patch_diff += np.abs(self._shift(padded, H, W, py, px) -
                                         self._shift(padded, H, W, dy + py, dx + px))
                weights = np.exp(-(patch_diff * patch_diff) / denom)
                gaussian_weights.append(weights)
        return np.stack(gaussian_weights, axis=0)

    def _compute_neighbours(self, padded: np.ndarray, H: int, W: int) -> np.ndarray:
        """Return the 63 shifted neighbour pixel arrays for a padded image.

        Args:
            padded: float32 reflect-padded array of shape (H+2*PAD_Y, W+2*PAD_X).
            H: Original image height.
            W: Original image width.

        Returns:
            float32 array of shape (63, H, W).
        """
        return np.stack(
            [self._shift(padded, H, W, dy, dx).copy()
             for dy in self.Y_DISPS for dx in self.X_DISPS],
            axis=0,
        )

    def _compute_std(self, padded: np.ndarray) -> np.ndarray:
        """Return per-pixel noise std estimated from the local 3×3 average.

        Args:
            padded: Reflect-padded image array.

        Returns:
            Per-pixel std array, clamped to a minimum of 1.0.
        """
        H = padded.shape[0] - 2 * self.PAD_Y
        W = padded.shape[1] - 2 * self.PAD_X
        avg = sum(self._shift(padded, H, W, py, px) for py, px in self.PATCH_OFFSETS) / len(self.PATCH_OFFSETS)
        avg -= self.black_level
        avg_sqrt = np.sqrt(np.maximum(avg, 0))
        # Old: std = self.noise_constant + self.noise_slope * avg_sqrt
        std = self.noise_calibration[0] + self.noise_calibration[1] * avg_sqrt + self.noise_calibration[2] * avg
        return np.maximum(std, 1.0)

    def process(self, bayer: np.ndarray) -> np.ndarray:
        """Denoise a raw Bayer image.

        For each pixel the method compares the 3×3 patch around it with the
        3×3 patch around each of 63 neighbouring pixels (at displacements that
        are multiples of 2, so neighbours share the same Bayer channel).  A
        Gaussian weight is derived from how similar each neighbour's patch is
        to the central patch, and the output pixel is the weighted average of
        all neighbours.

        Args:
            bayer: uint16 Bayer array, same shape as :attr:`DNG.bayer`.

        Returns:
            Denoised Bayer array with the same dtype and shape as *bayer*.
        """
        img = bayer.astype(np.float32)
        H, W = img.shape
        padded = np.pad(img, ((self.PAD_Y, self.PAD_Y), (self.PAD_X, self.PAD_X)), mode="reflect")

        gaussian_weights = self.compute_gaussian_weights(bayer)  # (63, H, W)
        neighbours = self._compute_neighbours(padded, H, W)      # (63, H, W)

        # Weighted average over all neighbours.
        weight_total = gaussian_weights.sum(axis=0)
        denoised = (gaussian_weights * neighbours).sum(axis=0) / weight_total

        # Blend denoised result with the original according to strength.
        output = self.strength * denoised + (1.0 - self.strength) * img
        return np.clip(output, 0, self.white_level).astype(np.uint16)
