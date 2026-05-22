"""Neural-network Bayer denoiser with the same interface as DenoiseHW."""

from __future__ import annotations

import numpy as np
import torch

from denoise_hw import DenoiseHW
from dng import DNG
from network import DenoiseNet


class DenoiseNetwork(DenoiseHW):
    """UNet-based denoiser with the same process() interface as DenoiseHW.

    Inherits __init__ from DenoiseHW to obtain black_level, white_level,
    and noise_calibration for computing the per-pixel noise-std channel.

    Args:
        dng: Loaded DNG image used to configure noise parameters.
        factor: Scaling factor applied to noise_calibration coefficients.
        weights_file: Optional path to a saved state-dict (.pth) file.
    """

    def __init__(
        self,
        dng: DNG,
        factor: float = 1.0,
        weights_file: str | None = None,
    ) -> None:
        super().__init__(dng, factor=factor)
        self.model = DenoiseNet()
        if weights_file is not None:
            self.model.load_state_dict(torch.load(weights_file, map_location="cpu"))
        self.model.eval()

    def process(self, bayer: np.ndarray) -> np.ndarray:
        """Denoise a raw Bayer image using the UNet model.

        Args:
            bayer: uint16 Bayer array of shape (H, W).

        Returns:
            Denoised uint16 Bayer array of the same shape.
        """
        img = bayer.astype(np.float32)

        avg = (img[0::2, 0::2] + img[0::2, 1::2] +
               img[1::2, 0::2] + img[1::2, 1::2]) / 4

        avg_bl = np.maximum(avg - self.black_level, 0)
        std_ds = (self.noise_calibration[0]
                  + self.noise_calibration[1] * np.sqrt(avg_bl)
                  + self.noise_calibration[2] * avg_bl)
        std_ds = np.maximum(std_ds, 1.0)

        ch = np.stack([img[0::2, 0::2], img[0::2, 1::2],
                       img[1::2, 0::2], img[1::2, 1::2]], axis=0)
        net_in = np.concatenate([ch, std_ds[np.newaxis]], axis=0)
        x = torch.from_numpy(net_in / self.white_level).unsqueeze(0).float()

        with torch.no_grad():
            out = self.model(x)

        ch_out = out.squeeze(0).numpy() * self.white_level
        result = np.empty_like(img)
        result[0::2, 0::2] = ch_out[0]
        result[0::2, 1::2] = ch_out[1]
        result[1::2, 0::2] = ch_out[2]
        result[1::2, 1::2] = ch_out[3]
        return np.clip(result, 0, self.white_level).astype(np.uint16)
