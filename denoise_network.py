"""Neural-network Bayer denoiser with the same interface as DenoiseHW."""

from __future__ import annotations

import numpy as np
import torch

from denoise_hw import DenoiseHW
from dng import DNG
from network import DenoiseNet, depth_to_space, space_to_depth


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
        ch, scaling_factors, num_encoders, state_dict = None, None, None, None
        if weights_file is not None:
            ckpt = torch.load(weights_file, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ch              = ckpt.get("ch")
                scaling_factors = ckpt.get("scaling_factors")
                num_encoders    = ckpt.get("num_encoders")
                state_dict      = ckpt["state_dict"]
            else:
                state_dict = ckpt
        self.model = DenoiseNet(ch=ch, scaling_factors=scaling_factors, num_encoders=num_encoders)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.eval()

    def process(self, bayer: np.ndarray) -> np.ndarray:
        """Denoise a raw Bayer image using the UNet model.

        Args:
            bayer: uint16 Bayer array of shape (H, W).

        Returns:
            Denoised uint16 Bayer array of the same shape.
        """
        img = bayer.astype(np.float32)

        ch = np.stack([img[0::2, 0::2], img[0::2, 1::2],
                       img[1::2, 0::2], img[1::2, 1::2]], axis=0)
        x_bayer = torch.from_numpy(ch / self.white_level).unsqueeze(0).float()
        x = space_to_depth(x_bayer, 2)

        with torch.no_grad():
            out = self.model(x)

        ch_out = depth_to_space(out, 2).squeeze(0).numpy() * self.white_level
        result = np.empty_like(img)
        result[0::2, 0::2] = ch_out[0]
        result[0::2, 1::2] = ch_out[1]
        result[1::2, 0::2] = ch_out[2]
        result[1::2, 1::2] = ch_out[3]
        return np.clip(result, 0, self.white_level).astype(np.uint16)


class NcnnDenoiseNetwork(DenoiseHW):
    """Same interface as DenoiseNetwork, but runs a converted ncnn .param/.bin
    pair (e.g. from convert.py --format ncnn --quantize int8) instead of the
    PyTorch model -- for comparing a quantized export's real image quality
    against the original checkpoint's.

    Args:
        dng: Loaded DNG image used to configure noise parameters.
        factor: Scaling factor applied to noise_calibration coefficients.
        param_file: Path to the ncnn .param file.
        bin_file: Path to the ncnn .bin file.
    """

    def __init__(
        self,
        dng: DNG,
        factor: float = 1.0,
        param_file: str | None = None,
        bin_file: str | None = None,
    ) -> None:
        super().__init__(dng, factor=factor)
        import ncnn

        self.net = ncnn.Net()
        self.net.load_param(str(param_file))
        self.net.load_model(str(bin_file))

    def process(self, bayer: np.ndarray) -> np.ndarray:
        """Denoise a raw Bayer image using the ncnn model.

        Args:
            bayer: uint16 Bayer array of shape (H, W).

        Returns:
            Denoised uint16 Bayer array of the same shape.
        """
        import ncnn

        img = bayer.astype(np.float32)

        ch = np.stack([img[0::2, 0::2], img[0::2, 1::2],
                       img[1::2, 0::2], img[1::2, 1::2]], axis=0)
        x_bayer = torch.from_numpy(ch / self.white_level).unsqueeze(0).float()
        x = space_to_depth(x_bayer, 2).squeeze(0).numpy()

        ex = self.net.create_extractor()
        ex.input(self.net.input_names()[0], ncnn.Mat(x))
        ret, mat_out = ex.extract(self.net.output_names()[0])
        if ret != 0:
            raise RuntimeError(f"ncnn extraction failed (ret={ret})")

        out = torch.from_numpy(mat_out.numpy()).unsqueeze(0)
        ch_out = depth_to_space(out, 2).squeeze(0).numpy() * self.white_level
        result = np.empty_like(img)
        result[0::2, 0::2] = ch_out[0]
        result[0::2, 1::2] = ch_out[1]
        result[1::2, 0::2] = ch_out[2]
        result[1::2, 1::2] = ch_out[3]
        return np.clip(result, 0, self.white_level).astype(np.uint16)
