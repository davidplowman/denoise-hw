"""Selects which denoiser architecture the rest of the codebase uses.

Switch networks by changing this import alone -- everything else (train.py,
convert.py, denoise_network.py, ...) imports DenoiseNet from here, not from
a specific architecture's module.
"""

from denoisenet import DenoiseNet, depth_to_space, space_to_depth

__all__ = ["DenoiseNet", "space_to_depth", "depth_to_space"]
