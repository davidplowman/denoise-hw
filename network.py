"""Selects which denoiser architecture the rest of the codebase uses.

Switch networks by changing this import alone -- everything else (train.py,
convert.py, denoise_network.py, ...) imports DenoiseNet from here, not from
a specific architecture's module.
"""

from nafnet import DenoiseNet

__all__ = ["DenoiseNet"]
