"""
Add realistic sensor noise to a noise-free raw (Bayer) image from the imx662.

All values are in the 16-bit DNG units used by the camera: the sensor's 12-bit
values multiplied by 16. The same model applies to every Bayer channel.

The noise has two independent parts:

1. Shot noise. Light arrives as a whole number of photo-electrons, which follows
   a Poisson distribution. Each electron is worth 10 * gain DN. This is the part
   that dominates in lit regions.

2. Dark (read) noise. This is present even with no light. Its RMS is
   9.9 + 7.05 * gain DN. It is mostly Gaussian, but about 2.4% of samples are
   much larger ("outliers"). We model this as a mix of two Gaussians: a narrow
   one for most pixels and a wide one for the occasional outliers.

Because both parts are always added, the same code handles lit and dark regions:
in lit regions the shot noise swamps the dark-noise outliers, and in dark
regions the outliers show through, just as on the real sensor.

The black level is not passed in: it is worked out from the gain, using a
piecewise linear curve measured from dark frames. The measured curves for R, G
and B were very similar, so a single curve (their average) is used for all
channels.

The result is rounded to the sensor's 12-bit steps (16 DN) and clipped to the
valid range, so it can go below the black level and clip at 0 like real data.

Defect maps (optional)
----------------------
On the real sensor the dark-noise outliers are not scattered at random: they
come from particular pixels. A defect map records which pixels those are, so the
same pixels misbehave in every image it is used with. It holds:

- a noise scale for every pixel. Most pixels are slightly quieter than average
  (about 0.85x) and a few are much noisier (up to about 10x).
- a list of "telegraph" (RTS) pixels, about 1.65% of the sensor. In each image
  such a pixel is either at its normal level or raised by a fixed step of a few
  electrons, chosen at random each time with its own probability.

Make a map once with make_defect_map() and save_defect_map(), then load it with
load_defect_map() and pass it to add_noise(..., defects=map) for every image.
Without a map, add_noise() uses the two-Gaussian mix described above, which has
the same overall statistics but puts the outliers on random pixels.
"""

import numpy as np

# Constants measured from the calibration images (16-bit DNG units).
DN_PER_ELECTRON_PER_GAIN = 10.0   # one photo-electron = 10 * gain DN
DARK_RMS_CONSTANT = 9.9           # dark noise RMS = 9.9 + 7.05 * gain
DARK_RMS_PER_GAIN = 7.05
OUTLIER_FRACTION = 0.024          # 2.4% of dark-noise samples are outliers...
OUTLIER_WIDTH = 3.5               # ...drawn from a Gaussian 3.5x the dark RMS
# Width of the narrow Gaussian, chosen so the overall dark RMS comes out exactly
# right (about 0.85).
CORE_WIDTH = np.sqrt((1 - OUTLIER_FRACTION * OUTLIER_WIDTH**2) / (1 - OUTLIER_FRACTION))
ADC_STEP = 16                     # 12-bit sensor values stored as multiples of 16
MAX_VALUE = 65520                 # largest possible value (4095 * 16)

# Defect map statistics, measured from 240 dark frames.
SENSOR_SHAPE = (1096, 1936)
# Per-pixel noise scale (a multiple of the dark RMS), given as a table of its
# distribution: a fraction NOISE_SCALE_FRACTIONS[i] of pixels have a scale below
# NOISE_SCALE_VALUES[i].
NOISE_SCALE_FRACTIONS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                         0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 1.0]
NOISE_SCALE_VALUES = [0.575, 0.668, 0.706, 0.742, 0.762, 0.789, 0.810, 0.831, 0.853, 0.878,
                      0.913, 0.966, 1.077, 1.242, 1.437, 1.658, 2.159, 2.763, 3.636, 4.305,
                      4.974, 6.546, 10.832]
RTS_FRACTION = 0.0165             # 1.65% of pixels are telegraph (RTS) pixels
RTS_STEP_MEDIAN = 5.0             # their step, in electrons: lognormal around 5 e-
RTS_STEP_SPREAD = 0.40            # (log standard deviation; 90% lie in 2.6-9.6 e-)
RTS_HIGH_MEDIAN = 0.083           # fraction of images in which each is raised:
RTS_HIGH_SPREAD = 0.87            # lognormal around 8.3% (90% lie in 2-35%)...
RTS_HIGH_MAX = 0.5                # ...but never more than half the time

# Black level versus gain, as (gain, black level) knee points. This is the
# average of the R, G and B curves measured from the dark frames. Between knee
# points the value is interpolated linearly; outside them it is held flat.
BLACK_LEVEL_GAINS = [4, 16, 64, 128, 256, 512]
BLACK_LEVEL_VALUES = [3203, 3192, 3171, 3131, 3120, 3063]


def black_level(gain):
    """The black level for an image captured at the given gain."""
    return np.interp(gain, BLACK_LEVEL_GAINS, BLACK_LEVEL_VALUES)


def noise_std(pixel, gain):
    """The standard deviation the added noise will have (for reference)."""
    signal = np.maximum(np.asarray(pixel, dtype=float) - black_level(gain), 0)
    dark_rms = DARK_RMS_CONSTANT + DARK_RMS_PER_GAIN * gain
    return np.sqrt(dark_rms**2 + DN_PER_ELECTRON_PER_GAIN * gain * signal)


def make_defect_map(shape=SENSOR_SHAPE, seed=None):
    """
    Create a random defect map for a sensor of the given shape (rows, columns).

    Use the same seed to get the same map again.
    """
    rng = np.random.default_rng(seed)

    # A noise scale for every pixel, drawn from the measured distribution.
    noise_scale = np.interp(rng.random(shape), NOISE_SCALE_FRACTIONS, NOISE_SCALE_VALUES)

    # Choose the telegraph pixels, and give each a step size and a probability
    # of being raised.
    rts_rows, rts_cols = np.nonzero(rng.random(shape) < RTS_FRACTION)
    count = len(rts_rows)
    rts_step = RTS_STEP_MEDIAN * np.exp(RTS_STEP_SPREAD * rng.standard_normal(count))
    rts_high = RTS_HIGH_MEDIAN * np.exp(RTS_HIGH_SPREAD * rng.standard_normal(count))
    rts_high = np.minimum(rts_high, RTS_HIGH_MAX)

    return {"noise_scale": noise_scale.astype(np.float32),
            "rts_rows": rts_rows, "rts_cols": rts_cols,
            "rts_step": rts_step, "rts_high": rts_high}


def save_defect_map(defects, filename):
    """Save a defect map to a .npz file."""
    np.savez_compressed(filename, **defects)


def load_defect_map(filename):
    """Load a defect map saved by save_defect_map()."""
    with np.load(filename) as f:
        return {name: f[name] for name in f.files}


def add_noise(pixel, gain, rng=None, defects=None):
    """
    Return a noisy version of `pixel`.

    pixel:   noise-free value(s) P, including the black level, as a single number
             or a numpy array (for example a whole Bayer mosaic).
    gain:    the linear analogue gain G the image was captured with.
    defects: optional defect map from make_defect_map() or load_defect_map().
             `pixel` must then be a whole image of the same shape as the map.
    """
    if rng is None:
        rng = np.random.default_rng()
    pixel = np.asarray(pixel, dtype=float)
    black = black_level(gain)

    # 1. Shot noise: turn the signal into a whole number of electrons, drawn
    #    from a Poisson distribution, and convert back to DN.
    dn_per_electron = DN_PER_ELECTRON_PER_GAIN * gain
    signal = pixel - black
    lit = np.maximum(signal, 0)      # a noise-free image shouldn't be below black,
    below = np.minimum(signal, 0)    # but keep any small negative part unchanged
    electrons = rng.poisson(lit / dn_per_electron)
    noisy_signal = electrons * dn_per_electron + below

    # 2. Dark noise.
    dark_rms = DARK_RMS_CONSTANT + DARK_RMS_PER_GAIN * gain
    if defects is None:
        # No map: each pixel picks the narrow or the wide Gaussian at random.
        is_outlier = rng.random(pixel.shape) < OUTLIER_FRACTION
        width = np.where(is_outlier, OUTLIER_WIDTH, CORE_WIDTH) * dark_rms
        dark_noise = rng.standard_normal(pixel.shape) * width
    else:
        if pixel.shape != defects["noise_scale"].shape:
            raise ValueError("image and defect map must be the same shape")
        # Each pixel has its own noise level, taken from the map...
        dark_noise = rng.standard_normal(pixel.shape) * defects["noise_scale"] * dark_rms
        # ...and each telegraph pixel is raised by its step if it comes up "high".
        raised = rng.random(len(defects["rts_step"])) < defects["rts_high"]
        steps = np.where(raised, defects["rts_step"] * dn_per_electron, 0)
        dark_noise[defects["rts_rows"], defects["rts_cols"]] += steps

    # 3. Put it back together the way the sensor would report it: rounded to
    #    the 12-bit step and clipped to the valid range.
    noisy = black + noisy_signal + dark_noise
    noisy = np.round(noisy / ADC_STEP) * ADC_STEP
    return np.clip(noisy, 0, MAX_VALUE)


if __name__ == "__main__":
    import sys

    # "python3 synthetic_noise.py make-map defects.npz [seed]" creates and saves a
    # defect map.
    if len(sys.argv) >= 3 and sys.argv[1] == "make-map":
        seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
        defects = make_defect_map(seed=seed)
        save_defect_map(defects, sys.argv[2])
        print(f"saved {sys.argv[2]}: {len(defects['rts_step'])} telegraph pixels")
        sys.exit()

    # Otherwise run a demonstration.
    # Example: a flat grey patch and a black patch at gain 5.82.
    rng = np.random.default_rng(1)
    G = 5.82
    B = black_level(G)
    print(f"gain {G}: black level {B:.1f}")
    for P in (B, B + 200, B + 5000, B + 30000):
        clean = np.full(100_000, float(P))
        noisy = add_noise(clean, G, rng)
        print(f"P - B = {P - B:6.0f}: measured std {noisy.std():7.1f}, "
              f"formula {noise_std(P, G):7.1f}, mean shift {noisy.mean() - P:+5.1f}")

    # Example: make a defect map and use it for a dark image. (Normally you would
    # save it with save_defect_map() and reload it with load_defect_map().)
    defects = make_defect_map(seed=1)
    print(f"defect map: {len(defects['rts_step'])} telegraph pixels")
    dark = add_noise(np.full(SENSOR_SHAPE, B), G, rng, defects=defects)
    print(f"dark image with defect map: std {dark.std():.1f}, formula {noise_std(B, G):.1f}")
