"""Load and expose sensor tuning parameters from pisp JSON tuning files."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

_TUNINGS_DIR = Path(__file__).parent / "tunings"


@dataclass
class GeqParams:
    """Green Equalisation (GEQ) parameters from the rpi.geq block."""
    offset: float
    slope: float


@dataclass
class Noise2Entry:
    """One calibration point for the rpi.noise2 model."""
    gain: int
    coefficients: tuple[float, float, float]  # (constant, slope1, slope2)


@dataclass
class Noise2Model:
    """Per-gain polynomial noise model: std = c + s1*sqrt(mean) + s2*mean."""
    calibration: list[Noise2Entry]


@dataclass
class NoiseModel:
    """Noise variance model: var = constant + slope * gain."""
    constant: float
    slope: float


@dataclass
class CdnParams:
    """Colour Denoise (CDN) parameters from the tuning file."""
    deviation: float
    strength: float


@dataclass
class SdnParams:
    """Spatial Denoise (SDN) parameters from the tuning file."""
    deviation: float
    strength: float
    deviation2: float
    deviation_no_tdn: float
    strength_no_tdn: float


@dataclass
class DenoiseMode:
    """Denoise parameters for a single operating mode (normal / hdr / night)."""
    sdn: SdnParams
    cdn: CdnParams | None


class Tuning:
    """Parsed sensor tuning from a pisp v2.0 JSON file.

    Attributes:
        sensor_name: Name of the sensor (e.g. "imx219").
        noise_model: Noise variance model coefficients.
        geq: Green equalisation parameters, or ``None`` if absent from the
            tuning file.
    """

    def __init__(self, sensor_name: str, data: dict) -> None:
        self.sensor_name = sensor_name
        self.noise_model: NoiseModel = self._parse_noise(data)
        self.noise2_model: Noise2Model | None = self._parse_noise2(data)
        self._denoise_modes: dict[str, DenoiseMode] = self._parse_denoise(data)
        self.geq: GeqParams | None = self._parse_geq(data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_noise2_calibration(self, gain: float) -> tuple[float, float, float]:
        """Return interpolated noise2 polynomial coefficients for the given gain.

        Finds the two bracketing calibration entries and interpolates linearly
        between them in std space at three sample points (mean = 0, 4096, 16384),
        then re-fits to recover (constant, slope1, slope2). Each entry's std
        values are scaled by (gain / entry.gain) before interpolating.

        If gain lies outside the calibration range, the nearest entry is used
        directly (with the same power scaling) and no interpolation is performed.

        Args:
            gain: Analogue gain value.

        Returns:
            (constant, slope1, slope2) coefficients such that
            std = constant + slope1 * sqrt(mean) + slope2 * mean.
        """
        def eval_std(mean, coeffs):
            return coeffs[0] + coeffs[1] * math.sqrt(mean) + coeffs[2] * mean

        cal = self.noise2_model.calibration
        lo = next((e for e in reversed(cal) if e.gain <= gain), None)
        hi = next((e for e in cal if e.gain >= gain), None)

        if lo is None:
            return tuple([coeff * math.sqrt(gain / hi.gain) for coeff in hi.coefficients])
        elif hi is None:
            return tuple([coeff * math.sqrt(gain / lo.gain) for coeff in lo.coefficients])

        if lo is hi:
            return tuple(lo.coefficients)

        sample_means = [0, 4096, 16384]
        lo_stds = [eval_std(m, lo.coefficients) * math.sqrt(gain / lo.gain) for m in sample_means]
        hi_stds = [eval_std(m, hi.coefficients) * math.sqrt(gain / hi.gain) for m in sample_means]

        # Interpolate from lo_stds to hi_stds as gain goes from lo.gain to hi.gain.
        t = (gain - lo.gain) / (hi.gain - lo.gain)
        s0, s1, s2 = [ls + t * (hs - ls) for ls, hs in zip(lo_stds, hi_stds)]

        # Exact re-fit: std = a + b*sqrt(mean) + c*mean at means [0, 4096, 16384]
        a = s0
        c = (s2 - 2 * s1 + s0) / 8192
        b = (s1 - s0 - 4096 * c) / 64
        return (a, b, c)

    def denoise_mode(self, mode: str = "normal") -> DenoiseMode:
        """Return SDN parameters for the requested operating mode.

        Args:
            mode: One of "normal", "hdr", or "night".

        Returns:
            DenoiseMode containing SDN parameters for that mode.

        Raises:
            KeyError: If *mode* is not present in the tuning file.
        """
        if mode not in self._denoise_modes:
            raise KeyError(
                f"Denoise mode '{mode}' not found for sensor '{self.sensor_name}'. "
                f"Available: {list(self._denoise_modes)}"
            )
        return self._denoise_modes[mode]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_algo(data: dict, name: str) -> dict:
        for entry in data.get("algorithms", []):
            if name in entry:
                return entry[name]
        raise KeyError(f"Algorithm '{name}' not found in tuning data")

    def _parse_noise2(self, data: dict) -> Noise2Model | None:
        try:
            n = self._find_algo(data, "rpi.noise2")
        except KeyError:
            return None
        entries = [
            Noise2Entry(gain=int(e["gain"]), coefficients=tuple(e["coefficients"]))
            for e in n["calibration"]
        ]
        return Noise2Model(calibration=entries)

    def _parse_geq(self, data: dict) -> GeqParams | None:
        try:
            g = self._find_algo(data, "rpi.geq")
        except KeyError:
            return None
        return GeqParams(offset=float(g["offset"]), slope=float(g["slope"]))

    def _parse_noise(self, data: dict) -> NoiseModel:
        n = self._find_algo(data, "rpi.noise")
        return NoiseModel(
            constant=float(n["reference_constant"]),
            slope=float(n["reference_slope"]),
        )

    def _parse_denoise(self, data: dict) -> dict[str, DenoiseMode]:
        d = self._find_algo(data, "rpi.denoise")
        modes: dict[str, DenoiseMode] = {}
        for mode_name in ("normal", "hdr", "night"):
            if mode_name not in d:
                continue
            sdn = d[mode_name]["sdn"]
            cdn_raw = d[mode_name].get("cdn")
            modes[mode_name] = DenoiseMode(
                sdn=SdnParams(
                    deviation=float(sdn["deviation"]),
                    strength=float(sdn["strength"]),
                    deviation2=float(sdn["deviation2"]),
                    deviation_no_tdn=float(sdn["deviation_no_tdn"]),
                    strength_no_tdn=float(sdn["strength_no_tdn"]),
                ),
                cdn=CdnParams(
                    deviation=float(cdn_raw["deviation"]),
                    strength=float(cdn_raw["strength"]),
                ) if cdn_raw is not None else None,
            )
        return modes


def load_tuning(sensor_name: str) -> Tuning:
    """Load tuning parameters for the named sensor.

    Args:
        sensor_name: Sensor identifier, e.g. "imx219".  Must match a file
                     in the ``tunings/`` directory (without the .json suffix).

    Returns:
        A :class:`Tuning` instance populated from the JSON file.

    Raises:
        FileNotFoundError: If no tuning file exists for *sensor_name*.
    """
    path = _TUNINGS_DIR / f"{sensor_name}.json"
    if not path.is_file():
        available = sorted(p.stem for p in _TUNINGS_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"No tuning file for sensor '{sensor_name}'. "
            f"Available sensors: {available}"
        )
    with path.open() as fh:
        data = json.load(fh)
    return Tuning(sensor_name, data)
