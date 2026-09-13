"""Savitzky-Golay smoothing, fitted in log-log.

A least-squares polynomial through a sliding window, evaluated at its centre.
Having a shape to fit with, it flattens a peak far less than a moving average
of the same width.

Two adaptations make it suit a spectrum, and both are applied here. The fit is
to ``log10(amp)`` against ``log10(f)``, where a source spectrum is close to two
straight lines and a knee, so a low order goes a long way and a negative
amplitude is unreachable. And the window is constant in log frequency rather
than in samples: the spectrum is resampled onto a uniform log-frequency grid,
filtered, and interpolated back, which makes the window a constant fraction of
a decade.

It rings. A polynomial fitted across a sharp transition overshoots on both
sides of it, which on a spectrum shows at the corner and at the Nyquist
roll-off. :class:`~specmod.smoothing.log_window.LogWindow` is the safer
default; this is for when the corner is the measurement.

References
----------
Savitzky, A. & Golay, M.J.E. (1964). Smoothing and differentiation of data by
simplified least squares procedures. *Analytical Chemistry* 36(8), 1627-1639.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from ..core.spectrum import Spectrum
from .base import record_smoothing

__all__ = ["SavitzkyGolay"]


@dataclass(frozen=True)
class SavitzkyGolay:
    """Sliding polynomial fit on a uniform log-frequency grid.

    Parameters
    ----------
    window_length
        Window width in resampled points. Must be odd and greater than
        ``polyorder``. With the default ``points_per_decade`` it is about
        ``window_length / 200`` of a decade, so 41 is roughly a fifth of a
        decade.
    polyorder
        Degree of the polynomial. 2 or 3 is usual; higher follows the noise.
    points_per_decade
        Density of the uniform log-frequency grid the filter runs on.
    """

    window_length: int = 41
    polyorder: int = 3
    points_per_decade: int = 200
    name: str = "savitzky_golay"

    def __post_init__(self) -> None:
        if self.window_length < 3 or self.window_length % 2 == 0:
            raise ValueError(
                f"window_length must be odd and at least 3, got {self.window_length}"
            )
        if self.polyorder < 1:
            raise ValueError(f"polyorder must be at least 1, got {self.polyorder}")
        if self.polyorder >= self.window_length:
            raise ValueError(
                f"polyorder ({self.polyorder}) must be below window_length "
                f"({self.window_length}); the fit is otherwise underdetermined"
            )
        if self.points_per_decade < 2:
            raise ValueError(
                f"points_per_decade must be at least 2, got {self.points_per_decade}"
            )

    def smooth(self, spectrum: Spectrum) -> Spectrum:
        freq = np.asarray(spectrum.freq, dtype=np.float64)
        amp = np.asarray(spectrum.amp, dtype=np.float64)
        out = amp.copy()

        # DC is outside log frequency; it passes through, as in `LogWindow`.
        positive = freq > 0.0
        f, a = freq[positive], amp[positive]
        if f.size < 2:
            return self._recorded(spectrum, out, resampled=0)

        log_f = np.log10(f)
        span = float(log_f[-1] - log_f[0])
        n_grid = max(int(np.ceil(span * self.points_per_decade)) + 1, 3)
        if n_grid < self.window_length:
            raise ValueError(
                f"The spectrum spans {span:.3g} decades, which is "
                f"{n_grid} points at {self.points_per_decade} per decade — "
                f"fewer than window_length={self.window_length}. Use a shorter "
                f"window or a denser grid."
            )

        grid = np.linspace(log_f[0], log_f[-1], n_grid)
        # Interpolating log10(amp) rather than amp keeps every step of this in
        # the domain the fit is defined in, so the round trip is one change of
        # variable rather than two.
        log_a = np.log10(a)
        on_grid = np.interp(grid, log_f, log_a)
        filtered = savgol_filter(
            on_grid, window_length=self.window_length, polyorder=self.polyorder
        )
        out[positive] = 10 ** np.interp(log_f, grid, filtered)
        return self._recorded(spectrum, out, resampled=n_grid)

    def _recorded(
        self, spectrum: Spectrum, amp: NDArray[np.float64], resampled: int
    ) -> Spectrum:
        return replace(
            spectrum,
            amp=amp,
            meta=record_smoothing(
                spectrum.meta,
                self.name,
                window_length=self.window_length,
                polyorder=self.polyorder,
                points_per_decade=self.points_per_decade,
                grid_points=resampled,
            ),
        )
