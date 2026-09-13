"""Savitzky-Golay smoothing, fitted in log-log.

A least-squares polynomial through a sliding window, evaluated at its centre.
Unlike a moving average it has a shape to fit with, so it flattens a peak far
less: the classic use is preserving the height and width of a spectral line
that a running mean would erode.

Two adaptations are needed before it suits a seismic spectrum, and both are
here rather than left to the caller:

**It is fitted to** ``log10(amp)`` **against** ``log10(f)``. A source spectrum
spans orders of magnitude in both, and a polynomial fitted to raw amplitude
against raw frequency spends its degrees of freedom on the falloff and has
nothing left for the corner. In log-log the same model is close to the thing
being fitted — two straight lines and a knee — so a low order goes a long way.
It also cannot return a negative amplitude, which a polynomial fitted to raw
amplitude can and does in the noise floor.

**The window is constant in log frequency, not in samples.** Savitzky-Golay
assumes uniform spacing, and a Fourier axis is uniform in Hz, so a fixed sample
count spans a decade at the bottom of the axis and a per-cent at the top. The
spectrum is resampled onto a uniform log-frequency grid, filtered there, and
interpolated back — which makes the window a constant fraction of a decade, the
same constant-relative-bandwidth idea as
:class:`~specmod.smoothing.log_window.LogWindow` and Konno-Ohmachi.

What it costs
-------------
- **Ringing.** A polynomial fitted across a sharp step overshoots on both sides
  of it. On a spectrum that mostly matters at the corner and at the Nyquist
  roll-off, where a visible lobe can appear that the data does not have.
- **Two resamplings.** Going onto the log grid and back is interpolation, so
  the result is not exactly a filtered version of the input samples. Raising
  ``points_per_decade`` reduces that and costs time.
- **It is not a weighted mean**, so unlike the window smoothers it has no
  guarantee of staying inside the range of its input.

Worth it where the corner frequency is the measurement and the spectrum is
noisy enough to need smoothing at all; ``LogWindow`` is the safer default.

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
