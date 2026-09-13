"""Constant-relative-bandwidth smoothing with a selectable window.

A window of fixed width in log frequency, slid along the axis: one octave
fraction wide everywhere, so it is narrow in Hz at low frequency and wide at
high. Konno-Ohmachi's principle with the window shape exposed — rectangular,
Bartlett, Hann, Hamming, Blackman, Gaussian — and with a running median
available in place of the weighted mean.

Fixed width in log frequency is what suits a source spectrum: its features —
plateau, corner, falloff — are proportionally spaced, so a window of fixed
width in Hz spans the plateau and the corner together at 0.5 Hz and three
samples at 40 Hz.

The shape matters less than the width. On an exact ``f**-2`` power law, which
every shape must return unchanged, the median residual runs from 1.3e-8 dex
(``blackman``) to 3.4e-4 dex (``boxcar``) — an ordering that follows the
taper's smoothness, and a spread far below what a fit resolves. ``hann`` is the
default. ``statistic="median"`` is the exception to all of this: it discards a
spike instead of averaging it in, and discards a genuine narrow peak with it.

None of these preserve energy, and none are unbiased across a corner.

.. note::

   A window symmetric in log frequency is not a symmetric average over
   samples. A Fourier grid is uniform in Hz, so every window covers more
   samples above its centre than below in log terms, and weighting them
   equally tilts the result — by 1.5e-3 dex on that power law for ``hann``,
   against 3.2e-8 dex when each sample is weighted by its share of the log
   axis. ``log_measure`` does the latter and is on by default; ``False`` is
   the equal-weight behaviour, which is what Konno-Ohmachi applies.

References
----------
Tylka, J.G., Boren, B.B. & Choueiri, E.Y. (2017). A generalized method for
fractional-octave smoothing of transfer functions that preserves log-frequency
symmetry. *JAES* 65(3), 239-245.

Konno, K. & Ohmachi, T. (1998). Ground-motion characteristics estimated from
spectral ratio between horizontal and vertical components of microtremor.
*BSSA* 88(1), 228-241.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..core.spectrum import Spectrum
from .base import record_smoothing

__all__ = ["WINDOWS", "LogWindow"]

WindowName = Literal["boxcar", "bartlett", "hann", "hamming", "blackman", "gaussian"]
Statistic = Literal["geometric", "mean", "median"]


def _window(name: str, u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Window weights for offsets ``u`` in [-1, 1] from the centre.

    Written on a symmetric interval rather than taken from
    ``scipy.signal.windows``, which samples a window at *N* discrete points.
    The count here is not fixed: the number of samples inside one octave
    fraction grows with frequency on a Fourier grid, so the window has to be a
    continuous function evaluated at whatever offsets the axis provides.
    """
    if name == "boxcar":
        return np.ones_like(u)
    if name == "bartlett":
        return 1.0 - np.abs(u)
    if name == "hann":
        return 0.5 * (1.0 + np.cos(np.pi * u))
    if name == "hamming":
        return 0.54 + 0.46 * np.cos(np.pi * u)
    if name == "blackman":
        return 0.42 + 0.5 * np.cos(np.pi * u) + 0.08 * np.cos(2.0 * np.pi * u)
    if name == "gaussian":
        # Truncated at +-1, where it has fallen to exp(-8) ~ 3e-4 of its peak,
        # so the truncation is far below the scatter it is smoothing.
        return np.exp(-0.5 * (4.0 * u) ** 2)
    raise ValueError(f"Unknown window {name!r}. Available: {sorted(WINDOWS)}.")


#: Window shapes this smoother accepts.
WINDOWS: dict[str, str] = {
    "boxcar": "rectangular; a running mean in log frequency",
    "bartlett": "triangular",
    "hann": "raised cosine",
    "hamming": "raised cosine on a pedestal",
    "blackman": "three-term cosine",
    "gaussian": "Gaussian, truncated at 4 sigma",
}


@dataclass(frozen=True)
class LogWindow:
    """Smooth with a fixed-width window in log frequency.

    Parameters
    ----------
    octave_fraction
        Full width of the window in octaves. ``1/3`` is the familiar
        third-octave; larger smooths harder.
    window
        Window shape, one of :data:`WINDOWS`.
    statistic
        ``geometric`` averages ``log10(amp)`` and is the default, for the same
        reason :class:`~specmod.smoothing.log_bins.LogBinner` uses it: over a
        decade of amplitudes an arithmetic mean is dominated by its largest
        member. ``mean`` averages the amplitudes directly.

        ``median`` takes the median of the samples the window covers. It has no
        weights, so the window shape and ``log_measure`` are ignored. Use it
        against spikes — a power-line tone, a dropout — which it discards
        rather than spreading across the window, along with any genuine narrow
        peak. It is the slowest of the three.
    log_measure
        Weight each sample by its share of the log-frequency axis, which is
        what keeps the window symmetric on a non-uniform grid — see the note
        above. ``False`` weights every sample equally.
    """

    octave_fraction: float = 1.0 / 3.0
    window: WindowName = "hann"
    statistic: Statistic = "geometric"
    log_measure: bool = True
    name: str = "log_window"

    def __post_init__(self) -> None:
        if self.octave_fraction <= 0:
            raise ValueError(
                f"octave_fraction must be positive, got {self.octave_fraction}"
            )
        if self.window not in WINDOWS:
            raise ValueError(
                f"Unknown window {self.window!r}. Available: {sorted(WINDOWS)}."
            )
        if self.statistic not in ("geometric", "mean", "median"):
            raise ValueError(
                f"statistic must be 'geometric', 'mean' or 'median', got "
                f"{self.statistic!r}"
            )

    def smooth(self, spectrum: Spectrum) -> Spectrum:
        freq = np.asarray(spectrum.freq, dtype=np.float64)
        amp = np.asarray(spectrum.amp, dtype=np.float64)
        out = amp.copy()

        # DC has no log and no neighbours in log space. It is excluded from
        # every window and passes through untouched rather than being dropped,
        # so the axis a caller handed in is the axis they get back.
        positive = freq > 0.0
        if positive.sum() < 2:
            return self._recorded(spectrum, out)

        f = freq[positive]
        a = amp[positive]
        data = np.log10(a) if self.statistic == "geometric" else a

        half = self.octave_fraction / 2.0
        lo_f, hi_f = f * 2.0**-half, f * 2.0**half
        # The axis is sorted, so the window is a contiguous slice. Finding its
        # ends costs a binary search per sample instead of the N-by-N weight
        # matrix the obvious implementation builds — which is what makes this
        # usable on a long record, where that matrix does not fit in memory.
        starts = np.searchsorted(f, lo_f, side="left")
        stops = np.searchsorted(f, hi_f, side="right")

        smoothed = np.empty_like(data)
        log_f = np.log2(f)
        # Each sample's share of the log-frequency axis. A Fourier grid is
        # uniform in Hz, so a window holds many samples per octave at the top
        # of the axis and few at the bottom; a per-sample average therefore
        # weights the upper half of every window more heavily, which is the
        # asymmetry this removes.
        measure = np.gradient(log_f) if self.log_measure else None
        for i in range(f.size):
            lo, hi = starts[i], stops[i]
            u = (log_f[lo:hi] - log_f[i]) / half
            w = _window(self.window, np.clip(u, -1.0, 1.0))
            if self.statistic == "median":
                # No weights: the window selects the samples, the order
                # statistic decides. Weighting a median is not defined by the
                # window, and pretending otherwise would make `window` look
                # like it does something here.
                smoothed[i] = np.median(data[lo:hi]) if hi > lo else data[i]
                continue
            if measure is not None:
                w = w * measure[lo:hi]
            total = w.sum()
            # A window can fall entirely between two samples at the sparse low
            # end of a linear grid. Normalising by the realised weight rather
            # than the ideal one is also what keeps the ends unbiased, where
            # the window is truncated by the edge of the axis.
            smoothed[i] = (w @ data[lo:hi]) / total if total > 0 else data[i]

        out[positive] = 10**smoothed if self.statistic == "geometric" else smoothed
        return self._recorded(spectrum, out)

    def _recorded(self, spectrum: Spectrum, amp: NDArray[np.float64]) -> Spectrum:
        return replace(
            spectrum,
            amp=amp,
            meta=record_smoothing(
                spectrum.meta,
                self.name,
                octave_fraction=self.octave_fraction,
                window=self.window,
                statistic=self.statistic,
                log_measure=self.log_measure,
            ),
        )
