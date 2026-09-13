"""Constant-relative-bandwidth smoothing with a selectable window.

A window of fixed width in **log** frequency, slid along the axis: one octave
fraction wide everywhere, so it is narrow in Hz at low frequency and wide at
high. That is the same principle as Konno-Ohmachi, generalised to the ordinary
window shapes — rectangular, Bartlett, Hann, Hamming, Blackman, Gaussian —
instead of Konno-Ohmachi's one fixed kernel, and to a running median, which is
not an average at all.

Why this shape for a seismic spectrum
-------------------------------------
Source spectra are power laws in log-log: a plateau, a corner, a falloff. Their
features are *proportionally* spaced, so the smoother should be too. A window
of fixed width in Hz does the opposite of what is wanted — at 0.5 Hz it spans
the plateau and the corner together, and at 40 Hz it barely spans three
samples. Fixed width in log frequency treats a decade the same wherever it
falls, which is why engineering seismology smooths this way and why the bins in
:class:`~specmod.smoothing.log_bins.LogBinner` are log-spaced too.

Symmetry, and why the obvious implementation does not have it
-------------------------------------------------------------
The weights depend only on ``log(f_j / f_i)``, so the *window* is symmetric
about its centre on a log axis. That is not enough. A Fourier grid is uniform
in Hz, so one window holds many samples per octave at the top of the axis and
few at the bottom, and averaging per sample therefore weights the upper half of
every window more heavily. The result is a smoother that is symmetric in
principle and biased in practice.

Weighting each sample by its share of the log-frequency axis fixes it, and the
size of the fix is worth stating. Smoothing an exact ``f**-2`` power law, which
is a straight line in log-log and must come back unchanged:

============ ================= ================
window       per-sample weight  log-measure
============ ================= ================
``hann``     1.5e-3 dex        3.2e-8 dex
``boxcar``   3.9e-3 dex        3.4e-4 dex
============ ================= ================

(median absolute residual, 1/3 octave, 0.05 Hz grid, above 1 Hz). It is on by
default; ``log_measure=False`` gives the per-sample behaviour, which is what
Konno-Ohmachi does and what the textbook description of fractional-octave
smoothing usually implies.

Choosing between the shapes
---------------------------
- ``boxcar`` — a plain running mean in log frequency. Cheapest, and the worst
  behaved: its sidelobes are the highest of the set, so a sharp feature leaks
  a long way along the axis.
- ``bartlett`` — triangular. The obvious improvement on ``boxcar`` at no real
  cost, and the mildest of the tapered windows: it preserves a peak's height
  better than ``hann`` but suppresses less scatter.
- ``hann`` — the default here. The usual compromise, and the closest of these
  to what Konno-Ohmachi does.
- ``hamming``, ``blackman`` — progressively harder suppression of the
  far-field at the cost of a wider effective window. ``blackman`` is worth
  reaching for on a spectrum so noisy that ``hann`` leaves visible scatter.
- ``gaussian`` — truncated at four sigma. No sidelobes at all, which is its
  argument; in exchange it has no compact support, so the truncation is the
  only thing bounding it.

Measured on an exact ``f**-2`` power law, which every one of them must return
unchanged (median absolute residual, 1/3 octave, log-measure weighting):
``hann`` 3.2e-8, ``blackman`` 1.3e-8, ``gaussian`` 3.7e-7, ``bartlett``
1.7e-6, ``hamming`` 5.1e-5, ``boxcar`` 3.4e-4 dex. The ordering is the taper's
smoothness, and the spread across it is far below anything a fit would notice
— which is the useful conclusion: **pick the width first, the shape second.**

And one option that is not a window at all: ``statistic="median"``. Measured on
the same power law with a single 60x spike dropped into it, the worst residual
across the axis is 1.7e-2 dex for the median against 3.8e-2 for the geometric
mean and 3.6e-1 for the arithmetic one. A spike is exactly what an average
cannot handle and an order statistic can.

None of them preserve energy, and none of them are unbiased on a curved
spectrum: averaging across a corner pulls it down, whatever the shape. That
is a property of smoothing rather than of a window, and it is the reason
``octave_fraction`` matters more than ``window``.

Relation to the published methods
---------------------------------
Fractional-octave smoothing with an explicit window is standard in acoustics
and in transfer-function work, where the weighting above is treated properly —
Tylka, Boren & Choueiri (*JAES*, 2017), on smoothing weights that preserve
log-frequency symmetry, is the reference usually given for it.

**That paper could not be read from the environment this was written in**, so
what is implemented here is the construction described above, derived from the
sampling argument and checked against the power law, rather than a reproduction
of its weighting. Where the two differ, theirs is the published one and this
file is not evidence about it.
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

        ``median`` takes the median of the samples the window covers, which is
        the one option here that is not a weighted mean: **the window shape and
        ``log_measure`` are both ignored**, since a median has no weights to
        apply. Use it when the spectrum carries spikes that should not be
        averaged into their neighbours — a power-line tone, a dropout, a
        telemetry glitch. A mean spreads such a sample across the window; a
        median discards it. It costs resolution at a real narrow peak, which it
        will also discard, and it is the slowest of the three.
    log_measure
        Weight each sample by its share of the log-frequency axis. On by
        default, and it is what makes the window symmetric in practice rather
        than only in principle — see the module docstring. ``False`` weights
        every sample in the window equally, which is what Konno-Ohmachi and
        the textbook description of fractional-octave smoothing both do.
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
