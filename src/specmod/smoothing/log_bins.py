"""Log-spaced frequency binning.

Replaces ``Spectrum.__bin_spectrum``, which had four problems:

1. Bin edges were hardcoded to 0.001-200 Hz regardless of the record. For a
   100 Hz trace that puts a third of the bins above Nyquist and a third below
   ``1/T``, where there is no data — so they came out empty.
2. Empty bins produced ``nan`` and a ``RuntimeWarning`` per bin, then were
   dropped. Dropping means the binned axis has a *different length* for
   different traces, which the SNR code then compares element-wise.
3. The bin count actually used came from a config dict the caller could not
   override per-call.
4. It ran unconditionally inside ``__init__``.

Here the default edges are derived from the record — ``1/T`` to Nyquist, the
band where data actually exists — and empty bins are handled explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..core.spectrum import Spectrum
from .base import record_smoothing

__all__ = ["LogBinner"]

Statistic = Literal["geometric", "mean", "median"]


@dataclass(frozen=True)
class LogBinner:
    """Average a spectrum into log-spaced frequency bins.

    Parameters
    ----------
    f_min, f_max
        Bin edges in Hz. ``None`` derives them from the record: ``1/T`` for
        ``f_min`` and Nyquist for ``f_max``. Deriving is the sensible default
        because it is exactly the band the record can represent.
    n_bins
        Number of bins.
    statistic
        How to combine samples within a bin. ``geometric`` is the mean of
        ``log10(amp)``, which is what the pre-refactor code computed and the
        right choice for a quantity spanning orders of magnitude — an
        arithmetic mean over a decade of amplitudes is dominated by its largest
        member.
    min_count
        Minimum samples for a bin to be kept when ``drop_empty`` is set.
    drop_empty
        Drop bins holding fewer than ``min_count`` samples. Log bins over a
        linearly-spaced frequency axis are inevitably sparse at the low end —
        a bin is only reliably populated above roughly ``1 / (2.3 * dlog10f * T)``
        — so dropping is usually what you want for plotting or fitting.

        Set ``False`` to keep a **fixed-length** axis with ``nan`` in the empty
        bins. Combined with explicit ``f_min``/``f_max`` that guarantees two
        spectra bin onto identical axes, which is what an element-wise
        signal-to-noise ratio requires.
    """

    f_min: float | None = None
    f_max: float | None = None
    n_bins: int = 151
    statistic: Statistic = "geometric"
    min_count: int = 1
    drop_empty: bool = True
    name: str = "log_bins"

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be at least 1, got {self.n_bins}")
        if self.min_count < 1:
            raise ValueError(f"min_count must be at least 1, got {self.min_count}")
        if self.f_min is not None and self.f_min <= 0:
            raise ValueError(
                f"f_min must be positive for log spacing, got {self.f_min}"
            )
        if (
            self.f_min is not None
            and self.f_max is not None
            and self.f_min >= self.f_max
        ):
            raise ValueError(f"f_min ({self.f_min}) must be below f_max ({self.f_max})")

    def edges_for(self, spectrum: Spectrum) -> NDArray[np.float64]:
        """Bin edges for a given spectrum.

        Explicit ``f_min``/``f_max`` are honoured **exactly** and never clamped
        to the spectrum's own range. That matters more than it looks: signal and
        noise windows have different durations, so clamping would bin them onto
        different axes, and the SNR ratio compares them element-wise. Pinning
        both edges is how a caller guarantees a shared axis.

        Only derived bounds are clamped, since deriving already means "whatever
        this record supports".
        """
        if self.f_min is not None:
            lo = self.f_min
        else:
            lo = max(spectrum.frequency_resolution, float(spectrum.freq[0]))
        if self.f_max is not None:
            hi = self.f_max
        else:
            hi = min(spectrum.nyquist, float(spectrum.freq[-1]))
        if lo >= hi:
            raise ValueError(
                f"Binning range [{lo:.4g}, {hi:.4g}] Hz is empty for a spectrum "
                f"spanning [{spectrum.freq[0]:.4g}, {spectrum.freq[-1]:.4g}] Hz."
            )
        return np.logspace(np.log10(lo), np.log10(hi), self.n_bins + 1)

    def smooth(self, spectrum: Spectrum) -> Spectrum:
        edges = self.edges_for(spectrum)
        # Bin centres are the geometric midpoints, matching the log spacing.
        centres = np.sqrt(edges[:-1] * edges[1:])

        idx = np.digitize(spectrum.freq, edges) - 1
        valid = (idx >= 0) & (idx < self.n_bins)
        if not valid.any():
            raise ValueError(
                f"Binning range [{edges[0]:.4g}, {edges[-1]:.4g}] Hz does not "
                f"overlap the spectrum, which spans "
                f"[{spectrum.freq[0]:.4g}, {spectrum.freq[-1]:.4g}] Hz."
            )
        idx, amp = idx[valid], spectrum.amp[valid]

        counts = np.bincount(idx, minlength=self.n_bins)
        values = np.full(self.n_bins, np.nan)

        if self.statistic == "median":
            for b in np.flatnonzero(counts):
                values[b] = np.median(amp[idx == b])
        else:
            data = np.log10(amp) if self.statistic == "geometric" else amp
            sums = np.bincount(idx, weights=data, minlength=self.n_bins)
            with np.errstate(invalid="ignore", divide="ignore"):
                means = sums / counts
            values = 10**means if self.statistic == "geometric" else means

        sparse = counts < self.min_count
        values[sparse] = np.nan
        if sparse.all():
            raise ValueError(
                f"No bin reached min_count={self.min_count}. The spectrum has "
                f"{len(spectrum)} samples across {self.n_bins} bins spanning "
                f"[{edges[0]:.4g}, {edges[-1]:.4g}] Hz; use fewer bins, a lower "
                f"min_count, or a narrower range."
            )
        keep = ~sparse if self.drop_empty else np.ones(self.n_bins, dtype=bool)

        return replace(
            spectrum,
            freq=centres[keep],
            amp=values[keep],
            meta=record_smoothing(
                spectrum.meta,
                self.name,
                n_bins=self.n_bins,
                statistic=self.statistic,
                f_min=float(edges[0]),
                f_max=float(edges[-1]),
                counts=counts[keep].tolist(),
                drop_empty=self.drop_empty,
            ),
        )
