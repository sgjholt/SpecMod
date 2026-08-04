"""The time-frequency surface a CWT produces, and the QC it makes possible.

A :class:`Scalogram` is deliberately *not* an amplitude spectrum. Its ``power``
is ``|W(a,b)|**2`` in the L2-Morlet convention, which carries units of
``[signal]**2 * time`` — documented, but not comparable to a Fourier amplitude
spectrum and not something to fit a source model to.

The conversion happens in exactly one place, :meth:`Scalogram.time_average`,
which applies the ``C_delta`` and ``dj*dt`` bridge and returns an ordinary
:class:`~specmod.core.spectrum.Spectrum`. One normalisation path, one test. A
second "already normalised" surface would be a second thing to get wrong.

References
----------
Torrence, C. and Compo, G.P. (1998). A practical guide to wavelet analysis.
*Bulletin of the American Meteorological Society* 79(1), 61-78.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .spectrum import Spectrum
from .units import AmplitudeKind, Motion

__all__ = ["Scalogram", "ScalogramQC"]


@dataclass(frozen=True)
class ScalogramQC:
    """Quality checks a time-frequency surface makes possible.

    An amplitude-only signal-to-noise test cannot see any of these: it collapses
    the time axis before looking. Every field is computed and recorded rather
    than acted on — a trace is never silently dropped, the numbers travel with
    the result so they can be filtered downstream.
    """

    #: Lowest frequency with usable coverage outside the cone of influence.
    #: Window length imposes this limit and nothing else in the pipeline
    #: enforces it, so a short window can otherwise report usable bandwidth
    #: where the transform has no support.
    lowest_resolved_frequency: float

    #: Fraction of the window free of edge effects, per frequency, summarised
    #: as the median across the band.
    median_coi_coverage: float

    #: Normalised Gini coefficient of energy over time, in ``[0, 1]``. Near 0
    #: is stationary; near 1 means essentially all the energy is in a handful
    #: of samples, which is a glitch rather than an arrival.
    temporal_concentration: float

    #: Ratio of spectral energy in the first half of the window to the second.
    #: Far from 1 suggests coda contamination, a second arrival, or a window
    #: that started late.
    half_window_ratio: float

    def to_dict(self) -> dict[str, float]:
        """Flat mapping, for landing in a results table as columns."""
        return {
            "qc_lowest_resolved_frequency": self.lowest_resolved_frequency,
            "qc_median_coi_coverage": self.median_coi_coverage,
            "qc_temporal_concentration": self.temporal_concentration,
            "qc_half_window_ratio": self.half_window_ratio,
        }


@dataclass(frozen=True)
class Scalogram:
    """Full time-frequency surface from a continuous wavelet transform.

    Parameters
    ----------
    time
        Sample times, shape ``(n_times,)``.
    freq
        Fourier-equivalent frequencies, shape ``(n_scales,)``. These are true
        Fourier frequencies via the analytic Morlet relation, not scales, so the
        axis means the same thing as an FFT's.
    power
        ``|W(a,b)|**2``, shape ``(n_scales, n_times)``.
    scales
        Wavelet scales in seconds, shape ``(n_scales,)``. Needed by the
        normalisation bridge, which divides by scale.
    coi
        Longest resolvable period at each time, shape ``(n_times,)``. A
        frequency is inside the cone of influence where ``1/freq > coi``.
    c_delta
        Reconstruction constant for the wavelet actually used. Computed rather
        than tabulated, so a non-default ``omega0`` stays correct.
    dj
        Spacing of the log-scale grid, in octaves.
    """

    time: NDArray[np.float64]
    freq: NDArray[np.float64]
    power: NDArray[np.float64]
    scales: NDArray[np.float64]
    coi: NDArray[np.float64]
    c_delta: float
    dj: float
    dt: float
    motion: Motion
    meta: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        for name in ("time", "freq", "power", "scales", "coi"):
            array = getattr(self, name)
            array.setflags(write=False)

    @property
    def duration(self) -> float:
        return float(self.time.size * self.dt)

    def coi_mask(self) -> NDArray[np.bool_]:
        """``True`` where a coefficient is free of edge effects."""
        period = 1.0 / self.freq
        return period[:, None] <= self.coi[None, :]

    def coi_coverage(self) -> NDArray[np.float64]:
        """Fraction of the window free of edge effects, per frequency.

        This is the number that says whether a window is long enough to
        constrain the low-frequency plateau, which is what sets ``Omega``.
        """
        coverage: NDArray[np.float64] = self.coi_mask().mean(axis=1)
        return coverage

    def time_average(self, *, mask_coi: bool = True) -> Spectrum:
        """Collapse to an ordinary amplitude spectrum.

        Applies the Torrence & Compo normalisation so that the result satisfies
        the same Parseval contract as every other estimator: summing the
        wavelet power over scales, weighted by ``dj*dt/C_delta`` and divided by
        scale, returns the record's energy.

        Parameters
        ----------
        mask_coi
            Exclude coefficients inside the cone of influence and rescale by
            the surviving fraction. Without this a short window reads low at
            low frequency — precisely the band that constrains ``Omega``.
        """
        freq, scales = self.freq, self.scales
        if mask_coi:
            valid = self.coi_mask()
            counts = valid.sum(axis=1)
            # A scale with no edge-free sample is not measured by this record.
            # Dropping it is the honest answer: emitting zero would read as "no
            # energy here" rather than "no measurement here", and would take a
            # log-space fit to -inf. The axis therefore depends on record
            # length, exactly as the cone of influence says it must.
            usable = counts > 0
            if not usable.any():
                raise ValueError(
                    f"No frequency in {freq.min():.3g}-{freq.max():.3g} Hz is "
                    f"free of edge effects over a {self.duration:.3g} s record. "
                    f"The window is too short for this scale range; raise "
                    f"f_min, lengthen the window, or pass mask_coi=False to "
                    f"accept edge-contaminated coefficients."
                )
            freq, scales = freq[usable], scales[usable]
            counts = counts[usable]
            summed = (self.power * valid).sum(axis=1)[usable]
            # Rescale the survivors up to the full window length, so masking
            # changes the variance of the estimate rather than its level.
            summed = summed * self.time.size / counts
        else:
            summed = self.power.sum(axis=1)

        # Torrence & Compo eq. 14: the scale sum of |W|**2 / s, times
        # dj*dt/C_delta, recovers the variance. Multiplying by dt again turns
        # variance into energy, matching sum(x**2)*dt.
        energy_per_scale = summed / scales * (self.dj * self.dt / self.c_delta)
        energy_per_scale = energy_per_scale * self.dt

        # The scale grid is logarithmic, so each frequency bin subtends
        # df = f * ln(2) * dj. Spreading each scale's energy over its own bin
        # gives a density on a frequency axis.
        bin_width = freq * np.log(2.0) * self.dj
        psd = energy_per_scale / (bin_width * self.duration)

        # Ascending frequency, matching every other estimator's axis.
        order = np.argsort(freq)
        spectrum = Spectrum(
            freq=np.ascontiguousarray(freq[order]),
            amp=np.ascontiguousarray(psd[order]),
            motion=self.motion,
            kind=AmplitudeKind.PSD,
            duration=self.duration,
            sampling_rate=1.0 / self.dt,
            meta=MappingProxyType({**dict(self.meta), "coi_masked": mask_coi}),
        )
        return spectrum.to_kind(AmplitudeKind.FAS)

    def qc(self) -> ScalogramQC:
        """Compute the §4.4.2 checks."""
        coverage = self.coi_coverage()
        resolved = self.freq[coverage > 0.5]
        lowest = float(resolved.min()) if resolved.size else float(self.freq.max())

        # Energy over time, summed across the band, as a Gini coefficient.
        over_time = self.power.sum(axis=0)
        total = float(over_time.sum())
        if total > 0:
            sorted_energy = np.sort(over_time)
            n = sorted_energy.size
            index = np.arange(1, n + 1)
            gini = float(
                (2.0 * (index * sorted_energy).sum()) / (n * sorted_energy.sum())
                - (n + 1.0) / n
            )
        else:
            gini = 0.0

        half = self.time.size // 2
        first = float(self.power[:, :half].sum())
        second = float(self.power[:, half:].sum())
        ratio = first / second if second > 0 else np.inf

        return ScalogramQC(
            lowest_resolved_frequency=lowest,
            median_coi_coverage=float(np.median(coverage)),
            temporal_concentration=gini,
            half_window_ratio=ratio,
        )
