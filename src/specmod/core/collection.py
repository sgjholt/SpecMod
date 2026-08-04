"""Pairing a signal against its noise, and the band that survives the comparison.

This is the typed replacement for ``spectral.SNP`` and ``spectral.Spectra``.
The numerics are identical — ``tests/test_golden_reference.py`` holds both
paths to the same 140 window-estimator results — but three structural
properties change, and they are the reason the rewrite is worth doing.

**Configuration is an argument, not an import-time global.** ``spectral.py``
binds every setting at module import (``BW_METHOD = cfg.SPECTRAL[...]``, and
eight more). That is why a Brune and a Boatwright model cannot be fitted in one
session, why tests cannot vary configuration without reimporting, and why they
cannot run in parallel. Everything here takes its settings as parameters.

**Nothing mutates.** The legacy classes rescale, rotate, interpolate and
integrate in place, which is what made ``core.Spectrum``'s read-only arrays
break the pipeline when the estimators were rewired: the containers were
mutating arrays they did not own. Each step here returns a new object, so a
spectrum cannot change under a reference someone else is holding.

**The pieces are separable.** The binning, the Parseval rescale, the
interpolation and the band search are module-level functions over arrays. They
were private methods reachable only by constructing a full pair from two obspy
traces, so the only way to test the band search was to run the whole pipeline.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

from .spectrum import Spectrum

__all__ = [
    "BinnedSpectrum",
    "SpectrumPair",
    "SpectrumSet",
    "find_bandwidth",
    "interpolate_onto",
    "log_bin",
    "parseval_scale",
]


@dataclass(frozen=True)
class BinnedSpectrum:
    """A spectrum averaged into log-spaced bins.

    Separate from :class:`~specmod.core.spectrum.Spectrum` because it is not
    one: the bin centres are geometric midpoints of the edges rather than
    Fourier frequencies, so record geometry (``duration``, ``sampling_rate``)
    no longer determines the axis and the Parseval contract does not hold on
    it. Conflating the two is how a binned spectrum ends up being handed to
    something that assumes an FFT grid.
    """

    freq: NDArray[np.float64]
    amp: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.freq.shape != self.amp.shape:
            raise ValueError(
                f"freq {self.freq.shape} and amp {self.amp.shape} must match"
            )

    def __len__(self) -> int:
        return int(self.freq.size)


def log_bin(
    freq: NDArray[np.float64],
    amp: NDArray[np.float64],
    *,
    f_min: float = 0.001,
    f_max: float = 200.0,
    n_bins: int = 101,
) -> BinnedSpectrum:
    """Average ``amp`` into ``n_bins`` log-spaced bins, dropping empty ones.

    The requested range is clamped to the record's own, which is what makes
    the requested bin count the count you get. Unclamped, the shipped defaults
    (0.001 Hz to 200 Hz) sit far outside any real record — on the PNR data
    roughly a third of the bins fall below the lowest frequency present and a
    third above the highest, all of them empty — which is why the surviving
    axis was always far shorter than ``n_bins``.

    The average is geometric (the mean of ``log10(amp)``), matching the log
    scale the bins themselves are spaced on. Empty bins are expected rather
    than exceptional — log bins over a linear grid are inevitably sparse at the
    low end — so they are dropped silently rather than warned about per bin.
    """
    lo = max(f_min, float(freq.min()))
    hi = min(f_max, float(freq.max()))
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins)

    amps = np.full(edges.size - 1, np.nan, dtype=np.float64)
    centres = np.zeros(edges.size - 1, dtype=np.float64)
    for i, (left, right) in enumerate(itertools.pairwise(edges)):
        inside = amp[(freq >= left) & (freq <= right)]
        if inside.size:
            amps[i] = 10 ** np.log10(inside).mean()
        centres[i] = 10 ** np.mean([np.log10(left), np.log10(right)])

    keep = ~np.isnan(amps)
    return BinnedSpectrum(freq=centres[keep], amp=amps[keep])


def parseval_scale(n_signal: int, n_noise: int) -> float:
    """Factor putting a noise spectrum on the signal's energy footing.

    The two windows are rarely the same length — 1.2 to 1.6 s of noise against
    1.8 to 3.5 s of signal on the PNR data — and a shorter record spreads the
    same power over fewer bins. Comparing them without this compares spectra
    computed over different durations.
    """
    return float(np.sqrt(n_signal / n_noise))


def interpolate_onto(
    target_freq: NDArray[np.float64],
    freq: NDArray[np.float64],
    amp: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Resample ``amp`` onto ``target_freq``.

    .. warning::

       ``np.interp`` does not extrapolate — it repeats the edge value. Below
       ``freq.min()`` the result is therefore a flat continuation rather than a
       measurement, and a signal-to-noise ratio computed there has an invented
       denominator. :meth:`SpectrumPair.resolution_floor` is what keeps the
       selected band out of that region; this function does not, and must not
       be used without it.
    """
    return np.interp(target_freq, freq, amp)


def find_bandwidth(
    freq: NDArray[np.float64],
    snr: NDArray[np.float64],
    threshold: float,
    *,
    percentile: float = 0.99,
    min_width: int = 3,
) -> tuple[float, float] | None:
    """Widest band whose signal-to-noise stays above ``threshold``.

    The ratio is mapped to ``{-1, +1}`` by ``sign(snr - threshold)`` and
    integrated, so the integral rises through stretches that pass and falls
    through stretches that fail. The band is read off where that integral
    crosses the requested percentiles, which finds the largest passing run
    rather than the first — a spectrum can dip below threshold at a single
    noisy bin without that ending the usable band.

    Returns ``None`` when no band of at least ``min_width`` bins survives.
    """
    integral = cumulative_trapezoid(np.sign(snr - threshold))
    if integral.size == 0 or integral.max() <= 0:
        return None
    integral = integral / integral.max()
    integral[integral <= 0] = -1

    high = int(np.abs(integral - percentile).argmin()) - 1
    low = int(np.abs(integral - (1 - percentile)).argmin())

    for _ in range(3):
        if low < high and low != 0:
            break
        integral[low] = 1
        low = int(np.abs(integral + 1 - percentile).argmin())
    else:
        return None

    if high - low < min_width:
        return None
    return float(freq[low]), float(freq[high])


@dataclass(frozen=True)
class SpectrumPair:
    """A signal spectrum and the noise it is judged against.

    Build with :meth:`compare`, which runs the rescale, the interpolation, the
    binning and the band search in the order they depend on each other.
    """

    signal: Spectrum
    noise: Spectrum
    binned_signal: BinnedSpectrum
    binned_noise: BinnedSpectrum
    snr: NDArray[np.float64]
    resolution_floor: float
    band: tuple[float, float] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passes(self) -> bool:
        """Whether a usable band survived."""
        return self.band is not None

    @classmethod
    def compare(
        cls,
        signal: Spectrum,
        noise: Spectrum,
        *,
        threshold: float = 3.0,
        f_min: float = 0.001,
        f_max: float = 200.0,
        n_bins: int = 101,
        scale_parseval: bool = True,
        resolution_floor: bool = True,
        meta: Mapping[str, Any] | None = None,
    ) -> SpectrumPair:
        """Pair the two and select the band.

        The order matters and is not arbitrary. The noise is rescaled and moved
        onto the signal's frequency axis *before* binning, which is what makes
        the two binned arrays share bin edges — the element-wise ratio below is
        only meaningful because of it, and it holds for every estimator
        including those whose native axes differ in length.

        The floor is captured from the two spectra before the interpolation,
        because afterwards the noise carries the signal's axis and its own
        lowest resolvable frequency is unrecoverable.
        """
        floor = max(
            float(signal.freq.min()) if signal.freq.size else 0.0,
            float(noise.freq.min()) if noise.freq.size else 0.0,
        )

        noise_amp = np.asarray(noise.amp, dtype=np.float64)
        if scale_parseval:
            noise_amp = noise_amp * parseval_scale(signal.amp.size, noise.amp.size)
        noise_amp = interpolate_onto(signal.freq, noise.freq, noise_amp)

        binned_signal = log_bin(
            signal.freq,
            np.asarray(signal.amp),
            f_min=f_min,
            f_max=f_max,
            n_bins=n_bins,
        )
        binned_noise = log_bin(
            signal.freq, noise_amp, f_min=f_min, f_max=f_max, n_bins=n_bins
        )

        snr = binned_signal.amp / binned_noise.amp
        band = find_bandwidth(binned_signal.freq, snr, threshold)
        if band is not None and resolution_floor:
            band = _clamp_to_floor(band, floor)

        aligned_noise = Spectrum(
            freq=signal.freq,
            amp=noise_amp,
            motion=noise.motion,
            kind=noise.kind,
            duration=noise.duration,
            sampling_rate=noise.sampling_rate,
            meta=dict(noise.meta),
        )
        return cls(
            signal=signal,
            noise=aligned_noise,
            binned_signal=binned_signal,
            binned_noise=binned_noise,
            snr=snr,
            resolution_floor=floor,
            band=band,
            meta=dict(meta or {}),
        )


def _clamp_to_floor(
    band: tuple[float, float], floor: float
) -> tuple[float, float] | None:
    """Refuse the part of a band that rests on an extrapolated noise level.

    Below the floor the noise is ``np.interp``'s repeated edge value, so the
    ratio there is measured against nothing. Raising the low edge is the
    conservative response; if the floor swallows the band entirely there is no
    usable measurement and the answer is ``None`` rather than a narrower band
    that would look like a result.
    """
    low, high = band
    if low >= floor:
        return band
    if floor >= high:
        return None
    return floor, high


@dataclass(frozen=True)
class SpectrumSet:
    """The pairs for one event, keyed by trace id.

    Replaces ``spectral.Spectra``. A mapping rather than a class with a
    ``group`` attribute, so the obvious operations — iterate, filter, count —
    are the ones that work.
    """

    pairs: Mapping[str, SpectrumPair]
    event: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> SpectrumPair:
        return self.pairs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def passing(self) -> SpectrumSet:
        """Only the pairs that yielded a usable band."""
        return SpectrumSet(
            pairs={k: v for k, v in self.pairs.items() if v.passes},
            event=self.event,
            meta=dict(self.meta),
        )

    def ids(self) -> Sequence[str]:
        return sorted(self.pairs)
