"""Pairing a signal against its noise, and the band that survives the comparison.

This is the typed replacement for ``spectral.SNP`` and ``spectral.Spectra``.
The numerics are identical — ``tests/test_golden_reference.py`` holds both
paths to the same 140 window-estimator results — but three structural
properties change, and they are the reason the rewrite is worth doing.

**Configuration is an argument, not an import-time global.** ``spectral.py``
binds every setting at module import (``BW_METHOD``, ``ROT_METHOD`` and
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

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .bandwidth import get_bandwidth_selector
from .noise import BoostNoise, NoiseModel, get_noise_model
from .spectrum import Spectrum

__all__ = [
    "BinnedSpectrum",
    "FittableView",
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

    **Membership is computed, not tested.** The bin index comes from the
    position of ``log10(f)`` along the range, which puts every sample in
    exactly one bin. The previous version tested ``f >= left and f <= right``
    against each edge in turn: both ends closed, so a sample landing on an
    interior edge belonged to *two* bins, and which of the two comparisons
    succeeded depended on the last bit of ``np.logspace``. That is one of the
    three places where a last-bit difference changed a result — it moved the
    surviving bin count by one, and with it the length of ``bsnr``. Computing
    the index removes the double membership and the edge comparison together.
    """
    lo = max(f_min, float(freq.min()))
    hi = min(f_max, float(freq.max()))
    n_intervals = n_bins - 1

    log_lo, log_hi = np.log10(lo), np.log10(hi)
    width = (log_hi - log_lo) / n_intervals

    # Index by position rather than by comparison against edges. The clip puts
    # the sample sitting exactly at `hi` into the last bin rather than one past
    # it, which is the only place the half-open rule needs an exception.
    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.floor((np.log10(freq) - log_lo) / width).astype(int)
    inside = (freq >= lo) & (freq <= hi)
    index = np.clip(index, 0, n_intervals - 1)

    amps = np.full(n_intervals, np.nan, dtype=np.float64)
    log_amp = np.log10(amp)
    for i in range(n_intervals):
        selected = log_amp[inside & (index == i)]
        if selected.size:
            amps[i] = 10 ** selected.mean()

    edges = np.logspace(log_lo, log_hi, n_bins)
    centres = 10 ** (0.5 * (np.log10(edges[:-1]) + np.log10(edges[1:])))

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
    method: str = "peak",
) -> tuple[float, float] | None:
    """Select the usable band with a named strategy.

    A thin front for :data:`specmod.core.bandwidth.BANDWIDTH_SELECTORS`. The
    default is ``"peak"``, which is what the shipped configuration has always
    used — the legacy ``BW_METHOD = 2``. See that module for what the
    strategies assume and why the choice matters.
    """
    return get_bandwidth_selector(method).select(freq, snr, threshold)


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

    def for_fitting(self, id: str = "") -> FittableView:
        """This pair as the flat view a fitter reads. See :class:`FittableView`."""
        return FittableView(pair=self, id=id)

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
        rotate_noise: bool = True,
        noise_model: str | NoiseModel = "boost",
        bandwidth: str = "peak",
        rotation_inc: float = 0.05,
        rotation_space: tuple[float, float] = (0.001, 1.001),
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

        if rotate_noise:
            # Derived on the binned axis and applied to both, rather than
            # computed twice: the unbinned factor is the binned one
            # interpolated up, which is what the legacy code does and what
            # keeps the two representations of "the noise" consistent.
            model = _resolve_noise_model(noise_model, rotation_space)
            factor = model.factor(
                binned_noise.freq, binned_noise.amp, binned_signal.amp
            )
            binned_noise = BinnedSpectrum(
                freq=binned_noise.freq, amp=binned_noise.amp * factor
            )
            noise_amp = noise_amp * interpolate_onto(
                signal.freq, binned_noise.freq, factor
            )

        snr = binned_signal.amp / binned_noise.amp
        band = find_bandwidth(binned_signal.freq, snr, threshold, method=bandwidth)
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


def _resolve_noise_model(
    noise_model: str | NoiseModel, space: tuple[float, float]
) -> NoiseModel:
    """Turn a name or an instance into a model, honouring the legacy ``space``.

    ``space`` is a parameter of the boost method alone, and it arrives here as
    a loose keyword rather than on the model because that is how the legacy
    configuration stored it. Passing an already-constructed model instead is
    the way to say what you mean; then the keyword is ignored, because the
    instance already carries its own.
    """
    if not isinstance(noise_model, str):
        return noise_model
    model = get_noise_model(noise_model)
    if isinstance(model, BoostNoise) and space != model.space:
        return BoostNoise(space=space)
    return model


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
class FittableView:
    """A pair presented as the flat thing a fitter reads.

    ``SpectrumPair`` keeps the unbinned spectrum and its binned form as
    separate objects, which is right for the comparison — they are different
    kinds of thing, and conflating them is how a binned axis ends up somewhere
    that assumes an FFT grid. A fitter wants them side by side, so this is the
    view that puts them there.

    A view rather than a conversion: it holds the pair and reads through, so
    there is one copy of the arrays and no question of which is authoritative.
    """

    pair: SpectrumPair
    id: str = ""

    @property
    def meta(self) -> dict[str, Any]:
        # A plain dict, not the Spectrum's `MappingProxyType`. The proxy is
        # right for an immutable spectrum but cannot be deepcopied, and the
        # fitter deepcopies metadata so a fit cannot write back into the
        # spectrum it was built from. Converting here is the adapter earning
        # its keep.
        return dict(self.pair.signal.meta)

    @property
    def freq(self) -> NDArray[np.float64]:
        return self.pair.signal.freq

    @property
    def amp(self) -> NDArray[np.float64]:
        return self.pair.signal.amp

    @property
    def bfreq(self) -> NDArray[np.float64]:
        return self.pair.binned_signal.freq

    @property
    def bamp(self) -> NDArray[np.float64]:
        return self.pair.binned_signal.amp

    @property
    def band(self) -> tuple[float, float] | None:
        return self.pair.band

    @property
    def passes(self) -> bool:
        return self.pair.passes


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
