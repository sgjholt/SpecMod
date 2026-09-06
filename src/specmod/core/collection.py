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
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from .bandwidth import BandwidthSelector, cap_to_nyquist, get_bandwidth_selector
from .noise import NOISE_MODELS, BoostNoise, NoiseModel, get_noise_model
from .spectrum import Spectrum
from .units import Motion

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

    The requested range is clamped to the record's own, which is what stops
    bins being wasted outside it. Unclamped, the shipped defaults (0.001 Hz to
    200 Hz) sit far outside any real record — on the PNR data roughly a third
    of the bins fall below the lowest frequency present and a third above the
    highest, all of them empty.

    Clamping does not make the surviving count equal ``n_bins``, and nothing
    does: ``n_bins`` counts bin *edges*, so there are at most ``n_bins - 1``
    intervals, and empty ones are dropped from those. Ask for 151 and a typical
    PNR window returns 104.

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


def _resolution_floor(spectrum: Spectrum) -> float:
    """The lowest frequency ``spectrum`` can actually resolve.

    Read from ``meta`` when it is there, and only derived from the axis when it
    is not. That order is the whole point. Deriving it works exactly once: the
    noise is interpolated onto the signal's axis before binning, so from then
    on ``noise.freq.min()`` is the *signal's* lowest frequency and the noise's
    own is gone. :func:`specmod.pipeline.spectrum_from_trace` records it on the
    spectrum for that reason, and until now nothing read it.

    The consequence was that a converted pair had a lower floor than the pair
    it came from — the shorter noise window's limit silently replaced by the
    longer signal window's. `to_motion` therefore let the band open into the
    region below the noise's resolution, where :func:`interpolate_onto` is
    repeating an edge value rather than reporting a measurement, and the
    signal-to-noise ratio has an invented denominator.

    Falls back to the axis for a spectrum built by hand rather than by the
    pipeline, which is the only case where the axis is still the truth.
    """
    recorded = spectrum.meta.get("resolution_floor")
    if recorded is not None:
        return float(recorded)
    return float(spectrum.freq.min()) if spectrum.freq.size else 0.0


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
    method: str | BandwidthSelector = "peak",
) -> tuple[float, float] | None:
    """Select the usable band with a named strategy.

    A thin front for :data:`specmod.core.bandwidth.BANDWIDTH_SELECTORS`. The
    default is ``"peak"``, which is what the shipped configuration has always
    used — the legacy ``BW_METHOD = 2``. See that module for what the
    strategies assume and why the choice matters.

    An already-constructed selector may be passed instead of a name, which is
    how a strategy carrying parameters — :class:`FixedBandwidth`, or a
    :class:`WidestBandwidth` with a different ``min_width`` — is used without
    routing it through configuration.
    """
    selector = method if not isinstance(method, str) else get_bandwidth_selector(method)
    return selector.select(freq, snr, threshold)


@dataclass(frozen=True)
class SpectrumPair:
    """A signal spectrum and the noise it is judged against.

    Build with :meth:`compare`, which runs the rescale, the interpolation, the
    binning and the band search in the order they depend on each other.
    """

    #: Where :meth:`compare` records its own arguments inside ``meta``.
    SETTINGS_KEY: ClassVar[str] = "compare_settings"

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

    def with_band(self, band: tuple[float, float] | None) -> SpectrumPair:
        """This pair with a band you chose, as a new pair.

        For the case configuration cannot reach: you have looked at the plot
        for one station, disagreed with the automatic band, and want to refit
        that one without changing what every other station did.

        >>> corrected = spectra["LV.L001..HHE"].with_band((1.0, 25.0))

        The band is intersected with the binned frequencies present, so it
        cannot claim resolution the record does not have, and ``None`` is
        accepted to reject a pair the selector accepted. Everything else —
        the spectra, the binning, the ratio — is carried over untouched,
        because the comparison itself is not in question; only the reading of
        it is.

        ``meta["band_imposed"]`` is set to ``"manual"``, for the same reason
        :class:`~specmod.core.bandwidth.FixedBandwidth` sets it: a band that
        no ratio chose should not be indistinguishable later from one that was
        measured. The recorded ``compare`` settings are left alone, so
        :meth:`to_motion` still replays how the pair was *built* — a domain
        change re-runs the selector and the hand-chosen band does not survive
        it, which is honest rather than convenient.
        """
        if band is not None:
            low, high = float(band[0]), float(band[1])
            if not low < high:
                raise ValueError(f"a band needs low < high, got {band!r}")
            freq = self.binned_signal.freq
            if freq.size:
                low = max(low, float(freq.min()))
                high = min(high, float(freq.max()))
            if high <= low:
                raise ValueError(
                    f"band {band!r} does not overlap the frequencies present "
                    f"({float(freq.min()):.3g}-{float(freq.max()):.3g} Hz)"
                )
            band = (low, high)

        meta = dict(self.meta)
        meta["band_imposed"] = "manual"
        return type(self)(
            signal=self.signal,
            noise=self.noise,
            binned_signal=self.binned_signal,
            binned_noise=self.binned_noise,
            snr=self.snr,
            resolution_floor=self.resolution_floor,
            band=band,
            meta=meta,
        )

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
        bandwidth: str | BandwidthSelector = "peak",
        max_nyquist_fraction: float | None = None,
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
        floor = max(_resolution_floor(signal), _resolution_floor(noise))

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
            # The factor is derived on the binned axis — that is where the
            # method is defined — and applied to the *unbinned* noise, which
            # then becomes the single source the binned noise is derived from.
            #
            # The order matters and used to be the other way round: the lift
            # multiplied the bins directly and, separately, the unbinned array
            # by the factor interpolated up. Those two operations do not agree.
            # A bin holds the geometric mean of `log10(amp)`, so binning the
            # lifted noise gives `mean(log a) + mean(log f)` while lifting the
            # bin gives `mean(log a) + log f(centre)` — equal only where the
            # factor is flat across the bin.
            #
            # The result was that a stored pair's `binned_noise` was not the
            # binning of its own `noise`, by up to 18.8% on the PNR windows.
            # Every pair was born inconsistent; a domain change re-bins, so
            # `to_motion` silently *repaired* it and looked like the culprit.
            model = _resolve_noise_model(noise_model, rotation_space)
            factor = model.factor(
                binned_noise.freq, binned_noise.amp, binned_signal.amp
            )
            noise_amp = noise_amp * interpolate_onto(
                signal.freq, binned_noise.freq, factor
            )
            binned_noise = log_bin(
                signal.freq, noise_amp, f_min=f_min, f_max=f_max, n_bins=n_bins
            )

        snr = binned_signal.amp / binned_noise.amp
        selector = _resolve_bandwidth(bandwidth)
        band = find_bandwidth(binned_signal.freq, snr, threshold, method=selector)
        if band is not None and resolution_floor:
            band = _clamp_to_floor(band, floor)
        # After the floor, and after the selector, because the ceiling is a
        # statement about the recording rather than about how the band was
        # chosen — so it has to constrain `fixed` as well as the two walks.
        if band is not None and max_nyquist_fraction is not None:
            band = cap_to_nyquist(band, signal.sampling_rate, max_nyquist_fraction)

        aligned_noise = Spectrum(
            freq=signal.freq,
            amp=noise_amp,
            motion=noise.motion,
            kind=noise.kind,
            duration=noise.duration,
            sampling_rate=noise.sampling_rate,
            meta=dict(noise.meta),
        )
        recorded = dict(meta or {})
        # How this pair was made, kept so it can be remade. `to_motion` needs
        # to replay the binning and the band search on converted amplitudes,
        # and a pair that cannot say what settings produced it could only do
        # that by being told again — which is how the settings of a stored
        # result drift from the settings it was actually computed with.
        recorded[cls.SETTINGS_KEY] = {
            "threshold": threshold,
            "f_min": f_min,
            "f_max": f_max,
            "n_bins": n_bins,
            "scale_parseval": scale_parseval,
            "resolution_floor": resolution_floor,
            "rotate_noise": rotate_noise,
            "noise_model": _strategy_record(noise_model),
            "bandwidth": _strategy_record(bandwidth),
            "max_nyquist_fraction": max_nyquist_fraction,
            "rotation_inc": rotation_inc,
            "rotation_space": rotation_space,
        }
        # A band the ratio did not choose is marked as such, so a stored result
        # — or one read back a year later — still distinguishes a measurement
        # from an assertion. `passes` cannot make that distinction and was
        # never meant to: it answers "is there a band", not "did the data
        # choose it".
        if getattr(selector, "name", None) == "fixed":
            recorded["band_imposed"] = "fixed"
        return cls(
            signal=signal,
            noise=aligned_noise,
            binned_signal=binned_signal,
            binned_noise=binned_noise,
            snr=snr,
            resolution_floor=floor,
            band=band,
            meta=recorded,
        )

    def to_motion(self, motion: Motion | str) -> SpectrumPair:
        """This pair in another ground-motion domain, re-binned and re-banded.

        Replaces ``spectral.SNP.integrate``/``differentiate``, which mutated
        in place and had no way to express "the same event, as displacement"
        other than destroying the velocity one. This returns a new pair.

        **The noise is not lifted again.** ``self.noise`` already carries the
        lift from the comparison that built this pair, and applying it a second
        time would compound on every conversion — narrowing the band each time.
        The pre-refactor code guarded this with a ``ROTATED`` flag; here it
        falls out of the settings being replayed with ``rotate_noise=False``.

        **The band can move, and not for the reason it first appears.** The
        *unbinned* signal-to-noise ratio is invariant under a domain change —
        both spectra are multiplied by the same power of ``2*pi*f``. The
        *binned* ratio is not, because a bin holds the geometric mean of
        ``log10(amp)`` and averaging ``log10(a/f)`` over a bin is not
        ``log10(a)`` averaged minus ``log10(f_centre)`` unless the centre is
        the geometric mean of the frequencies in it. Measured on the 28 PNR
        windows: the binned ratio moves by up to 16%, and 3 of the 28 bands
        with it.
        """
        settings = dict(self.meta.get(self.SETTINGS_KEY, {}))
        settings["rotate_noise"] = False
        meta = {k: v for k, v in self.meta.items() if k != self.SETTINGS_KEY}
        return type(self).compare(
            self.signal.to_motion(motion),
            self.noise.to_motion(motion),
            meta=meta,
            **settings,
        )


def _strategy_record(strategy: Any) -> Any:
    """A JSON-safe settings entry for a strategy that may be an instance.

    :meth:`SpectrumPair.compare` takes a name *or* an object for both the
    noise model and the bandwidth selector, and records what it was given so
    :meth:`SpectrumPair.to_motion` can replay it. An object does not survive
    ``json.dumps``, so a pair built by passing one could not be saved at all —
    ``io.save`` raised from inside its metadata dump, naming the dataclass
    rather than the argument that caused it.

    Recorded instead as the registered name plus the strategy's own fields,
    which serialises, rebuilds, and makes the replay survive a save and reload
    rather than only working in memory.
    """
    if isinstance(strategy, str):
        return strategy
    if is_dataclass(strategy) and not isinstance(strategy, type):
        return {"name": getattr(strategy, "name", ""), **asdict(strategy)}
    return getattr(strategy, "name", strategy)


def _split_record(spec: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    params = {k: v for k, v in spec.items() if k != "name"}
    return str(spec["name"]), params


def _resolve_noise_model(
    noise_model: str | NoiseModel | Mapping[str, Any], space: tuple[float, float]
) -> NoiseModel:
    """Turn a name, a record or an instance into a model, honouring ``space``.

    ``space`` is a parameter of the boost method alone, and it arrives here as
    a loose keyword rather than on the model because that is how the legacy
    configuration stored it. Passing an already-constructed model instead is
    the way to say what you mean; then the keyword is ignored, because the
    instance already carries its own — and a record written by
    :func:`_strategy_record` carries it too, which is what keeps that true
    across a save and reload.
    """
    if isinstance(noise_model, Mapping):
        name, params = _split_record(noise_model)
        rebuilt: NoiseModel = NOISE_MODELS[name](**params)
        return rebuilt
    if not isinstance(noise_model, str):
        return noise_model
    model = get_noise_model(noise_model)
    if isinstance(model, BoostNoise) and space != model.space:
        return BoostNoise(space=space)
    return model


def _resolve_bandwidth(
    bandwidth: str | BandwidthSelector | Mapping[str, Any],
) -> str | BandwidthSelector:
    """Turn a record back into a selector; names and instances pass through."""
    if isinstance(bandwidth, Mapping):
        name, params = _split_record(bandwidth)
        return get_bandwidth_selector(name, **params)
    return bandwidth


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
        meta = dict(self.pair.signal.meta)
        # The band and the gate belong in a flat fit table: a fitted corner
        # frequency without the band it was read over is not interpretable,
        # and it is the first thing anyone comparing two runs asks for. Under
        # the legacy names, so a flatfile written from either container has the
        # same columns.
        meta["pass_snr"] = self.pair.passes
        if self.pair.band is not None:
            meta["lower-f-bound"] = float(self.pair.band[0])
            meta["upper-f-bound"] = float(self.pair.band[1])
        if self.id:
            meta.setdefault("id", self.id)
        return meta

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

    @property
    def motion(self) -> Motion:
        """The ground-motion domain the arrays are in.

        Read through like the rest, because a fitter has to know it: the
        initial guess for ``fc`` is the frequency of the spectral peak, and
        that is the corner only in velocity.
        """
        return self.pair.signal.motion


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

    def to_motion(self, motion: Motion | str) -> SpectrumSet:
        """The whole event in another ground-motion domain.

        Replaces ``spectral.Spectra.inte``/``diff``. See
        :meth:`SpectrumPair.to_motion` for what is recomputed and what is not.
        """
        return SpectrumSet(
            pairs={k: v.to_motion(motion) for k, v in self.pairs.items()},
            event=self.event,
            meta=dict(self.meta),
        )
