"""Choosing the frequency band a spectrum is fitted over.

The band is the most consequential choice in the pipeline after the transform
itself: it is what constrains ``Omega``, and therefore ``M0`` and ``Mw``. There
is more than one defensible way to pick it, so this is a **set** of strategies
behind one signature — given frequencies, a signal-to-noise ratio and a
threshold, return the band or ``None`` — resolved through
:data:`BANDWIDTH_SELECTORS`, the same way :mod:`specmod.core.noise` handles
noise models and :mod:`specmod.transforms` handles estimators.

``peak``
    The shipped default, and what the legacy ``BW_METHOD = 2`` did. Walks
    outward from the highest signal-to-noise bin until the ratio drops below
    threshold in each direction. Anchoring on the peak is a real modelling
    choice: it says the usable band is the one containing the strongest part
    of the signal, even if a wider passing run exists elsewhere.

``widest``
    The widest contiguous run above threshold, bridging single-bin dips.
    Anchors on nothing, so it finds a wide low-frequency run that ``peak``
    would miss if the peak sits elsewhere.

``fixed``
    The band you name. Not a selection at all — see :class:`FixedBandwidth`
    for when ignoring the data is the right answer and what it costs.

Both automatic selectors walk while the *ratio* holds, which is why they can
run to the top of the record: near Nyquist the signal and the noise are dying
into the anti-alias roll-off together, so their ratio survives a region where
neither carries information. Nothing in the walk knows that. The cap in
``SnrConfig.max_nyquist_fraction`` is the answer to it, applied after
selection rather than inside it — see :func:`cap_to_nyquist`.

Returning ``None`` on failure is deliberate and differs from both legacy
methods, which returned a band anyway and set a ``pass_snr`` flag beside it —
so a caller reading the band without checking the flag got numbers that looked
like a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "BANDWIDTH_SELECTORS",
    "BandwidthSelector",
    "FixedBandwidth",
    "PeakBandwidth",
    "WidestBandwidth",
    "cap_to_nyquist",
    "get_bandwidth_selector",
]


@runtime_checkable
class BandwidthSelector(Protocol):
    """Anything that picks a band from a signal-to-noise curve."""

    @property
    def name(self) -> str:
        """Short identifier, recorded alongside the result."""
        ...

    def select(
        self,
        freq: NDArray[np.float64],
        snr: NDArray[np.float64],
        threshold: float,
    ) -> tuple[float, float] | None:
        """The usable band, or ``None`` if none survives."""
        ...


@dataclass(frozen=True)
class PeakBandwidth:
    """Walk outward from the strongest bin until the ratio fails each way.

    Ported from the legacy ``find_optimal_signal_bandwidth_2``, which is what
    the shipped configuration has always used.

    .. note::

       **A latent bug in the original is fixed here, and it changes results.**
       The legacy indexed the bin before the first failure with ``[...][0] - 1``
       on a raw index array. When the failure was the bin *immediately* above
       the peak that index is ``0``, so ``0 - 1`` wrapped to ``-1`` and
       selected the *highest* frequency in the record instead of failing —
       returning a band far wider than the data supports, silently. The same
       wrap could not happen at the low end, where ``+ 1`` is used.

       Here the walk stops where it should and returns ``None`` when there is
       no room to walk.
    """

    @property
    def name(self) -> str:
        return "peak"

    def select(
        self,
        freq: NDArray[np.float64],
        snr: NDArray[np.float64],
        threshold: float,
    ) -> tuple[float, float] | None:
        if freq.size == 0 or snr.size != freq.size:
            return None

        peak = int(np.argmax(snr))
        if snr[peak] < threshold:
            return None

        # Walk out from the peak while the ratio holds.
        low = peak
        while low - 1 >= 0 and snr[low - 1] >= threshold:
            low -= 1
        high = peak
        while high + 1 < snr.size and snr[high + 1] >= threshold:
            high += 1

        if high <= low:
            return None
        return float(freq[low]), float(freq[high])


@dataclass(frozen=True)
class WidestBandwidth:
    """The widest contiguous run above threshold, bridging short dips.

    Replaces the legacy ``find_optimal_signal_bandwidth`` (``BW_METHOD = 1``),
    which took percentiles of an integrated sign function with a retry loop.
    Every step of that was discontinuous and they compounded: an edge could
    move 13 bins between machines. It also lagged — on a clean 5-30 Hz passing
    region it put the low edge at 9.41 Hz, and the low edge is what constrains
    ``Omega``. This lands within one bin of the truth.
    """

    max_gap: int = 1
    min_width: int = 3

    @property
    def name(self) -> str:
        return "widest"

    def select(
        self,
        freq: NDArray[np.float64],
        snr: NDArray[np.float64],
        threshold: float,
    ) -> tuple[float, float] | None:
        if freq.size == 0 or snr.size != freq.size:
            return None

        passing = snr >= threshold
        if not passing.any():
            return None

        # Bridge short gaps, so one noisy bin does not split a band in two.
        bridged = passing.copy()
        (failing,) = np.where(~passing)
        for i in failing:
            left, right = i - 1, i + self.max_gap
            if left >= 0 and right < passing.size and passing[left] and passing[right]:
                bridged[i : i + self.max_gap] = True

        edges = np.diff(np.concatenate(([0], bridged.view(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        if starts.size == 0:
            return None

        widths = ends - starts
        best = int(np.argmax(widths))
        if widths[best] < self.min_width:
            return None

        low, high = int(starts[best]), int(ends[best]) - 1
        return float(freq[low]), float(freq[high])


@dataclass(frozen=True)
class FixedBandwidth:
    """The band you name, whatever the signal-to-noise ratio says.

    For when the band is a decision rather than a measurement: a corner the
    instrument response imposes, a band held fixed across a study so its
    events stay comparable, or one station whose automatic band you have
    looked at and rejected.

    **The ratio is ignored, not consulted and overruled.** A band chosen this
    way carries no evidence that the data support it, which is the whole point
    of it and also the whole risk: :meth:`~specmod.core.SpectrumPair.compare`
    records ``band_imposed`` in the pair's metadata so a stored result still
    says the band was asserted rather than found.

    Intersected with the frequencies actually present, and ``None`` if that
    leaves nothing. A band reaching past the end of the record would claim
    resolution that is not there, and everything downstream slices its arrays
    with these two numbers.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(
                f"a fixed band needs low < high, got low={self.low!r} "
                f"high={self.high!r}"
            )
        if self.low < 0:
            raise ValueError(f"a fixed band needs low >= 0, got {self.low!r}")

    @property
    def name(self) -> str:
        return "fixed"

    def select(
        self,
        freq: NDArray[np.float64],
        snr: NDArray[np.float64],
        threshold: float,
    ) -> tuple[float, float] | None:
        del snr, threshold  # deliberately unused; that is what "fixed" means
        if freq.size == 0:
            return None
        low = max(self.low, float(freq.min()))
        high = min(self.high, float(freq.max()))
        if high <= low:
            return None
        return low, high


#: Registered selectors, by the name configuration refers to them by.
BANDWIDTH_SELECTORS: dict[
    str, type[PeakBandwidth] | type[WidestBandwidth] | type[FixedBandwidth]
] = {
    "peak": PeakBandwidth,
    "widest": WidestBandwidth,
    "fixed": FixedBandwidth,
}


def get_bandwidth_selector(name: str, **kwargs: object) -> BandwidthSelector:
    """Resolve a registered selector by name, with its defaults.

    Takes keyword arguments like :func:`specmod.transforms.get_estimator` and
    :func:`specmod.smoothing.get_smoother`, because ``"fixed"`` has no
    defaults to fall back on — the band is the whole of it.
    """
    try:
        cls = BANDWIDTH_SELECTORS[name]
    except KeyError:
        raise ValueError(
            f"Unknown bandwidth selector {name!r}. "
            f"Available: {sorted(BANDWIDTH_SELECTORS)}."
        ) from None
    try:
        return cls(**kwargs)  # type: ignore[arg-type]
    except TypeError as exc:
        # The common case by far: `bandwidth_method = "fixed"` set in
        # configuration with no band beside it. `TypeError: missing 2 required
        # positional arguments` names the dataclass, not the setting.
        raise ValueError(
            f"bandwidth selector {name!r} cannot be built from {kwargs!r}: "
            f"{exc}. The 'fixed' selector needs a band — set "
            f"`[snr] fixed_band = [low, high]`, or pass "
            f"FixedBandwidth(low, high) directly."
        ) from exc


def cap_to_nyquist(
    band: tuple[float, float], sampling_rate: float, fraction: float
) -> tuple[float, float] | None:
    """Refuse the part of a band that sits too near the Nyquist frequency.

    Applied *after* a selector has run, in the same way
    ``_clamp_to_floor`` handles the other end, rather than being pushed into
    each selector: it is a statement about the recording, not about how the
    band was chosen, and it should therefore constrain every strategy the same
    way — including ``fixed``.

    The problem it answers is that both automatic selectors walk while the
    signal-to-noise *ratio* holds. Approaching Nyquist the signal and the
    noise roll off together, so the ratio can stay above threshold through a
    region where neither is informative; on the shipped tutorial event the
    median high edge lands at 66% of Nyquist and five of 28 pairs run past
    90%, which is the anti-alias filter being fitted as though it were a
    source spectrum.

    ``None`` when the cap swallows the band entirely — the same answer the
    selectors give when nothing survives, because a band whose low edge is
    already above the ceiling is not a narrower measurement, it is no
    measurement.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"max_nyquist_fraction must be in (0, 1], got {fraction!r}")
    ceiling = (sampling_rate / 2) * fraction
    low, high = band
    if low >= ceiling:
        return None
    return low, min(high, ceiling)
