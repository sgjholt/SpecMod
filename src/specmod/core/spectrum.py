"""The :class:`Spectrum` container.

Immutable, self-describing, and normalisation-aware. Every operation returns a
new object rather than mutating in place, so a spectrum cannot be silently
integrated twice — the pre-refactor ``Spectrum.integrate()`` mutated, and its
only inverse was ``differentiate()``, which is neither exact nor recorded.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .units import AmplitudeKind, Motion

__all__ = ["Spectrum"]

#: Tolerance on the derived sample count. ``duration`` is ``n * dt`` and
#: ``sampling_rate`` is ``1 / dt``, so their product is ``n`` exactly up to
#: floating-point representation — anything further out is a real mismatch,
#: not rounding.
_SAMPLE_COUNT_TOL = 1e-6


def _validate_record_geometry(
    freq: NDArray[np.float64], duration: float, sampling_rate: float
) -> None:
    """Check the three quantities every correction is built on agree.

    Sample count, duration and sampling rate are not independent: ``duration =
    n * dt`` and ``sampling_rate = 1 / dt``, so any two determine the third and
    the frequency axis they imply. Every normalisation in this package — the
    ``2T`` between amplitude and power, the fold at DC and Nyquist, the taper
    corrections, the wavelet scale grid — is a function of them.

    That makes an inconsistent triple the most dangerous thing a caller can
    construct: it produces a spectrum that is wrong by a clean factor
    everywhere, which looks like a plausible spectrum and survives every check
    that inspects shape rather than scale. Catching it here is cheap; catching
    it downstream has historically meant noticing that a magnitude looks odd.
    """
    implied = duration * sampling_rate
    if abs(implied - round(implied)) > _SAMPLE_COUNT_TOL * max(1.0, implied):
        raise ValueError(
            f"duration={duration} s at {sampling_rate} Hz implies "
            f"{implied} samples, which is not a whole number. These are not "
            f"independent: duration = n_samples / sampling_rate. One of them "
            f"is wrong, and every amplitude conversion depends on both."
        )

    if freq.size == 0:
        return
    if freq[0] < 0.0:
        raise ValueError(f"frequencies must be non-negative, got {freq.min()}")
    if freq.size > 1 and not np.all(np.diff(freq) > 0):
        raise ValueError(
            "freq must be strictly increasing; band() and the smoothers both "
            "assume it, and an unsorted axis integrates to nonsense"
        )

    nyquist = sampling_rate / 2.0
    if freq[-1] > nyquist * (1.0 + 1e-9):
        raise ValueError(
            f"frequency axis reaches {freq[-1]} Hz but the Nyquist frequency "
            f"for {sampling_rate} Hz sampling is {nyquist} Hz. Either "
            f"sampling_rate is wrong or the axis does not belong to this "
            f"record; energy() would silently integrate over a band the "
            f"record cannot represent."
        )


@dataclass(frozen=True)
class Spectrum:
    """A one-sided spectrum that knows its own units.

    Parameters
    ----------
    freq
        Frequency axis in Hz, strictly increasing, excluding DC by default.
    amp
        Amplitude in whatever :attr:`kind` declares.
    motion
        Ground-motion domain.
    kind
        What ``amp`` represents.
    duration
        **Physical** record duration in seconds, ``n_samples * dt``. Carried
        explicitly because every conversion between kinds needs it and it
        cannot be recovered from ``len(freq)`` once padding is involved.
    sampling_rate
        Samples per second of the source record, in Hz.
    meta
        Arbitrary trace metadata. Stored read-only so a shared mapping cannot be
        mutated through one spectrum and observed through another.
    """

    freq: NDArray[np.float64]
    amp: NDArray[np.float64]
    motion: Motion
    kind: AmplitudeKind
    duration: float
    sampling_rate: float
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        freq = np.ascontiguousarray(self.freq, dtype=np.float64)
        amp = np.ascontiguousarray(self.amp, dtype=np.float64)
        if freq.ndim != 1 or amp.ndim != 1:
            raise ValueError("freq and amp must be one-dimensional")
        if freq.shape != amp.shape:
            raise ValueError(
                f"freq and amp must be the same length, got "
                f"{freq.shape[0]} and {amp.shape[0]}"
            )
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.sampling_rate <= 0:
            raise ValueError(
                f"sampling_rate must be positive, got {self.sampling_rate}"
            )
        _validate_record_geometry(freq, self.duration, self.sampling_rate)
        freq.setflags(write=False)
        amp.setflags(write=False)
        object.__setattr__(self, "freq", freq)
        object.__setattr__(self, "amp", amp)
        object.__setattr__(self, "motion", Motion(self.motion))
        object.__setattr__(self, "kind", AmplitudeKind(self.kind))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    # ------------------------------------------------------------------ units

    @property
    def unit(self) -> str:
        """Unit string, e.g. ``m/s*s`` for a velocity FAS."""
        return self.kind.unit(self.motion)

    @property
    def n_samples(self) -> int:
        """Samples in the source record, ``duration * sampling_rate``.

        The third of the triple, derived rather than stored so it cannot
        disagree with the other two. Validated on construction — see
        :func:`_validate_record_geometry` for why that matters.

        Note this is the *record* length, not ``len(freq)``. Zero-padding
        changes the second and not the first, and confusing them is the §2.2
        bug.
        """
        return round(self.duration * self.sampling_rate)

    @property
    def nyquist(self) -> float:
        return self.sampling_rate / 2.0

    @property
    def frequency_resolution(self) -> float:
        """``1/T`` — the narrowest frequency difference the record can resolve.

        This is the low-frequency floor the SNR bandwidth search must respect;
        nothing enforced it before, so a short window could report usable
        bandwidth below what it could physically resolve.
        """
        return 1.0 / self.duration

    def to_kind(self, kind: AmplitudeKind | str) -> Spectrum:
        """Convert between FAS, MAGNITUDE, PSD and ASD.

        Conversions go via FAS rather than being enumerated pairwise, so the
        factors of ``2T`` and of the fold each live in exactly one place.

        ``MAGNITUDE`` is the conversion to reach for when reading a long-period
        level: ``Omega`` is defined on ``|X|``, not on the folded ``FAS``, and
        the two differ by two. Asking for it by name is the point — the factor
        is easy to apply by hand and easy to apply twice, or not at all.
        """
        target = AmplitudeKind(kind)
        if target is self.kind:
            return self
        fas = self._to_fas()
        if target is AmplitudeKind.FAS:
            return fas
        if target is AmplitudeKind.MAGNITUDE:
            # Undo the fold: FAS carries the negative-frequency half, |X| does not.
            return replace(fas, amp=fas.amp / self._fold_factor(), kind=target)
        two_t = 2.0 * self.duration
        if target is AmplitudeKind.PSD:
            amp = fas.amp**2 / two_t
        else:  # ASD
            amp = fas.amp / np.sqrt(two_t)
        return replace(fas, amp=amp, kind=target)

    def _fold_factor(self) -> NDArray[np.float64]:
        """Per-bin ratio between the folded ``FAS`` and the unfolded ``|X|``.

        Two everywhere except DC and Nyquist, which have no negative-frequency
        twin to fold in — a real signal's transform is conjugate-symmetric, and
        those two bins are their own mirror image. A blanket factor of two is
        therefore wrong at both ends, by exactly two.

        **Parity matters here.** An ``rfft`` of an even-length record ends
        exactly on Nyquist; an odd-length one ends half a bin below it, at
        ``fs/2 * (n-1)/n``, and that bin *does* have a twin and *is* folded. So
        which bins are special depends on the record length as well as on
        ``drop_dc``, and is read off the axis rather than assumed.

        The tolerance is derived from the axis's own bin spacing rather than
        being a fixed relative one. For an odd-length record the top bin sits
        ``df/2`` below Nyquist, and ``df`` shrinks as the record lengthens — so
        a fixed ``rtol`` eventually swallows the gap and folds that bin wrongly.
        With ``numpy``'s default it does so from about 200000 samples, which at
        1000 Hz is a 200 s record. Scaling with ``df`` keeps the two cases
        separated at any length.
        """
        factor = np.full(self.freq.shape, 2.0)
        if self.freq.size == 0:
            return factor
        # A hundredth of the narrowest spacing: far tighter than the df/2 gap
        # that separates an odd-length top bin from Nyquist, and far looser
        # than floating-point error on an even-length one, which lands exactly.
        spacing = float(np.min(np.diff(self.freq))) if self.freq.size > 1 else 0.0
        tol = 0.01 * spacing if spacing > 0 else 1e-12
        factor[self.freq < tol] = 1.0
        factor[np.abs(self.freq - self.nyquist) < tol] = 1.0
        return factor

    def _to_fas(self) -> Spectrum:
        if self.kind is AmplitudeKind.FAS:
            return self
        if self.kind is AmplitudeKind.MAGNITUDE:
            return replace(
                self, amp=self.amp * self._fold_factor(), kind=AmplitudeKind.FAS
            )
        two_t = 2.0 * self.duration
        if self.kind is AmplitudeKind.PSD:
            amp = np.sqrt(self.amp * two_t)
        else:  # ASD
            amp = self.amp * np.sqrt(two_t)
        return replace(self, amp=amp, kind=AmplitudeKind.FAS)

    def to_motion(self, motion: Motion | str) -> Spectrum:
        """Integrate or differentiate to another ground-motion domain.

        Multiplies by ``(2*pi*f)`` per order of differentiation. Only valid on
        an amplitude-like kind, so a PSD is converted to FAS, transformed, and
        converted back — squaring the frequency factor would otherwise be
        silently wrong.
        """
        target = Motion(motion)
        if target is self.motion:
            return self
        if self.kind is not AmplitudeKind.FAS:
            return self.to_kind(AmplitudeKind.FAS).to_motion(target).to_kind(self.kind)
        order = target.derivative_order - self.motion.derivative_order
        factor = (2.0 * np.pi * self.freq) ** order
        return replace(self, amp=self.amp * factor, motion=target)

    # ------------------------------------------------------------- operations

    def band(self, fmin: float | None = None, fmax: float | None = None) -> Spectrum:
        """Restrict to a frequency band, inclusive of both bounds."""
        mask = np.ones(self.freq.shape, dtype=bool)
        if fmin is not None:
            mask &= self.freq >= fmin
        if fmax is not None:
            mask &= self.freq <= fmax
        if not mask.any():
            raise ValueError(
                f"No samples in band [{fmin}, {fmax}] Hz; spectrum spans "
                f"[{self.freq[0]:.4g}, {self.freq[-1]:.4g}] Hz."
            )
        return replace(self, freq=self.freq[mask], amp=self.amp[mask])

    def energy(self) -> float:
        """Total signal energy, ``sum(x^2) * dt``, recovered from the spectrum.

        This is the quantity Parseval's theorem ties to the time domain, and it
        is what the cross-estimator normalisation test asserts. For a one-sided
        FAS the two-sided integral folds to ``integral of A^2 / 2 df``.
        """
        fas = self.to_kind(AmplitudeKind.FAS)
        return float(np.trapezoid(fas.amp**2 / 2.0, fas.freq))

    def __len__(self) -> int:
        return int(self.freq.size)

    def __repr__(self) -> str:
        sid = self.meta.get("id", "")
        where = f" {sid}" if sid else ""
        return (
            f"Spectrum({self.kind.value}, {self.motion.value},{where} "
            f"n={len(self)}, {self.freq[0]:.3g}-{self.freq[-1]:.3g} Hz, "
            f"T={self.duration:.4g} s, [{self.unit}])"
        )
