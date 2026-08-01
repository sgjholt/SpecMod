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
        """Convert between FAS, PSD and ASD.

        Conversions go via FAS rather than being enumerated pairwise, so there
        is one place where the factor of ``2T`` lives.
        """
        target = AmplitudeKind(kind)
        if target is self.kind:
            return self
        fas = self._to_fas()
        if target is AmplitudeKind.FAS:
            return fas
        two_t = 2.0 * self.duration
        if target is AmplitudeKind.PSD:
            amp = fas.amp**2 / two_t
        else:  # ASD
            amp = fas.amp / np.sqrt(two_t)
        return replace(fas, amp=amp, kind=target)

    def _to_fas(self) -> Spectrum:
        if self.kind is AmplitudeKind.FAS:
            return self
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
