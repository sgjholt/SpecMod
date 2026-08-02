"""Backend for Prieto's ``multitaper`` package.

The direct successor to the Fortran library that ``mtspec`` wrapped, by the same
author and pure Python. It is worth having alongside
:class:`~specmod.transforms.multitaper.MultitaperEstimator` for three reasons:

1. **It is the closest available proxy for pre-refactor behaviour.** ``mtspec``
   wrapped the same lineage, so when the phase-0 comparison asks "did the old
   code do this", this is the cheapest way to ask.
2. **Jackknife confidence intervals** and the **F-test for spectral lines**,
   which the native implementation does not provide.
3. It is an independent implementation, so disagreement between the two is
   informative rather than invisible.

Installed with ``pip install specmod[multitaper]``.

Normalisation
-------------
**This backend always rescales the spectrum to integrate to the record
variance** (``mtspec.py``: ``sscal = xvar / (sum(spec)*df)``). That is baked
into ``MTSpec`` and cannot be switched off.

The consequence matters: ``Spectrum.energy()`` on the result recovers the input
energy *by construction*, whatever the taper weighting did. It is therefore not
an independent check that the estimate is sound, and it means this backend is
immune to the position-dependent energy bias described in
:mod:`specmod.transforms.multitaper` — but only for total energy, not for the
shape of the spectrum. See ``docs/choosing_a_transform.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.spectrum import Spectrum
from ..core.units import AmplitudeKind, Motion
from .base import build_spectrum, prepare_record

__all__ = ["PrietoMultitaperEstimator"]

Weighting = Literal["adaptive", "constant", "eigenvalue"]

#: Prieto's ``iadapt`` codes.
_IADAPT: dict[str, int] = {"adaptive": 0, "constant": 1, "eigenvalue": 2}


def _require_multitaper() -> Any:
    try:
        from multitaper import MTSpec
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The 'multitaper' package is not installed. It is an optional "
            "backend; install it with `pip install specmod[multitaper]`."
        ) from exc
    return MTSpec


def _one_sided(
    freq: NDArray[np.float64], spec: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fold ``MTSpec``'s output into SpecMod's one-sided PSD convention.

    Two conversions, both easy to get wrong silently:

    - ``MTSpec.freq`` comes back in **FFT order** (0, +f..., -f...), not sorted.
      Integrating over it without sorting runs backwards through the negative
      half and returns a partly-cancelled, sometimes negative, result.
    - The spectrum is **two-sided**, normalised so the full integral equals the
      variance. SpecMod's PSD is one-sided with the same total, so folding
      doubles every non-DC bin.
    """
    order = np.argsort(freq)
    freq, spec = freq[order], spec[order]
    positive = freq > 0
    return freq[positive], spec[positive] * 2.0


@dataclass(frozen=True)
class PrietoMultitaperEstimator:
    """Multitaper estimate via Prieto's ``multitaper`` package.

    Parameters
    ----------
    time_bandwidth, n_tapers
        As for :class:`~specmod.transforms.multitaper.MultitaperEstimator`.
    weighting
        ``adaptive`` is Prieto's default. ``constant`` weights the tapers
        equally; ``eigenvalue`` weights by the concentration ratios.
    n_fft
        Transform length. ``None`` leaves ``MTSpec``'s own default, which pads.
        Padding refines the frequency grid; ``duration`` stays the physical
        record length either way.
    """

    time_bandwidth: float = 3.0
    n_tapers: int = 5
    weighting: Weighting = "adaptive"
    n_fft: int | None = None
    drop_dc: bool = True
    name: str = "prieto"

    def __post_init__(self) -> None:
        if self.weighting not in _IADAPT:
            raise ValueError(
                f"Unknown weighting {self.weighting!r}; "
                f"expected one of {sorted(_IADAPT)}."
            )
        if self.time_bandwidth <= 0:
            raise ValueError(
                f"time_bandwidth must be positive, got {self.time_bandwidth}"
            )
        limit = int(2 * self.time_bandwidth - 1)
        if not 1 <= self.n_tapers <= limit:
            raise ValueError(
                f"n_tapers={self.n_tapers} must be between 1 and 2*NW-1={limit} "
                f"for time_bandwidth={self.time_bandwidth}."
            )

    def _mtspec(self, x: NDArray[np.float64], dt: float) -> Any:
        MTSpec = _require_multitaper()
        return MTSpec(
            x,
            nw=self.time_bandwidth,
            kspec=self.n_tapers,
            dt=dt,
            nfft=0 if self.n_fft is None else int(self.n_fft),
            iadapt=_IADAPT[self.weighting],
        )

    def estimate(
        self,
        data: ArrayLike,
        dt: float,
        *,
        motion: Motion | str = Motion.VELOCITY,
        meta: dict[str, Any] | None = None,
    ) -> Spectrum:
        x, _n, duration = prepare_record(data, dt)
        mt = self._mtspec(x, dt)
        freq, psd = _one_sided(
            np.asarray(mt.freq, dtype=np.float64).ravel(),
            np.asarray(mt.spec, dtype=np.float64).ravel(),
        )
        if not self.drop_dc:  # DC was dropped by the f > 0 mask
            freq = np.concatenate([[0.0], freq])
            psd = np.concatenate([[0.0], psd])

        spectrum = build_spectrum(
            freq,
            psd,
            kind=AmplitudeKind.PSD,
            motion=motion,
            duration=duration,
            sampling_rate=1.0 / dt,
            meta={
                **(meta or {}),
                "time_bandwidth": self.time_bandwidth,
                "n_tapers": self.n_tapers,
                "weighting": self.weighting,
                # Not optional in this backend — see the module docstring.
                "normalize_to_variance": True,
            },
            estimator=self.name,
        )
        return spectrum.to_kind(AmplitudeKind.FAS)

    def confidence_interval(
        self, data: ArrayLike, dt: float, *, motion: Motion | str = Motion.VELOCITY
    ) -> tuple[Spectrum, Spectrum]:
        """Jackknife 95% confidence bounds, as two FAS spectra.

        The main reason to reach for this backend rather than the native one.
        The interval is log-symmetric about the estimate — measured at roughly
        2.5x either way for ``nw=3, kspec=5``.

        .. note::

           Upstream inverts the two bounds for a small fraction of bins (~2%
           in testing), so ``low <= high`` does not hold everywhere. Sort the
           pair per-bin if you need that guarantee.

        Raises
        ------
        NotImplementedError
            For ``weighting="constant"``. This is an upstream bug, not a
            limitation: in ``multitaper.utils.jackspec`` the degrees-of-freedom
            array ``se`` keeps shape ``(nfft, 1)`` under constant weighting, so
            ``t.ppf(...) * sqrt(var[:, 0])`` broadcasts to ``(nfft, nfft)``
            instead of elementwise and raises. ``adaptive`` and ``eigenvalue``
            are unaffected.
        """
        if self.weighting == "constant":
            raise NotImplementedError(
                "Jackknife intervals are unavailable for weighting='constant' "
                "due to a shape bug in multitaper.utils.jackspec. Use "
                "weighting='adaptive' or 'eigenvalue'."
            )
        x, _n, duration = prepare_record(data, dt)
        mt = self._mtspec(x, dt)
        ci = np.asarray(mt.jackspec(), dtype=np.float64)
        raw_freq = np.asarray(mt.freq, dtype=np.float64).ravel()

        bounds = []
        for column in (0, 1):
            freq, psd = _one_sided(raw_freq, ci[:, column].ravel())
            bounds.append(
                build_spectrum(
                    freq,
                    psd,
                    kind=AmplitudeKind.PSD,
                    motion=motion,
                    duration=duration,
                    sampling_rate=1.0 / dt,
                    meta={
                        "weighting": self.weighting,
                        "bound": ("low", "high")[column],
                    },
                    estimator=self.name,
                ).to_kind(AmplitudeKind.FAS)
            )
        return bounds[0], bounds[1]

    def f_test(
        self, data: ArrayLike, dt: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Thomson's F-test for periodic (line) components.

        Returns ``(freq, F, p)`` on the one-sided axis. Useful for spotting
        instrumental or cultural tones that would otherwise be mistaken for
        source structure.
        """
        x, _n, _duration = prepare_record(data, dt)
        mt = self._mtspec(x, dt)
        f_statistic, p_value = mt.ftest()
        raw_freq = np.asarray(mt.freq, dtype=np.float64).ravel()
        order = np.argsort(raw_freq)
        positive = raw_freq[order] > 0
        return (
            raw_freq[order][positive],
            np.asarray(f_statistic, dtype=np.float64).ravel()[order][positive],
            np.asarray(p_value, dtype=np.float64).ravel()[order][positive],
        )
