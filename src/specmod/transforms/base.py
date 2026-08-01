"""The estimator interface and the one place normalisation is defined.

Every backend — FFT, Welch, multitaper, and later the CWT — returns a
:class:`~specmod.core.spectrum.Spectrum` obeying the same contract, so a single
test suite pins all of them. That is the point of the abstraction: the
pre-refactor code called ``mtspec`` directly from ``Spectrum.__init__`` and
threaded backend keyword arguments through three layers of public API, so there
was nowhere for a shared contract to live.

The contract
------------
Given a real record ``x`` of ``N`` samples at interval ``dt``:

- The returned spectrum is **one-sided**, spanning ``(0, f_Nyquist]``.
- Its default kind is :attr:`~specmod.core.units.AmplitudeKind.FAS`, in
  ``[x] * s``.
- ``duration`` is ``N * dt``, the **physical** record length. Normalisation is
  keyed off it and never off ``len(freq)``, so zero-padding changes resolution
  without changing amplitude.
- Energy is preserved: ``spectrum.energy()`` recovers ``sum(x^2) * dt`` to
  within the accuracy of the estimator.

Taper correction
----------------
A taper attenuates the record, and the correction depends on what you are
measuring. The two are not interchangeable:

``energy`` (default)
    Divide by ``sqrt(mean(w^2))``, preserving total power. Parseval then holds
    exactly, which is what the shared test asserts, and it is the right choice
    for a transient — which is what a seismic arrival is.

``amplitude``
    Divide by ``mean(w)``, preserving the peak of a coherent sinusoid so that
    it reads ``A0 * T``. Right when measuring a monochromatic line.

For a Tukey taper with ``alpha=0.05`` the two differ by well under a percent,
but the choice is explicit rather than implied.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window

from ..core.spectrum import Spectrum
from ..core.units import AmplitudeKind, Motion

__all__ = [
    "SpectralEstimator",
    "TaperCorrection",
    "make_window",
    "prepare_record",
    "window_correction",
]

TaperCorrection = Literal["energy", "amplitude"]


@runtime_checkable
class SpectralEstimator(Protocol):
    """Anything that turns a real record into a :class:`Spectrum`."""

    @property
    def name(self) -> str:
        """Short identifier, recorded in the spectrum's metadata.

        Declared read-only so the frozen dataclasses implementing this protocol
        satisfy it; a plain mutable attribute would not.
        """
        ...

    def estimate(
        self,
        data: ArrayLike,
        dt: float,
        *,
        motion: Motion | str = Motion.VELOCITY,
        meta: dict[str, Any] | None = None,
    ) -> Spectrum:
        """Estimate the spectrum of ``data`` sampled at interval ``dt``."""
        ...


def prepare_record(
    data: ArrayLike, dt: float
) -> tuple[NDArray[np.float64], int, float]:
    """Validate and demean a record, returning ``(x, n_samples, duration)``.

    Demeaning is unconditional: a non-zero mean puts all of its energy in the
    DC bin, which is then discarded, so leaving it in loses energy the Parseval
    check would report as a failure.
    """
    x = np.ascontiguousarray(np.asarray(data, dtype=np.float64).squeeze())
    if x.ndim != 1:
        raise ValueError(f"Expected a 1-D record, got shape {x.shape}")
    if x.size < 2:
        raise ValueError(f"Record too short: {x.size} sample(s)")
    if not np.all(np.isfinite(x)):
        raise ValueError("Record contains NaN or Inf")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    return x - x.mean(), int(x.size), float(x.size * dt)


def make_window(kind: str, n: int, alpha: float = 0.05) -> NDArray[np.float64]:
    """Build a taper of length ``n``."""
    if kind == "boxcar":
        return np.ones(n, dtype=np.float64)
    spec: str | tuple[str, float] = ("tukey", alpha) if kind == "tukey" else kind
    return np.asarray(get_window(spec, n, fftbins=False), dtype=np.float64)


def window_correction(
    window: NDArray[np.float64], correction: TaperCorrection
) -> float:
    """Scale factor that undoes a taper's attenuation.

    See the module docstring for why there are two.
    """
    if correction == "energy":
        return float(np.sqrt(np.mean(window**2)))
    if correction == "amplitude":
        return float(np.mean(window))
    raise ValueError(
        f"Unknown taper correction {correction!r}; expected 'energy' or 'amplitude'."
    )


def one_sided_fas(
    x: NDArray[np.float64],
    dt: float,
    duration: float,
    *,
    window: NDArray[np.float64],
    correction: TaperCorrection,
    n_fft: int | None,
    drop_dc: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Core FFT path: taper, transform, fold, normalise.

    Returns ``(freq, fas)`` with ``fas`` in ``[x] * s``.

    The normalisation is ``dt`` per sample, doubled to fold the negative
    frequencies, and divided by the taper correction. ``dt`` is used rather
    than anything derived from the transform length, which is what makes
    padding change only the frequency sampling.
    """
    n = x.size
    nfft = int(n_fft) if n_fft is not None else n
    if nfft < n:
        raise ValueError(f"n_fft ({nfft}) is shorter than the record ({n} samples)")

    spec = np.fft.rfft(x * window, n=nfft)
    freq: NDArray[np.float64] = np.fft.rfftfreq(nfft, d=dt).astype(np.float64)

    fas = np.abs(spec) * dt / window_correction(window, correction)
    # Fold: every bin except DC and, for even nfft, Nyquist has a negative twin.
    fas[1:] *= 2.0
    if nfft % 2 == 0:
        fas[-1] /= 2.0

    # Deliberately no rescaling for zero-padding. The DFT sums over the N
    # non-zero samples whatever nfft is, so it approximates the same continuous
    # transform `dt * sum(x_n exp(-2 pi i f t_n))` and merely evaluates it on a
    # finer grid. Normalising by `dt` alone is what makes that true; scaling by
    # any length ratio would reintroduce exactly the padding sensitivity that
    # made the pre-refactor `psd_to_amp` wrong (§2.2).
    del duration, n  # normalisation depends on neither; kept for the signature

    if drop_dc:
        return freq[1:], fas[1:]
    return freq, fas


def build_spectrum(
    freq: NDArray[np.float64],
    amp: NDArray[np.float64],
    *,
    kind: AmplitudeKind,
    motion: Motion | str,
    duration: float,
    sampling_rate: float,
    meta: dict[str, Any] | None,
    estimator: str,
) -> Spectrum:
    """Assemble a Spectrum, recording which estimator produced it."""
    info = dict(meta or {})
    info.setdefault("estimator", estimator)
    return Spectrum(
        freq=freq,
        amp=amp,
        motion=Motion(motion),
        kind=kind,
        duration=duration,
        sampling_rate=sampling_rate,
        meta=info,
    )
