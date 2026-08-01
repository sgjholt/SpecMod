"""Plain FFT and Welch estimators.

``FFTEstimator`` is the direct route: taper, transform, fold, normalise. Paired
with a smoother (:mod:`specmod.smoothing`) it is the conventional
engineering-seismology approach and the fastest option here.

``WelchEstimator`` averages over overlapping segments, trading frequency
resolution for variance. That makes it a good default for *noise* windows,
where a stable level matters more than resolving structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import welch

from ..core.spectrum import Spectrum
from ..core.units import AmplitudeKind, Motion
from .base import (
    TaperCorrection,
    build_spectrum,
    make_window,
    one_sided_fas,
    prepare_record,
)

__all__ = ["FFTEstimator", "WelchEstimator"]


@dataclass(frozen=True)
class FFTEstimator:
    """One-sided Fourier amplitude spectrum via :func:`numpy.fft.rfft`.

    Parameters
    ----------
    taper, taper_alpha
        Window applied before transforming. ``tukey`` with a small ``alpha``
        suppresses edge discontinuities while leaving the body of the record
        untouched.
    taper_correction
        ``"energy"`` preserves Parseval, ``"amplitude"`` preserves the peak of a
        coherent sinusoid. See :mod:`specmod.transforms.base`.
    n_fft
        Transform length. ``None`` means no padding. Padding refines the
        frequency grid without changing amplitude — the property that the
        pre-refactor normalisation got wrong by keying off ``len(freq)``.
    drop_dc
        Discard the zero-frequency bin.
    """

    taper: str = "tukey"
    taper_alpha: float = 0.05
    taper_correction: TaperCorrection = "energy"
    n_fft: int | None = None
    drop_dc: bool = True
    name: str = "fft"

    def estimate(
        self,
        data: ArrayLike,
        dt: float,
        *,
        motion: Motion | str = Motion.VELOCITY,
        meta: dict[str, Any] | None = None,
    ) -> Spectrum:
        x, _n, duration = prepare_record(data, dt)
        window = make_window(self.taper, x.size, self.taper_alpha)
        freq, fas = one_sided_fas(
            x,
            dt,
            duration,
            window=window,
            correction=self.taper_correction,
            n_fft=self.n_fft,
            drop_dc=self.drop_dc,
        )
        return build_spectrum(
            freq,
            fas,
            kind=AmplitudeKind.FAS,
            motion=motion,
            duration=duration,
            sampling_rate=1.0 / dt,
            meta=meta,
            estimator=self.name,
        )


@dataclass(frozen=True)
class WelchEstimator:
    """Segment-averaged PSD via :func:`scipy.signal.welch`, returned as FAS.

    Averaging reduces variance at the cost of frequency resolution, which is
    the right trade for a noise window. Note that ``duration`` remains the full
    record length: it is the physical property of the record, not of the
    segments, and every kind conversion depends on it.
    """

    segment_length: int | None = None
    overlap: float = 0.5
    taper: str = "hann"
    drop_dc: bool = True
    name: str = "welch"

    def estimate(
        self,
        data: ArrayLike,
        dt: float,
        *,
        motion: Motion | str = Motion.VELOCITY,
        meta: dict[str, Any] | None = None,
    ) -> Spectrum:
        x, n, duration = prepare_record(data, dt)
        nperseg = min(self.segment_length or n, n)
        freq, psd = welch(
            x,
            fs=1.0 / dt,
            window=self.taper,
            nperseg=nperseg,
            noverlap=int(nperseg * self.overlap),
            detrend=False,
            return_onesided=True,
            scaling="density",
        )
        # scipy's density is one-sided [x]^2/Hz, matching AmplitudeKind.PSD.
        if self.drop_dc:
            freq, psd = freq[1:], psd[1:]
        spectrum = build_spectrum(
            np.asarray(freq, dtype=np.float64),
            np.asarray(psd, dtype=np.float64),
            kind=AmplitudeKind.PSD,
            motion=motion,
            duration=duration,
            sampling_rate=1.0 / dt,
            meta=meta,
            estimator=self.name,
        )
        return spectrum.to_kind(AmplitudeKind.FAS)
