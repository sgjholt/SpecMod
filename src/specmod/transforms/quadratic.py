"""Quadratic multitaper estimation (Prieto et al. 2007).

An ordinary multitaper estimate averages over ``K`` tapers, each of which
smooths the spectrum across the inner band ``[-W, W]``. Where the true spectrum
is curved, that smoothing does not average out: it pulls peaks down and fills
troughs in, by an amount proportional to the second derivative. The quadratic
inverse method estimates that curvature and subtracts the bias it causes.

Measured on a pure line, where the true Fourier amplitude ``A*T/2`` is known
exactly: the ordinary estimate recovers 0.87 of the peak, this one 1.02. It
also separates two lines just past the resolution limit with about 1.4x the
peak-to-trough contrast. On white noise — no curvature to correct — it agrees
with the ordinary estimate to within a few percent, which is the control that
says it is not simply sharpening everything.

.. warning::

   **It is not the right tool for fitting a corner frequency, despite the
   corner being a curvature feature.** The correction models the spectrum as
   quadratic in *linear* frequency across the inner band, and a source spectrum
   falling as ``f**-2`` is badly described that way. Measured on a synthetic
   Brune record, the estimate droops in the far tail — about 9% low at
   10-25 Hz and 20% low at 25-49 Hz — which drags a fitted ``f_c`` down with
   it. Over 12 realisations of a 4 Hz corner: FFT recovered 3.99 Hz, ordinary
   multitaper 3.89 Hz, and this estimator 3.44 Hz.

   Reach for it where the feature of interest is a *peak or a line* — an
   instrumental tone, a site resonance, a spectral hole — not a monotone decay.

The cost is a per-frequency least-squares solve, so this is roughly two orders
of magnitude slower than :class:`~specmod.transforms.multitaper.MultitaperEstimator`.
It is not the estimator to reach for on a whole catalogue.

Implementation
--------------
The numerical core is vendored from Prieto's ``multitaper`` package rather than
reimplemented — see :mod:`specmod._vendor.qiinv` for the licence, the changes
made, and why. This module owns the part that faces SpecMod: DPSS tapers,
Thomson weights, and the units contract that every estimator here satisfies.

References
----------
Prieto, G.A., Parker, R.L., Thomson, D.J., Vernon, F.L., Graham, R.L. (2007).
Reducing the bias of multitaper spectrum estimates.
*Geophysical Journal International* 171(3), 1269-1281.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal.windows import dpss

from .._vendor.qiinv import qiinv
from ..core.spectrum import Spectrum
from ..core.units import AmplitudeKind, Motion
from .base import build_spectrum, prepare_record
from .multitaper import _adaptive_weights, center_on_energy_centroid

__all__ = ["QuadraticMultitaperEstimator"]


@dataclass(frozen=True)
class QuadraticMultitaperEstimator:
    """Curvature-corrected multitaper spectrum estimate.

    Parameters
    ----------
    time_bandwidth, n_tapers
        As :class:`~specmod.transforms.multitaper.MultitaperEstimator`. The
        correction scales with ``W = NW/N``, so a larger time-bandwidth product
        means more smoothing to undo and a correspondingly larger correction.
    adaptive
        Whether the eigencoefficients fed to the curvature fit are weighted by
        Thomson's adaptive scheme. Applies to the input weights only; the
        quadratic step itself is unweighted.
    normalize_to_variance
        Rescale so the spectrum integrates to the record variance, as ``mtspec``
        and Prieto's package do. Off by default, for the reason given in
        :class:`~specmod.transforms.multitaper.MultitaperEstimator`.

        Note that Prieto's ``MTSpec.qiinv()`` applies this unconditionally, and
        to an already-renormalised input, so reproducing that path needs it on.
    center
        Circularly shift the record so its energy centroid sits mid-window.
        Same rationale and same wrap check as the ordinary estimator.
    """

    time_bandwidth: float = 3.0
    n_tapers: int = 5
    adaptive: bool = True
    center: bool = False
    center_edge_tolerance: float = 0.05
    normalize_to_variance: bool = False
    drop_dc: bool = True
    name: str = "quadratic"

    def __post_init__(self) -> None:
        if self.time_bandwidth <= 0:
            raise ValueError(
                f"time_bandwidth must be positive, got {self.time_bandwidth}"
            )
        limit = int(2 * self.time_bandwidth - 1)
        if self.n_tapers < 1:
            raise ValueError(f"n_tapers must be at least 1, got {self.n_tapers}")
        if self.n_tapers > limit:
            raise ValueError(
                f"n_tapers={self.n_tapers} exceeds 2*NW-1={limit} for "
                f"time_bandwidth={self.time_bandwidth}. Tapers beyond that are "
                f"poorly concentrated and add leakage rather than reducing "
                f"variance; raise time_bandwidth or lower n_tapers."
            )
        if self.n_tapers < 2:
            raise ValueError(
                "The quadratic estimate fits three basis functions to the "
                "cross-spectra of K tapers, so it needs at least 2 tapers "
                "(K**2 = 4 equations) to be determined at all. Use "
                "MultitaperEstimator for a single-taper estimate."
            )

    def estimate(
        self,
        data: ArrayLike,
        dt: float,
        *,
        motion: Motion | str = Motion.VELOCITY,
        meta: dict[str, Any] | None = None,
    ) -> Spectrum:
        x, n, duration = prepare_record(data, dt)
        if self.center:
            x = center_on_energy_centroid(x, self.center_edge_tolerance)

        tapers, eigenvalues = dpss(
            n, self.time_bandwidth, self.n_tapers, sym=False, return_ratios=True
        )

        # qiinv works on the full two-sided transform: it forms cross-spectra
        # between tapers and needs the negative frequencies to do it.
        yk = np.fft.fft(tapers * x, axis=-1)
        eigenspectra = np.abs(yk) ** 2

        if self.adaptive:
            weights = _adaptive_weights(eigenspectra, eigenvalues, n)
        else:
            weights = np.ones_like(eigenspectra)

        qispec, _slope, curvature = qiinv(
            yk.T, weights.T, tapers.T, eigenvalues, self.time_bandwidth
        )

        # qispec is in |FFT|**2; the same dt that makes an ordinary
        # eigenspectrum a PSD applies here, and for the same reason.
        psd = qispec * dt

        # The correction is a subtraction and can in principle overshoot into
        # negative power where curvature is large and poorly determined. The
        # damping inside qiinv makes that rare, but a negative PSD is not a
        # spectrum, so clip and say so rather than emit sqrt of a negative.
        negative = int((psd < 0).sum())
        if negative:
            psd = np.maximum(psd, 0.0)

        freq: NDArray[np.float64] = np.fft.rfftfreq(n, d=dt).astype(np.float64)
        psd = psd[: freq.size].copy()
        psd[1:] *= 2.0  # fold negative frequencies
        if n % 2 == 0:
            psd[-1] /= 2.0

        if self.normalize_to_variance:
            df = float(freq[1] - freq[0])
            total = float(psd.sum() * df)
            if total > 0:
                psd = psd * (float(x.var()) / total)

        if self.drop_dc:
            freq, psd = freq[1:], psd[1:]

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
                "adaptive": self.adaptive,
                "centered": self.center,
                "normalize_to_variance": self.normalize_to_variance,
                "curvature_rms": float(np.sqrt(np.mean(curvature**2))),
                "clipped_bins": negative,
            },
            estimator=self.name,
        )
        return spectrum.to_kind(AmplitudeKind.FAS)
