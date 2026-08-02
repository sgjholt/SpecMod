"""Thomson multitaper estimation on DPSS tapers.

This replaces ``mtspec``, which is a ctypes wrapper around Fortran, ships as
source with no wheels, has had no release since 2016, and does not build
without a Fortran compiler — so it made the whole package uninstallable.

The implementation here uses :func:`scipy.signal.windows.dpss`, in SciPy since
1.1, and adds Thomson's adaptive weighting. That is a few dozen lines and no
new dependency, and it puts the normalisation under our control, which is what
lets one contract cover every backend.

Prieto's pure-Python ``multitaper`` package remains available behind the
``specmod[multitaper]`` extra for the things it does that this does not:
jackknife confidence intervals, the F-test for spectral lines, and coherence.

.. warning::

   **Multitaper assumes stationarity, and estimated energy depends on where in
   the window a transient sits.** This is about the analysis window's own
   contents, not about signal leaking into a noise window.

   Measured through this class on an identical 10%-wide burst moved across a
   2000-sample window (estimated energy / true energy):

   ======== ============ ======== ==============
   Position ``adaptive`` flat     sum of taper^2
   ======== ============ ======== ==============
   10%      **0.203**    0.974    0.948
   25%      1.253        1.076    1.071
   50%      1.317        1.146    1.148
   75%      1.244        1.067    1.072
   90%      **0.149**    0.917    0.951
   uniform  0.957        --       --
   ======== ============ ======== ==============

   Two separate effects, and they behave differently:

   *Without* adaptive weighting the bias is modest, +/-15%, and tracks the
   summed taper envelope almost exactly — compare the last two columns. That is
   simply the taper weighting the middle of the record more than the ends.

   *With* adaptive weighting an edge-located transient loses 80-85% of its
   energy. That is far too large to be taper shape, and it is **not understood
   yet**. The likely cause is that the weights are seeded from the two
   lowest-order tapers, which are the most centre-concentrated and see almost
   none of an edge-located burst, so the iteration starts near zero and the
   weights collapse onto exactly the tapers with no signal in them. Treat the
   adaptive path as suspect for strongly off-centre arrivals until this is
   resolved; ``adaptive=False`` is well-behaved throughout.

   Note this matters for the published workflow specifically: refining the
   window to the 1st-99th percentiles of cumulative energy trims the quiet
   lead-in while the coda tail remains, which pushes the arrival away from
   centre.

   The taper-shape part applied equally to ``mtspec``, which uses the same
   tapers, so it is present in pre-refactor results rather than introduced
   here. Whether Prieto's Fortran adaptive routine collapses the same way is
   untested. Nothing is corrected here, because changing it would move
   published numbers — see ``docs/REFACTOR_PLAN.md`` §5.2.6.

   If absolute energy fidelity matters more than variance reduction, prefer
   :class:`~specmod.transforms.fft.FFTEstimator` with a light taper: it holds
   1.03x regardless of position. The temporal-concentration QC check planned in
   §4.4.2 measures the property that drives all of this.

References
----------
Thomson, D.J. (1982). Spectrum estimation and harmonic analysis.
*Proc. IEEE* 70(9), 1055-1096.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal.windows import dpss

from ..core.spectrum import Spectrum
from ..core.units import AmplitudeKind, Motion
from .base import build_spectrum, prepare_record

__all__ = ["MultitaperEstimator"]


def _adaptive_weights(
    eigenspectra: NDArray[np.float64],
    eigenvalues: NDArray[np.float64],
    variance: float,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> NDArray[np.float64]:
    """Thomson's adaptive weights.

    Higher-order tapers leak more, so weighting them equally lets out-of-band
    power contaminate the estimate. The weights downweight a taper wherever its
    broadband leakage would dominate the local signal, which matters here: a
    seismic spectrum spans orders of magnitude in amplitude, so leakage from the
    peak can swamp the high-frequency tail entirely.

    Iterates ``w_k = sqrt(lambda_k) * S / (lambda_k * S + (1 - lambda_k) * var)``
    to convergence.
    """
    # Start from the two least-leaky tapers, as Thomson recommends.
    spectrum = eigenspectra[:2].mean(axis=0)
    weights = np.ones_like(eigenspectra)
    for _ in range(max_iter):
        denom = (
            eigenvalues[:, None] * spectrum + (1.0 - eigenvalues[:, None]) * variance
        )
        weights = np.sqrt(eigenvalues[:, None]) * spectrum / np.maximum(denom, 1e-300)
        w2 = weights**2
        updated = (w2 * eigenspectra).sum(axis=0) / np.maximum(w2.sum(axis=0), 1e-300)
        if np.allclose(updated, spectrum, rtol=tol):
            spectrum = updated
            break
        spectrum = updated
    return weights


@dataclass(frozen=True)
class MultitaperEstimator:
    """Multitaper spectrum estimate.

    Parameters
    ----------
    time_bandwidth
        The time-bandwidth product ``NW``. Larger values reduce variance and
        leakage at the cost of frequency resolution. In the pre-refactor code
        this was the literal ``3`` passed positionally to ``mtspec``, with no
        way to configure it.
    n_tapers
        Number of DPSS tapers. Must not exceed ``2*NW - 1``, beyond which the
        tapers are poorly concentrated and add leakage rather than reducing
        variance; exceeding it raises rather than silently degrading.
    adaptive
        Apply Thomson's adaptive weighting. When ``False``, tapers are averaged
        with equal weight.
    """

    time_bandwidth: float = 3.0
    n_tapers: int = 5
    adaptive: bool = True
    drop_dc: bool = True
    name: str = "multitaper"

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

    def estimate(
        self,
        data: ArrayLike,
        dt: float,
        *,
        motion: Motion | str = Motion.VELOCITY,
        meta: dict[str, Any] | None = None,
    ) -> Spectrum:
        x, n, duration = prepare_record(data, dt)

        tapers, eigenvalues = dpss(
            n, self.time_bandwidth, self.n_tapers, sym=False, return_ratios=True
        )
        # DPSS tapers are unit-norm; scale so each eigenspectrum is a PSD in
        # [x]^2/Hz, then fold to one-sided.
        spectra = np.fft.rfft(tapers * x, axis=-1)
        eigenspectra = (np.abs(spectra) ** 2) * dt

        if self.adaptive and self.n_tapers > 1:
            weights = _adaptive_weights(eigenspectra, eigenvalues, float(x.var()))
            w2 = weights**2
            psd = (w2 * eigenspectra).sum(axis=0) / np.maximum(w2.sum(axis=0), 1e-300)
        else:
            psd = eigenspectra.mean(axis=0)

        freq: NDArray[np.float64] = np.fft.rfftfreq(n, d=dt).astype(np.float64)
        psd[1:] *= 2.0  # fold negative frequencies
        if n % 2 == 0:
            psd[-1] /= 2.0

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
            },
            estimator=self.name,
        )
        return spectrum.to_kind(AmplitudeKind.FAS)
