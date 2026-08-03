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

   ======== ============ ======== ============== ==============
   Position ``adaptive`` flat     sum of taper^2 FFT, 5% Tukey
   ======== ============ ======== ============== ==============
   10%      0.956        0.973    0.948          1.031
   25%      1.079        1.075    1.071          1.031
   50%      1.151        1.145    1.148          1.031
   75%      1.070        1.066    1.072          1.031
   90%      0.898        0.916    0.951          1.031
   uniform  1.010        1.009    --             0.995
   ======== ============ ======== ============== ==============

   The bias is modest, +/-15%, it is the *same* under either weighting, and it
   tracks the summed taper envelope almost exactly — compare against the
   ``taper^2`` column. That is simply the taper weighting the middle of the
   record more than the ends, and it applied equally to ``mtspec``, which uses
   the same tapers. It is therefore present in pre-refactor results rather than
   introduced here, and nothing is silently corrected — see
   ``docs/REFACTOR_PLAN.md`` §5.2.6.

   It matters for the published workflow specifically: refining the window to
   the 1st-99th percentiles of cumulative energy trims the quiet lead-in while
   the coda tail remains, which pushes the arrival away from centre.

   Two ways out, both opt-in. Set ``center=True`` to remove the position
   dependence entirely — a circular shift leaves ``|FFT|`` unchanged, so it is
   free. Or prefer :class:`~specmod.transforms.fft.FFTEstimator` with a light
   taper, which holds 1.03x regardless of position, when absolute energy
   fidelity matters more than variance reduction.

.. note::

   **Fixed in this version.** Adaptive weighting previously collapsed for
   off-centre transients, recovering 0.203 of the true energy at 10% and 0.149
   at 90% where the table above now reads 0.956 and 0.898.

   The cause was a units mismatch, not anything in Thomson's method. Thomson's
   Eq. 5.1b regularises each weight with ``b_k = (1 - lambda_k) * sigma^2``,
   where ``sigma^2`` is the broadband power **in the units of the spectrum
   being weighted**. This module scales its eigenspectra to PSD (multiplying by
   ``dt``) but passed the record's raw time-domain variance as ``sigma^2``,
   overstating the leakage floor by a factor of ``1/dt`` — a hundredfold at
   100 sps. The regularisation term then dominated the denominator, every
   weight collapsed towards zero, and the collapse was worst exactly where the
   signal term was smallest: for a burst the tapers barely see. That is why it
   looked position-dependent, and why stationary noise — where the signal term
   is large at every frequency — passed cleanly and hid it.

   :func:`_adaptive_weights` now derives ``sigma^2`` from the eigenspectra
   themselves, which makes it invariant to how they were scaled, and clips the
   weights at unity as Thomson specifies. With
   ``normalize_to_variance=True`` putting both on the same absolute scale, the
   result now matches Prieto's ``multitaper`` to within 0.3% across the band,
   under both adaptive and flat weighting, for stationary noise and for bursts
   at 10%, 50% and 90%.

   Because the defect was ours and not the method's, ``adaptive`` defaults back
   to ``True`` — see the parameter documentation for why that is the better
   estimator once it works.

References
----------
Thomson, D.J. (1982). Spectrum estimation and harmonic analysis.
*Proc. IEEE* 70(9), 1055-1096.

Prieto, G.A. (2022). The multitaper spectrum analysis package in Python.
*Seismological Research Letters* 93(3), 1922-1929.
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


def _broadband_power(eigenspectra: NDArray[np.float64], n: int) -> float:
    """Mean eigenspectral power per two-sided bin — Thomson's ``sigma^2``.

    This is the white level that a taper's out-of-band leakage would deliver,
    and it is what Thomson's Eq. 5.1b multiplies by ``1 - lambda_k``. It must
    be expressed in the *same units as the eigenspectra*, which is the whole
    reason it is computed from them here rather than taken from the record.

    ``eigenspectra`` holds only the non-negative-frequency bins, so the
    two-sided sum is recovered by doubling and removing the double-counted DC
    and (for even ``n``) Nyquist terms.
    """
    total = 2.0 * eigenspectra.sum(axis=-1) - eigenspectra[..., 0]
    if n % 2 == 0:
        total -= eigenspectra[..., -1]
    return float((total / n).mean())


def _adaptive_weights(
    eigenspectra: NDArray[np.float64],
    eigenvalues: NDArray[np.float64],
    n: int,
    *,
    max_iter: int = 1000,
    tol: float = 9.5e-7,
) -> NDArray[np.float64]:
    """Thomson's adaptive weights.

    Higher-order tapers leak more, so weighting them equally lets out-of-band
    power contaminate the estimate. The weights downweight a taper wherever its
    broadband leakage would dominate the local signal, which matters here: a
    seismic spectrum spans orders of magnitude in amplitude, so leakage from the
    peak can swamp the high-frequency tail entirely.

    Iterates ``w_k = sqrt(lambda_k) * S / (lambda_k * S + b_k)`` to convergence,
    with ``b_k = (1 - lambda_k) * sigma^2`` from Thomson Eq. 5.1b, and clips the
    weights at unity so that an unbiased taper is weighted exactly once.

    ``sigma^2`` is derived from the eigenspectra rather than supplied, which
    makes the routine invariant to how the caller has scaled them. That is not
    a stylistic choice: passing the record's *time-domain* variance against
    eigenspectra already scaled to PSD overstates ``b_k`` by a factor of
    ``1/dt``, which drives every weight towards zero for any record whose
    energy is concentrated away from the taper centre. See the module docstring.
    """
    sigma2 = _broadband_power(eigenspectra, n)
    b_k = (1.0 - eigenvalues)[:, None] * sigma2
    sqrt_lambda = np.sqrt(eigenvalues)[:, None]
    lam = eigenvalues[:, None]

    # Start from the two least-leaky tapers, as Thomson recommends.
    spectrum = eigenspectra[:2].mean(axis=0)
    weights = np.ones_like(eigenspectra)
    for _ in range(max_iter):
        denom = np.maximum(lam * spectrum + b_k, 1e-300)
        weights = np.minimum(sqrt_lambda * spectrum / denom, 1.0)
        w2 = weights**2
        updated = (w2 * eigenspectra).sum(axis=0) / np.maximum(w2.sum(axis=0), 1e-300)
        change = np.abs(updated - spectrum) / np.maximum(updated + spectrum, 1e-300)
        spectrum = updated
        if change.max() < tol:
            break
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

        On by default, because leakage suppression is the reason to reach for
        multitaper at all and flat weighting does not provide it. Measured on a
        2 Hz line 10^6 times stronger than the background — a mild version of
        what a seismic spectrum does across its band — the recovered noise
        floor between 20 and 49 Hz sits **287x** above the truth with flat
        weighting and **1.1x** with adaptive. A Brune fit reads ``t*`` and
        ``f_c`` off exactly that high-frequency decay, so an inflated floor is
        not a cosmetic problem.

        The cost is resolution: adaptive weighting downweights the higher-order
        tapers wherever leakage would dominate, so it uses fewer effective
        degrees of freedom and gives a noisier estimate in bands where the
        signal is strong. Turn it off for a well-conditioned record with little
        dynamic range, where the extra averaging is worth more than the leakage
        rejection.

        This defaulted to ``False`` in earlier versions of the refactor, while
        the implementation was collapsing for off-centre transients. That was
        our bug and it is fixed; see the note above.
    center
        Circularly shift the record so its energy centroid sits mid-window
        before estimating.

        This removes the position dependence **entirely** rather than reducing
        it: measured across start positions from 2% to 78%, the recovered
        energy ratio is identical to three decimal places once centred, and the
        spectrum matches the naturally-centred case exactly. It is legitimate
        because ``|FFT|`` is invariant under a circular shift, so the quantity
        being estimated does not change.

        What remains after centring is the taper concentration itself — a
        compact centred transient still reads about 1.16x high with flat
        weighting. That bias is *consistent* rather than position-dependent,
        which matters: a consistent multiplicative bias cancels in any ratio
        (signal-to-noise, spectral ratios, relative amplitudes between stations)
        and can be calibrated, where a position-dependent one cannot.

        Off by default because a circular shift wraps. It is safe when the
        window edges are quiet, and refuses when they are not — see
        ``center_edge_tolerance``.
    center_edge_tolerance
        Maximum amplitude at the wrap point, as a fraction of the record's
        peak, before centring raises rather than introducing a discontinuity.
        A window whose coda is still strong at the end cannot be safely rolled.
    normalize_to_variance
        Rescale the whole spectrum so it integrates to the record's variance,
        as Prieto's ``multitaper`` package does (``mtspec.py``: ``sscal =
        xvar / (sum(spec)*df)``). ``mtspec`` wrapped the same lineage, so
        **enable this when reproducing pre-refactor results.**

        It is off by default for one reason: with it on, ``Spectrum.energy()``
        recovers the input energy *by construction*, so the Parseval check in
        the test suite stops being a falsifiable contract and starts being a
        tautology. Leaving it off keeps that check meaningful.

        It is a calibration, not a derivation. It forces total power to be
        right and lets the spectral *shape* absorb whatever error remains — so
        it does not make the long-period level position-independent, only much
        less position-dependent. See the warning above for measured numbers.
    """

    time_bandwidth: float = 3.0
    n_tapers: int = 5
    adaptive: bool = True
    center: bool = False
    center_edge_tolerance: float = 0.05
    normalize_to_variance: bool = False
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

    def _centered(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        energy = x**2
        centroid = int((np.arange(x.size) * energy).sum() / energy.sum())
        rolled: NDArray[np.float64] = np.roll(x, x.size // 2 - centroid)
        peak = float(np.abs(rolled).max())
        if peak > 0:
            step = abs(float(rolled[0]) - float(rolled[-1])) / peak
            if step > self.center_edge_tolerance:
                raise ValueError(
                    f"Centring would introduce a discontinuity of {step:.3f} of "
                    f"the record's peak at the wrap point, above "
                    f"center_edge_tolerance={self.center_edge_tolerance}. The "
                    f"window edges are not quiet enough to roll; taper first, "
                    f"widen the window, or use FFTEstimator, which is "
                    f"position-stable without centring."
                )
        return rolled

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
            x = self._centered(x)

        tapers, eigenvalues = dpss(
            n, self.time_bandwidth, self.n_tapers, sym=False, return_ratios=True
        )
        # DPSS tapers are unit-norm; scale so each eigenspectrum is a PSD in
        # [x]^2/Hz, then fold to one-sided.
        spectra = np.fft.rfft(tapers * x, axis=-1)
        eigenspectra = (np.abs(spectra) ** 2) * dt

        if self.adaptive and self.n_tapers > 1:
            weights = _adaptive_weights(eigenspectra, eigenvalues, n)
            w2 = weights**2
            psd = (w2 * eigenspectra).sum(axis=0) / np.maximum(w2.sum(axis=0), 1e-300)
        else:
            psd = eigenspectra.mean(axis=0)

        freq: NDArray[np.float64] = np.fft.rfftfreq(n, d=dt).astype(np.float64)
        psd[1:] *= 2.0  # fold negative frequencies
        if n % 2 == 0:
            psd[-1] /= 2.0

        if self.normalize_to_variance:
            # Prieto's convention: pin the integral to the record variance.
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
            },
            estimator=self.name,
        )
        return spectrum.to_kind(AmplitudeKind.FAS)
