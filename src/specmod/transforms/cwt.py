"""Continuous wavelet transform on an L2-normalised Morlet.

The reason this is written here rather than delegated to PyWavelets: the
normalisation is the entire difficulty. CWT coefficients carry units of
``[signal] * sqrt(time)`` under L2 and ``[signal]`` under L1, and neither is the
``[signal] * s`` of a Fourier amplitude spectrum. A naive ``|W|`` plotted
against scale-derived frequency looks entirely reasonable and gives the right
corner frequency with the wrong ``Omega`` — so the wrong seismic moment, with
nothing anywhere to catch it. ``pywt.cwt``'s conventions are not documented to
the precision that requires, so using it would mean reverse-engineering them.

The bridge back to amplitude lives in
:meth:`specmod.core.scalogram.Scalogram.time_average`, and
``tests/test_cwt.py`` is its specification: the CWT has to agree with the FFT
and multitaper estimators on a sinusoid of known amplitude and on the energy of
a record. The derivation is not trusted; the test is.

References
----------
Torrence, C. and Compo, G.P. (1998). A practical guide to wavelet analysis.
*Bulletin of the American Meteorological Society* 79(1), 61-78.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.scalogram import Scalogram
from ..core.spectrum import Spectrum
from ..core.units import Motion
from .base import prepare_record

__all__ = ["CWTEstimator"]


def _morlet_fourier_factor(omega0: float) -> float:
    """Scale-to-Fourier-period conversion, T&C Table 1.

    ``period = factor * scale``. This is what lets the output frequency axis
    mean the same thing as an FFT's rather than being "scale, roughly".
    """
    return float(4.0 * np.pi / (omega0 + np.sqrt(2.0 + omega0**2)))


@dataclass(frozen=True)
class CWTEstimator:
    """Continuous wavelet transform, time-averaged to an amplitude spectrum.

    Produces both outputs from one transform: :meth:`scalogram` returns the
    full time-frequency surface, and :meth:`estimate` returns its time average
    as an ordinary :class:`~specmod.core.spectrum.Spectrum`, so the fitting
    pipeline does not know the difference. The surface is what you look at when
    a fit comes out wrong.

    Parameters
    ----------
    omega0
        Morlet central frequency, dimensionless. Trades time resolution against
        frequency resolution; 6 is the conventional choice and the value for
        which the analytic scale-frequency relation is usually quoted.
    dj
        Scale spacing in octaves. Smaller resolves the frequency axis more
        finely at proportionally more cost. The normalisation carries an
        explicit ``dj`` factor, so the recovered energy does not depend on it.
    mask_coi
        Exclude the cone of influence from the time average. On by default:
        without it a short window reads low at low frequency, which is the band
        that constrains ``Omega``.
    f_min, f_max
        Frequency range to cover. Defaults span ``1/T`` — the longest period
        the record can represent — up to the Nyquist frequency.
    """

    omega0: float = 6.0
    dj: float = 0.125
    mask_coi: bool = True
    f_min: float | None = None
    f_max: float | None = None
    name: str = "cwt"

    def __post_init__(self) -> None:
        if self.omega0 <= 0:
            raise ValueError(f"omega0 must be positive, got {self.omega0}")
        if self.dj <= 0:
            raise ValueError(f"dj must be positive, got {self.dj}")
        if (
            self.f_min is not None
            and self.f_max is not None
            and self.f_min >= self.f_max
        ):
            raise ValueError(f"f_min={self.f_min} must be below f_max={self.f_max}")

    def _scales(self, n: int, dt: float) -> NDArray[np.float64]:
        factor = _morlet_fourier_factor(self.omega0)
        f_max = self.f_max if self.f_max is not None else 0.5 / dt
        f_min = self.f_min if self.f_min is not None else 1.0 / (n * dt)
        # scale = period / factor, and period = 1/f.
        s_min = 1.0 / (f_max * factor)
        s_max = 1.0 / (f_min * factor)
        n_scales = int(np.floor(np.log2(s_max / s_min) / self.dj)) + 1
        return np.asarray(s_min * 2.0 ** (np.arange(n_scales) * self.dj), dtype=float)

    def _c_delta(self, dt: float) -> float:
        """Reconstruction constant, computed against *this* scale grid.

        Two reasons not to use the tabulated value, and the second is the one
        that matters.

        First, T&C tabulate 0.776 for ``omega0=6`` only, and this class lets
        ``omega0`` vary. The constant moves with it — 1.05 at 4, 0.55 at 8 —
        so a hardcoded 0.776 silently mis-normalises every non-default choice.

        Second, and less obvious: the value derived here is deliberately *not*
        0.776 even at ``omega0=6``. It comes out near 0.72 and drifts with
        ``dj``, because the reconstruction sum is a discrete approximation to a
        continuous integral over scale, and this measures the approximation
        actually being used rather than its limit. That is what keeps the
        Parseval contract: with the computed constant, recovered energy is
        0.97-1.07 of the truth; with 0.776 it is systematically ~7% low, and
        the error grows as ``dj`` coarsens.

        So the ``dj`` dependence is the point, not a defect — do not "fix" it
        by substituting the published number. ``tests/test_cwt.py`` pins this.
        """
        n = 1024
        delta = np.zeros(n)
        delta[n // 2] = 1.0
        scales = self._scales(n, dt)
        coefficients = self._transform(delta, dt, scales)
        # T&C eq. 11 rearranged: the delta reconstructs to 1 at its own sample.
        summed = float(np.sum(np.real(coefficients[:, n // 2]) / np.sqrt(scales)))
        psi0_at_zero = np.pi**-0.25
        return float(self.dj * np.sqrt(dt) / psi0_at_zero * summed)

    def _transform(
        self, x: NDArray[np.float64], dt: float, scales: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Wavelet coefficients, L2-normalised, computed in the frequency domain."""
        n = x.size
        x_hat = np.fft.fft(x)
        omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)

        # Morlet is analytic: it has no support at negative frequency, which is
        # what makes |W| an envelope rather than an oscillation.
        heaviside = (omega > 0).astype(float)
        scaled = scales[:, None] * omega[None, :]
        psi_hat = (
            np.pi**-0.25
            * heaviside[None, :]
            * np.exp(-((scaled - self.omega0) ** 2) / 2.0)
        )
        # T&C eq. 6: the sqrt(2*pi*s/dt) factor is the L2 normalisation, and is
        # what makes the coefficients comparable across scales.
        psi_hat = psi_hat * np.sqrt(2.0 * np.pi * scales[:, None] / dt)

        coefficients: NDArray[np.complex128] = np.fft.ifft(
            x_hat[None, :] * np.conjugate(psi_hat), axis=-1
        )
        return coefficients

    def scalogram(
        self,
        data: ArrayLike,
        dt: float,
        *,
        motion: Motion | str = Motion.VELOCITY,
        meta: dict[str, Any] | None = None,
    ) -> Scalogram:
        """The full time-frequency surface."""
        x, n, duration = prepare_record(data, dt)
        scales = self._scales(n, dt)
        coefficients = self._transform(x, dt, scales)

        factor = _morlet_fourier_factor(self.omega0)
        freq = 1.0 / (factor * scales)

        # T&C section 3g: the e-folding time for Morlet is sqrt(2)*s, so the
        # longest resolvable period at a given sample grows with its distance
        # from the nearest edge.
        edge_distance = np.minimum(np.arange(n), n - 1 - np.arange(n)) * dt
        coi = factor * np.sqrt(2.0) * edge_distance

        return Scalogram(
            time=(np.arange(n) * dt).astype(np.float64),
            freq=freq,
            power=np.abs(coefficients) ** 2,
            scales=scales,
            coi=coi,
            c_delta=self._c_delta(dt),
            dj=self.dj,
            dt=dt,
            motion=Motion(motion),
            meta=MappingProxyType(
                {
                    **(meta or {}),
                    "estimator": self.name,
                    "omega0": self.omega0,
                    "dj": self.dj,
                    "duration": duration,
                }
            ),
        )

    def estimate(
        self,
        data: ArrayLike,
        dt: float,
        *,
        motion: Motion | str = Motion.VELOCITY,
        meta: dict[str, Any] | None = None,
    ) -> Spectrum:
        """Time-averaged amplitude spectrum, interchangeable with the others."""
        surface = self.scalogram(data, dt, motion=motion, meta=meta)
        return surface.time_average(mask_coi=self.mask_coi)
