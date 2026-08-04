"""Acceptance tests for the continuous wavelet transform.

Written before the implementation, because §4.4 of the refactor plan says to:
the CWT's normalisation is where this is easy to get subtly wrong, and a wrong
derivation produces a plausible-looking spectrum with a wrong ``Omega`` — the
one error that propagates straight into ``M0`` without anything complaining.

So the specification is not the derivation, it is this file. A CWT amplitude
spectrum has to agree with the FFT and multitaper estimators on quantities
whose true values are known independently: the amplitude of a sinusoid, and the
energy of a record.
"""

from __future__ import annotations

import numpy as np
import pytest

from specmod.core import AmplitudeKind, Scalogram
from specmod.transforms import CWTEstimator, FFTEstimator, MultitaperEstimator

FS = 100.0
DT = 1.0 / FS
N = 2048
DURATION = N * DT


def noise(sigma: float = 1e-6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, N)


def time_domain_energy(x: np.ndarray) -> float:
    return float(np.sum((x - x.mean()) ** 2) * DT)


# ------------------------------------------------------- the normalisation


def test_recovers_the_energy_of_a_record() -> None:
    """The Parseval bridge, which is the whole point of §4.4.

    ``C_delta`` and the ``dj*dt`` factors have to combine to return the record's
    energy, exactly as they do for every other estimator here. This is the
    check that says the wavelet coefficients were converted to a spectrum and
    not merely plotted.
    """
    x = noise()
    spectrum = CWTEstimator().estimate(x, DT)
    assert spectrum.energy() == pytest.approx(time_domain_energy(x), rel=0.15)


def test_agrees_with_the_other_estimators_on_a_known_sinusoid() -> None:
    """A sinusoid of amplitude ``A`` has Fourier amplitude ``A*T/2``.

    All three estimators must find the same peak at the same frequency. The CWT
    is broader in frequency than the FFT, so the tolerance is loose — what is
    being tested is that the *normalisation convention* matches, not the
    resolution.
    """
    t = np.arange(N) * DT
    amplitude = 2.5
    x = amplitude * np.sin(2 * np.pi * 5.0 * t)
    truth = time_domain_energy(x)

    found = {}
    for name, est in (
        ("fft", FFTEstimator(taper="boxcar")),
        ("multitaper", MultitaperEstimator()),
        ("cwt", CWTEstimator()),
    ):
        spectrum = est.estimate(x, DT)
        band = spectrum.band(3.0, 8.0)
        found[name] = (
            float(spectrum.freq[np.argmax(spectrum.amp)]),
            float(np.trapezoid(band.amp**2 / 2.0, band.freq)),
        )

    for name, (freq, _) in found.items():
        assert freq == pytest.approx(5.0, rel=0.10), f"{name} peaked at {freq:.2f} Hz"

    # Energy in a band around the line, *not* peak height. Each estimator
    # spreads a pure line over its own resolution bandwidth, so the peaks
    # legitimately differ by a factor of four here and comparing them would
    # assert almost nothing. The integral under the line does not care about
    # bandwidth, so it is the quantity on which the three must actually agree —
    # and it is what catches a normalisation out by 2*pi, by dt, or by sqrt(dt).
    for name, (_, energy) in found.items():
        assert energy == pytest.approx(truth, rel=0.10), (
            f"{name} put {energy:.3f} in the 3-8 Hz band against a true {truth:.3f}"
        )


def test_amplitude_scales_linearly_with_the_record() -> None:
    """Doubling the record must double the amplitude spectrum, not its square."""
    x = noise()
    one = CWTEstimator().estimate(x, DT)
    two = CWTEstimator().estimate(2.0 * x, DT)
    ratio = float(np.median(two.amp / one.amp))
    assert ratio == pytest.approx(2.0, rel=0.01)


def test_energy_is_independent_of_scale_resolution() -> None:
    """``dj`` sets how finely the scales are sampled, not how much energy exists.

    A normalisation that forgot the ``dj`` factor would pass every
    single-resolution test and fail this one.
    """
    x = noise()
    energies = [
        CWTEstimator(dj=dj).estimate(x, DT).energy() for dj in (0.0625, 0.125, 0.25)
    ]
    assert max(energies) / min(energies) < 1.1


def test_reports_units_like_every_other_estimator() -> None:
    spectrum = CWTEstimator().estimate(noise(), DT, motion="velocity")
    assert spectrum.kind is AmplitudeKind.FAS
    assert spectrum.unit == "m/s*s"
    assert spectrum.meta["estimator"] == "cwt"
    assert spectrum.meta["omega0"] == 6.0


def test_the_reconstruction_constant_beats_the_tabulated_one() -> None:
    """``C_delta`` is computed against our own scale grid, and must stay that way.

    Torrence & Compo tabulate 0.776 for Morlet at ``omega0=6``, and substituting
    it here looks like an obvious tidy-up. It is not: the tabulated figure is
    the continuum limit, while the reconstruction sum is discrete, so using it
    leaves recovered energy systematically low. The computed value measures the
    approximation actually in use.

    This is why :meth:`CWTEstimator._c_delta` drifts with ``dj`` — it is
    absorbing that discretisation, which is also why recovered energy does not
    drift with ``dj``.
    """
    import dataclasses

    x = noise()
    truth = time_domain_energy(x)
    surface = CWTEstimator().scalogram(x, DT)

    computed = surface.time_average().energy() / truth
    tabulated = (
        dataclasses.replace(surface, c_delta=0.776).time_average().energy() / truth
    )
    assert abs(computed - 1.0) < abs(tabulated - 1.0), (
        f"the computed constant ({computed:.3f}) should recover energy better "
        f"than the tabulated 0.776 ({tabulated:.3f})"
    )


def test_the_reconstruction_constant_tracks_omega0() -> None:
    """It is a property of the wavelet, so changing the wavelet must move it."""
    constants = [CWTEstimator(omega0=w)._c_delta(DT) for w in (4.0, 6.0, 8.0)]
    assert constants[0] > constants[1] > constants[2], (
        "C_delta should fall as omega0 rises; a hardcoded constant would not"
    )


# ------------------------------------------------------------ the scalogram


def test_produces_a_time_frequency_surface() -> None:
    """The QC gate the plan asks for: both outputs from one transform."""
    x = noise()
    scalogram = CWTEstimator().scalogram(x, DT)
    assert isinstance(scalogram, Scalogram)
    assert scalogram.power.shape == (scalogram.freq.size, scalogram.time.size)
    assert scalogram.time.size == N
    assert np.isfinite(scalogram.power).all()
    assert scalogram.time_average().kind is AmplitudeKind.FAS


def test_the_surface_localises_a_transient_in_time() -> None:
    """What the scalogram is for: a time average cannot show this."""
    t = np.arange(N) * DT
    x = np.zeros(N)
    start = N // 2
    width = N // 16
    x[start : start + width] = np.sin(2 * np.pi * 10.0 * t[:width])

    scalogram = CWTEstimator().scalogram(x, DT)
    band = np.argmin(np.abs(scalogram.freq - 10.0))
    energy_in_time = scalogram.power[band]
    centroid = float((np.arange(N) * energy_in_time).sum() / energy_in_time.sum() / N)
    assert 0.45 < centroid < 0.62, f"transient centroid recovered at {centroid:.2%}"


def test_cone_of_influence_masks_the_unresolvable_low_frequencies() -> None:
    """A window cannot resolve a period longer than itself.

    The COI is the only estimator here that makes that limit explicit, which is
    the §4.4.2 argument for feeding it into bandwidth selection.
    """
    scalogram = CWTEstimator().scalogram(noise(), DT)
    coverage = scalogram.coi_coverage()
    assert coverage.shape == scalogram.freq.shape
    assert (coverage <= 1.0).all()
    assert (coverage >= 0.0).all()
    # High frequencies are fully resolved; the lowest are not.
    assert coverage[np.argmax(scalogram.freq)] > 0.95
    assert coverage[np.argmin(scalogram.freq)] < 0.5


def test_masking_the_coi_changes_the_low_frequency_end_only() -> None:
    """High frequencies sit well inside the cone, so masking must not move them.

    Masking drops the frequencies the record cannot resolve, so the two spectra
    have different axis lengths — which is the point, and why this compares on
    the masked axis rather than element-wise.
    """
    x = noise()
    scalogram = CWTEstimator().scalogram(x, DT)
    masked = scalogram.time_average(mask_coi=True)
    unmasked = scalogram.time_average(mask_coi=False)

    assert masked.freq.size < unmasked.freq.size, "masking should drop scales"
    assert masked.freq.min() > unmasked.freq.min(), "it should drop the low end"

    high = masked.freq > 10.0
    reference = np.interp(masked.freq[high], unmasked.freq, unmasked.amp)
    assert masked.amp[high] == pytest.approx(reference, rel=0.05)


def test_qc_reports_the_checks_the_plan_specifies() -> None:
    """§4.4.2. Compute and record, never silently drop."""
    x = noise()
    qc = CWTEstimator().scalogram(x, DT).qc()

    assert qc.lowest_resolved_frequency >= 0.0
    assert 0.0 <= qc.temporal_concentration <= 1.0
    assert qc.half_window_ratio > 0.0
    assert isinstance(qc.to_dict(), dict)


def test_qc_flags_a_spike_that_amplitude_snr_would_pass() -> None:
    """The failure mode §4.4.2 exists to catch.

    A single glitch has a large broadband amplitude, so an amplitude-only SNR
    test sees a strong, wideband "signal". Its energy is concentrated in one
    sample, which is exactly what the temporal-concentration measure sees.
    """
    spike = np.zeros(N)
    spike[N // 2] = 1.0
    spread = noise(sigma=1.0)

    est = CWTEstimator()
    concentrated = est.scalogram(spike, DT).qc().temporal_concentration
    diffuse = est.scalogram(spread, DT).qc().temporal_concentration
    assert concentrated > diffuse * 2.0
