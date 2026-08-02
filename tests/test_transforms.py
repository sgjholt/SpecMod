"""The normalisation contract, applied to every estimator.

This is the load-bearing test suite of the refactor. The plan's premise is that
one contract, pinned by one set of tests, covers every backend — so that
swapping mtspec for a native multitaper, or adding a CWT later, cannot quietly
change the science. See docs/REFACTOR_PLAN.md §4.1 and §5.

Tolerances differ by estimator on purpose and are justified where they are set.
A boxcar FFT of a bin-centred sinusoid is exact to floating point; a multitaper
deliberately trades a little bias for a large variance reduction.
"""

from __future__ import annotations

import numpy as np
import pytest

from specmod.core import AmplitudeKind, Motion, Spectrum
from specmod.transforms import (
    ESTIMATORS,
    FFTEstimator,
    MultitaperEstimator,
    WelchEstimator,
    get_estimator,
)

FS = 100.0
DT = 1.0 / FS
DURATION = 20.0
N = int(FS * DURATION)


def sinusoid(amplitude: float = 2.5, frequency: float = 5.0) -> np.ndarray:
    """A whole number of cycles, so the line falls on a bin centre."""
    return amplitude * np.sin(2 * np.pi * frequency * np.arange(N) * DT)


def noise(sigma: float = 1e-6, seed: int = 42) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, N)


def time_domain_energy(x: np.ndarray) -> float:
    """``sum(x^2) * dt`` on the demeaned record, which is what Parseval ties to."""
    return float(np.sum((x - x.mean()) ** 2) * DT)


ALL_ESTIMATORS = [
    pytest.param(FFTEstimator(taper="boxcar"), 1e-9, id="fft-boxcar"),
    pytest.param(FFTEstimator(), 0.02, id="fft-tukey"),
    pytest.param(MultitaperEstimator(), 0.03, id="multitaper"),
    pytest.param(MultitaperEstimator(adaptive=True), 0.03, id="multitaper-adaptive"),
    pytest.param(WelchEstimator(), 0.03, id="welch"),
]


# ------------------------------------------------------------------- Parseval


@pytest.mark.parametrize(("estimator", "rtol"), ALL_ESTIMATORS)
def test_energy_is_conserved_for_noise(estimator, rtol: float) -> None:
    """The contract: spectrum.energy() recovers sum(x^2)*dt.

    Broadband noise is the fair test — a continuous spectrum is what these
    estimators are designed for.
    """
    x = noise()
    spectrum = estimator.estimate(x, DT)
    assert spectrum.energy() == pytest.approx(time_domain_energy(x), rel=rtol)


@pytest.mark.parametrize(("estimator", "rtol"), ALL_ESTIMATORS)
def test_energy_is_conserved_for_a_sinusoid(estimator, rtol: float) -> None:
    x = sinusoid()
    spectrum = estimator.estimate(x, DT)
    assert spectrum.energy() == pytest.approx(
        time_domain_energy(x), rel=max(rtol, 0.02)
    )


@pytest.mark.parametrize(("estimator", "rtol"), ALL_ESTIMATORS)
def test_energy_scales_with_the_square_of_amplitude(estimator, rtol: float) -> None:
    """Linearity: doubling the record quadruples its energy."""
    one = estimator.estimate(noise(), DT).energy()
    two = estimator.estimate(2.0 * noise(), DT).energy()
    assert two == pytest.approx(4.0 * one, rel=1e-6)


# ---------------------------------------------------------- amplitude recovery


def test_fas_peak_of_a_sinusoid_is_amplitude_times_duration() -> None:
    """A Fourier *amplitude* spectrum is an integral, so it grows with T.

    This is why Omega has units of m*s, and it is the property that makes the
    long-period level comparable between records only after this normalisation.
    """
    a0 = 2.5
    spectrum = FFTEstimator(taper="boxcar").estimate(sinusoid(a0), DT)
    assert spectrum.amp.max() == pytest.approx(a0 * DURATION, rel=1e-9)


def test_amplitude_correction_recovers_a_tapered_peak() -> None:
    """The two taper corrections are not interchangeable.

    'amplitude' preserves a coherent line; 'energy' preserves total power. See
    specmod.transforms.base.
    """
    a0 = 2.5
    tapered = FFTEstimator(taper="tukey", taper_correction="amplitude")
    peak = tapered.estimate(sinusoid(a0), DT).amp.max()
    assert peak == pytest.approx(a0 * DURATION, rel=0.02)


def test_peak_lands_at_the_right_frequency() -> None:
    f0 = 7.0
    spectrum = FFTEstimator(taper="boxcar").estimate(sinusoid(frequency=f0), DT)
    assert spectrum.freq[spectrum.amp.argmax()] == pytest.approx(f0, abs=0.05)


# ------------------------------------------------- §2.2 padding regression


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_zero_padding_changes_resolution_not_amplitude(factor: int) -> None:
    """The bug that motivated typing the units.

    The pre-refactor conversion divided by ``len(freq)``, so padding to 2N
    inflated amplitudes by sqrt(2) — and the tutorial ships a commented-out
    padding block ready to be uncommented. Normalising by ``dt`` and carrying
    the physical duration makes padding a pure interpolation.
    """
    a0 = 2.5
    x = sinusoid(a0)
    spectrum = FFTEstimator(taper="boxcar", n_fft=N * factor).estimate(x, DT)

    assert len(spectrum) == N * factor // 2
    assert spectrum.duration == pytest.approx(DURATION)
    assert spectrum.amp.max() == pytest.approx(a0 * DURATION, rel=1e-9)
    assert spectrum.energy() == pytest.approx(time_domain_energy(x), rel=1e-3)


def test_padding_shorter_than_the_record_is_rejected() -> None:
    with pytest.raises(ValueError, match="shorter than the record"):
        FFTEstimator(n_fft=N // 2).estimate(sinusoid(), DT)


# ------------------------------------------------------ cross-estimator accord


def brune_realisation(seed: int, fc: float = 3.0, omega: float = 1e-4) -> np.ndarray:
    """A synthetic record whose spectrum is flat below ``fc`` and falls as f^-2."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(N, DT)
    target = omega / (1.0 + (freqs / fc) ** 2)
    phase = rng.uniform(0.0, 2 * np.pi, freqs.size)
    return np.fft.irfft(target * np.exp(1j * phase) * N, n=N)


def test_estimators_agree_on_the_long_period_level() -> None:
    """The quantity the science actually reads off: the flat low-frequency level.

    If backends disagreed here, swapping mtspec out would move every Omega and
    therefore every seismic moment.

    Averaged over realisations on purpose. A single periodogram bin is
    chi-square with 2 degrees of freedom, so one record cannot distinguish a
    normalisation difference from ordinary estimator variance — which is
    exactly what an earlier version of this test did, and it failed for that
    reason rather than for a real one. Averaging targets *bias*, which is what
    "do the backends agree" actually means.
    """
    estimators = {
        "fft": FFTEstimator(),
        "multitaper": MultitaperEstimator(),
        # A real Welch: short enough segments that it actually averages.
        "welch": WelchEstimator(segment_length=N // 4),
    }
    levels: dict[str, float] = {}
    for name, est in estimators.items():
        per_seed = [
            float(np.median(est.estimate(brune_realisation(s), DT).band(0.4, 2.0).amp))
            for s in range(24)
        ]
        levels[name] = float(np.mean(per_seed))

    reference = levels["fft"]
    for name, level in levels.items():
        assert level == pytest.approx(reference, rel=0.10), (
            f"{name} disagrees on the long-period level: {level:.4g} vs {reference:.4g}"
        )


# ------------------------------------------------------------- estimator setup


def test_multitaper_rejects_too_many_tapers() -> None:
    """Beyond 2*NW-1 the tapers leak rather than reduce variance."""
    with pytest.raises(ValueError, match=r"exceeds 2\*NW-1"):
        MultitaperEstimator(time_bandwidth=3.0, n_tapers=6)


def test_multitaper_time_bandwidth_is_configurable() -> None:
    """It was the literal 3 passed positionally to mtspec."""
    spectrum = MultitaperEstimator(time_bandwidth=4.0, n_tapers=7).estimate(noise(), DT)
    assert spectrum.meta["time_bandwidth"] == 4.0
    assert spectrum.meta["n_tapers"] == 7


def test_registry_resolves_every_estimator() -> None:
    for name in ESTIMATORS:
        est = get_estimator(name)
        assert est.estimate(noise(), DT).kind is AmplitudeKind.FAS


def test_unknown_estimator_names_the_alternatives() -> None:
    with pytest.raises(ValueError, match="Unknown estimator"):
        get_estimator("mtspec")


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (np.array([1.0]), "too short"),
        (np.array([1.0, np.nan, 2.0]), "NaN or Inf"),
        (np.ones((4, 4)), "1-D record"),
    ],
)
def test_bad_records_are_rejected(bad: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FFTEstimator().estimate(bad, DT)


def test_non_positive_dt_is_rejected() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        FFTEstimator().estimate(sinusoid(), 0.0)


# --------------------------------------------------------------- Spectrum type


def test_spectrum_records_its_units() -> None:
    s = FFTEstimator().estimate(sinusoid(), DT, motion=Motion.VELOCITY)
    assert s.kind is AmplitudeKind.FAS
    assert s.motion is Motion.VELOCITY
    assert s.unit == "m/s*s"
    assert s.duration == pytest.approx(DURATION)
    assert s.frequency_resolution == pytest.approx(1 / DURATION)


@pytest.mark.parametrize("kind", list(AmplitudeKind))
def test_kind_conversions_round_trip(kind: AmplitudeKind) -> None:
    original = FFTEstimator().estimate(noise(), DT)
    assert original.to_kind(kind).to_kind(AmplitudeKind.FAS).amp == pytest.approx(
        original.amp, rel=1e-12
    )


def test_energy_is_invariant_under_kind_conversion() -> None:
    """A PSD and the FAS it came from describe the same record."""
    s = FFTEstimator().estimate(noise(), DT)
    for kind in AmplitudeKind:
        assert s.to_kind(kind).energy() == pytest.approx(s.energy(), rel=1e-10)


@pytest.mark.parametrize("target", list(Motion))
def test_motion_conversions_round_trip(target: Motion) -> None:
    original = FFTEstimator().estimate(noise(), DT, motion=Motion.VELOCITY)
    there_and_back = original.to_motion(target).to_motion(Motion.VELOCITY)
    assert there_and_back.amp == pytest.approx(original.amp, rel=1e-10)
    assert there_and_back.motion is Motion.VELOCITY


def test_integration_divides_by_angular_frequency() -> None:
    """Velocity -> displacement is division by 2*pi*f, once."""
    vel = FFTEstimator().estimate(noise(), DT, motion=Motion.VELOCITY)
    disp = vel.to_motion(Motion.DISPLACEMENT)
    assert disp.amp == pytest.approx(vel.amp / (2 * np.pi * vel.freq), rel=1e-12)
    assert disp.unit == "m*s"


def test_motion_conversion_of_a_psd_squares_correctly() -> None:
    """A PSD is amplitude squared, so it must not take a linear factor.

    Converting via FAS is what stops this being silently wrong by (2*pi*f).
    """
    vel = FFTEstimator().estimate(noise(), DT, motion=Motion.VELOCITY)
    psd_then = vel.to_kind(AmplitudeKind.PSD).to_motion(Motion.DISPLACEMENT)
    motion_then = vel.to_motion(Motion.DISPLACEMENT).to_kind(AmplitudeKind.PSD)
    assert psd_then.amp == pytest.approx(motion_then.amp, rel=1e-10)


def test_spectrum_is_immutable() -> None:
    """The old Spectrum mutated in place, so a double integrate was silent."""
    s = FFTEstimator().estimate(noise(), DT)
    with pytest.raises((AttributeError, ValueError)):
        s.amp[0] = 1.0
    with pytest.raises(AttributeError):
        s.motion = Motion.DISPLACEMENT  # type: ignore[misc]


def test_conversions_return_new_objects() -> None:
    s = FFTEstimator().estimate(noise(), DT)
    before = s.amp.copy()
    s.to_motion(Motion.DISPLACEMENT)
    s.to_kind(AmplitudeKind.PSD)
    assert s.amp == pytest.approx(before)


def test_mismatched_arrays_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        Spectrum(
            freq=np.arange(10.0),
            amp=np.arange(5.0),
            motion=Motion.VELOCITY,
            kind=AmplitudeKind.FAS,
            duration=1.0,
            sampling_rate=100.0,
        )


def test_band_selection_reports_the_available_range() -> None:
    s = FFTEstimator().estimate(noise(), DT)
    assert s.band(1.0, 10.0).freq.max() <= 10.0
    with pytest.raises(ValueError, match="No samples in band"):
        s.band(200.0, 300.0)


# ------------------------------------------------- stationarity and transients


def transient(fraction: float = 0.07, seed: int = 3) -> np.ndarray:
    """Energy concentrated in a small, central part of the window.

    A crude stand-in for a seismic arrival after the window refinement the
    published workflow applies, which deliberately tightens onto the energetic
    part of the record.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(N)
    width = int(N * fraction)
    start = (N - width) // 2
    x[start : start + width] = rng.normal(0.0, 1e-6, width)
    return x


def transient_at(position: float, width: float = 0.10, seed: int = 1) -> np.ndarray:
    """A burst of fixed energy and width, placed anywhere in the window."""
    rng = np.random.default_rng(seed)
    x = np.zeros(N)
    w = int(N * width)
    start = max(0, min(N - w, int(N * position) - w // 2))
    x[start : start + w] = rng.normal(0.0, 1e-6, w)
    return x


def test_flat_multitaper_bias_tracks_the_taper_envelope() -> None:
    """Without adaptive weighting the bias is modest and explicable.

    It follows the summed DPSS envelope: the tapers weight the middle of the
    record above the ends, so a centred transient reads high and an edge one
    low, by roughly +/-15%.
    """
    est = MultitaperEstimator(adaptive=False)
    ratios = {
        p: est.estimate(transient_at(p), DT).energy()
        / time_domain_energy(transient_at(p))
        for p in (0.10, 0.50, 0.90)
    }
    assert 0.85 < ratios[0.10] < 1.05
    assert 1.05 < ratios[0.50] < 1.30
    assert ratios[0.50] > ratios[0.10], "centre should read higher than the edge"


def test_adaptive_weighting_collapses_for_edge_transients() -> None:
    """A known, unexplained deficiency — pinned so it cannot regress silently.

    An edge-located burst loses 80-85% of its energy under adaptive weighting,
    far more than taper shape accounts for. The suspected cause is the weights
    being seeded from the two lowest-order tapers, which see almost none of it.
    This is documented rather than fixed because changing it would move
    published numbers; see the module docstring and REFACTOR_PLAN §5.2.6.
    """
    edge = transient_at(0.10)
    centre = transient_at(0.50)

    adaptive = MultitaperEstimator(adaptive=True)
    edge_ratio = adaptive.estimate(edge, DT).energy() / time_domain_energy(edge)
    centre_ratio = adaptive.estimate(centre, DT).energy() / time_domain_energy(centre)

    assert edge_ratio < 0.35, "the collapse should be severe, not marginal"
    assert centre_ratio > 1.15
    # Turning adaptive off restores sane behaviour — which is why it is now
    # the shipped default.
    flat = MultitaperEstimator(adaptive=False).estimate(edge, DT).energy()
    assert flat / time_domain_energy(edge) > 0.85
    assert MultitaperEstimator().adaptive is False, "default must stay off"


def test_light_taper_fft_tracks_transient_energy_far_better() -> None:
    """The practical consequence: prefer FFT when energy fidelity matters."""
    x = transient()
    expected = time_domain_energy(x)
    fft = FFTEstimator(taper="tukey", taper_alpha=0.05).estimate(x, DT).energy()
    multitaper = MultitaperEstimator().estimate(x, DT).energy()
    assert abs(fft / expected - 1.0) < abs(multitaper / expected - 1.0) / 2


def test_the_bias_is_specific_to_transients() -> None:
    """Stationary noise is recovered correctly, which is why §5's Parseval
    tests use it — and why they did not catch this."""
    x = noise()
    ratio = MultitaperEstimator().estimate(x, DT).energy() / time_domain_energy(x)
    assert ratio == pytest.approx(1.0, rel=0.03)
