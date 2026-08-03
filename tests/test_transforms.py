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


@pytest.mark.parametrize("adaptive", [False, True])
def test_multitaper_bias_tracks_the_taper_envelope(adaptive: bool) -> None:
    """The bias is modest, explicable, and the same under either weighting.

    It follows the summed DPSS envelope: the tapers weight the middle of the
    record above the ends, so a centred transient reads high and an edge one
    low, by roughly +/-15%. Adaptive weighting is held to the same standard as
    flat because the two now agree — which is the whole point of the fix.
    """
    est = MultitaperEstimator(adaptive=adaptive)
    ratios = {
        p: est.estimate(transient_at(p), DT).energy()
        / time_domain_energy(transient_at(p))
        for p in (0.10, 0.50, 0.90)
    }
    assert 0.85 < ratios[0.10] < 1.05
    assert 1.05 < ratios[0.50] < 1.30
    assert ratios[0.50] > ratios[0.10], "centre should read higher than the edge"


def test_adaptive_weighting_does_not_collapse_for_edge_transients() -> None:
    """Regression test for a units bug that made adaptive weighting unusable.

    Thomson's Eq. 5.1b regularises the weights with ``(1 - lambda_k) *
    sigma^2``, and ``sigma^2`` has to be in the units of the spectrum being
    weighted. Passing the record's time-domain variance against PSD-scaled
    eigenspectra overstated it by ``1/dt``, so the regularisation term swamped
    the signal term and every weight collapsed towards zero — worst exactly
    where the tapers saw least of the burst. At 10% this recovered 0.203 of the
    true energy and at 90%, 0.149.

    The tolerance below is deliberately tight against flat weighting rather
    than against 1.0: what is being asserted is that the two weightings see the
    same energy, since any residual is taper shape and common to both.
    """
    for position in (0.10, 0.50, 0.90):
        x = transient_at(position)
        expected = time_domain_energy(x)
        adaptive = MultitaperEstimator(adaptive=True).estimate(x, DT).energy()
        flat = MultitaperEstimator(adaptive=False).estimate(x, DT).energy()
        assert adaptive / expected > 0.85, f"collapse has returned at {position:.0%}"
        assert adaptive == pytest.approx(flat, rel=0.03), (
            f"weightings disagree at {position:.0%}; they should differ in "
            f"variance and leakage, not in total energy"
        )


def test_adaptive_weighting_suppresses_leakage_and_flat_does_not() -> None:
    """Why adaptive is the default: it is the only one that does this job.

    A strong low-frequency peak over a weak high-frequency tail is the ordinary
    shape of a seismic spectrum, and ``t*`` and ``f_c`` are both read off that
    tail. Flat weighting lets the peak leak into it through the higher-order
    tapers; adaptive weighting is precisely the mechanism for not doing that.
    """
    rng = np.random.default_rng(11)
    t = np.arange(N) * DT
    weak = rng.normal(0.0, 1e-9, N)
    x = 1e-3 * np.sin(2 * np.pi * 2.0 * t) + weak

    band = (20.0, 49.0)
    truth = FFTEstimator(taper="tukey", taper_alpha=0.05).estimate(weak, DT).band(*band)
    flat = MultitaperEstimator(adaptive=False).estimate(x, DT).band(*band)
    adaptive = MultitaperEstimator(adaptive=True).estimate(x, DT).band(*band)

    assert float(np.median(flat.amp / truth.amp)) > 50.0
    assert float(np.median(adaptive.amp / truth.amp)) < 2.0


def test_adaptive_weighting_is_the_default() -> None:
    assert MultitaperEstimator().adaptive is True
    assert MultitaperEstimator().estimate(transient(), DT).meta["adaptive"] is True


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


@pytest.mark.parametrize("adaptive", [False, True])
def test_variance_normalisation_is_exactly_a_scalar_multiply(adaptive: bool) -> None:
    """Prieto's convention, offered explicitly because mtspec used it.

    Asserting that it recovers the input energy would be worthless — it pins
    the integral by construction, so that check cannot fail while the feature
    exists at all. What *is* falsifiable is that it does so by scaling the
    whole spectrum uniformly: a frequency-dependent normalisation would pass an
    energy check and still be wrong.

    This is also what makes the next test true rather than merely observed. A
    scalar multiply cannot change any ratio *within* a spectrum, so it cannot
    move the low-frequency plateau relative to the rest, and so it cannot fix
    the position dependence that plateau inherits.
    """
    x = transient_at(0.10)
    raw = MultitaperEstimator(adaptive=adaptive).estimate(x, DT)
    normalised = MultitaperEstimator(
        adaptive=adaptive, normalize_to_variance=True
    ).estimate(x, DT)

    assert normalised.freq == pytest.approx(raw.freq)
    ratio = normalised.amp / raw.amp
    assert ratio.std() / ratio.mean() < 1e-12, "normalisation must not reshape"
    # ...and the constant is the one that corrects total energy, which is the
    # only thing the caller is actually buying.
    expected = np.sqrt(time_domain_energy(x) / raw.energy())
    assert float(ratio.mean()) == pytest.approx(expected, rel=0.02)


def test_variance_normalisation_is_off_by_default() -> None:
    assert MultitaperEstimator().normalize_to_variance is False
    assert (
        MultitaperEstimator()
        .estimate(transient_at(0.10), DT)
        .meta["normalize_to_variance"]
        is False
    )


def test_variance_normalisation_does_not_fix_the_plateau() -> None:
    """The distinction the docs turn on: it pins energy, not spectral shape.

    Omega is read off the low-frequency plateau, so a caller who enables this
    should not assume the level is now position-independent.
    """
    est = MultitaperEstimator(normalize_to_variance=True)
    band = (1.0, 4.0)
    edge = float(np.median(est.estimate(transient_at(0.10), DT).band(*band).amp))
    centre = float(np.median(est.estimate(transient_at(0.50), DT).band(*band).amp))
    assert edge != pytest.approx(centre, rel=0.05), (
        "if these agree, variance normalisation has become a full fix and the "
        "documentation in docs/choosing_a_transform.md needs revisiting"
    )


# ---------------------------------------------------- centring the transient


def test_centring_removes_position_dependence_entirely() -> None:
    """The bias is positional, not phase-related, so centring is a complete fix.

    Distinguishing the two mattered: a *symmetric* (zero-phase) envelope
    collapses identically at 10% and 90%, so symmetry does not rescue it, and
    the cause is where the energy sits relative to the tapers. A circular shift
    to mid-window therefore fixes it exactly — and is legitimate because |FFT|
    is invariant under circular shift, so the estimated quantity is unchanged.
    """
    rng = np.random.default_rng(3)
    width = 400
    burst = rng.normal(0.0, 1.0, width) * np.exp(-np.arange(width) / 60.0)

    def at(start: int) -> np.ndarray:
        x = np.zeros(N)
        x[start : start + width] = burst
        return x

    centred = MultitaperEstimator(center=True)
    ratios = [
        centred.estimate(at(s), DT).energy() / time_domain_energy(at(s))
        for s in (40, 200, 600, 1000, 1400, 1560)
    ]
    assert max(ratios) - min(ratios) < 1e-6, "centring should make position irrelevant"

    # Without it, the same sweep spans a factor of three.
    plain = MultitaperEstimator()
    raw = [
        plain.estimate(at(s), DT).energy() / time_domain_energy(at(s))
        for s in (40, 200, 600, 1000, 1400, 1560)
    ]
    assert max(raw) / min(raw) > 2.5


def test_what_remains_after_centring_is_the_taper_concentration() -> None:
    """A consistent multiplicative bias, not a position-dependent one.

    The distinction matters: a consistent factor cancels in any ratio — SNR,
    spectral ratios, relative station amplitudes — and can be calibrated.
    """
    rng = np.random.default_rng(3)
    width = 400
    x = np.zeros(N)
    x[600 : 600 + width] = rng.normal(0.0, 1.0, width) * np.exp(
        -np.arange(width) / 60.0
    )
    ratio = MultitaperEstimator(center=True).estimate(
        x, DT
    ).energy() / time_domain_energy(x)
    assert 1.05 < ratio < 1.30


def test_centring_refuses_when_the_edges_are_not_quiet() -> None:
    """A circular shift wraps. Rolling a record whose coda still runs at the
    window edge would splice a discontinuity into the middle of the arrival."""
    with pytest.raises(ValueError, match="discontinuity"):
        MultitaperEstimator(center=True).estimate(noise(), DT)


def test_centring_is_recorded_in_metadata() -> None:
    rng = np.random.default_rng(3)
    x = np.zeros(N)
    x[600:1000] = rng.normal(0.0, 1.0, 400) * np.exp(-np.arange(400) / 60.0)
    assert MultitaperEstimator(center=True).estimate(x, DT).meta["centered"] is True
    assert MultitaperEstimator().estimate(x, DT).meta["centered"] is False
