"""Tests for the optional Prieto ``multitaper`` backend.

Skipped entirely when the package is absent — it is an optional extra
(``pip install specmod[multitaper]``), so CI must stay green without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from specmod.core import AmplitudeKind, Motion
from specmod.transforms import FFTEstimator, MultitaperEstimator

pytest.importorskip("multitaper", reason="optional extra: specmod[multitaper]")

from specmod.transforms import PrietoMultitaperEstimator

FS = 100.0
DT = 1.0 / FS
N = 2000


def noise(sigma: float = 1e-6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, N)


def energy(x: np.ndarray) -> float:
    return float(np.sum((x - x.mean()) ** 2) * DT)


# ------------------------------------------------------- convention conversion


@pytest.mark.parametrize("weighting", ["adaptive", "constant", "eigenvalue"])
def test_energy_matches_the_input(weighting: str) -> None:
    """Checks the fold, not the estimator.

    MTSpec returns a two-sided spectrum in FFT order, normalised to the record
    variance. Getting either the sort or the factor of two wrong would show up
    here — and integrating without sorting can even return a negative number.
    """
    x = noise()
    est = PrietoMultitaperEstimator(weighting=weighting)
    assert est.estimate(x, DT).energy() == pytest.approx(energy(x), rel=0.02)


def test_frequency_axis_is_sorted_and_positive() -> None:
    spectrum = PrietoMultitaperEstimator().estimate(noise(), DT)
    assert np.all(np.diff(spectrum.freq) > 0)
    assert spectrum.freq[0] > 0
    assert spectrum.freq[-1] <= spectrum.nyquist


def test_agrees_with_the_native_multitaper_on_the_long_period_level() -> None:
    """Two independent implementations of the same method.

    Compared against each other rather than against the FFT: a periodogram's
    median is biased low relative to its mean (chi-square with 2 d.o.f.), so an
    FFT median sits systematically below either multitaper. That is a property
    of the statistic, not a normalisation difference.
    """
    x = noise()
    band = (1.0, 5.0)
    ours = float(np.median(MultitaperEstimator().estimate(x, DT).band(*band).amp))
    theirs = float(
        np.median(PrietoMultitaperEstimator().estimate(x, DT).band(*band).amp)
    )
    assert theirs == pytest.approx(ours, rel=0.10)


def test_units_are_carried_through() -> None:
    spectrum = PrietoMultitaperEstimator().estimate(
        noise(), DT, motion=Motion.ACCELERATION
    )
    assert spectrum.kind is AmplitudeKind.FAS
    assert spectrum.motion is Motion.ACCELERATION
    assert spectrum.duration == pytest.approx(N * DT)


def test_variance_normalisation_is_recorded_as_unavoidable() -> None:
    """It is baked into MTSpec, so callers must be able to see that it applied."""
    spectrum = PrietoMultitaperEstimator().estimate(noise(), DT)
    assert spectrum.meta["normalize_to_variance"] is True


# ---------------------------------------------------------------- the extras


def test_jackknife_interval_brackets_the_estimate() -> None:
    """The headline reason to reach for this backend."""
    x = noise()
    est = PrietoMultitaperEstimator(weighting="adaptive")
    spectrum = est.estimate(x, DT)
    low, high = est.confidence_interval(x, DT)

    assert len(low) == len(high) == len(spectrum)
    inside = ((spectrum.amp >= low.amp) & (spectrum.amp <= high.amp)).mean()
    assert inside > 0.95

    # Upstream inverts the bounds for a small fraction of bins — measured at
    # 2.3% here. Asserted as a majority rather than universally, because the
    # alternative is to assert something that is not true of the package.
    assert np.mean(low.amp <= high.amp) > 0.95


def test_jackknife_reports_the_upstream_bug_rather_than_crashing() -> None:
    """Constant weighting hits a shape bug in multitaper.utils.jackspec.

    Worth a specific message: the traceback from upstream is a bare broadcast
    error that gives no hint the fix is to change the weighting.
    """
    est = PrietoMultitaperEstimator(weighting="constant")
    with pytest.raises(NotImplementedError, match="weighting='constant'"):
        est.confidence_interval(noise(), DT)


def test_f_test_finds_a_planted_tone() -> None:
    """Spotting instrumental or cultural lines masquerading as source structure."""
    f0 = 12.0
    x = noise() + 3e-6 * np.sin(2 * np.pi * f0 * np.arange(N) * DT)
    freq, f_statistic, p_value = PrietoMultitaperEstimator().f_test(x, DT)

    assert freq.shape == f_statistic.shape == p_value.shape
    assert np.all(np.diff(freq) > 0)
    assert freq[np.argmax(f_statistic)] == pytest.approx(f0, abs=0.1)


def test_f_test_finds_nothing_in_plain_noise() -> None:
    _freq, f_statistic, _p = PrietoMultitaperEstimator().f_test(noise(), DT)
    planted = PrietoMultitaperEstimator().f_test(
        noise() + 3e-6 * np.sin(2 * np.pi * 12.0 * np.arange(N) * DT), DT
    )[1]
    assert f_statistic.max() < planted.max() / 5


# --------------------------------------------------------------------- setup


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"weighting": "jackknife"}, "Unknown weighting"),
        ({"time_bandwidth": 0.0}, "time_bandwidth must be positive"),
        ({"n_tapers": 99}, "must be between 1 and"),
    ],
)
def test_rejects_bad_setup(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PrietoMultitaperEstimator(**kwargs)


def test_registry_exposes_it() -> None:
    from specmod.transforms import ESTIMATORS, get_estimator

    assert "prieto" in ESTIMATORS
    assert get_estimator("prieto").estimate(noise(), DT).kind is AmplitudeKind.FAS


def test_position_independence_from_the_baked_in_normalisation() -> None:
    """Why this backend is the proxy for pre-refactor behaviour.

    mtspec applied the same rescaling, so the position-dependent energy bias
    documented for the native estimator is very likely absent from results
    produced before this refactor. See docs/choosing_a_transform.md.
    """
    rng = np.random.default_rng(1)
    burst = rng.normal(0.0, 1e-6, 200)

    ratios = []
    for start in (60, N // 2 - 100, N - 260):
        x = np.zeros(N)
        x[start : start + 200] = burst
        ratios.append(PrietoMultitaperEstimator().estimate(x, DT).energy() / energy(x))

    assert max(ratios) / min(ratios) < 1.05, (
        "variance normalisation should make total energy position-independent"
    )
    # The native estimator, without that rescaling, does not manage this.
    native = []
    for start in (60, N // 2 - 100, N - 260):
        x = np.zeros(N)
        x[start : start + 200] = burst
        native.append(MultitaperEstimator().estimate(x, DT).energy() / energy(x))
    assert max(native) / min(native) > 1.2


def test_fft_is_still_the_most_position_stable() -> None:
    """Normalisation fixes energy, not shape — the distinction the docs turn on."""
    rng = np.random.default_rng(1)
    burst = rng.normal(0.0, 1e-6, 200)
    band = (2.0, 8.0)

    def plateau(est, start: int) -> float:
        x = np.zeros(N)
        x[start : start + 200] = burst
        return float(np.median(est.estimate(x, DT).band(*band).amp))

    starts = (60, N // 2 - 100, N - 260)
    for est in (FFTEstimator(), PrietoMultitaperEstimator()):
        levels = [plateau(est, s) for s in starts]
        spread = max(levels) / min(levels)
        if est.name == "fft":
            fft_spread = spread
        else:
            prieto_spread = spread
    assert fft_spread < prieto_spread
