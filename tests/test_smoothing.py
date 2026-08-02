"""Tests for the smoothing layer (docs/REFACTOR_PLAN.md §4.3)."""

from __future__ import annotations

import numpy as np
import pytest

from specmod.core import AmplitudeKind, Motion
from specmod.smoothing import (
    SMOOTHERS,
    KonnoOhmachi,
    LogBinner,
    get_smoother,
    is_smoothed,
)
from specmod.transforms import FFTEstimator

FS = 100.0
DT = 1.0 / FS


def spectrum(n: int = 2000, seed: int = 0, sigma: float = 1e-6):
    x = np.random.default_rng(seed).normal(0.0, sigma, n)
    return FFTEstimator().estimate(x, DT)


# ------------------------------------------------------------------- log bins


def test_default_edges_come_from_the_record() -> None:
    """The old binner hardcoded 0.001-200 Hz whatever the record was.

    For a 100 Hz, 20 s trace that put a third of the bins above Nyquist and a
    third below 1/T, where there is no data at all.
    """
    s = spectrum()
    edges = LogBinner().edges_for(s)
    assert edges[0] == pytest.approx(s.frequency_resolution, rel=1e-9)
    assert edges[-1] == pytest.approx(s.nyquist, rel=1e-9)


def test_binning_reduces_the_axis_and_stays_in_range() -> None:
    s = spectrum()
    binned = LogBinner(n_bins=60).smooth(s)
    assert len(binned) <= 60
    assert binned.freq[0] >= s.freq[0]
    assert binned.freq[-1] <= s.freq[-1]
    assert np.all(np.diff(binned.freq) > 0)


def test_bin_centres_are_geometric_midpoints() -> None:
    """Log bins want log centres; an arithmetic midpoint biases high."""
    s = spectrum()
    binner = LogBinner(f_min=1.0, f_max=10.0, n_bins=4, drop_empty=False)
    edges = binner.edges_for(s)
    expected = np.sqrt(edges[:-1] * edges[1:])
    assert binner.smooth(s).freq == pytest.approx(expected)


def test_geometric_statistic_matches_the_pre_refactor_definition() -> None:
    """The old code computed 10**mean(log10(amp)) — a geometric mean."""
    s = spectrum()
    binner = LogBinner(f_min=5.0, f_max=6.0, n_bins=1, drop_empty=False)
    inside = s.band(5.0, 6.0)
    expected = 10 ** np.mean(np.log10(inside.amp))
    assert binner.smooth(s).amp[0] == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("statistic", ["geometric", "mean", "median"])
def test_every_statistic_lands_inside_the_data_range(statistic: str) -> None:
    s = spectrum()
    binned = LogBinner(
        f_min=5.0, f_max=6.0, n_bins=1, statistic=statistic, drop_empty=False
    ).smooth(s)
    inside = s.band(5.0, 6.0)
    assert inside.amp.min() <= binned.amp[0] <= inside.amp.max()


def test_explicit_edges_are_not_clamped_to_the_spectrum() -> None:
    """Signal and noise windows differ in duration.

    Clamping explicit bounds to each spectrum's own range would bin them onto
    different axes, and the SNR ratio compares them element-wise.
    """
    long_record, short_record = spectrum(2000), spectrum(700, seed=1)
    assert long_record.duration != short_record.duration

    binner = LogBinner(f_min=0.5, f_max=40.0, n_bins=60, drop_empty=False)
    a, b = binner.smooth(long_record), binner.smooth(short_record)

    assert len(a) == len(b) == 60
    assert a.freq == pytest.approx(b.freq)


def test_derived_edges_do_differ_between_records() -> None:
    """The counterpart: deriving means "whatever this record supports"."""
    binner = LogBinner()
    a = binner.smooth(spectrum(2000))
    b = binner.smooth(spectrum(700, seed=1))
    assert len(a) != len(b)


def test_drop_empty_false_keeps_a_fixed_length_axis() -> None:
    s = spectrum()
    binned = LogBinner(n_bins=200, drop_empty=False).smooth(s)
    assert len(binned) == 200
    assert np.isnan(binned.amp).any(), "expected sparse low-frequency bins"


def test_drop_empty_true_removes_the_gaps() -> None:
    s = spectrum()
    binned = LogBinner(n_bins=200, drop_empty=True).smooth(s)
    assert len(binned) < 200
    assert not np.isnan(binned.amp).any()


def test_impossible_binning_explains_itself() -> None:
    s = spectrum()
    with pytest.raises(ValueError, match="No bin reached min_count"):
        LogBinner(n_bins=400, min_count=50).smooth(s)
    with pytest.raises(ValueError, match="does not overlap the spectrum"):
        LogBinner(f_min=200.0, f_max=300.0).smooth(s)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_bins": 0}, "n_bins must be at least 1"),
        ({"min_count": 0}, "min_count must be at least 1"),
        ({"f_min": -1.0}, "f_min must be positive"),
        ({"f_min": 10.0, "f_max": 1.0}, "must be below f_max"),
    ],
)
def test_binner_rejects_bad_setup(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LogBinner(**kwargs)


# -------------------------------------------------------------- konno-ohmachi


def test_konno_ohmachi_preserves_the_frequency_axis() -> None:
    """Unlike binning — often what you want before fitting on the raw grid."""
    s = spectrum()
    smoothed = KonnoOhmachi().smooth(s)
    assert smoothed.freq == pytest.approx(s.freq)
    assert len(smoothed) == len(s)


def test_konno_ohmachi_reduces_scatter() -> None:
    s = spectrum()
    smoothed = KonnoOhmachi(bandwidth=20.0).smooth(s)
    rough = np.std(np.diff(np.log10(s.amp)))
    smooth = np.std(np.diff(np.log10(smoothed.amp)))
    assert smooth < rough


def test_smaller_bandwidth_smooths_harder() -> None:
    """Konno-Ohmachi's b is inverse: smaller means a wider window."""
    s = spectrum()
    hard = np.std(np.diff(np.log10(KonnoOhmachi(bandwidth=10.0).smooth(s).amp)))
    soft = np.std(np.diff(np.log10(KonnoOhmachi(bandwidth=80.0).smooth(s).amp)))
    assert hard < soft


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bandwidth": 0.0}, "bandwidth must be positive"),
        ({"count": 0}, "count must be"),
    ],
)
def test_konno_ohmachi_rejects_bad_setup(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        KonnoOhmachi(**kwargs)


# ------------------------------------------------------ metadata and contract


@pytest.mark.parametrize("smoother", [LogBinner(), KonnoOhmachi()])
def test_units_survive_smoothing(smoother) -> None:
    s = spectrum()
    out = smoother.smooth(s)
    assert out.kind is AmplitudeKind.FAS
    assert out.motion is Motion.VELOCITY
    assert out.duration == s.duration
    assert out.sampling_rate == s.sampling_rate


@pytest.mark.parametrize("smoother", [LogBinner(), KonnoOhmachi()])
def test_smoothing_is_recorded(smoother) -> None:
    """Smoothing breaks Parseval, so downstream needs to be able to tell."""
    s = spectrum()
    assert not is_smoothed(s)
    out = smoother.smooth(s)
    assert is_smoothed(out)
    assert out.meta["smoothing"][0]["method"] == smoother.name


def test_chained_smoothing_leaves_a_trail() -> None:
    s = KonnoOhmachi().smooth(spectrum())
    both = LogBinner().smooth(s)
    assert [m["method"] for m in both.meta["smoothing"]] == [
        "konno_ohmachi",
        "log_bins",
    ]


@pytest.mark.parametrize("smoother", [LogBinner(), KonnoOhmachi()])
def test_smoothing_does_not_mutate_its_input(smoother) -> None:
    s = spectrum()
    before = s.amp.copy()
    smoother.smooth(s)
    assert s.amp == pytest.approx(before)


def test_registry_resolves_every_smoother() -> None:
    s = spectrum()
    for name in SMOOTHERS:
        assert len(get_smoother(name).smooth(s)) > 0


def test_unknown_smoother_names_the_alternatives() -> None:
    with pytest.raises(ValueError, match="Unknown smoother"):
        get_smoother("savgol")
