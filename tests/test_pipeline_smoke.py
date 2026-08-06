"""End-to-end run of the real pipeline on the committed PNR waveforms.

Every bug in the estimator-wiring work was found by running the pipeline, not
by the unit tests — most sharply a ``ValueError: output array is read-only``
that only appeared when a *real* noise stream was present. The unit suite could
not see it: it exercises estimators, and the break was in how their output
crossed into the containers.

So this covers the seam rather than the science. It runs
``preprocess -> spectrum_set_from_streams -> SNR -> bandwidth`` on 28 real
windows and asserts the things that are cheap to check and expensive to get
wrong. Numerical claims belong in the focused suites; what this asserts is that
the pipeline *runs* and that its outputs are structurally sane.

**The read-only tests here now assert the opposite of what they used to.** They
existed because the legacy classes mutated ``amp`` in place and had to be
handed copies, so the pipeline broke if a frozen array reached them. Nothing
mutates any more, so the guarantee to protect is the one that used to be the
hazard: no caller can change a spectrum under a reference someone else holds.

The data is committed (224 KB under ``Tutorial/Data``), so this runs in CI
without a network fetch and is not marked ``dataset``.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

from specmod.pipeline import spectrum_set_from_streams  # noqa: E402

ESTIMATORS = ["fft", "welch", "multitaper", "quadratic", "cwt"]


@functools.cache
def _run(estimator: str, windows: Any) -> Any:
    signal, noise = windows()
    return spectrum_set_from_streams(signal, noise, estimator=estimator)


@pytest.fixture(scope="module")
def spectra(pnr_windows: Any) -> Any:
    """One full pipeline run on the configured default, shared below."""
    return _run("multitaper", pnr_windows)


# --------------------------------------------------------------- it runs


def test_the_pipeline_produces_a_spectrum_for_every_window(spectra: Any) -> None:
    assert len(spectra) == 28


def test_every_spectrum_is_structurally_sane(spectra: Any) -> None:
    """Cheap invariants that a broken conversion would violate loudly."""
    for name in spectra.ids():
        pair = spectra[name]
        for label, spectrum in (("signal", pair.signal), ("noise", pair.noise)):
            where = f"{name} {label}"
            assert spectrum.freq.size > 0, where
            assert np.isfinite(spectrum.amp).all(), where
            assert (spectrum.amp > 0).all(), f"{where}: non-positive amplitude"
            assert np.all(np.diff(spectrum.freq) > 0), f"{where}: unsorted axis"
            assert spectrum.freq.max() <= spectrum.nyquist * (1 + 1e-9), where


def test_spectra_report_their_provenance(spectra: Any) -> None:
    """Which estimator produced a result is part of the result."""
    for name in spectra.ids():
        signal = spectra[name].signal
        assert signal.meta["estimator"] == "multitaper"
        assert str(signal.motion) == "velocity"
        assert signal.meta["id"] == name


# ----------------------------------------------- the seam, the other way up


def test_amplitude_arrays_are_read_only(spectra: Any) -> None:
    """The inverse of what this file used to assert, and deliberately so.

    It read ``assert spectrum.amp.flags.writeable``, because the legacy classes
    mutated in place — the noise rescale, the lift, integrate, differentiate —
    and a frozen array reaching them raised. That is the hazard the containers
    were rewritten to remove, so the property worth pinning is now the one that
    used to break the pipeline.
    """
    for name in spectra.ids():
        pair = spectra[name]
        for spectrum in (pair.signal, pair.noise):
            assert not spectrum.amp.flags.writeable, name
            assert not spectrum.freq.flags.writeable, name


def test_a_domain_change_leaves_the_original_alone(spectra: Any) -> None:
    """What immutability buys, exercised rather than merely flagged.

    ``Spectra.inte()`` overwrote the event in place, so "the same event as
    displacement" could only be had by destroying the velocity one. Both now
    exist at once, and integrate/differentiate remain inverses.
    """
    name = spectra.ids()[0]
    before = spectra[name].signal.amp.copy()

    displacement = spectra.to_motion("displacement")
    assert spectra[name].signal.amp == pytest.approx(before)
    assert not np.allclose(displacement[name].signal.amp, before)

    back = displacement.to_motion("velocity")
    assert back[name].signal.amp == pytest.approx(before, rel=1e-9)


# ------------------------------------------------------- SNR and bandwidth


def test_signal_to_noise_is_computed_for_every_pair(spectra: Any) -> None:
    for name in spectra.ids():
        assert spectra[name].snr.size > 0, name
        assert np.isfinite(spectra[name].snr).all(), name


def test_selected_bandwidth_lies_inside_what_the_record_resolves(
    spectra: Any,
) -> None:
    """A window cannot resolve a period longer than itself.

    Nothing enforced this before; it now holds because the bins are clamped to
    the record's own frequency range. That is an implicit guarantee resting on
    an unrelated change, so it is worth an explicit check — if the binning
    changes, this is what notices.
    """
    checked = 0
    for name in spectra.ids():
        pair = spectra[name]
        if pair.band is None:
            continue
        low, high = pair.band
        assert low >= 1.0 / pair.signal.duration, (
            f"{name}: band opens at {low:.3f} Hz but 1/T is "
            f"{1.0 / pair.signal.duration:.3f} Hz"
        )
        assert high <= pair.signal.nyquist, name
        assert low < high, name
        checked += 1
    assert checked > 20, f"only {checked} windows yielded a bandwidth"


# ------------------------------------------------------- every estimator


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_whole_pipeline_runs_on_each_estimator(
    estimator: str, pnr_windows: Any
) -> None:
    """The point of the rewiring, checked against real data rather than noise.

    A backend returning a differently-shaped or differently-scaled spectrum
    breaks here.

    ``cwt`` is included, having been excluded when this module was written on
    the belief that an element-wise signal-to-noise needed pinned bin edges to
    compare a cone-of-influence-masked axis against a full one. It does not,
    for two reasons that only became true together: the noise is re-binned
    *after* being moved onto the signal's axis, so the two binned arrays align
    by construction whatever the estimator; and the resolution floor clamps the
    band, which is what keeps the shortened axis from being compared against
    the flat edge value ``np.interp`` leaves below the noise window's minimum.
    """
    result = _run(estimator, pnr_windows)

    assert len(result) == 28
    for name in result.ids():
        pair = result[name]
        assert pair.signal.meta["estimator"] == estimator, name
        assert np.isfinite(pair.signal.amp).all(), name
        assert (pair.signal.amp > 0).all(), name
        # The alignment that made the exclusion unnecessary. It is an implicit
        # consequence of re-binning after interpolation, so it is asserted
        # rather than assumed: without it the ratio is silently comparing
        # different frequencies.
        assert pair.binned_signal.amp.shape == pair.binned_noise.amp.shape, name
        assert pair.binned_signal.freq == pytest.approx(pair.binned_noise.freq), name
        assert np.isfinite(pair.snr).all(), name


# ------------------------------------------------ the noise resolution floor


def test_the_band_respects_the_noise_window_too(spectra: Any) -> None:
    """The noise window is the shorter one, and it sets the limit.

    The noise is put on the signal's frequency axis, and ``np.interp`` does not
    extrapolate — it repeats the edge value — so below the noise window's own
    lowest frequency the "noise level" is a flat continuation rather than a
    measurement, and the signal-to-noise computed there has an invented
    denominator.

    On these 28 pairs the noise windows run 1.1-1.7 s against 1.8-3.7 s
    signals, and 6 selected a band opening below the noise window's ``1/T``
    before this floor existed.
    """
    checked = 0
    for name in spectra.ids():
        pair = spectra[name]
        if pair.band is None:
            continue
        assert pair.band[0] >= pair.resolution_floor - 1e-12, (
            f"{name}: band opens at {pair.band[0]:.3f} Hz, below the "
            f"resolution floor {pair.resolution_floor:.3f} Hz"
        )
        checked += 1
    assert checked > 20


def test_the_floor_is_the_stricter_of_the_two_windows(spectra: Any) -> None:
    for name in spectra.ids():
        pair = spectra[name]
        assert pair.resolution_floor == pytest.approx(
            max(float(pair.signal.freq.min()), pair.noise.meta["resolution_floor"])
        ), name


def test_the_cwt_floor_is_stricter_than_the_multitaper_one(
    pnr_windows: Any,
) -> None:
    """A wavelet needs several cycles inside the window, not one.

    The cone of influence is about 1.4x stricter than ``1/T`` here, and taking
    each spectrum's floor from its own frequency axis means that rule applies
    without the pipeline knowing which estimator produced it.
    """
    floors = {}
    for estimator in ("multitaper", "cwt"):
        pairs = _run(estimator, pnr_windows).pairs.values()
        floors[estimator] = float(np.median([p.resolution_floor for p in pairs]))
    assert floors["cwt"] > floors["multitaper"] * 1.2, floors
