"""End-to-end run of the real pipeline on the committed PNR waveforms.

Every bug in the estimator-wiring work was found by running the pipeline, not
by the unit tests — most sharply a ``ValueError: output array is read-only``
that only appears when a *real* noise stream is present, because
``SNP.__scale_noise_parseval`` mutates in place. The unit suite could not see
it: it exercises estimators, and the break was in how their output crosses into
the legacy classes.

So this covers the seam rather than the science. It runs
``preprocess -> Spectra.from_streams -> SNR -> bandwidth`` on 28 real windows
and asserts the things that are cheap to check and expensive to get wrong.
Numerical claims belong in the focused suites; what this asserts is that the
pipeline *runs*, that its outputs are structurally sane, and that the legacy
in-place operations still work on what the new estimators hand them.

The data is committed (224 KB under ``Tutorial/Data``), so this runs in CI
without a network fetch and is not marked ``dataset``.
"""

from __future__ import annotations

import contextlib
import functools
import glob
import io
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

import specmod.preprocess as pre  # noqa: E402
from specmod.spectral import Spectra  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Tutorial" / "Data" / "2019-08-26T07:30:47.0"
INVENTORY = ROOT / "Tutorial" / "MetaData" / "pnr_inventory.xml"

#: The Preston New Road event the tutorial is built around.
ORIGIN = "2019-08-26T07:49:24.2"
LATITUDE, LONGITUDE, DEPTH_KM = 53.784, -2.967, 2.1

pytestmark = pytest.mark.skipif(
    not DATA.is_dir() or not INVENTORY.is_file(),
    reason="tutorial waveforms not present",
)


def _cut_windows() -> tuple[Any, Any]:
    """Signal and noise streams, cut with the published Magna workflow.

    Copies of the cached originals, so a test that mutates a stream cannot
    affect the next. ``Spectra.from_streams`` does mutate.
    """
    signal, noise = _build_windows()
    return signal.copy(), noise.copy()


@functools.cache
def _build_windows() -> tuple[Any, Any]:
    """Response removal and window refinement dominate this module's runtime,
    and every test wants the same windows — so do it once."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inventory = obspy.read_inventory(str(INVENTORY))
        stream = obspy.read(os.path.join(str(DATA), "*HH[EN]*"))
        pre.set_stream_distance(
            stream,
            LATITUDE,
            LONGITUDE,
            DEPTH_KM,
            obspy.UTCDateTime(ORIGIN),
            inventory=inventory,
            dtype="mseed",
        )
        pre.set_picks_from_pyrocko(
            stream, glob.glob(os.path.join(str(DATA), "*.picks"))[0]
        )
        stream = obspy.Stream([tr for tr in stream if "s_time" in tr.stats])
        stream.detrend("linear")
        stream.detrend("demean")
        stream.taper(0.05)
        stream.remove_response(inventory, output="VEL")

        signal = pre.get_signal(
            stream,
            pre.cut_s,
            rafp=0.8,
            tafs=20,
            time_after="absolute_time",
            refine_window=True,
        )
        return signal, pre.get_noise_p(stream, signal)


@pytest.fixture(scope="module")
def spectra() -> Any:
    """One full pipeline run, shared across the assertions below."""
    signal, noise = _cut_windows()
    with contextlib.redirect_stdout(io.StringIO()):
        return Spectra.from_streams(signal, noise)


# --------------------------------------------------------------- it runs


def test_the_pipeline_produces_a_spectrum_for_every_window(spectra: Any) -> None:
    assert len(spectra.group) == 28


def test_every_spectrum_is_structurally_sane(spectra: Any) -> None:
    """Cheap invariants that a broken conversion would violate loudly."""
    for name, snp in spectra.group.items():
        for label, spectrum in (("signal", snp.signal), ("noise", snp.noise)):
            where = f"{name} {label}"
            assert spectrum.freq.size > 0, where
            assert np.isfinite(spectrum.amp).all(), where
            assert (spectrum.amp > 0).all(), f"{where}: non-positive amplitude"
            assert np.all(np.diff(spectrum.freq) > 0), f"{where}: unsorted axis"
            nyquist = spectrum.meta["sampling_rate"] / 2.0
            assert spectrum.freq.max() <= nyquist * (1 + 1e-9), where


def test_spectra_report_their_provenance(spectra: Any) -> None:
    """Which estimator produced a result is part of the result."""
    for snp in spectra.group.values():
        assert snp.signal.estimator == "multitaper"
        assert snp.signal.motion == "velocity"


# ------------------------------------------- the seam the unit tests missed


def test_amplitude_arrays_are_writable(spectra: Any) -> None:
    """Regression test for the read-only break.

    ``core.Spectrum`` marks its arrays read-only so a spectrum cannot be
    mutated behind its own back. The legacy classes here mutate ``amp`` in
    place — the noise rescale, the rotation, integrate, differentiate — so they
    must be handed copies. Passing the frozen array through makes every one of
    those raise, and only on an event that has a real noise stream.
    """
    for snp in spectra.group.values():
        for spectrum in (snp.signal, snp.noise):
            assert spectrum.amp.flags.writeable
            assert spectrum.bamp.flags.writeable
            # `freq` too: it was frozen for a while as a side effect of the
            # amplitude conversion handing its array straight to
            # `core.Spectrum`, which was harmless only because nothing happened
            # to write to it.
            assert spectrum.freq.flags.writeable
            assert spectrum.bfreq.flags.writeable


def test_the_legacy_in_place_operations_still_work(spectra: Any) -> None:
    """Exercise them rather than merely checking the flag.

    ``integrate`` and ``differentiate`` are inverses, so a round trip is a
    cheap end-to-end check that the arrays are both writable and correctly
    shaped against their frequency axes.
    """
    snp = next(iter(spectra.group.values()))
    before = snp.signal.amp.copy()
    snp.signal.integrate()
    assert not np.allclose(snp.signal.amp, before), "integrate did nothing"
    snp.signal.differentiate()
    assert snp.signal.amp == pytest.approx(before, rel=1e-9)


def test_the_amplitude_conversion_round_trips_on_real_data(spectra: Any) -> None:
    snp = next(iter(spectra.group.values()))
    before = snp.signal.amp.copy()
    snp.signal.amp_to_psd()
    snp.signal.psd_to_amp()
    assert snp.signal.amp == pytest.approx(before, rel=1e-9)


# ------------------------------------------------------- SNR and bandwidth


def test_signal_to_noise_is_computed_for_every_pair(spectra: Any) -> None:
    for name, snp in spectra.group.items():
        assert snp.bsnr.size > 0, name
        assert np.isfinite(snp.bsnr).all(), name


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
    for name, snp in spectra.group.items():
        band = getattr(snp, "ubfreqs", None)
        if band is None or len(band) != 2:
            continue
        duration = snp.signal.meta["npts"] * snp.signal.meta["delta"]
        low, high = float(band[0]), float(band[1])
        assert low >= 1.0 / duration, (
            f"{name}: band opens at {low:.3f} Hz but 1/T is {1.0 / duration:.3f} Hz"
        )
        assert high <= snp.signal.meta["sampling_rate"] / 2.0, name
        assert low < high, name
        checked += 1
    assert checked > 20, f"only {checked} windows yielded a bandwidth"


# ------------------------------------------------------- every estimator


@pytest.mark.parametrize(
    "estimator", ["fft", "welch", "multitaper", "quadratic", "cwt"]
)
def test_the_whole_pipeline_runs_on_each_estimator(estimator: str) -> None:
    """The point of the rewiring, checked against real data rather than noise.

    Each backend goes through ``Spectra.from_streams``, which is where the
    conversions, the noise rescale and the SNR search all live. A backend that
    returns a differently-shaped or differently-scaled spectrum breaks here.

    ``cwt`` is included, having been excluded when this module was written on
    the belief that ``SNP``'s element-wise signal-to-noise needed pinned bin
    edges to compare a cone-of-influence-masked axis against a full one. It
    does not, for two reasons that only became true together:
    ``__interp_noise_to_signal`` re-bins the noise *after* moving it onto the
    signal's axis, so the two ``bamp`` arrays align by construction whatever
    the estimator; and the resolution floor now clamps the band, which is what
    keeps the shortened axis from being compared against the flat edge value
    ``np.interp`` leaves below the noise window's own minimum.
    """
    signal, noise = _cut_windows()
    with contextlib.redirect_stdout(io.StringIO()):
        result = Spectra.from_streams(signal, noise, estimator=estimator)

    assert len(result.group) == 28
    for name, snp in result.group.items():
        assert snp.signal.estimator == estimator, name
        assert np.isfinite(snp.signal.amp).all(), name
        assert (snp.signal.amp > 0).all(), name
        # The alignment that made the exclusion unnecessary. It is an implicit
        # consequence of re-binning after interpolation, so it is asserted
        # rather than assumed: without it the division below is silently
        # comparing different frequencies.
        assert snp.signal.bamp.shape == snp.noise.bamp.shape, name
        assert snp.signal.bfreq == pytest.approx(snp.noise.bfreq), name
        assert np.isfinite(snp.bsnr).all(), name


# ------------------------------------------------ the noise resolution floor


def test_the_band_respects_the_noise_window_too(spectra: Any) -> None:
    """The noise window is the shorter one, and it sets the limit.

    ``__interp_noise_to_signal`` puts the noise on the signal's frequency axis.
    ``np.interp`` does not extrapolate — it repeats the edge value — so below
    the noise window's own lowest frequency the "noise level" is a flat
    continuation rather than a measurement, and the signal-to-noise computed
    there has an invented denominator.

    On these 28 pairs the noise windows run 1.1-1.7 s against 1.8-3.7 s
    signals, and 6 selected a band opening below the noise window's ``1/T``
    before this floor existed.
    """
    checked = 0
    for name, snp in spectra.group.items():
        band = getattr(snp, "ubfreqs", None)
        if band is None or len(band) != 2:
            continue
        assert float(band[0]) >= snp.resolution_floor - 1e-12, (
            f"{name}: band opens at {float(band[0]):.3f} Hz, below the "
            f"resolution floor {snp.resolution_floor:.3f} Hz"
        )
        checked += 1
    assert checked > 20


def test_the_floor_is_the_stricter_of_the_two_windows(spectra: Any) -> None:
    for name, snp in spectra.group.items():
        assert snp.resolution_floor == pytest.approx(
            max(snp.signal.resolution_floor, snp.noise.resolution_floor)
        ), name


def test_the_cwt_floor_is_stricter_than_the_multitaper_one() -> None:
    """A wavelet needs several cycles inside the window, not one.

    The cone of influence is about 1.4x stricter than ``1/T`` here, and taking each
    spectrum's floor from its own frequency axis means that rule applies
    without the pipeline knowing which estimator produced it.
    """
    floors = {}
    for estimator in ("multitaper", "cwt"):
        signal, noise = _cut_windows()
        with contextlib.redirect_stdout(io.StringIO()):
            result = Spectra.from_streams(signal, noise, estimator=estimator)
        floors[estimator] = np.median(
            [snp.resolution_floor for snp in result.group.values()]
        )
    assert floors["cwt"] > floors["multitaper"] * 1.2, floors
