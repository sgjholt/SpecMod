"""The direct waveform-to-``SpectrumSet`` path, against the legacy one.

:mod:`specmod.pipeline` goes trace to :class:`~specmod.core.Spectrum` to
:class:`~specmod.core.SpectrumPair` without touching ``spectral``'s mutable
classes. The legacy route reaches the same place by a longer one: the
estimator's spectrum is converted to a PSD, copied out into a mutable object,
converted back to magnitude, and wrapped in a ``core.Spectrum`` again by
``SNP``.

Whether those two agree is the question this file exists to answer, and it
answers it by measurement over all 28 PNR windows and all five estimators
rather than by argument about whether the conversions cancel. They are not
expected to be bit-identical — the legacy takes ``FAS -> PSD -> MAGNITUDE``
where this takes ``FAS -> MAGNITUDE``, which is the same map composed
differently — so the bound is 1 part in 1e12, twelve orders tighter than
anything that would matter seismologically and tight enough that a genuine
change in the arithmetic cannot hide under it.
"""

from __future__ import annotations

import contextlib
import functools
import io
from typing import Any

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

from specmod.core import Spectrum, SpectrumPair, SpectrumSet  # noqa: E402
from specmod.pipeline import (  # noqa: E402
    pair_from_traces,
    spectrum_from_trace,
    spectrum_set_from_streams,
)
from specmod.spectral import Spectra  # noqa: E402

#: The legacy path composes the same conversion in two steps rather than one,
#: so agreement is to floating-point round-off, not to the bit.
RTOL = 1e-12

ESTIMATORS = ["fft", "welch", "multitaper", "quadratic", "cwt"]


@functools.cache
def _legacy_container(estimator: str, windows: Any) -> Spectra:
    signal, noise = windows()
    with contextlib.redirect_stdout(io.StringIO()):
        return Spectra.from_streams(signal, noise, estimator=estimator)


def _legacy(estimator: str, windows: Any) -> SpectrumSet:
    return _legacy_container(estimator, windows).as_spectrum_set()


@functools.cache
def _direct(estimator: str, windows: Any) -> SpectrumSet:
    signal, noise = windows()
    return spectrum_set_from_streams(signal, noise, estimator=estimator)


# ------------------------------------------------------------ the unit itself


class TestSpectrumFromTrace:
    def test_it_returns_the_unfolded_convention(self, pnr_windows: Any) -> None:
        """``Omega`` is defined on ``|X|``, not on the folded ``2|X|``.

        Reading it off a folded spectrum puts every moment out by a factor of
        two — 0.2 magnitude units — so this is pinned rather than left to the
        default.
        """
        signal, _ = pnr_windows()
        spectrum = spectrum_from_trace(signal[0])
        assert str(spectrum.kind) == "magnitude"

    def test_it_carries_the_trace_identity(self, pnr_windows: Any) -> None:
        signal, _ = pnr_windows()
        spectrum = spectrum_from_trace(signal[0])
        assert spectrum.meta["id"] == signal[0].id
        assert spectrum.meta["station"] == signal[0].stats.station

    def test_the_window_edges_survive_as_strings(self, pnr_windows: Any) -> None:
        """A spectrum that cannot say what window it came from is not
        reproducible. ``UTCDateTime`` does not survive a plain mapping, so the
        edges are stringified rather than dropped."""
        signal, _ = pnr_windows()
        spectrum = spectrum_from_trace(signal[0])
        assert spectrum.meta["wstart"] == str(signal[0].stats["wstart"])
        assert spectrum.meta["wend"] == str(signal[0].stats["wend"])

    def test_unstable_stats_are_dropped(self, pnr_windows: Any) -> None:
        """``processing`` grows with every ObsPy call, so two runs of the same
        pipeline would compare unequal on it alone."""
        signal, _ = pnr_windows()
        spectrum = spectrum_from_trace(signal[0])
        assert "processing" in signal[0].stats
        assert "processing" not in spectrum.meta

    def test_the_resolution_floor_is_the_axis_minimum(self, pnr_windows: Any) -> None:
        signal, _ = pnr_windows()
        spectrum = spectrum_from_trace(signal[0])
        assert spectrum.meta["resolution_floor"] == float(spectrum.freq.min())

    def test_the_cwt_floor_is_stricter_than_the_fft_one(self, pnr_windows: Any) -> None:
        """The cone of influence, not ``1/T``: a wavelet needs several cycles in
        the window where a Fourier bin needs one."""
        signal, _ = pnr_windows()
        fft = spectrum_from_trace(signal[0], estimator="fft")
        cwt = spectrum_from_trace(signal[0], estimator="cwt")
        assert cwt.meta["resolution_floor"] > fft.meta["resolution_floor"]

    def test_the_arrays_are_read_only(self, pnr_windows: Any) -> None:
        """The property the legacy classes could not have, and the reason the
        round trip through them needed copies at every hand-off."""
        signal, _ = pnr_windows()
        spectrum = spectrum_from_trace(signal[0])
        with pytest.raises(ValueError, match="read-only"):
            spectrum.amp[0] = 0.0


# --------------------------------------------------- against the legacy path


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_same_windows_come_out(estimator: str, pnr_windows: Any) -> None:
    assert (
        _direct(estimator, pnr_windows).ids() == _legacy(estimator, pnr_windows).ids()
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize(
    "attribute", ["signal", "noise", "binned_signal", "binned_noise"]
)
def test_the_spectra_are_unchanged(
    estimator: str, attribute: str, pnr_windows: Any
) -> None:
    direct, legacy = _direct(estimator, pnr_windows), _legacy(estimator, pnr_windows)
    problems = []
    for id in legacy.ids():
        want = getattr(legacy[id], attribute)
        got = getattr(direct[id], attribute)
        for field in ("freq", "amp"):
            a, b = getattr(got, field), getattr(want, field)
            if a.shape != b.shape:
                problems.append(f"{id} {attribute}.{field}: {b.shape} -> {a.shape}")
            elif a != pytest.approx(b, rel=RTOL):
                rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
                problems.append(f"{id} {attribute}.{field}: max rel {rel:.2e}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_signal_to_noise_is_unchanged(estimator: str, pnr_windows: Any) -> None:
    direct, legacy = _direct(estimator, pnr_windows), _legacy(estimator, pnr_windows)
    for id in legacy.ids():
        assert direct[id].snr == pytest.approx(legacy[id].snr, rel=RTOL), id


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_selected_bands_are_unchanged(estimator: str, pnr_windows: Any) -> None:
    """The one that matters most: the band is what every fitted parameter is
    read over, so a band that moved is a result that moved."""
    direct, legacy = _direct(estimator, pnr_windows), _legacy(estimator, pnr_windows)
    problems = []
    for id in legacy.ids():
        want, got = legacy[id].band, direct[id].band
        if (want is None) != (got is None) or (
            want is not None and got != pytest.approx(want, rel=RTOL)
        ):
            problems.append(f"{id}: {want} -> {got}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_resolution_floors_are_unchanged(estimator: str, pnr_windows: Any) -> None:
    direct, legacy = _direct(estimator, pnr_windows), _legacy(estimator, pnr_windows)
    for id in legacy.ids():
        assert direct[id].resolution_floor == pytest.approx(
            legacy[id].resolution_floor, rel=RTOL
        ), id


# ------------------------------------------------------------------ contract


def test_it_refuses_to_pair_different_stations(pnr_windows: Any) -> None:
    """A signal compared against another station's noise is not a degraded
    measurement, it is a meaningless one — so this raises rather than warns."""
    signal, noise = pnr_windows()
    with pytest.raises(ValueError, match="not in the same order"):
        spectrum_set_from_streams(signal[:2], obspy.Stream(noise[1:3]))


def test_it_refuses_streams_of_different_lengths(pnr_windows: Any) -> None:
    signal, noise = pnr_windows()
    with pytest.raises(ValueError, match="argument 2 is shorter"):
        spectrum_set_from_streams(signal, obspy.Stream(noise[:-1]))


def test_the_event_defaults_to_the_origin_time(pnr_windows: Any) -> None:
    signal, noise = pnr_windows()
    got = spectrum_set_from_streams(signal[:2], obspy.Stream(noise[:2]))
    assert got.event == str(signal[0].stats["otime"])


def test_a_single_pair_matches_the_stream_path(pnr_windows: Any) -> None:
    """The two entry points share their settings, so they cannot disagree about
    what the configuration said — asserted, because they are separate code."""
    signal, noise = pnr_windows()
    one = pair_from_traces(signal[0], noise[0])
    many = spectrum_set_from_streams(signal[:1], obspy.Stream(noise[:1]))
    assert isinstance(one, SpectrumPair)
    assert one.band == many[signal[0].id].band
    assert one.snr == pytest.approx(many[signal[0].id].snr)


def test_the_result_is_the_immutable_container(pnr_windows: Any) -> None:
    signal, noise = pnr_windows()
    got = spectrum_set_from_streams(signal[:2], obspy.Stream(noise[:2]))
    assert isinstance(got, SpectrumSet)
    assert all(isinstance(p.signal, Spectrum) for p in got.pairs.values())


# ------------------------------------------------------- through the fitter


@functools.cache
def _fits(estimator: str, windows: Any) -> tuple[Any, Any]:
    """The same event fitted through both containers.

    The fit is where a difference in the spectra would be amplified rather
    than merely carried: a nonlinear minimiser near a shallow minimum can walk
    a visibly different path from an input difference far below the noise
    floor. Running it is the only way to know how far that goes.
    """
    import numpy as np  # noqa: PLC0415

    from specmod.fitting import FitSpectra, fittable_signal  # noqa: PLC0415

    def guesses(container: Any) -> dict[str, dict[str, float]]:
        out = {}
        for id in container:
            signal = fittable_signal(container[id], id)
            if signal is not None:
                out[id] = {
                    "llpsp": float(np.log10(np.median(signal.amp[:5]))),
                    "fc": 5.0,
                    "ts": 0.01,
                }
        return out

    tables = []
    for container in (
        _direct(estimator, windows),
        _legacy_container(estimator, windows),
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            fit = FitSpectra(container, guess=guesses(container))
            fit.fit_spectra()
        tables.append(fit.table)
    return tables[0], tables[1]


def test_both_containers_fit_every_station(pnr_windows: Any) -> None:
    direct, legacy = _fits("fft", pnr_windows)
    assert len(direct) == len(legacy) == 28


def test_the_goodness_of_fit_is_unchanged(pnr_windows: Any) -> None:
    """The tell that the spectra really are the same.

    ``chisqr`` is evaluated at whatever parameters the minimiser reached, so
    two runs landing on slightly different parameters of the *same* surface
    still agree here to round-off. If the surface itself had moved, this is
    what would show it.
    """
    direct, legacy = _fits("fft", pnr_windows)
    for column in ("chisqr", "redchi", "bic"):
        assert direct[column].to_numpy() == pytest.approx(
            legacy[column].to_numpy(), rel=1e-9
        ), column


def test_the_fitted_parameters_agree_to_far_better_than_they_are_known(
    pnr_windows: Any,
) -> None:
    """Not to 1e-12, and the reason is the minimiser rather than the spectra.

    Powell on a shallow minimum amplifies the 1e-12 input difference between
    the two paths. Measured, the worst is ``fc`` at 7.5e-6 relative — 4e-5 Hz
    on a 5 Hz corner, which propagates to 2e-5 in stress drop. The bound here
    is 1e-4, loose enough not to fail on a different BLAS and four orders
    tighter than anything a corner frequency is ever known to.
    """
    direct, legacy = _fits("fft", pnr_windows)
    for column in ("fc", "llpsp", "ts"):
        a = direct[column].to_numpy(dtype=float)
        b = legacy[column].to_numpy(dtype=float)
        assert a == pytest.approx(b, rel=1e-4), column


def test_the_flatfile_records_the_band_each_fit_used(pnr_windows: Any) -> None:
    """A corner frequency without the band it was read over is not
    interpretable, and it is the first thing anyone comparing runs asks for.

    The legacy container wrote these from ``SNP.__update_lims_to_meta``. The
    frozen pair cannot write back into its own signal, so ``FittableView``
    supplies them — under the legacy column names, so a flatfile from either
    container has the same schema.
    """
    direct, legacy = _fits("fft", pnr_windows)
    for column in ("lower-f-bound", "upper-f-bound", "pass_snr"):
        assert column in direct.columns, column
        assert direct[column].to_numpy() == pytest.approx(
            legacy[column].to_numpy(dtype=float)
        ), column


def test_the_flatfile_gains_the_provenance_the_legacy_one_lacked(
    pnr_windows: Any,
) -> None:
    """Which estimator, which station, and what the window could resolve."""
    direct, legacy = _fits("fft", pnr_windows)
    for column in ("estimator", "id", "resolution_floor"):
        assert column in direct.columns
        assert column not in legacy.columns
    assert set(direct["estimator"]) == {"fft"}
    assert len(set(direct["id"])) == 28
