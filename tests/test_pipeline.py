"""The waveform-to-``SpectrumSet`` path: trace, spectrum, pair, event.

:mod:`specmod.pipeline` is the single file that knows what an ObsPy ``Trace``
is. ``core`` and ``transforms`` below it take arrays, a duration and a sampling
rate, which is what makes them testable without constructing a Stream.

This covers the adapter and the domain change. The numerical guarantee — that
these are the same numbers the pre-refactor pipeline produced — is carried by
``tests/test_golden_reference.py``, against a committed artefact rather than
against live legacy code, so that it survives the legacy code being removed.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
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

#: Tolerance for comparisons between two routes to the same quantity. Tight
#: enough that a change in the arithmetic cannot hide under it, loose enough
#: not to demand bit-identity from operations composed in a different order.
RTOL = 1e-12

ESTIMATORS = ["fft", "welch", "multitaper", "quadratic", "cwt"]

#: Fixed probabilities the golden summaries are sampled at. Must match
#: ``tools/make_golden.py``; a mismatch would compare different quantities.
QUANTILES = np.linspace(0.0, 1.0, 33)


def _summary(a: Any) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "max": float(a.max()),
        "sum": float(a.sum()),
        "quantiles": [float(q) for q in np.quantile(a, QUANTILES)],
    }


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


# The legacy comparison lived here: both paths run over the same 28 windows and
# all five estimators, spectra, SNR, floors and bands held to 1e-12, plus the
# fitter run on each container. It found the one real gap — the flatfile's band
# columns — and it agreed everywhere else.
#
# `spectral` is deleted, so it has no subject. Deleting a comparison is a real
# loss of evidence, and the honest statement of what carries it now is:
# `tests/golden/pipeline_reference.json` was generated by the legacy path and
# has never been regenerated. `specmod.pipeline` reproduces it to 1e-15 across
# all 140 window-estimator results, and `tests/test_golden_reference.py` checks
# that on every run. That is why the reference is summaries of a committed
# artefact rather than a comparison against live code: it is the form of the
# claim that survives the code being removed.
#

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


# ------------------------------------------------------- ground-motion domain


class TestToMotion:
    """`SpectrumSet.to_motion`, replacing `Spectra.inte()`/`diff()`.

    The legacy pair mutated in place, so "the same event as displacement" could
    only be had by destroying the velocity one. These return new objects.
    """

    def test_it_matches_the_recorded_displacement(self, pnr_windows: Any) -> None:
        """Against ``tests/golden/motion_reference.json``.

        That file was captured while ``spectral.Spectra.inte()`` still existed
        and was validated against it: **2.2e-15 on the summaries and identical
        bands on all 28**. The comparison itself could not be kept — one side
        of it is deleted — so this is its durable form, the same numbers held
        against a committed artefact.
        """
        reference = json.loads(
            (Path(__file__).parent / "golden" / "motion_reference.json").read_text()
        )["displacement"]
        displacement = _direct("fft", pnr_windows).to_motion("displacement")

        assert sorted(displacement.ids()) == sorted(reference)
        problems = []
        for id in displacement.ids():
            pair, want = displacement[id], reference[id]
            assert str(pair.signal.motion) == want["motion"], id
            for key, array in (
                ("amp", pair.signal.amp),
                ("noise_amp", pair.noise.amp),
                ("bsnr", pair.snr),
            ):
                got = _summary(array)
                for field in ("median", "max", "sum"):
                    if got[field] != pytest.approx(want[key][field], rel=1e-9):
                        problems.append(
                            f"{id} {key}.{field}: {want[key][field]:.6e} -> "
                            f"{got[field]:.6e}"
                        )
            if want["band"] is None:
                if pair.band is not None:
                    problems.append(f"{id} band: none -> {pair.band}")
            elif pair.band != pytest.approx(want["band"], rel=1e-9):
                problems.append(f"{id} band: {want['band']} -> {pair.band}")
        assert not problems, "\n".join(problems)

    def test_the_original_is_untouched(self, pnr_windows: Any) -> None:
        """The property the legacy could not have. `Spectra.inte()` overwrote
        the event; here both domains exist at once."""
        direct = _direct("fft", pnr_windows)
        id = direct.ids()[0]
        before = direct[id].signal.amp.copy()
        displacement = direct.to_motion("displacement")
        assert direct[id].signal.amp == pytest.approx(before)
        assert displacement[id].signal.amp != pytest.approx(before)

    def test_the_domain_label_follows(self, pnr_windows: Any) -> None:
        direct = _direct("fft", pnr_windows)
        for pair in direct.to_motion("acceleration").pairs.values():
            assert str(pair.signal.motion) == "acceleration"
            assert str(pair.noise.motion) == "acceleration"

    def test_the_noise_is_not_lifted_a_second_time(self, pnr_windows: Any) -> None:
        """`self.noise` already carries the lift, so replaying the comparison
        with it on would compound on every conversion, narrowing the band each
        time. The pre-refactor code guarded this with a `ROTATED` flag; here it
        falls out of the recorded settings being replayed with the lift off."""
        direct = _direct("fft", pnr_windows)
        settings = direct[direct.ids()[0]].meta[SpectrumPair.SETTINGS_KEY]
        assert settings["rotate_noise"] is True

        once = direct.to_motion("displacement")
        twice = once.to_motion("velocity").to_motion("displacement")
        for id in once.ids():
            assert twice[id].noise.amp == pytest.approx(once[id].noise.amp, rel=RTOL)

    def test_the_settings_travel_with_the_pair(self, pnr_windows: Any) -> None:
        """A pair that could not say how it was made could only be remade by
        being told again, which is how a stored result's recorded settings drift
        from the ones it was computed with."""
        pair = _direct("fft", pnr_windows)[_direct("fft", pnr_windows).ids()[0]]
        settings = pair.meta[SpectrumPair.SETTINGS_KEY]
        assert settings["n_bins"] == 151
        assert settings["bandwidth"] == "peak"
        assert settings["noise_model"] == "boost"

    def test_the_unbinned_arrays_round_trip_exactly(self, pnr_windows: Any) -> None:
        direct = _direct("fft", pnr_windows)
        back = direct.to_motion("displacement").to_motion("velocity")
        for id in direct.ids():
            assert back[id].signal.amp == pytest.approx(
                direct[id].signal.amp, rel=RTOL
            ), id
            assert back[id].noise.amp == pytest.approx(
                direct[id].noise.amp, rel=RTOL
            ), id

    def test_the_binned_noise_does_not_round_trip(self, pnr_windows: Any) -> None:
        """A pre-existing inconsistency, inherited rather than introduced.

        The lift is applied to the binned noise directly but to the unbinned
        noise by interpolating the factor onto the finer axis, so the two are
        **not related by the binning operation**: re-binning the lifted
        unbinned noise does not reproduce the lifted binned one. A domain
        change re-bins, so the round trip restores the unbinned arrays exactly
        and lands the binned noise 18.8% away at worst, moving 3 of 28 bands.

        Measured identical to the legacy path — same percentage, same three
        stations — so `to_motion` reproduces it rather than causing it. Fixing
        it moves `snr`, and therefore bands, and therefore every fitted
        parameter; that is a science decision. See REFACTOR_PLAN §4.6.
        """
        direct = _direct("fft", pnr_windows)
        back = direct.to_motion("displacement").to_motion("velocity")
        worst = max(
            float(
                np.max(
                    np.abs(back[id].binned_noise.amp / direct[id].binned_noise.amp - 1)
                )
            )
            for id in direct.ids()
        )
        assert 0.1 < worst < 0.5, worst

        moved = [id for id in direct.ids() if back[id].band != direct[id].band]
        assert 0 < len(moved) < len(direct)
