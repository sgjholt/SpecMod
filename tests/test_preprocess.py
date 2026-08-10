"""Characterisation tests for :mod:`specmod.preprocess`.

``preprocess`` sits *upstream* of the golden spectral reference, which starts
from cut windows. A change to how windows are chosen therefore moves that
reference rather than failing against it — the one shape of regression the
existing safety net is blind to. These tests close that gap so the module can
be decomposed with the same confidence as the rest.

Two kinds of test live here, and they answer different questions:

* **Pure-function tests** pin behaviour that needs no data. They are the
  documentation of what each function actually does, including the places
  where that differs from what its name or docstring promises.
* **Reference tests** pin where all 28 tutorial windows land, against
  ``tests/golden/window_reference.json``. Regenerate with
  ``python tools/make_golden.py``.

Characterisation means *current* behaviour, not correct behaviour. Where the
two differ the test says so in its name and its comment, and the fix is a
later commit whose diff then shows exactly what observable behaviour changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

import specmod.preprocess as pre  # noqa: E402

REFERENCE = Path(__file__).parent / "golden" / "window_reference.json"

#: Window edges are quantised to the sample interval, so the only way a
#: last-bit difference in the response-removed data can show up is by flipping
#: an ``argmin`` inside ``signal_intensity`` — which needs two adjacent samples
#: equidistant from a percentile. One sample of slack covers that without
#: admitting any change large enough to matter seismologically.
SAMPLE_SLACK = 1


def _reference() -> dict[str, Any]:
    if not REFERENCE.is_file():
        pytest.skip("window reference not generated")
    return json.loads(REFERENCE.read_text())["windows"]


# --------------------------------------------------------------- synthetic


def _trace(
    npts: int = 1000,
    sampling_rate: float = 100.0,
    data: np.ndarray | None = None,
    station: str = "TEST",
) -> Any:
    """A trace with picks set, so the cutting functions have something to use."""
    if data is None:
        data = np.zeros(npts)
    tr = obspy.Trace(
        np.asarray(data, dtype=np.float64),
        header={"sampling_rate": sampling_rate, "station": station, "network": "XX"},
    )
    start = tr.stats.starttime
    tr.stats["otime"] = start
    tr.stats["p_time"] = start + 2.0
    tr.stats["s_time"] = start + 4.0  # p-s differential of 2 s
    return tr


def _stream(**kwargs: Any) -> Any:
    return obspy.Stream([_trace(**kwargs)])


# ----------------------------------------------------------- pure functions


class TestNormalise:
    def test_maps_to_the_unit_interval(self) -> None:
        got = pre.normalise(np.array([0.0, 1.0, 2.0, 3.0]))
        assert got == pytest.approx([0.0, 1.0 / 3, 2.0 / 3, 1.0])

    def test_honours_a_custom_space(self) -> None:
        got = pre.normalise(np.array([0.0, 1.0, 2.0]), space=[-1, 1])
        assert got == pytest.approx([-1.0, 0.0, 1.0])

    def test_is_invariant_to_offset_and_scale(self) -> None:
        x = np.array([3.0, -1.0, 7.5, 2.25])
        assert pre.normalise(x) == pytest.approx(pre.normalise(5.0 * x + 11.0))

    def test_a_constant_array_maps_to_the_top_of_the_space(self) -> None:
        # Not obviously the right answer — 0, 1 and "undefined" are all
        # defensible for an array with no range. `np.interp` with a degenerate
        # `xp` returns the last `fp`, so it is 1. Pinned because
        # `signal_intensity` calls this on a cumulative integral, and a dead
        # trace is exactly the case where that integral is constant.
        assert pre.normalise(np.ones(5)) == pytest.approx(np.ones(5))


class TestSignalIntensity:
    def test_brackets_where_the_energy_is(self) -> None:
        # Energy confined to samples 300-700 of a 1000-sample, 100 Hz trace.
        data = np.zeros(1000)
        data[300:700] = 1.0
        start, end = pre.signal_intensity(_trace(data=data))
        # Trimmed slightly inside the box at both ends: the default 1st and
        # 99th percentiles of the cumulative energy are reached a few samples
        # after the box opens and a few before it closes.
        assert start == pytest.approx(3.0, abs=0.06)
        assert end == pytest.approx(7.0, abs=0.06)
        assert start > 3.0
        assert end < 7.0

    def test_returns_offsets_in_seconds_not_samples(self) -> None:
        data = np.zeros(1000)
        data[300:700] = 1.0
        slow = _trace(data=data, sampling_rate=100.0)
        fast = _trace(data=data, sampling_rate=200.0)
        assert pre.signal_intensity(slow)[1] == pytest.approx(
            2 * pre.signal_intensity(fast)[1], abs=0.02
        )

    def test_percentiles_are_configurable(self) -> None:
        data = np.zeros(1000)
        data[300:700] = 1.0
        wide = pre.signal_intensity(_trace(data=data), pctls=[1, 99])
        narrow = pre.signal_intensity(_trace(data=data), pctls=[25, 75])
        assert narrow[0] > wide[0]
        assert narrow[1] < wide[1]

    def test_start_never_follows_end(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            start, end = pre.signal_intensity(_trace(data=rng.normal(size=500)))
            assert start <= end


class TestGetStaShift:
    def test_returns_the_shift_for_a_listed_station(self) -> None:
        assert pre.get_sta_shift("AQ01", {"AQ01": 0.5}) == 0.5

    def test_returns_zero_for_an_unlisted_station(self) -> None:
        assert pre.get_sta_shift("AQ02", {"AQ01": 0.5}) == 0


class TestBasicSetTheoreticals:
    def test_arrival_times_follow_distance_over_velocity(self) -> None:
        st = _stream()
        st[0].stats["repi"] = 11.8
        otime = st[0].stats.starttime
        pre.basic_set_theoreticals(st, otime, p=5.9, s=2.9)
        assert st[0].stats["p_time"] - otime == pytest.approx(2.0)
        assert st[0].stats["s_time"] - otime == pytest.approx(11.8 / 2.9)

    def test_the_distance_metric_is_selectable(self) -> None:
        st = _stream()
        st[0].stats["repi"], st[0].stats["rhyp"] = 10.0, 20.0
        otime = st[0].stats.starttime
        pre.basic_set_theoreticals(st, otime, p=5.0, dmetric="rhyp")
        assert st[0].stats["p_time"] - otime == pytest.approx(4.0)


class TestCutS:
    def test_the_window_starts_at_a_fraction_of_the_p_s_time(self) -> None:
        st = _stream()
        pre.cut_s(st, rafp=0.8, tafs=5, refine_window=False)
        # p at 2 s, p-s of 2 s, so the start is at 2 + 0.8*2 = 3.6 s.
        assert st[0].stats["wstart"] - st[0].stats["otime"] == pytest.approx(3.6)
        assert st[0].stats["wend"] - st[0].stats["wstart"] == pytest.approx(5.0)

    def test_relative_ps_scales_the_length_by_the_p_s_time(self) -> None:
        st = _stream()
        pre.cut_s(st, rafp=0.8, tafs=3, time_after="relative_ps", refine_window=False)
        assert st[0].stats["wend"] - st[0].stats["wstart"] == pytest.approx(6.0)

    def test_the_window_is_clamped_to_the_end_of_the_record(self) -> None:
        st = _stream(npts=500)  # 5 s of data
        pre.cut_s(st, rafp=0.8, tafs=20, refine_window=False)
        assert st[0].stats["wend"] == st[0].stats["otime"] + 4.99

    def test_a_station_shift_moves_the_window(self) -> None:
        plain, shifted = _stream(), _stream()
        pre.cut_s(plain, refine_window=False)
        pre.cut_s(shifted, sta_shift={"TEST": 0.5}, refine_window=False)
        assert shifted[0].stats["wstart"] - plain[0].stats["wstart"] == pytest.approx(
            0.5
        )

    def test_an_unrecognised_time_after_is_named_in_the_error(self) -> None:
        # Was an `UnboundLocalError` from the middle of the loop: `time_after`
        # was free text tested by two un-chained `if`s, so a typo left `s_end`
        # unbound and the message never mentioned the argument at fault.
        with pytest.raises(ValueError, match="time_after='relatve_ps'"):
            pre.cut_s(_stream(), time_after="relatve_ps", refine_window=False)

    def test_it_accepts_cut_p_s_spelling_of_the_relative_mode(self) -> None:
        # The two functions used to disagree — `cut_p` said `"relative_time"`,
        # `cut_s` said `"relative_ps"` — and neither took the other's word for
        # the same idea. Both now take both.
        by_ps, by_time = _stream(), _stream()
        pre.cut_s(by_ps, tafs=3, time_after="relative_ps", refine_window=False)
        pre.cut_s(by_time, tafs=3, time_after="relative_time", refine_window=False)
        assert by_ps[0].stats["wend"] == by_time[0].stats["wend"]

    def test_refinement_shrinks_the_window_onto_the_energy(self) -> None:
        data = np.zeros(2000)
        data[500:600] = 1.0  # 1 s of energy, 5 s into the record
        st = _stream(npts=2000, data=data)
        pre.cut_s(st, rafp=0.8, tafs=10, refine_window=True)
        assert st[0].stats["wstart"] - st[0].stats["otime"] == pytest.approx(
            5.0, abs=0.05
        )
        assert st[0].stats["wend"] - st[0].stats["otime"] == pytest.approx(
            6.0, abs=0.05
        )


class TestCutP:
    def test_relative_time_scales_the_length_by_the_p_s_time(self) -> None:
        st = _stream()
        pre.cut_p(st, tafp=0.5, time_after="relative_time", refine_window=False)
        assert st[0].stats["wend"] - st[0].stats["wstart"] == pytest.approx(1.0)

    def test_absolute_time_takes_the_length_in_seconds(self) -> None:
        st = _stream()
        pre.cut_p(st, tafp=0.8, time_after="absolute_time", refine_window=False)
        assert st[0].stats["wend"] - st[0].stats["wstart"] == pytest.approx(0.8)

    def test_the_lead_shifts_the_window_earlier(self) -> None:
        st = _stream()
        pre.cut_p(st, bf=0.5, refine_window=False)
        assert st[0].stats["wstart"] - st[0].stats["p_time"] == pytest.approx(-0.5)

    def test_it_accepts_cut_s_s_spelling_of_the_relative_mode(self) -> None:
        by_time, by_ps = _stream(), _stream()
        pre.cut_p(by_time, tafp=0.5, time_after="relative_time", refine_window=False)
        pre.cut_p(by_ps, tafp=0.5, time_after="relative_ps", refine_window=False)
        assert by_time[0].stats["wend"] == by_ps[0].stats["wend"]

    def test_an_unrecognised_time_after_is_named_in_the_error(self) -> None:
        with pytest.raises(ValueError, match="time_after='absolute'"):
            pre.cut_p(_stream(), time_after="absolute", refine_window=False)


class TestNoiseWindows:
    def test_the_noise_window_is_asked_for_the_signal_s_length(self) -> None:
        st = _stream(npts=3000)
        sig = pre.get_signal(st, pre.cut_s, tafs=4, refine_window=False)
        noise = pre.get_noise_p(st, sig)
        stats = noise[0].stats
        assert stats["wend_requested"] - stats["wstart_requested"] == pytest.approx(4.0)

    def test_the_noise_window_ends_before_the_p_arrival(self) -> None:
        st = _stream(npts=3000)
        sig = pre.get_signal(st, pre.cut_s, tafs=1, refine_window=False)
        noise = pre.get_noise_p(st, sig, bshift=0.2)
        assert noise[0].stats["wend"] == noise[0].stats["p_time"] - 0.2

    def test_a_window_predating_the_record_is_recorded_as_what_it_delivered(
        self,
    ) -> None:
        # `link_window_to_trace` used to record only what was *asked for*, so
        # `wend - wstart` overstated the data on every noise trace running off
        # the front of the record — all 28 of the tutorial windows. It now
        # records both, with `wstart`/`wend` describing the trace you have.
        st = _stream(npts=3000)  # P arrives 2 s in, so 1.8 s of noise exists
        sig = pre.get_signal(st, pre.cut_s, tafs=10, refine_window=False)
        stats = pre.get_noise_p(st, sig)[0].stats
        delivered = stats.endtime - stats.starttime
        assert stats["wend_requested"] - stats["wstart_requested"] == pytest.approx(
            10.0
        )
        assert stats["wend"] - stats["wstart"] == pytest.approx(delivered)
        assert delivered == pytest.approx(1.79, abs=0.02)
        assert stats["wstart"] > stats["wstart_requested"]

    def test_an_untruncated_window_records_the_same_pair_twice(self) -> None:
        st = _stream(npts=3000)
        sig = pre.get_signal(st, pre.cut_s, tafs=1, refine_window=False)
        stats = pre.get_noise_p(st, sig)[0].stats
        assert stats["wstart"] == stats["wstart_requested"]
        assert stats["wend"] == stats["wend_requested"]

    def test_get_noise_s_without_a_signal_uses_a_fixed_length(self) -> None:
        st = _stream(npts=3000)
        noise = pre.get_noise_s(st, bf=1.5)
        assert noise[0].stats["wend"] - noise[0].stats["wstart"] == pytest.approx(1.5)

    def test_get_noise_s_with_a_signal_matches_its_length(self) -> None:
        st = _stream(npts=3000)
        sig = pre.get_signal(st, pre.cut_s, tafs=1.2, refine_window=False)
        noise = pre.get_noise_s(st, sig=sig)
        assert noise[0].stats["wend"] - noise[0].stats["wstart"] == pytest.approx(1.2)

    def test_a_shorter_signal_stream_raises_instead_of_passing_records_through(
        self,
    ) -> None:
        # The two streams are paired by position, so a signal stream of a
        # different length is not a shorter answer but a wrong one. With
        # `zip(..., strict=False)` the unpaired traces stayed in the result
        # whole and unlinked — full-length records presented as noise windows,
        # no error, no warning.
        st = obspy.Stream(
            [_trace(npts=3000, station="A"), _trace(npts=3000, station="B")]
        )
        sig = pre.get_signal(st, pre.cut_s, tafs=1, refine_window=False)
        with pytest.raises(ValueError, match="argument 2 is shorter"):
            pre.get_noise_p(st, obspy.Stream(sig[:1]))


_PICK_HEADER = "# Snuffler Markers File Version 0.2\n"
_PICK_LINE = (
    "phase: {date} {time}  1 XX.TEST..HHZ  None  None  None  {phase}  None False\n"
)


def _picks_file(tmp_path: Path, phases: dict[str, str]) -> str:
    """A minimal Snuffler marker file, one line per phase."""
    path = tmp_path / "test.picks"
    body = "".join(
        _PICK_LINE.format(date="2020-01-01", time=t, phase=p) for p, t in phases.items()
    )
    path.write_text(_PICK_HEADER + body)
    return str(path)


def _picked_stream(otime_offset: float = 0.0) -> Any:
    st = _stream()
    for tr in st:
        tr.stats.starttime = obspy.UTCDateTime("2020-01-01T00:00:00")
        tr.stats["otime"] = tr.stats.starttime + otime_offset
        del tr.stats["p_time"]
        del tr.stats["s_time"]
    return st


class TestPicks:
    def test_both_phases_are_read_when_present(self, tmp_path: Path) -> None:
        st = _picked_stream()
        pre.set_picks(st, _picks_file(tmp_path, {"P": "00:00:10.0", "S": "00:00:20.0"}))
        start = st[0].stats.starttime
        assert st[0].stats["p_time"] - start == pytest.approx(10.0)
        assert st[0].stats["s_time"] - start == pytest.approx(20.0)

    def test_a_missing_s_pick_is_extrapolated_from_the_origin(
        self, tmp_path: Path
    ) -> None:
        st = _picked_stream()
        pre.set_picks(
            st, _picks_file(tmp_path, {"P": "00:00:10.0"}), emergency_ratio=1.7
        )
        # S = P + (P - otime) * ratio = 10 + 10*1.7 = 27 s after the origin.
        assert st[0].stats["s_time"] - st[0].stats.starttime == pytest.approx(27.0)

    def test_a_station_with_no_picks_at_all_is_left_alone(self, tmp_path: Path) -> None:
        st = _picked_stream()
        st[0].stats.station = "OTHER"
        pre.set_picks(st, _picks_file(tmp_path, {"P": "00:00:10.0"}))
        assert "p_time" not in st[0].stats
        assert "s_time" not in st[0].stats

    def test_an_origin_after_the_p_pick_leaves_s_unset_and_warns(
        self, tmp_path: Path
    ) -> None:
        # The emergency S pick is `p + (p - otime) * ratio`, which only lands
        # after P when the origin precedes the pick. Nothing used to check
        # that, and the tutorial was for a time configured with an origin 18
        # minutes *after* its picks: a station missing an S pick there would
        # have got S before P — silently, and every window cut from it would
        # be nonsense.
        # The pipeline's own idiom for "unusable" is a trace without `s_time`,
        # and callers already filter on it, so that is what is left behind.
        st = _picked_stream(otime_offset=100.0)
        with pytest.warns(UserWarning, match="cannot be extrapolated"):
            pre.set_picks(st, _picks_file(tmp_path, {"P": "00:00:10.0"}))
        assert "s_time" not in st[0].stats
        assert "p_time" in st[0].stats  # the P pick that was read is kept

    def test_the_old_pyrocko_name_still_works_and_warns(self, tmp_path: Path) -> None:
        st = _picked_stream()
        with pytest.warns(DeprecationWarning, match="use set_picks"):
            pre.set_picks_from_pyrocko(
                st, _picks_file(tmp_path, {"P": "00:00:10.0", "S": "00:00:20.0"})
            )
        start = st[0].stats.starttime
        assert st[0].stats["p_time"] - start == pytest.approx(10.0)
        assert st[0].stats["s_time"] - start == pytest.approx(20.0)


class TestStreamDistance:
    def _by_list(self, dtype: str = "list") -> Any:
        st = _stream()
        pre.set_stream_distance(
            st,
            53.0,
            -3.0,
            2.0,
            st[0].stats.starttime,
            stlats=[53.1],
            stlons=[-3.1],
            stelvs=[100.0],
            dtype=dtype,
        )
        return st

    def test_mseed_needs_an_inventory(self) -> None:
        # Was an `AttributeError` from inside `get_channel_metadata`, several
        # frames down from the argument actually at fault.
        st = _stream()
        with pytest.raises(ValueError, match="requires an inventory"):
            pre.set_stream_distance(
                st, 0.0, 0.0, 1.0, st[0].stats.starttime, dtype="mseed"
            )

    def test_the_list_path_computes_a_distance(self) -> None:
        # Was unreachable: `STREAM_DISTANCE_METHODS` held `"list"` while the
        # branch reading `stlats`/`stlons`/`stelvs` tested for `"none"`, so
        # `"none"` failed the guard and did nothing while `"list"` passed it
        # and fell through to a printed "invalid method choice".
        st = self._by_list()
        assert st[0].stats["slat"] == 53.1
        assert st[0].stats["selv"] == 100.0
        # 0.1 deg of latitude plus 0.1 deg of longitude at 53 N.
        assert st[0].stats["repi"] == pytest.approx(13.0, abs=0.2)
        expected = np.sqrt((2.0 + 0.1) ** 2 + st[0].stats["repi"] ** 2)
        assert st[0].stats["rhyp"] == pytest.approx(expected)

    def test_none_still_works_as_a_deprecated_alias(self) -> None:
        with pytest.warns(DeprecationWarning, match="deprecated alias"):
            st = self._by_list(dtype="none")
        assert st[0].stats["repi"] == pytest.approx(self._by_list()[0].stats["repi"])

    def test_the_list_path_needs_its_coordinates(self) -> None:
        st = _stream()
        with pytest.raises(ValueError, match="requires stlats"):
            pre.set_stream_distance(
                st, 53.0, -3.0, 2.0, st[0].stats.starttime, dtype="list"
            )

    def test_an_unrecognised_dtype_is_named_in_the_error(self) -> None:
        # Was a printed "invalid method choice" per trace, leaving every trace
        # with an origin but no distance and deferring the failure to whatever
        # first asked for `repi`.
        st = _stream()
        with pytest.raises(ValueError, match="dtype='miniseed'"):
            pre.set_stream_distance(
                st, 53.0, -3.0, 2.0, st[0].stats.starttime, dtype="miniseed"
            )
        assert "dep" not in st[0].stats


class TestPadTraces:
    def test_padding_extends_both_ends_with_the_fill_value(self) -> None:
        st = _stream(npts=1000)
        before = st[0].stats.starttime
        pre.pad_traces(st, pad_len=2, pad_val=0)
        assert st[0].stats.starttime == before - 2
        assert st[0].stats.npts == 1000 + 400


class TestCutC:
    def test_the_coda_starts_after_the_s_window_and_runs_to_the_record_end(
        self,
    ) -> None:
        # Used to raise `TypeError` for any input whatsoever: the coda start
        # was written `tafp * relps + s_start`, i.e. `float + UTCDateTime`,
        # and `UTCDateTime` defines no `__radd__`. The function had never run.
        st = _stream(npts=3000)
        endtime = st[0].stats.endtime
        pre.cut_c(st, raf=0.8, tafp=1.4)
        # p at 2 s, p-s of 2 s: s starts at 3.6 s, coda at 3.6 + 1.4*2 = 6.4 s.
        assert st[0].stats["wstart"] - st[0].stats["otime"] == pytest.approx(6.4)
        assert st[0].stats["wend"] == endtime


# ------------------------------------------------------- against real data


class TestTutorialWindows:
    """Where the 28 published windows land, pinned against the reference."""

    def test_the_same_traces_survive_the_pipeline(self, pnr_stream: Any) -> None:
        assert sorted(tr.id for tr in pnr_stream()) == sorted(_reference())

    def test_station_geometry_is_unchanged(self, pnr_stream: Any) -> None:
        want = _reference()
        for tr in pnr_stream():
            ref = want[tr.id]
            for key in (
                "slat",
                "slon",
                "selv",
                "repi",
                "rhyp",
                "azimuth",
                "back_azimuth",
            ):
                assert tr.stats[key] == pytest.approx(ref[key], rel=1e-12), (
                    f"{tr.id}: {key}"
                )

    def test_picks_are_unchanged(self, pnr_stream: Any) -> None:
        want = _reference()
        for tr in pnr_stream():
            otime = tr.stats["otime"]
            ref = want[tr.id]
            assert float(tr.stats["p_time"] - otime) == pytest.approx(ref["p_time"])
            assert float(tr.stats["s_time"] - otime) == pytest.approx(ref["s_time"])

    @pytest.mark.parametrize("refine", [False, True])
    def test_signal_windows_are_unchanged(self, pnr_stream: Any, refine: bool) -> None:
        key = "signal" if refine else "unrefined"
        want = _reference()
        stream = pnr_stream()
        cut = pre.get_signal(
            stream,
            pre.cut_s,
            rafp=0.8,
            tafs=20,
            time_after="absolute_time",
            refine_window=refine,
        )
        problems = []
        for tr in cut:
            ref, otime = want[tr.id][key], tr.stats["otime"]
            slack = SAMPLE_SLACK * tr.stats.delta
            for edge in ("start", "end"):
                got = float(tr.stats[f"w{edge}"] - otime)
                if abs(got - ref[edge]) > slack:
                    problems.append(f"{tr.id} {key} {edge}: {ref[edge]} -> {got}")
            if abs(tr.stats.npts - ref["npts"]) > SAMPLE_SLACK:
                problems.append(f"{tr.id} {key} npts: {ref['npts']} -> {tr.stats.npts}")
        assert not problems, "\n".join(problems)

    def test_refinement_offsets_are_unchanged(self, pnr_stream: Any) -> None:
        """What `signal_intensity` decided, isolated from where the cut began.

        Separate from the window test above because these two can move
        independently: a change to the unrefined cut shifts both edges
        together, a change to the percentile search changes only these.
        """
        want = _reference()
        stream = pnr_stream()
        common = {"rafp": 0.8, "tafs": 20, "time_after": "absolute_time"}
        raw = pre.get_signal(stream, pre.cut_s, refine_window=False, **common)
        fine = pre.get_signal(stream, pre.cut_s, refine_window=True, **common)
        problems = []
        for a, b in zip(raw, fine, strict=True):
            ref = want[a.id]["refinement"]
            slack = SAMPLE_SLACK * a.stats.delta
            for name, got in (
                ("lead", float(b.stats["wstart"] - a.stats["wstart"])),
                ("trail", float(b.stats["wend"] - a.stats["wstart"])),
            ):
                if abs(got - ref[name]) > slack:
                    problems.append(f"{a.id} {name}: {ref[name]} -> {got}")
        assert not problems, "\n".join(problems)

    def test_noise_windows_are_unchanged(self, pnr_stream: Any) -> None:
        want = _reference()
        stream = pnr_stream()
        sig = pre.get_signal(
            stream,
            pre.cut_s,
            rafp=0.8,
            tafs=20,
            time_after="absolute_time",
            refine_window=True,
        )
        noise = pre.get_noise_p(stream, sig)
        problems = []
        for tr in noise:
            ref, otime = want[tr.id]["noise"], tr.stats["otime"]
            slack = SAMPLE_SLACK * tr.stats.delta
            for edge in ("start", "end"):
                got = float(tr.stats[f"w{edge}"] - otime)
                if abs(got - ref[edge]) > slack:
                    problems.append(f"{tr.id} noise {edge}: {ref[edge]} -> {got}")
            if abs(tr.stats.npts - ref["npts"]) > SAMPLE_SLACK:
                problems.append(f"{tr.id} noise npts: {ref['npts']} -> {tr.stats.npts}")
        assert not problems, "\n".join(problems)

    def test_every_tutorial_noise_window_is_short(self, pnr_windows: Any) -> None:
        """All 28, not some — and by up to a factor of two.

        The tutorial's records begin roughly two seconds before the P arrival,
        so ``get_noise_p`` never gets the signal-length window it asks for.
        This is why noise and signal spectra do not share a frequency
        resolution, and why ``SpectrumPair`` carries a ``resolution_floor``
        rather than assuming they do. Pinned so that a change which quietly
        started delivering full-length noise windows would be noticed:
        it would move every noise spectrum in the reference.
        """
        signal, noise = pnr_windows()
        ratios = []
        for sig, noi in zip(signal, noise, strict=True):
            requested = float(
                noi.stats["wend_requested"] - noi.stats["wstart_requested"]
            )
            delivered = float(noi.stats["wend"] - noi.stats["wstart"])
            assert delivered == pytest.approx(
                float(noi.stats.endtime - noi.stats.starttime)
            )
            assert requested == pytest.approx(
                float(sig.stats.endtime - sig.stats.starttime), abs=1e-6
            )
            assert delivered < requested
            ratios.append(delivered / requested)
        assert max(ratios) < 1.0
        assert min(ratios) > 0.4

    def test_the_signal_window_is_delivered_as_requested(self, pnr_stream: Any) -> None:
        """Signal windows, unlike noise windows, get what they ask for.

        Up to half a sample of ``trim`` snapping, which matters because
        ``get_noise_p`` derives the noise window's length from the signal's
        recorded ``wend - wstart``. Now that those describe what was delivered
        rather than what was asked for, that derivation shifts by at most
        ``delta/2`` — and on this data not at all, because every noise window
        is truncated by the record start regardless.
        """
        stream = pnr_stream()
        cut = pre.get_signal(
            stream,
            pre.cut_s,
            rafp=0.8,
            tafs=20,
            time_after="absolute_time",
            refine_window=True,
        )
        for tr in cut:
            half = 0.5 * tr.stats.delta
            assert abs(tr.stats["wstart"] - tr.stats["wstart_requested"]) <= half
            assert abs(tr.stats["wend"] - tr.stats["wend_requested"]) <= half
