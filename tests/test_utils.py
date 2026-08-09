"""Characterisation tests for what is left of :mod:`specmod.utils`.

``read_pyrocko`` is the only part of this module the pipeline depends on, and
it sits upstream of both golden references — every window is cut relative to
the picks it returns. Like ``preprocess``, a change here *moves* those
references rather than failing against them, so it needs pinning separately.

The catalogue readers are covered here too. ``plot_traces`` is not: it draws,
and a test that only asserts it does not raise buys nothing that a broken
figure would not also pass.

Where current behaviour is wrong, the test says so and pins it anyway — see
``TestReadPyrocko``'s notes on weights and on repeated picks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

import specmod.utils as ut  # noqa: E402
from specmod.datasets import PNR_2019  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PATHS = PNR_2019.directory(ROOT)

HEADER = "# Snuffler Markers File Version 0.2\n"
LINE = "phase: {date} {time}  {weight} {sid}  None  None  None  {phase}  None False\n"


def _picks(tmp_path: Path, rows: list[dict[str, Any]]) -> str:
    path = tmp_path / "markers.picks"
    body = "".join(
        LINE.format(
            date=r.get("date", "2020-01-01"),
            time=r["time"],
            weight=r.get("weight", 1),
            sid=r.get("sid", "XX.TEST..HHZ"),
            phase=r["phase"],
        )
        for r in rows
    )
    path.write_text(HEADER + body)
    return str(path)


class TestReadPyrocko:
    def test_both_phases_land_under_one_station_key(self, tmp_path: Path) -> None:
        got = ut.read_pyrocko(
            _picks(
                tmp_path,
                [
                    {"time": "00:00:10.0", "phase": "P"},
                    {"time": "00:00:20.0", "phase": "S"},
                ],
            )
        )
        assert set(got) == {"XX.TEST"}
        assert got["XX.TEST"]["P"] == obspy.UTCDateTime("2020-01-01T00:00:10")
        assert got["XX.TEST"]["S"] == obspy.UTCDateTime("2020-01-01T00:00:20")

    def test_the_key_is_network_and_station_only(self, tmp_path: Path) -> None:
        """Location and channel are dropped, so picks on different channels of
        the same station collapse together. That is deliberate — a P arrival is
        a property of the station, not of the component it was picked on."""
        got = ut.read_pyrocko(
            _picks(
                tmp_path,
                [
                    {"time": "00:00:10.0", "phase": "P", "sid": "XX.TEST..HHZ"},
                    {"time": "00:00:20.0", "phase": "S", "sid": "XX.TEST.00.HHN"},
                ],
            )
        )
        assert set(got) == {"XX.TEST"}
        assert set(got["XX.TEST"]) == {"P", "S"}

    @pytest.mark.parametrize("marker", ["^", "v", "P"])
    def test_all_three_p_markers_read_as_p(self, tmp_path: Path, marker: str) -> None:
        """Snuffler distinguishes first motion — up, down, or unspecified — and
        the polarity is parsed out of the map and then thrown away."""
        got = ut.read_pyrocko(
            _picks(tmp_path, [{"time": "00:00:10.0", "phase": marker}])
        )
        assert set(got["XX.TEST"]) == {"P"}

    def test_the_first_line_is_treated_as_a_header_whatever_it_says(
        self, tmp_path: Path
    ) -> None:
        # DEFECT, pinned: the reader does `readlines()[1:]` unconditionally
        # rather than checking for the Snuffler magic. A marker file without
        # the header line silently loses its first pick.
        path = tmp_path / "no_header.picks"
        path.write_text(
            LINE.format(
                date="2020-01-01",
                time="00:00:10.0",
                weight=1,
                sid="XX.TEST..HHZ",
                phase="P",
            )
            + LINE.format(
                date="2020-01-01",
                time="00:00:20.0",
                weight=1,
                sid="XX.TEST..HHZ",
                phase="S",
            )
        )
        got = ut.read_pyrocko(str(path))
        assert set(got["XX.TEST"]) == {"S"}, "the P pick was eaten as a header"

    def test_weights_above_three_are_dropped(self, tmp_path: Path) -> None:
        got = ut.read_pyrocko(
            _picks(
                tmp_path,
                [
                    {"time": "00:00:10.0", "phase": "P", "weight": 1},
                    {"time": "00:00:20.0", "phase": "S", "weight": 4},
                ],
            )
        )
        assert set(got["XX.TEST"]) == {"P"}

    def test_a_repeated_phase_keeps_the_last_one_read(self, tmp_path: Path) -> None:
        # DEFECT, pinned: two P picks for one station — easy to produce by
        # picking on more than one component — silently resolve to whichever
        # appears last in the file, with no warning and no ordering guarantee
        # beyond file order. There is no weighting or averaging.
        got = ut.read_pyrocko(
            _picks(
                tmp_path,
                [
                    {"time": "00:00:10.0", "phase": "P"},
                    {"time": "00:00:11.0", "phase": "P"},
                ],
            )
        )
        assert got["XX.TEST"]["P"] == obspy.UTCDateTime("2020-01-01T00:00:11")

    def test_an_unknown_marker_raises(self, tmp_path: Path) -> None:
        # A `KeyError` on the marker character rather than a message. Pinned
        # because it is at least loud: a marker type this does not understand
        # stops the run instead of being dropped.
        with pytest.raises(KeyError):
            ut.read_pyrocko(_picks(tmp_path, [{"time": "00:00:10.0", "phase": "X"}]))

    @pytest.mark.skipif(
        not PATHS.picks.is_dir(), reason="tutorial waveforms not present"
    )
    def test_the_tutorial_picks_are_unchanged(self) -> None:
        """The 15 stations every window in both golden references is cut from."""
        got = ut.read_pyrocko(str(PATHS.picks_file()))
        assert len(got) == 15
        assert all(set(v) == {"P", "S"} for v in got.values())
        assert all(v["S"] > v["P"] for v in got.values())
        # All 30 picks fall inside an eight-second span of the record.
        times = [t for v in got.values() for t in v.values()]
        assert max(times) - min(times) == pytest.approx(7.877, abs=0.01)


class TestCatalogue:
    def test_cat2kstyle_drops_sub_second_precision(self) -> None:
        row = {"Date": "2020/03/18", "Time": "13:09:31.123"}
        assert ut.cat2kstyle(row) == "2020.03.18.13.09.31"

    def test_keith2utc_round_trips_a_catalogue_row(self) -> None:
        row = {"Date": "2020/03/18", "Time": "13:09:31.000"}
        assert ut.keith2utc(row) == obspy.UTCDateTime(2020, 3, 18, 13, 9, 31)

    def test_seconds_without_decimals_are_not_lost(self) -> None:
        """The silent case: the old fixed slice returned 13:09, not 13:09:31."""
        assert ut.cat2kstyle({"Date": "2020/03/18", "Time": "13:09:31"}) == (
            "2020.03.18.13.09.31"
        )

    def test_read_cat_splits_on_runs_of_whitespace(self, tmp_path: Path) -> None:
        # `sep=r"\s+"` rather than the `delim_whitespace=True` this used to
        # pass, which pandas 3.0 removed.
        path = tmp_path / "cat.txt"
        path.write_text("Date       Time          Mag\n2020/03/18 13:09:31.000  5.7\n")
        df = ut.read_cat(str(path))
        assert list(df.columns) == ["Date", "Time", "Mag"]
        assert df["Mag"].iloc[0] == pytest.approx(5.7)


class TestStreamDistanceSort:
    def _stream(self, distances: list[float]) -> Any:
        st = obspy.Stream()
        for i, d in enumerate(distances):
            tr = obspy.Trace(
                np.zeros(10),
                header={"station": f"S{i}", "sampling_rate": 1.0},
            )
            tr.stats["repi"] = d
            st += tr
        return st

    def test_it_sorts_by_distance(self) -> None:
        got = ut.stream_distance_sort(self._stream([30.0, 10.0, 20.0]))
        assert [tr.stats["repi"] for tr in got] == [10.0, 20.0, 30.0]

    def test_it_returns_a_copy_not_the_input(self) -> None:
        st = self._stream([30.0, 10.0])
        got = ut.stream_distance_sort(st)
        assert got is not st
        assert [tr.stats["repi"] for tr in st] == [30.0, 10.0], "input was mutated"

    def test_a_stream_without_distances_comes_back_unsorted(self) -> None:
        st = obspy.Stream(
            [
                obspy.Trace(np.zeros(10), header={"station": "B"}),
                obspy.Trace(np.zeros(10), header={"station": "A"}),
            ]
        )
        got = ut.stream_distance_sort(st)
        assert [tr.stats.station for tr in got] == ["B", "A"]
