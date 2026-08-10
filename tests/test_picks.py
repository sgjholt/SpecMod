"""Tests for :mod:`specmod.picks`.

Covers sensor identity and matching, the three resolution rules — event
selection, ambiguity, duplicates — and the conversion from an ObsPy catalogue.
``TestHypoDDEndToEnd`` runs a real phase file through :func:`set_picks`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

obspy = pytest.importorskip("obspy")

from obspy.core.event import (  # noqa: E402
    Catalog,
    Event,
    Origin,
    WaveformStreamID,
)
from obspy.core.event import Pick as ObsPyPick  # noqa: E402

import specmod.picks as pk  # noqa: E402
import specmod.preprocess as pre  # noqa: E402

ORIGIN = obspy.UTCDateTime("2019-08-26T07:30:47")


def _pick(
    station: str = "L001",
    phase: str = "P",
    offset: float = 3.0,
    network: str | None = "LV",
    location: str | None = "",
    **kwargs: Any,
) -> pk.Pick:
    return pk.Pick(
        sensor=pk.SensorID(network=network, station=station, location=location),
        phase=phase,
        time=ORIGIN + offset,
        **kwargs,
    )


def _trace(network: str = "LV", station: str = "L001", location: str = "") -> Any:
    return obspy.Trace(
        header={
            "network": network,
            "station": station,
            "location": location,
            "channel": "HHZ",
        }
    )


class TestSensorID:
    def test_an_unstated_field_is_not_an_empty_one(self) -> None:
        # Defect 2. `--` and "the format has no such field" are different
        # claims, and resolution needs the difference.
        assert pk.SensorID("LV", "L001", "") != pk.SensorID("LV", "L001", None)

    def test_parses_and_renders_the_flat_key(self) -> None:
        assert pk.SensorID.parse("LV.L001.--") == pk.SensorID("LV", "L001", "")
        assert str(pk.SensorID("LV", "L001", "")) == "LV.L001.--"
        assert str(pk.SensorID("LV", "L001", "00")) == "LV.L001.00"

    def test_an_unstated_field_renders_as_a_wildcard(self) -> None:
        assert str(pk.SensorID(None, "L001", None)) == "*.L001.*"

    def test_a_key_with_the_wrong_field_count_raises(self) -> None:
        with pytest.raises(ValueError, match=r"not NET\.STA\.LOC"):
            pk.SensorID.parse("LV.L001")

    @pytest.mark.parametrize(
        ("pick", "sensor", "expected"),
        [
            (("LV", "L001", ""), ("LV", "L001", ""), True),
            ((None, "L001", None), ("LV", "L001", ""), True),
            ((None, "L001", None), ("UR", "L001", "00"), True),
            (("LV", "L001", None), ("UR", "L001", ""), False),
            (("LV", "L001", "00"), ("LV", "L001", ""), False),
            ((None, "L002", None), ("LV", "L001", ""), False),
        ],
    )
    def test_matching_constrains_only_the_stated_fields(
        self,
        pick: tuple[str | None, str, str | None],
        sensor: tuple[str | None, str, str | None],
        expected: bool,
    ) -> None:
        assert pk.SensorID(*pick).matches(pk.SensorID(*sensor)) is expected

    def test_completeness_is_about_the_optional_fields(self) -> None:
        assert pk.SensorID("LV", "L001", "").is_complete
        assert not pk.SensorID(None, "L001", "").is_complete
        assert not pk.SensorID("LV", "L001", None).is_complete


class TestResolve:
    def test_a_station_only_pick_reaches_its_sensor(self) -> None:
        # Defect 1, the headline case: this is what a HypoDD, NonLinLoc or
        # bulletin pick looks like once ObsPy has parsed it.
        got = pk.resolve(
            pk.PickSet((_pick(network=None, location=None),)),
            [pk.SensorID("LV", "L001", "")],
        )
        assert got.n_attached == 1
        assert got.attached["LV.L001.--"]["P"].time == ORIGIN + 3.0

    def test_the_attached_pick_carries_the_sensor_it_reached(self) -> None:
        got = pk.resolve(
            pk.PickSet((_pick(network=None, location=None),)),
            [pk.SensorID("LV", "L001", "")],
        )
        assert got.attached["LV.L001.--"]["P"].sensor.is_complete

    def test_a_pick_for_an_absent_sensor_is_reported_not_dropped(self) -> None:
        got = pk.resolve(
            pk.PickSet((_pick(station="NOPE"),)), [pk.SensorID("LV", "L001", "")]
        )
        assert got.n_attached == 0
        assert len(got.unused) == 1
        assert "1 unused" in got.summary()

    def test_one_sensor_many_components_is_not_ambiguous(self) -> None:
        # A stream carries a trace per component. Counting those as separate
        # sensors would make every ordinary stream ambiguous.
        sensors = [pk.SensorID("LV", "L001", "")] * 3
        got = pk.resolve(pk.PickSet((_pick(network=None, location=None),)), sensors)
        assert got.n_attached == 1


class TestAmbiguity:
    #: One station, two instruments — the surface/borehole case the location
    #: code exists to separate.
    SENSORS: ClassVar = [pk.SensorID("LV", "L001", ""), pk.SensorID("LV", "L001", "00")]

    def test_a_partial_pick_matching_two_sensors_raises_by_default(self) -> None:
        with pytest.raises(ValueError, match="matches 2 sensors"):
            pk.resolve(pk.PickSet((_pick(location=None),)), self.SENSORS)

    def test_the_error_names_what_would_disambiguate(self) -> None:
        with pytest.raises(ValueError, match="states no location code"):
            pk.resolve(pk.PickSet((_pick(location=None),)), self.SENSORS)

    def test_skip_reports_it_instead(self) -> None:
        got = pk.resolve(
            pk.PickSet((_pick(location=None),)), self.SENSORS, on_ambiguous="skip"
        )
        assert got.n_attached == 0
        assert len(got.ambiguous) == 1

    def test_broadcast_attaches_to_every_match(self) -> None:
        got = pk.resolve(
            pk.PickSet((_pick(location=None),)), self.SENSORS, on_ambiguous="broadcast"
        )
        assert set(got.attached) == {"LV.L001.--", "LV.L001.00"}

    def test_a_complete_pick_is_never_ambiguous(self) -> None:
        got = pk.resolve(pk.PickSet((_pick(location="00"),)), self.SENSORS)
        assert set(got.attached) == {"LV.L001.00"}


class TestDuplicates:
    def test_prefer_reviewed_beats_an_earlier_automatic_pick(self) -> None:
        picks = pk.PickSet(
            (
                _pick(offset=3.0, automatic=True, reviewed=False),
                _pick(offset=4.0, automatic=False, reviewed=True),
            )
        )
        got = pk.resolve(picks, [pk.SensorID("LV", "L001", "")])
        assert got.attached["LV.L001.--"]["P"].time == ORIGIN + 4.0
        assert got.duplicated == (("LV.L001.--", "P"),)

    def test_earliest_ignores_provenance(self) -> None:
        picks = pk.PickSet(
            (_pick(offset=4.0, reviewed=True), _pick(offset=3.0, reviewed=False))
        )
        got = pk.resolve(picks, [pk.SensorID("LV", "L001", "")], duplicates="earliest")
        assert got.attached["LV.L001.--"]["P"].time == ORIGIN + 3.0

    def test_highest_weight_prefers_a_stated_weight_over_none(self) -> None:
        picks = pk.PickSet((_pick(offset=3.0), _pick(offset=4.0, weight=1.0)))
        got = pk.resolve(
            picks, [pk.SensorID("LV", "L001", "")], duplicates="highest_weight"
        )
        assert got.attached["LV.L001.--"]["P"].time == ORIGIN + 4.0

    def test_error_refuses_to_choose(self) -> None:
        picks = pk.PickSet((_pick(offset=3.0), _pick(offset=4.0)))
        with pytest.raises(ValueError, match="has 2 P picks"):
            pk.resolve(picks, [pk.SensorID("LV", "L001", "")], duplicates="error")

    def test_a_single_pick_is_not_reported_as_duplicated(self) -> None:
        got = pk.resolve(pk.PickSet((_pick(),)), [pk.SensorID("LV", "L001", "")])
        assert got.duplicated == ()


def _event(offset: float, event_id: str) -> Event:
    event = Event(resource_id=event_id)
    event.origins.append(Origin(time=ORIGIN + offset))
    event.picks.append(
        ObsPyPick(
            time=ORIGIN + offset + 3.0,
            phase_hint="Pg",
            waveform_id=WaveformStreamID(
                network_code="LV", station_code="L001", location_code=""
            ),
        )
    )
    return event


class TestEventSelection:
    #: Two events a day apart, sharing a sensor.
    CATALOG = Catalog(events=[_event(0.0, "smi:local/one"), _event(86400.0, "two")])

    def test_a_multi_event_source_raises_rather_than_merging(self) -> None:
        # Defect 3. This used to keep whichever event came last, silently.
        sets = pk.from_catalog(self.CATALOG)
        with pytest.raises(ValueError, match="holds 2 events"):
            pk.select_event(sets)

    def test_an_event_id_selects(self) -> None:
        chosen = pk.select_event(pk.from_catalog(self.CATALOG), event_id="two")
        assert chosen.origin == ORIGIN + 86400.0

    def test_an_unmatched_event_id_raises_and_lists_what_is_there(self) -> None:
        with pytest.raises(ValueError, match="matched 0 events"):
            pk.select_event(pk.from_catalog(self.CATALOG), event_id="three")

    def test_an_origin_time_selects_within_tolerance(self) -> None:
        chosen = pk.select_event(pk.from_catalog(self.CATALOG), near=ORIGIN + 5.0)
        assert chosen.event_id == "smi:local/one"

    def test_a_tolerance_matching_both_raises(self) -> None:
        with pytest.raises(ValueError, match="2 of 2 events"):
            pk.select_event(pk.from_catalog(self.CATALOG), near=ORIGIN, tolerance_s=1e6)

    def test_a_single_event_needs_no_selector(self) -> None:
        one = Catalog(events=[_event(0.0, "only")])
        assert len(pk.select_event(pk.from_catalog(one))) == 1

    def test_an_empty_source_raises(self) -> None:
        with pytest.raises(ValueError, match="no events"):
            pk.select_event([])


class TestFromCatalog:
    def test_a_phase_branch_folds_but_is_kept(self) -> None:
        picks = pk.from_catalog(Catalog(events=[_event(0.0, "one")]))[0]
        assert picks.picks[0].phase == "P"
        assert picks.picks[0].raw_phase == "Pg"

    def test_a_rejected_pick_is_dropped(self) -> None:
        event = _event(0.0, "one")
        event.picks[0].evaluation_status = "rejected"
        assert len(pk.from_catalog(Catalog(events=[event]))[0]) == 0

    def test_a_pick_with_no_phase_hint_is_dropped(self) -> None:
        event = _event(0.0, "one")
        event.picks[0].phase_hint = None
        assert len(pk.from_catalog(Catalog(events=[event]))[0]) == 0

    def test_a_non_ps_phase_is_dropped(self) -> None:
        event = _event(0.0, "one")
        event.picks[0].phase_hint = "Lg"
        assert len(pk.from_catalog(Catalog(events=[event]))[0]) == 0

    def test_an_empty_network_code_states_nothing(self) -> None:
        # Defect 2 at its source. An empty network is not a SEED network, so
        # it means "not supplied"; an empty location code is a real value.
        event = _event(0.0, "one")
        event.picks[0].waveform_id.network_code = ""
        picks = pk.from_catalog(Catalog(events=[event]))[0]
        assert picks.picks[0].sensor.network is None
        assert picks.picks[0].sensor.location == ""

    def test_evaluation_mode_becomes_provenance(self) -> None:
        event = _event(0.0, "one")
        event.picks[0].evaluation_mode = "automatic"
        assert pk.from_catalog(Catalog(events=[event]))[0].picks[0].automatic


class TestHypoDDEndToEnd:
    """The measured defect, as a regression test against a real format."""

    PHASE_FILE = (
        "# 2019  8 26  7 30 47.00  53.785021  -2.970780   2.04  2.90  "
        "0.0  0.0  0.0        1\n"
        "L001    3.520   1.000   P\n"
        "L001    5.870   1.000   S\n"
    )

    @pytest.fixture
    def source(self, tmp_path: Path) -> str:
        path = tmp_path / "event.pha"
        path.write_text(self.PHASE_FILE)
        return str(path)

    def test_obspy_supplies_no_network_code(self, source: str) -> None:
        """The premise. If this ever changes, the case below stops being a case."""
        waveform_id = obspy.read_events(source)[0].picks[0].waveform_id
        assert not waveform_id.network_code
        assert waveform_id.location_code is None

    def test_the_picks_reach_the_stream(self, source: str) -> None:
        stream = obspy.Stream([_trace()])
        for trace in stream:
            trace.stats["otime"] = ORIGIN
        pre.set_picks(stream, source)
        assert "p_time" in stream[0].stats
        assert "s_time" in stream[0].stats
        assert stream[0].stats["s_time"] > stream[0].stats["p_time"]

    def test_the_flat_mapping_still_cannot_match(self, source: str) -> None:
        # Why `set_picks` no longer goes through it: the mapping keys on the
        # pick's own identity, which here states no network.
        assert set(pre.read_picks(source)) == {"*.L001.*"}


class TestSetPicksReport:
    def test_the_resolution_is_available_to_the_caller(self, tmp_path: Path) -> None:
        path = tmp_path / "event.pha"
        path.write_text(TestHypoDDEndToEnd.PHASE_FILE)
        stream = obspy.Stream([_trace(), _trace(station="ABSENT")])
        for trace in stream:
            trace.stats["otime"] = ORIGIN

        report: list[pk.Resolution] = []
        pre.set_picks(stream, str(path), report=report)

        assert len(report) == 1
        assert report[0].n_attached == 2
        assert report[0].summary().startswith("2 attached to 1 sensors")
