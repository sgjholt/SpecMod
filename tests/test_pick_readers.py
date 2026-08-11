"""The reader contract, run against every registered reader.

Two halves. ``TestReaderContract`` is parameterised over :data:`PICK_READERS`
and is what a plugin author runs against their own reader. ``TestRoster``
writes each format ObsPy can write, reads it back, and records which of them
actually carry picks through the round trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

obspy = pytest.importorskip("obspy")

from obspy.core.event import (  # noqa: E402
    Catalog,
    Event,
    Magnitude,
    Origin,
    Pick,
    WaveformStreamID,
)

import specmod.picks as pk  # noqa: E402

ORIGIN = obspy.UTCDateTime("2019-08-26T07:30:47")

MARKER = """# Snuffler Markers File Version 0.2
phase: 2019-08-26 07:30:50.500  1 LV.L001..HHZ  None None None P None False
phase: 2019-08-26 07:30:52.900  1 LV.L001..HHN  None None None S None False
"""

HYPODD = (
    "# 2019  8 26  7 30 47.00  53.785021  -2.970780   2.04  2.90  "
    "0.0  0.0  0.0    1\n"
    """L001    3.520   1.000   P
L001    5.870   1.000   S
"""
)


def _catalog() -> Catalog:
    """One event, two sensors, P and S on each."""
    event = Event()
    origin = Origin(time=ORIGIN, latitude=53.785, longitude=-2.97, depth=2040.0)
    event.origins.append(origin)
    event.preferred_origin_id = origin.resource_id
    event.magnitudes.append(Magnitude(mag=2.9, magnitude_type="Mw"))
    for station, p, s in (("L001", 3.5, 5.9), ("L002", 4.1, 6.8)):
        for phase, offset in (("P", p), ("S", s)):
            event.picks.append(
                Pick(
                    time=ORIGIN + offset,
                    phase_hint=phase,
                    waveform_id=WaveformStreamID(
                        network_code="LV",
                        station_code=station,
                        location_code="",
                        channel_code="HHZ",
                    ),
                )
            )
    return Catalog(events=[event])


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One small file per format, plus the things a sniffer must survive."""
    root = tmp_path_factory.mktemp("corpus")
    files = {
        "snuffler": root / "event.picks",
        "quakeml": root / "event.xml",
        "hypodd": root / "event.pha",
    }
    files["snuffler"].write_text(MARKER)
    files["hypodd"].write_text(HYPODD)
    _catalog().write(str(files["quakeml"]), format="QUAKEML")

    files["empty"] = root / "empty.txt"
    files["empty"].write_bytes(b"")
    files["binary"] = root / "noise.bin"
    files["binary"].write_bytes(bytes(range(256)) * 8)
    files["truncated"] = root / "truncated.xml"
    files["truncated"].write_text(files["quakeml"].read_text()[:200])
    files["prose"] = root / "notes.txt"
    files["prose"].write_text("station L001 looked fine today\n")
    return files


#: Fixtures every reader must decline: not a pick file in any format.
NOT_PICKS = ("empty", "binary", "truncated", "prose")

#: Which reader owns which fixture.
OWNERS = {"snuffler": "snuffler", "quakeml": "obspy_events", "hypodd": "obspy_events"}


@pytest.mark.parametrize("reader", pk.PICK_READERS.values(), ids=pk.PICK_READERS)
class TestReaderContract:
    def test_it_satisfies_the_protocol(self, reader: pk.PickReader) -> None:
        assert isinstance(reader, pk.PickReader)
        assert reader.name in pk.PICK_READERS
        assert all(s.startswith(".") for s in reader.suffixes)

    @pytest.mark.parametrize("name", NOT_PICKS)
    def test_it_declines_what_is_not_a_pick_file(
        self, reader: pk.PickReader, corpus: dict[str, Path], name: str
    ) -> None:
        assert reader.can_read(corpus[name]) is False

    def test_it_declines_a_missing_file(
        self, reader: pk.PickReader, tmp_path: Path
    ) -> None:
        assert reader.can_read(tmp_path / "nope.xml") is False

    def test_it_claims_only_its_own_fixtures(
        self, reader: pk.PickReader, corpus: dict[str, Path]
    ) -> None:
        for name, owner in OWNERS.items():
            assert reader.can_read(corpus[name]) is (reader.name == owner)

    def test_what_it_claims_it_can_read(
        self, reader: pk.PickReader, corpus: dict[str, Path]
    ) -> None:
        for name, owner in OWNERS.items():
            if reader.name != owner:
                continue
            sets = reader.read(corpus[name])
            assert sets
            assert all(isinstance(s, pk.PickSet) for s in sets)
            picks = [p for s in sets for p in s.picks]
            assert picks
            assert all(p.phase in ("P", "S") for p in picks)
            assert all(isinstance(p.time, obspy.UTCDateTime) for p in picks)
            assert all(p.sensor.station for p in picks)


class TestDetection:
    def test_exactly_one_reader_claims_each_fixture(
        self, corpus: dict[str, Path]
    ) -> None:
        for name, owner in OWNERS.items():
            assert pk.detect_reader(corpus[name]).name == owner

    @pytest.mark.parametrize("name", NOT_PICKS)
    def test_nothing_claimed_is_an_error_naming_what_was_tried(
        self, corpus: dict[str, Path], name: str
    ) -> None:
        with pytest.raises(ValueError, match="no reader recognises"):
            pk.detect_reader(corpus[name])

    def test_an_explicit_format_skips_detection(self, corpus: dict[str, Path]) -> None:
        assert len(pk.read(corpus["hypodd"], format="obspy_events")[0]) == 2

    def test_an_unknown_format_name_raises(self, corpus: dict[str, Path]) -> None:
        with pytest.raises(ValueError, match="Unknown pick format"):
            pk.read(corpus["quakeml"], format="nonesuch")

    def test_a_catalog_needs_no_reader(self) -> None:
        assert len(pk.read(_catalog())[0]) == 4


@pytest.mark.filterwarnings(
    # ObsPy's NORDIC writer, on a pick that carries no evaluation mode. It is
    # third-party, it says nothing about what these tests assert, and it is
    # matched by message so a different NORDIC warning would still surface.
    "ignore:Evaluation mode None is not mappable:UserWarning"
)
class TestRoster:
    """What §4.9.6 claims, measured rather than assumed.

    Each format is written by ObsPy, read back, and checked for whether its
    picks survived. ``SCML`` is the row this corrected: the writer emits the
    picks and the reader drops them.
    """

    CARRIES_PICKS = ("QUAKEML", "NORDIC", "HYPODDPHA")

    @pytest.mark.parametrize("fmt", CARRIES_PICKS)
    def test_picks_survive_the_round_trip(self, fmt: str, tmp_path: Path) -> None:
        path = tmp_path / f"event.{fmt.lower()}"
        _catalog().write(str(path), format=fmt)

        sets = pk.read(path)
        assert len(sets) == 1
        picks = sets[0].picks
        assert len(picks) == 4
        assert {p.phase for p in picks} == {"P", "S"}
        assert {p.sensor.station for p in picks} == {"L001", "L002"}

    @pytest.mark.parametrize("fmt", CARRIES_PICKS)
    def test_the_picks_resolve_against_a_stream(self, fmt: str, tmp_path: Path) -> None:
        """The point of resolution: a partial identity still reaches its sensor."""
        path = tmp_path / f"event.{fmt.lower()}"
        _catalog().write(str(path), format=fmt)

        sensors = [pk.SensorID("LV", s, "") for s in ("L001", "L002")]
        got = pk.resolve(pk.select_event(pk.read(path)), sensors)
        assert got.n_attached == 4
        assert got.unused == ()

    def test_scml_loses_its_picks_on_the_way_back(self, tmp_path: Path) -> None:
        """Measured against obspy 1.5.0, and why SCML is not in the roster.

        The writer emits the picks — they are in the file — and the reader
        returns the origin without them. Recheck on an ObsPy upgrade: if this
        starts failing, SCML has gained pick support and the roster grows.
        """
        path = tmp_path / "event.scml"
        _catalog().write(str(path), format="SCML")
        assert path.read_text().count("<pick ") == 4

        sets = pk.read(path)
        assert sets[0].origin == ORIGIN
        assert len(sets[0]) == 0


class TestSnufflerReader:
    def test_a_plain_time_marker_is_not_a_phase(self, tmp_path: Path) -> None:
        path = tmp_path / "mixed.picks"
        path.write_text(MARKER + "2019-08-26 07:30:40.0 0 None None\n")
        assert len(pk.read(path)[0]) == 2

    def test_the_channel_is_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "event.picks"
        path.write_text(MARKER)
        assert {p.channel for p in pk.read(path)[0].picks} == {"HHZ", "HHN"}

    def test_a_rejected_weight_is_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "event.picks"
        path.write_text(MARKER.replace(".500  1 LV", ".500  4 LV"))
        assert len(pk.read(path)[0]) == 1


class TestShippedPicksAgree:
    """Both shipped forms of the tutorial picks read the same."""

    ROOT = Path(__file__).resolve().parent.parent

    @pytest.fixture
    def paths(self) -> Any:
        from specmod.datasets import PNR_2019  # noqa: PLC0415

        directory = PNR_2019.directory(self.ROOT)
        if not directory.picks.is_dir():
            pytest.skip("tutorial picks not present")
        return directory

    def test_the_marker_and_quakeml_forms_match(self, paths: Any) -> None:
        marker = next(iter(paths.picks.glob("*.picks")))
        quakeml = next(iter(paths.picks.glob("*.xml")))

        from_marker = pk.read(marker)[0].mapping()
        from_quakeml = pk.read(quakeml)[0].mapping()
        assert from_marker == from_quakeml
        assert len(from_marker) == 15

    def test_each_is_claimed_by_exactly_one_reader(self, paths: Any) -> None:
        assert pk.detect_reader(next(iter(paths.picks.glob("*.picks")))).name == (
            "snuffler"
        )
        assert pk.detect_reader(next(iter(paths.picks.glob("*.xml")))).name == (
            "obspy_events"
        )
