"""Registration, plugin discovery, and the configurable CSV reader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

obspy = pytest.importorskip("obspy")

import specmod.picks as pk  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike

CSV = """station,phase_type,arrival_time,net,loc,chan,prob
L001,P,2019-08-26T07:30:50.500,LV,,HHZ,0.94
L001,S,2019-08-26T07:30:52.900,LV,,HHN,0.81
L002,Pg,2019-08-26T07:30:51.100,LV,,HHZ,0.77
L002,Lg,2019-08-26T07:30:59.000,LV,,HHZ,0.30
"""

COLUMNS = {
    "station": "station",
    "phase": "phase_type",
    "time": "arrival_time",
    "network": "net",
    "location": "loc",
    "channel": "chan",
    "weight": "prob",
}


@pytest.fixture
def table(tmp_path: Path) -> Path:
    path = tmp_path / "picks.csv"
    path.write_text(CSV)
    return path


@pytest.fixture(autouse=True)
def _restore_registry() -> Any:
    """Every test here mutates a module-level registry; put it back."""
    before = dict(pk.PICK_READERS)
    loaded = pk._plugins_loaded
    yield
    pk.PICK_READERS.clear()
    pk.PICK_READERS.update(before)
    pk._plugins_loaded = loaded


class TestCSVPickReader:
    READER = pk.CSVPickReader(columns=COLUMNS, reader_name="my_lab")

    def test_it_reads_the_mapped_columns(self, table: Path) -> None:
        picks = self.READER.read(table)[0].picks
        assert len(picks) == 3  # the Lg row is not a P or S arrival
        first = picks[0]
        assert first.sensor == pk.SensorID("LV", "L001", "")
        assert first.phase == "P"
        assert first.channel == "HHZ"
        assert first.weight == 0.94

    def test_a_phase_branch_folds_and_is_kept(self, table: Path) -> None:
        pg = [p for p in self.READER.read(table)[0].picks if p.sensor.station == "L002"]
        assert pg[0].phase == "P"
        assert pg[0].raw_phase == "Pg"

    def test_an_unmapped_field_is_unstated_not_empty(self, tmp_path: Path) -> None:
        # The `location` column is what separates two sensors at a site. A
        # table without one states nothing, and resolution matches on the rest.
        path = tmp_path / "bare.csv"
        path.write_text("station,phase_type,arrival_time\nL001,P,2019-08-26T07:30:50\n")
        reader = pk.CSVPickReader(
            columns={
                "station": "station",
                "phase": "phase_type",
                "time": "arrival_time",
            },
            reader_name="bare",
        )
        sensor = reader.read(path)[0].picks[0].sensor
        assert sensor.network is None
        assert sensor.location is None

    def test_a_blank_location_column_states_an_empty_one(self, table: Path) -> None:
        assert self.READER.read(table)[0].picks[0].sensor.location == ""

    def test_it_claims_a_file_carrying_its_columns(self, table: Path) -> None:
        assert self.READER.can_read(table) is True

    def test_it_declines_a_table_with_different_headings(self, tmp_path: Path) -> None:
        path = tmp_path / "other.csv"
        path.write_text("a,b,c\n1,2,3\n")
        assert self.READER.can_read(path) is False

    @pytest.mark.parametrize("content", [b"", bytes(range(256)) * 4, b"not a table\n"])
    def test_can_read_never_raises(self, tmp_path: Path, content: bytes) -> None:
        path = tmp_path / "odd.bin"
        path.write_bytes(content)
        assert self.READER.can_read(path) is False

    def test_a_missing_required_column_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match=r"columns must map \['time'\]"):
            pk.CSVPickReader(columns={"station": "sta", "phase": "ph"})

    def test_an_unknown_field_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown pick fields"):
            pk.CSVPickReader(
                columns={
                    "station": "s",
                    "phase": "p",
                    "time": "t",
                    "magnitude": "m",
                }
            )

    def test_the_picks_resolve_against_a_stream(self, table: Path) -> None:
        got = pk.resolve(
            self.READER.read(table)[0], [pk.SensorID("LV", s, "") for s in ("L001",)]
        )
        assert got.n_attached == 2
        assert len(got.unused) == 1  # L002's P pick, whose sensor is absent


class TestRegistration:
    def test_a_registered_reader_is_detected(self, table: Path) -> None:
        reader = pk.CSVPickReader(columns=COLUMNS, reader_name="my_lab")
        pk.register_reader(reader)
        assert pk.detect_reader(table).name == "my_lab"
        assert len(pk.read(table)[0]) == 3

    def test_an_unregistered_table_is_claimed_by_nothing(self, table: Path) -> None:
        with pytest.raises(ValueError, match="no reader recognises"):
            pk.detect_reader(table)

    def test_a_builtin_name_cannot_be_taken(self) -> None:
        reader = pk.CSVPickReader(columns=COLUMNS, reader_name="snuffler")
        with pytest.raises(ValueError, match="built-in reader"):
            pk.register_reader(reader)

    def test_a_duplicate_name_needs_replace(self) -> None:
        reader = pk.CSVPickReader(columns=COLUMNS, reader_name="my_lab")
        pk.register_reader(reader)
        with pytest.raises(ValueError, match="already registered"):
            pk.register_reader(reader)
        pk.register_reader(reader, replace=True)

    def test_registering_does_not_disturb_the_builtins(self, table: Path) -> None:
        pk.register_reader(pk.CSVPickReader(columns=COLUMNS, reader_name="my_lab"))
        assert set(pk.BUILTIN_READERS) <= set(pk.PICK_READERS)


@dataclass(frozen=True)
class _GreedyReader:
    """Claims everything — stands in for a plugin that sniffs too loosely."""

    @property
    def name(self) -> str:
        return "greedy"

    @property
    def suffixes(self) -> tuple[str, ...]:
        return (".any",)

    def can_read(self, source: str | PathLike[str]) -> bool:
        return True

    def read(self, source: str | PathLike[str]) -> list[pk.PickSet]:
        return [pk.PickSet()]


class TestAmbiguousReaders:
    def test_two_claims_is_an_error_naming_both(self, table: Path) -> None:
        pk.register_reader(pk.CSVPickReader(columns=COLUMNS, reader_name="my_lab"))
        pk.register_reader(_GreedyReader())
        with pytest.raises(ValueError, match="2 readers claim"):
            pk.detect_reader(table)

    def test_an_explicit_format_still_works(self, table: Path) -> None:
        pk.register_reader(pk.CSVPickReader(columns=COLUMNS, reader_name="my_lab"))
        pk.register_reader(_GreedyReader())
        assert len(pk.read(table, format="my_lab")[0]) == 3


class _Broken:
    def __init__(self) -> None:
        raise RuntimeError("no")


class _FakeEntryPoint:
    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self._value = value
        self.dist = type("Dist", (), {"name": "some-plugin"})()

    def load(self) -> Any:
        return self._value


class TestPluginDiscovery:
    def _entry_points(self, monkeypatch: pytest.MonkeyPatch, *points: Any) -> None:
        import specmod.picks as module  # noqa: PLC0415

        monkeypatch.setattr(module, "_plugins_loaded", False)
        monkeypatch.setattr(
            "importlib.metadata.entry_points",
            lambda group=None: list(points) if group == pk.ENTRY_POINT_GROUP else [],
        )

    def test_a_plugin_is_registered_on_first_use(
        self, monkeypatch: pytest.MonkeyPatch, table: Path
    ) -> None:
        factory = lambda: pk.CSVPickReader(columns=COLUMNS, reader_name="my_lab")  # noqa: E731
        self._entry_points(monkeypatch, _FakeEntryPoint("my_lab", factory))
        assert pk.detect_reader(table).name == "my_lab"

    def test_discovery_happens_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def factory() -> pk.PickReader:
            calls.append(1)
            return pk.CSVPickReader(columns=COLUMNS, reader_name="my_lab")

        self._entry_points(monkeypatch, _FakeEntryPoint("my_lab", factory))
        pk.load_plugins()
        pk.load_plugins()
        assert len(calls) == 1

    def test_a_broken_plugin_warns_and_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._entry_points(monkeypatch, _FakeEntryPoint("bad", _Broken))
        with pytest.warns(UserWarning, match="could not be registered"):
            pk.load_plugins()
        assert set(pk.PICK_READERS) == set(pk.BUILTIN_READERS)

    def test_the_warning_names_the_distribution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._entry_points(monkeypatch, _FakeEntryPoint("bad", _Broken))
        with pytest.warns(UserWarning, match="some-plugin"):
            pk.load_plugins()

    def test_a_broken_plugin_does_not_stop_a_good_one(
        self, monkeypatch: pytest.MonkeyPatch, table: Path
    ) -> None:
        factory = lambda: pk.CSVPickReader(columns=COLUMNS, reader_name="my_lab")  # noqa: E731
        self._entry_points(
            monkeypatch,
            _FakeEntryPoint("bad", _Broken),
            _FakeEntryPoint("my_lab", factory),
        )
        with pytest.warns(UserWarning, match="could not be registered"):
            pk.load_plugins()
        assert pk.detect_reader(table).name == "my_lab"

    def test_a_plugin_cannot_shadow_a_builtin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = lambda: pk.CSVPickReader(columns=COLUMNS, reader_name="snuffler")  # noqa: E731
        self._entry_points(monkeypatch, _FakeEntryPoint("snuffler", factory))
        with pytest.warns(UserWarning, match="could not be registered"):
            pk.load_plugins()
        assert pk.PICK_READERS["snuffler"] is pk.BUILTIN_READERS["snuffler"]


class TestSetPicksTakesAFormat:
    def test_a_registered_reader_reaches_the_stream(self, table: Path) -> None:
        import specmod.preprocess as pre  # noqa: PLC0415

        pk.register_reader(pk.CSVPickReader(columns=COLUMNS, reader_name="my_lab"))
        stream = obspy.Stream(
            [obspy.Trace(header={"network": "LV", "station": "L001", "location": ""})]
        )
        stream[0].stats["otime"] = obspy.UTCDateTime("2019-08-26T07:30:47")

        pre.set_picks(stream, table, format="my_lab")
        assert "p_time" in stream[0].stats
        assert "s_time" in stream[0].stats


def test_importing_specmod_does_not_load_plugins() -> None:
    """Discovery is lazy: a caller who never reads a pick pays nothing.

    Asserted as a subprocess because this module has already used the registry.
    """
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    code = "import specmod.picks as pk; print(pk._plugins_loaded)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
