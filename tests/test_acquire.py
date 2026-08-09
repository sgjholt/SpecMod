"""Acquisition: config parsing, the written layout, and the manifest.

Every test drives a fake client. Nothing here touches the network, which is
not merely politeness — a suite that fetches live can silently change its own
expected answer, which is the failure regression tests exist to catch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

obspy = pytest.importorskip("obspy")

from specmod.acquire import (  # noqa: E402
    AcquisitionConfig,
    EventSpec,
    StationSpec,
    WindowSpec,
    fetch,
    read_config,
    verify,
)
from specmod.datasets import EventDirectory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

EXPLICIT = {
    "origin": "2019-08-26T07:30:47.000000Z",
    "latitude": 53.785021,
    "longitude": -2.970780,
    "depth_km": 2.04,
}


class FakeClient:
    """The three FDSN calls :func:`fetch` makes, and a record of the arguments.

    Deliberately not a mock library: the assertions worth making are about
    *what was asked for* — the window, the radius conversion, the level — and a
    hand-written double makes those readable.
    """

    def __init__(self, *, n_traces: int = 2, events: Any = None) -> None:
        self.calls: dict[str, dict[str, Any]] = {}
        self._n_traces = n_traces
        self._events = events

    def get_events(self, **kwargs: Any) -> Any:
        self.calls["get_events"] = kwargs
        return self._events

    def get_stations(self, **kwargs: Any) -> Any:
        self.calls["get_stations"] = kwargs
        return obspy.read_inventory()  # ObsPy's bundled example inventory

    def get_waveforms(self, **kwargs: Any) -> Any:
        self.calls["get_waveforms"] = kwargs
        stream = obspy.read()  # three traces of example data
        return obspy.Stream(stream[: self._n_traces])


def _config(**overrides: Any) -> AcquisitionConfig:
    base: dict[str, Any] = {
        "name": "test_event",
        "event": EventSpec(**EXPLICIT),
        "stations": StationSpec(network="LV", channel="HH*"),
        "window": WindowSpec(before_origin_s=5.0, after_origin_s=60.0),
    }
    base.update(overrides)
    return AcquisitionConfig(**base)


class TestConfig:
    def test_the_shipped_pnr_config_parses(self) -> None:
        """It is the worked example, so it has to remain loadable."""
        config = read_config(ROOT / "datasets" / "pnr_2019.toml")
        assert config.name == "pnr_2019"
        assert config.event.origin == "2019-08-26T07:30:47.000000Z"
        assert config.event.catalogue_magnitude_type == "Mw"
        assert config.stations.max_radius_km == 30.0

    def test_it_matches_the_shipped_event(self) -> None:
        """The config and `datasets.PNR_2019` must describe the same event.

        They are two statements of the same hypocentre, so they can disagree.
        """
        from specmod.datasets import PNR_2019  # noqa: PLC0415

        resolved = read_config(ROOT / "datasets" / "pnr_2019.toml").event.resolved()
        assert resolved == PNR_2019

    def test_an_unknown_key_is_refused(self) -> None:
        """A typo in a config should not be silently ignored."""
        with pytest.raises(ValueError, match="unknown key"):
            AcquisitionConfig.from_dict({"name": "x", "stations_": {}})

    def test_a_config_needs_a_name(self) -> None:
        with pytest.raises(ValueError, match="needs a `name`"):
            AcquisitionConfig.from_dict({"data_centre": "IRIS"})

    def test_an_event_needs_an_id_or_a_full_hypocentre(self) -> None:
        with pytest.raises(ValueError, match="eventid"):
            EventSpec(latitude=53.0, longitude=-2.0)

    def test_a_window_must_have_length(self) -> None:
        with pytest.raises(ValueError, match="positive length"):
            WindowSpec(before_origin_s=0.0, after_origin_s=0.0)

    def test_an_unresolved_event_refuses_to_pretend(self) -> None:
        """`eventid` alone is a request, not an answer."""
        with pytest.raises(ValueError, match="not been resolved"):
            EventSpec(eventid="usgs:12345").resolved()


class TestFetch:
    def test_it_writes_the_event_directory_layout(self, tmp_path: Path) -> None:
        """A fetched event and a committed one must be read by the same code."""
        manifest = fetch(_config(), out=tmp_path, client=FakeClient())

        paths = EventDirectory(tmp_path / EXPLICIT["origin"])
        assert paths.is_present()
        assert paths.inventory.is_file()
        assert paths.picks.is_dir()
        assert len(list(paths.waveforms.iterdir())) == 2
        assert manifest["name"] == "test_event"

    def test_the_window_is_taken_relative_to_the_origin(self, tmp_path: Path) -> None:
        client = FakeClient()
        fetch(_config(), out=tmp_path, client=client)

        origin = obspy.UTCDateTime(EXPLICIT["origin"])
        asked = client.calls["get_waveforms"]
        assert asked["starttime"] == origin - 5.0
        assert asked["endtime"] == origin + 60.0

    def test_it_asks_for_the_response(self, tmp_path: Path) -> None:
        """Raw counts are only useful with the response stored beside them."""
        client = FakeClient()
        fetch(_config(), out=tmp_path, client=client)
        assert client.calls["get_stations"]["level"] == "response"

    def test_a_radius_is_converted_to_degrees_about_the_epicentre(
        self, tmp_path: Path
    ) -> None:
        """FDSN takes degrees; a config in kilometres is the friendlier unit."""
        client = FakeClient()
        fetch(
            _config(stations=StationSpec(max_radius_km=111.195)),
            out=tmp_path,
            client=client,
        )

        asked = client.calls["get_stations"]
        assert asked["maxradius"] == pytest.approx(1.0, rel=1e-6)
        assert asked["latitude"] == EXPLICIT["latitude"]

    def test_no_radius_means_no_geographic_filter(self, tmp_path: Path) -> None:
        """Sending a centre with no radius would silently narrow the request."""
        client = FakeClient()
        fetch(_config(stations=StationSpec()), out=tmp_path, client=client)
        assert "maxradius" not in client.calls["get_stations"]
        assert "latitude" not in client.calls["get_stations"]

    def test_it_does_not_consult_a_catalogue_for_an_explicit_event(
        self, tmp_path: Path
    ) -> None:
        client = FakeClient()
        fetch(_config(), out=tmp_path, client=client)
        assert "get_events" not in client.calls


class TestManifest:
    def test_it_records_what_was_asked_and_what_came_back(self, tmp_path: Path) -> None:
        manifest = fetch(_config(), out=tmp_path, client=FakeClient())

        assert manifest["data_centre"] == "IRIS"
        assert manifest["obspy_version"] == obspy.__version__
        assert manifest["fetched_at"].endswith("+00:00")
        # After wildcard expansion: the config says `HH*`, the manifest says
        # which channels that turned out to be.
        assert len(manifest["resolved"]["channels"]) == 2
        assert manifest["resolved"]["latitude"] == EXPLICIT["latitude"]

    def test_the_config_travels_with_the_data(self, tmp_path: Path) -> None:
        """A downloaded dataset has to be self-describing."""
        source = ROOT / "datasets" / "pnr_2019.toml"
        manifest = fetch(source, out=tmp_path, client=FakeClient())
        assert manifest["config"] == source.read_text()

    def test_it_is_written_beside_the_data(self, tmp_path: Path) -> None:
        fetch(_config(), out=tmp_path, client=FakeClient())
        written = json.loads((tmp_path / "manifest.json").read_text())
        assert written["name"] == "test_event"

    def test_every_written_file_is_hashed(self, tmp_path: Path) -> None:
        manifest = fetch(_config(), out=tmp_path, client=FakeClient())
        # Two waveforms plus the inventory.
        assert len(manifest["files"]) == 3
        assert all(len(d) == 64 for d in manifest["files"].values())


class TestVerify:
    def test_an_untouched_fetch_verifies(self, tmp_path: Path) -> None:
        fetch(_config(), out=tmp_path, client=FakeClient())
        assert verify(tmp_path) == []

    def test_an_edited_file_is_reported(self, tmp_path: Path) -> None:
        fetch(_config(), out=tmp_path, client=FakeClient())
        paths = EventDirectory(tmp_path / EXPLICIT["origin"])
        target = next(iter(paths.waveforms.iterdir()))
        target.write_bytes(target.read_bytes() + b"\0")

        problems = verify(tmp_path)
        assert len(problems) == 1
        assert problems[0].startswith("changed:")

    def test_a_deleted_file_is_reported(self, tmp_path: Path) -> None:
        fetch(_config(), out=tmp_path, client=FakeClient())
        EventDirectory(tmp_path / EXPLICIT["origin"]).inventory.unlink()

        problems = verify(tmp_path)
        assert len(problems) == 1
        assert problems[0].startswith("missing:")


class _Origin:
    def __init__(self, time: str, lat: float, lon: float, depth_m: float) -> None:
        self.time, self.latitude, self.longitude, self.depth = time, lat, lon, depth_m


class _Magnitude:
    def __init__(self, mag: float, kind: str) -> None:
        self.mag, self.magnitude_type = mag, kind


class _CatalogueEvent:
    def __init__(self, origin: _Origin, magnitude: _Magnitude | None) -> None:
        self.origins = [origin]
        self.magnitudes = [] if magnitude is None else [magnitude]
        self._magnitude = magnitude

    def preferred_origin(self) -> _Origin:
        return self.origins[0]

    def preferred_magnitude(self) -> _Magnitude | None:
        return self._magnitude


class TestResolvingAnEventId:
    """The branch that takes the hypocentre from the catalogue rather than the
    config, which is the one worth preferring — a retyped origin is a second
    source of truth that can disagree silently."""

    @staticmethod
    def _catalogue(n: int = 1, magnitude: _Magnitude | None = None) -> list[Any]:
        origin = _Origin("2020-03-18T13:09:31.0", 40.751, -112.078, 9200.0)
        return [_CatalogueEvent(origin, magnitude) for _ in range(n)]

    def test_it_fills_the_hypocentre_from_the_catalogue(self, tmp_path: Path) -> None:
        client = FakeClient(
            events=self._catalogue(magnitude=_Magnitude(5.7, "Mww")),
        )
        manifest = fetch(
            _config(event=EventSpec(eventid="uu60363602")),
            out=tmp_path,
            client=client,
        )

        assert client.calls["get_events"] == {"eventid": "uu60363602"}
        resolved = manifest["resolved"]
        assert resolved["origin"] == "2020-03-18T13:09:31.0"
        assert resolved["latitude"] == pytest.approx(40.751)
        # Metres in the catalogue, kilometres in this package.
        assert resolved["depth_km"] == pytest.approx(9.2)
        assert resolved["catalogue_magnitude"] == pytest.approx(5.7)
        assert resolved["catalogue_magnitude_type"] == "Mww"
        assert resolved["eventid"] == "uu60363602"

    def test_an_event_with_no_magnitude_still_resolves(self, tmp_path: Path) -> None:
        """Not every catalogue entry carries one, and that is not a failure."""
        manifest = fetch(
            _config(event=EventSpec(eventid="x")),
            out=tmp_path,
            client=FakeClient(events=self._catalogue()),
        )
        assert manifest["resolved"]["catalogue_magnitude"] is None

    @pytest.mark.parametrize("n", [0, 2])
    def test_an_ambiguous_id_is_refused(self, n: int, tmp_path: Path) -> None:
        """Silently taking the first of two would pick an event at random."""
        with pytest.raises(ValueError, match="must identify exactly one"):
            fetch(
                _config(event=EventSpec(eventid="ambiguous")),
                out=tmp_path,
                client=FakeClient(events=self._catalogue(n)),
            )
