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

from obspy.geodetics import gps2dist_azimuth, locations2degrees  # noqa: E402

from specmod.acquire import (  # noqa: E402
    _KM_PER_DEGREE_MAX,
    _KM_PER_DEGREE_MIN,
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

# Most tests here fetch without priority lists, and ObsPy's example inventory
# has four instruments at GR.FUR, so the co-sited warning fires all over the
# suite and says nothing about the test that tripped it. `TestChannelPriorities`
# asserts it with `pytest.warns`, which ignores this filter.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*carry more than one instrument:UserWarning"
)

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

    def __init__(
        self, *, n_traces: int = 2, events: Any = None, inventory: Any = None
    ) -> None:
        self.calls: dict[str, dict[str, Any]] = {}
        self._n_traces = n_traces
        self._events = events
        self._inventory = inventory

    def get_events(self, **kwargs: Any) -> Any:
        self.calls["get_events"] = kwargs
        return self._events

    def get_stations(self, **kwargs: Any) -> Any:
        self.calls["get_stations"] = kwargs
        if self._inventory is not None:
            return self._inventory
        return obspy.read_inventory()  # ObsPy's bundled example inventory

    def get_waveforms_bulk(self, bulk: Any) -> Any:
        self.calls["get_waveforms_bulk"] = {"bulk": list(bulk)}
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
        bulk = client.calls["get_waveforms_bulk"]["bulk"]
        assert bulk
        # `UTCDateTime` is not usable in a set, so compare elementwise.
        assert all(line[4] == origin - 5.0 for line in bulk)
        assert all(line[5] == origin + 60.0 for line in bulk)

    def test_the_waveforms_are_asked_for_by_name_not_by_wildcard(
        self, tmp_path: Path
    ) -> None:
        """The request that carries the radius.

        FDSN dataselect has no geographic parameters, so a wildcard waveform
        request is unbounded whatever the station query asked for. A live fetch
        for Magna with `max_radius_km = 50` returned three BH channels from a
        station 225 km out, with no metadata in the archived inventory, because
        the station pattern went to the wire verbatim.
        """
        client = FakeClient(inventory=_inventory_around_the_event())
        fetch(
            _config(stations=StationSpec(max_radius_km=50.0)),
            out=tmp_path,
            client=client,
        )

        bulk = client.calls["get_waveforms_bulk"]["bulk"]
        assert bulk
        assert not [line for line in bulk if "*" in line[:4] or "?" in line[:4]]

    def test_no_waveform_is_asked_for_without_metadata_beside_it(
        self, tmp_path: Path
    ) -> None:
        """The property the bulk request gives for free.

        Whatever the station query returned is exactly what is requested, so
        the archive cannot hold a trace whose station is missing from
        `inventory.xml`.
        """
        client = FakeClient()
        fetch(_config(stations=StationSpec(station="*")), out=tmp_path, client=client)

        inventory = obspy.read_inventory()
        available = {
            (network.code, station.code, channel.location_code or "--", channel.code)
            for network in inventory
            for station in network
            for channel in station
        }
        asked = {line[:4] for line in client.calls["get_waveforms_bulk"]["bulk"]}
        assert asked == available

    def test_a_channel_with_two_epochs_is_one_request(self, tmp_path: Path) -> None:
        """An instrument replaced mid-window appears twice in the inventory."""
        doubled = obspy.read_inventory() + obspy.read_inventory()
        client = FakeClient(inventory=doubled)
        fetch(_config(), out=tmp_path, client=client)

        bulk = client.calls["get_waveforms_bulk"]["bulk"]
        assert len(bulk) == len({line[:4] for line in bulk})

    def test_an_empty_station_query_says_so(self, tmp_path: Path) -> None:
        """Better than a bulk request with no lines, which FDSN rejects with a
        message about syntax rather than about the radius."""
        client = FakeClient(inventory=obspy.Inventory())
        with pytest.raises(ValueError, match="no channels matched"):
            fetch(
                _config(stations=StationSpec(network="LV", max_radius_km=1.0)),
                out=tmp_path,
                client=client,
            )

    def test_it_asks_for_the_response(self, tmp_path: Path) -> None:
        """Raw counts are only useful with the response stored beside them."""
        client = FakeClient()
        fetch(_config(), out=tmp_path, client=client)
        assert client.calls["get_stations"]["level"] == "response"

    def test_a_radius_is_asked_about_the_epicentre(self, tmp_path: Path) -> None:
        """FDSN takes degrees; a config in kilometres is the friendlier unit."""
        client = FakeClient(inventory=_inventory_around_the_event())
        fetch(
            _config(stations=StationSpec(max_radius_km=111.195)),
            out=tmp_path,
            client=client,
        )

        asked = client.calls["get_stations"]
        assert asked["maxradius"] == pytest.approx(1.0, rel=0.01)
        assert asked["latitude"] == EXPLICIT["latitude"]
        assert asked["longitude"] == EXPLICIT["longitude"]

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


def _inventory_around_the_event(*, north_km: float = 10.0) -> Any:
    """The example inventory, moved to sit near the test event.

    ObsPy's example stations are in Bavaria and the test event is in
    Lancashire, so any radius at all excludes every one of them. Each station is
    shifted onto the event's meridian, spread northwards, and the tests then
    measure what they built rather than assuming a kilometres-per-degree.
    """
    inventory = obspy.read_inventory()
    # By code, so that BW.RJOB's three epochs stay one place.
    codes = sorted({s.code for network in inventory for s in network})
    for network in inventory:
        for station in network:
            step = codes.index(station.code) + 1
            station.latitude = EXPLICIT["latitude"] + step * north_km / 111.0
            station.longitude = EXPLICIT["longitude"]
    return inventory


def _epicentral_km(station: Any) -> float:
    metres, _, _ = gps2dist_azimuth(
        EXPLICIT["latitude"], EXPLICIT["longitude"], station.latitude, station.longitude
    )
    return float(metres) / 1000.0


def _two_sensors_at_one_station() -> Any:
    """The example inventory, with GR.FUR's broadband and short period split.

    ``HH`` moves to location ``00`` and ``BH`` to ``10``, which is the layout
    that makes the two priority lists distinguishable: channel priorities rank
    within a location, location priorities rank across them.
    """
    inventory = obspy.read_inventory()
    for network in inventory:
        for station in network:
            for channel in station:
                if channel.code.startswith("HH"):
                    channel.location_code = "00"
                elif channel.code.startswith("BH"):
                    channel.location_code = "10"
    return inventory


def _asked_for(client: FakeClient) -> set[tuple[str, ...]]:
    return {line[:4] for line in client.calls["get_waveforms_bulk"]["bulk"]}


class TestRadius:
    """`max_radius_km` means kilometres, at every latitude.

    FDSN takes a radius in degrees of arc, and a degree is not a fixed number of
    kilometres: measured on WGS84 across latitude and azimuth it runs from
    110.574 km at the equator to 111.691 km at the poles. Converting with one
    constant moves the boundary by up to half a percent — ±0.3 km on a 50 km
    radius, ±2 km on 400 km — so the query is widened and the cut made here.
    """

    def test_the_cut_is_the_true_epicentral_distance(self, tmp_path: Path) -> None:
        inventory = _inventory_around_the_event(north_km=10.0)
        client = FakeClient(inventory=inventory)
        limit = 25.0
        fetch(
            _config(stations=StationSpec(max_radius_km=limit)),
            out=tmp_path,
            client=client,
        )

        expected = {
            station.code
            for network in inventory
            for station in network
            if _epicentral_km(station) <= limit
        }
        assert {line[1] for line in _asked_for(client)} == expected

    @pytest.mark.parametrize("limit", [9.9, 10.1])
    def test_a_station_either_side_of_the_boundary(
        self, limit: float, tmp_path: Path
    ) -> None:
        """The nearest station sits at ~10 km, so the two limits straddle it."""
        inventory = _inventory_around_the_event(north_km=10.0)
        client = FakeClient(inventory=inventory)
        nearest = min((s for network in inventory for s in network), key=_epicentral_km)
        inside = _epicentral_km(nearest) <= limit

        if not inside:
            with pytest.raises(ValueError, match="no channels matched"):
                fetch(
                    _config(stations=StationSpec(max_radius_km=limit)),
                    out=tmp_path,
                    client=client,
                )
            return

        fetch(
            _config(stations=StationSpec(max_radius_km=limit)),
            out=tmp_path,
            client=client,
        )
        assert {line[1] for line in _asked_for(client)} == {nearest.code}

    def test_the_station_query_is_widened_so_it_cannot_clip(
        self, tmp_path: Path
    ) -> None:
        """The server's degrees must not exclude what the exact cut would keep.

        Sent at the loosest kilometres-per-degree, so the margin is on the side
        of asking for a few stations too many and dropping them here.
        """
        client = FakeClient(inventory=_inventory_around_the_event())
        fetch(
            _config(stations=StationSpec(max_radius_km=50.0)),
            out=tmp_path,
            client=client,
        )

        sent = client.calls["get_stations"]["maxradius"]
        assert sent >= 50.0 / 110.574  # the tightest degree on the ellipsoid
        assert sent <= 50.0 / 110.0  # but not so loose as to be a free-for-all

    def test_a_minimum_radius_alone_is_honoured(self, tmp_path: Path) -> None:
        """It used to be sent only alongside a maximum, so on its own it did nothing."""
        inventory = _inventory_around_the_event(north_km=10.0)
        client = FakeClient(inventory=inventory)
        fetch(
            _config(stations=StationSpec(min_radius_km=25.0)),
            out=tmp_path,
            client=client,
        )

        assert "minradius" in client.calls["get_stations"]
        assert client.calls["get_stations"]["latitude"] == EXPLICIT["latitude"]
        expected = {
            station.code
            for network in inventory
            for station in network
            if _epicentral_km(station) >= 25.0
        }
        assert {line[1] for line in _asked_for(client)} == expected

    def test_a_local_array_is_cut_to_the_metre(self, tmp_path: Path) -> None:
        """A sub-kilometre deployment is a radius like any other.

        Induced-seismicity and mine arrays span hundreds of metres, which is
        where converting kilometres to degrees with one constant is worst in
        relative terms: at 500 m the equator-to-pole spread is 5 m, and station
        spacings are smaller than that.
        """
        inventory = _inventory_around_the_event(north_km=0.2)
        client = FakeClient(inventory=inventory)
        fetch(
            _config(stations=StationSpec(max_radius_km=0.5)),
            out=tmp_path,
            client=client,
        )

        expected = {
            station.code
            for network in inventory
            for station in network
            if _epicentral_km(station) <= 0.5
        }
        assert expected  # the fixture must straddle the limit, not sit inside it
        assert len(expected) < 3
        assert {line[1] for line in _asked_for(client)} == expected

    @pytest.mark.parametrize("radius", [0.0, -1.0])
    def test_a_radius_that_selects_nothing_is_refused(self, radius: float) -> None:
        """`max_radius_km = 0` is a typo, not a request for an empty dataset."""
        with pytest.raises(ValueError, match="selects nothing"):
            StationSpec(max_radius_km=radius)

    def test_an_empty_annulus_is_refused(self) -> None:
        with pytest.raises(ValueError, match="annulus is empty"):
            StationSpec(min_radius_km=50.0, max_radius_km=10.0)

    def test_the_conversion_bounds_bracket_every_latitude(self) -> None:
        """The two constants are claims about WGS84, so measure them.

        If a later edit tightens them towards the 111.195 they replaced, the
        widened query starts clipping stations the exact cut would have kept.
        """
        measured = []
        for lat in range(0, 89):
            for dlat, dlon in ((1.0, 0.0), (0.0, 1.0), (0.7, 0.7)):
                if lat + dlat > 89.5:
                    continue
                degrees = locations2degrees(lat, 0.0, lat + dlat, dlon)
                metres, _, _ = gps2dist_azimuth(lat, 0.0, lat + dlat, dlon)
                measured.append(metres / 1000.0 / degrees)

        assert min(measured) >= _KM_PER_DEGREE_MIN
        assert max(measured) <= _KM_PER_DEGREE_MAX


class TestChannelPriorities:
    """One instrument per station, rather than every instrument at every station.

    A station query for `HH*,BH*,HN*,EN*` returns a broadband *and* an
    accelerometer wherever both are installed, and the waveform request is built
    from that answer, so both are downloaded and both reach the pipeline — where
    they are two records of one ground motion, weighted as two stations by
    anything that averages over them.
    """

    def test_the_first_matching_pattern_wins(self, tmp_path: Path) -> None:
        client = FakeClient()
        fetch(
            _config(
                stations=StationSpec(channel_priorities=["HH[ZNE]", "BH[ZNE]"]),
            ),
            out=tmp_path,
            client=client,
        )

        codes = {line[3] for line in _asked_for(client)}
        assert codes == {"HHZ", "HHN", "HHE"}

    def test_a_station_without_the_first_choice_falls_through(
        self, tmp_path: Path
    ) -> None:
        """The point of a ranking: BW.RJOB has only EH, and is still fetched."""
        client = FakeClient()
        fetch(
            _config(stations=StationSpec(channel_priorities=["HH[ZNE]", "EH[ZNE]"])),
            out=tmp_path,
            client=client,
        )

        by_station = {}
        for network, station, _, channel in _asked_for(client):
            by_station.setdefault(f"{network}.{station}", set()).add(channel[:2])
        assert by_station == {
            "GR.FUR": {"HH"},
            "GR.WET": {"HH"},
            "BW.RJOB": {"EH"},
        }

    def test_a_station_matching_no_pattern_is_dropped(self, tmp_path: Path) -> None:
        """A priority list is a restriction as well as a ranking."""
        client = FakeClient()
        fetch(
            _config(stations=StationSpec(channel_priorities=["HH[ZNE]"])),
            out=tmp_path,
            client=client,
        )
        assert "BW" not in {line[0] for line in _asked_for(client)}

    def test_the_written_inventory_holds_only_what_was_fetched(
        self, tmp_path: Path
    ) -> None:
        """The invariant the bulk request gives, kept in the other direction.

        Pruning the inventory as well as the request is what keeps the archive
        describing exactly what it holds — no metadata for an instrument that
        was ranked out and never downloaded.
        """
        client = FakeClient()
        fetch(
            _config(stations=StationSpec(channel_priorities=["HH[ZNE]"])),
            out=tmp_path,
            client=client,
        )

        written = obspy.read_inventory(
            str(EventDirectory(tmp_path / EXPLICIT["origin"]).inventory)
        )
        on_disk = {
            (network.code, station.code, channel.location_code or "--", channel.code)
            for network in written
            for station in network
            for channel in station
        }
        assert on_disk == _asked_for(client)

    def test_channels_are_ranked_within_each_location(self, tmp_path: Path) -> None:
        """Two sensors at one station are two selections, not one.

        ObsPy's `MassDownloader` groups by location before ranking channels, so
        a config carried over from it selects the same instruments.
        """
        client = FakeClient(inventory=_two_sensors_at_one_station())
        fetch(
            _config(stations=StationSpec(channel_priorities=["HH[ZNE]", "BH[ZNE]"])),
            out=tmp_path,
            client=client,
        )

        fur = {line[2:4] for line in _asked_for(client) if line[1] == "FUR"}
        assert {loc for loc, _ in fur} == {"00", "10"}

    def test_location_priorities_pick_one_of_them(self, tmp_path: Path) -> None:
        client = FakeClient(inventory=_two_sensors_at_one_station())
        fetch(
            _config(
                stations=StationSpec(
                    channel_priorities=["HH[ZNE]", "BH[ZNE]"],
                    location_priorities=["10", "00"],
                )
            ),
            out=tmp_path,
            client=client,
        )

        fur = {line[2:4] for line in _asked_for(client) if line[1] == "FUR"}
        assert {loc for loc, _ in fur} == {"10"}
        assert {code[:2] for _, code in fur} == {"BH"}

    def test_a_blank_location_is_the_empty_pattern(self, tmp_path: Path) -> None:
        """`""` in the config, `--` on the wire: the two conventions must meet."""
        client = FakeClient()
        fetch(
            _config(stations=StationSpec(location_priorities=[""])),
            out=tmp_path,
            client=client,
        )
        assert {line[2] for line in _asked_for(client)} == {"--"}

    def test_priorities_that_match_nothing_say_so(self, tmp_path: Path) -> None:
        """Otherwise this is an empty bulk request, which FDSN rejects on syntax."""
        client = FakeClient()
        with pytest.raises(ValueError, match="matched none of them"):
            fetch(
                _config(stations=StationSpec(channel_priorities=["ZZ[ZNE]"])),
                out=tmp_path,
                client=client,
            )

    def test_the_patterns_are_matched_against_upper_case_codes(
        self, tmp_path: Path
    ) -> None:
        """A lower-case pattern matches nothing, and the error says why."""
        client = FakeClient()
        with pytest.raises(ValueError, match="upper case"):
            fetch(
                _config(stations=StationSpec(channel_priorities=["hh[zne]"])),
                out=tmp_path,
                client=client,
            )

    def test_an_unranked_fetch_warns_about_co_sited_instruments(
        self, tmp_path: Path
    ) -> None:
        """The trap this exists to close, named at the moment it is sprung."""
        with pytest.warns(UserWarning, match="more than one instrument"):
            fetch(_config(), out=tmp_path, client=FakeClient())

    def test_one_instrument_per_station_is_not_warned_about(
        self, tmp_path: Path, recwarn: Any
    ) -> None:
        only_rjob = obspy.read_inventory().select(network="BW")
        fetch(_config(), out=tmp_path, client=FakeClient(inventory=only_rjob))
        assert not [w for w in recwarn if "more than one instrument" in str(w.message)]

    def test_keeping_everything_deliberately_is_not_warned_about(
        self, tmp_path: Path, recwarn: Any
    ) -> None:
        """`["*"]` is how a config says it wants both instruments and means it."""
        client = FakeClient()
        fetch(
            _config(stations=StationSpec(channel_priorities=["*"])),
            out=tmp_path,
            client=client,
        )

        assert not [w for w in recwarn if "more than one instrument" in str(w.message)]
        assert len(_asked_for(client)) == 24

    def test_the_manifest_counts_what_was_narrowed_away(self, tmp_path: Path) -> None:
        manifest = fetch(
            _config(stations=StationSpec(channel_priorities=["HH[ZNE]"])),
            out=tmp_path,
            client=FakeClient(),
        )

        resolved = manifest["resolved"]
        assert resolved["channels_available"] == 24
        assert resolved["channels_requested"] == 6  # HH at FUR and WET
        assert len(resolved["channels"]) == 2  # what the fake archive serves

    @pytest.mark.parametrize("field", ["channel_priorities", "location_priorities"])
    def test_one_pattern_is_still_a_list(self, field: str) -> None:
        """`channel_priorities = "HH[ZNE]"` would otherwise rank by character."""
        with pytest.raises(ValueError, match="not one pattern"):
            StationSpec(**{field: "HH[ZNE]"})

    @pytest.mark.parametrize("field", ["channel_priorities", "location_priorities"])
    def test_an_empty_list_is_refused(self, field: str) -> None:
        with pytest.raises(ValueError, match="select nothing"):
            StationSpec(**{field: []})

    def test_a_config_carries_them(self) -> None:
        config = AcquisitionConfig.from_dict(
            {
                "name": "x",
                "event": EXPLICIT,
                "stations": {"channel_priorities": ["HH[ZNE]", "HN[ZNE]"]},
            }
        )
        assert config.stations.channel_priorities == ("HH[ZNE]", "HN[ZNE]")


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
        # The shipped config has a 30 km radius, and it is now cut exactly.
        manifest = fetch(
            source,
            out=tmp_path,
            client=FakeClient(inventory=_inventory_around_the_event()),
        )
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


class _Catalogue(list):  # type: ignore[type-arg]
    """A catalogue that can be written, as ObsPy's can.

    `fetch` now saves the QuakeML rather than reading six numbers off it and
    dropping the rest, so the double has to be writable.
    """

    def __init__(self, events: list[Any]) -> None:
        super().__init__(events)
        self.written: list[str] = []

    def write(self, filename: str, format: str = "QUAKEML", **kwargs: Any) -> None:
        self.written.append(filename)
        Path(filename).write_text("<q:quakeml/>")


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
    def _catalogue(n: int = 1, magnitude: _Magnitude | None = None) -> _Catalogue:
        origin = _Origin("2020-03-18T13:09:31.0", 40.751, -112.078, 9200.0)
        return _Catalogue([_CatalogueEvent(origin, magnitude) for _ in range(n)])

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


class TestSeparateCatalogueAndArchive:
    """Event ids are issued per catalogue, so the two centres can differ.

    A USGS ComCat id like ``uu60363602`` means nothing to IRIS, and the Magna
    config depends on being able to say so.
    """

    def test_the_shipped_magna_config_names_both(self) -> None:
        config = read_config(ROOT / "datasets" / "magna_2020.toml")
        assert config.data_centre == "IRIS"
        assert config.event.catalogue == "USGS"
        assert config.event.eventid == "uu60363602"
        assert config.stations.max_radius_km == 400.0

    def test_the_catalogue_is_asked_not_the_archive(self, tmp_path: Path) -> None:
        archive, catalogue = FakeClient(), FakeClient(events=_magna_catalogue())
        fetch(
            _config(event=EventSpec(eventid="uu60363602", catalogue="USGS")),
            out=tmp_path,
            client=archive,
            event_client=catalogue,
        )
        assert "get_events" in catalogue.calls
        assert "get_events" not in archive.calls
        # And the waveforms still come from the archive.
        assert "get_waveforms_bulk" in archive.calls

    def test_the_manifest_records_both(self, tmp_path: Path) -> None:
        manifest = fetch(
            _config(event=EventSpec(eventid="uu60363602", catalogue="USGS")),
            out=tmp_path,
            client=FakeClient(),
            event_client=FakeClient(events=_magna_catalogue()),
        )
        assert manifest["data_centre"] == "IRIS"
        assert manifest["event_catalogue"] == "USGS"

    def test_one_centre_is_reused_when_no_catalogue_is_named(
        self, tmp_path: Path
    ) -> None:
        """The common case stays a single client."""
        client = FakeClient(events=_magna_catalogue())
        fetch(_config(event=EventSpec(eventid="x")), out=tmp_path, client=client)
        assert "get_events" in client.calls
        assert "get_waveforms_bulk" in client.calls


def _magna_catalogue() -> _Catalogue:
    """The epicentre USGS gives for uu60363602, per its KML."""
    return _Catalogue(
        [
            _CatalogueEvent(
                _Origin("2020-03-18T13:09:31.0", 40.751, -112.0783333, 9200.0),
                _Magnitude(5.7, "mww"),
            )
        ]
    )
