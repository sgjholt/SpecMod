"""Fetch an event from an FDSN data centre into the layout tests and users read.

The request is declared in TOML and the response is written as an
:class:`specmod.datasets.EventDirectory`, beside a manifest recording what was
asked for and what came back::

    from specmod.acquire import fetch
    fetch("datasets/pnr_2019.toml", out="build/pnr_2019")

or ``specmod fetch datasets/pnr_2019.toml -o build/pnr_2019``.

**Waveforms are stored raw.** Counts and the response, never a deconvolved
trace: baking ``remove_response`` into the artefact takes it out of test
coverage and freezes one ObsPy version's behaviour into the fixture.

**A config makes the request reproducible, not the response.** FDSN is not
content-addressed — responses are corrected retroactively, archives are
backfilled, catalogue solutions revised. That is what :func:`verify` and the
manifest are for, and why published artefacts are pinned by hash rather than
re-fetched. See §5.2.2 of ``docs/REFACTOR_PLAN.md``.

Every network call goes through the ``client`` argument, which defaults to an
ObsPy FDSN client and is injected in tests. Nothing here calls the network on
import.
"""

from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
import tomllib
import warnings
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .datasets import Event, EventDirectory

__all__ = [
    "AcquisitionConfig",
    "EventSpec",
    "StationSpec",
    "WindowSpec",
    "fetch",
    "read_config",
    "verify",
]


@dataclass(frozen=True)
class EventSpec:
    """Which earthquake, and where its parameters come from.

    ``eventid`` resolves the hypocentre from the data centre's catalogue, which
    is preferable to retyping it: a retyped origin is a second source of truth
    that can disagree with the catalogue silently. The explicit fields are for
    events the catalogue does not carry — induced sequences monitored privately,
    most often — and one of the two must be given.
    """

    eventid: str | None = None
    #: FDSN service to resolve ``eventid`` against, when it is not the one
    #: serving the waveforms. Event ids are issued per catalogue — a USGS
    #: ComCat id means nothing to IRIS — so the two are genuinely separable
    #: and the config has to be able to say so.
    catalogue: str | None = None
    origin: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    depth_km: float | None = None
    catalogue_magnitude: float | None = None
    catalogue_magnitude_type: str | None = None

    def __post_init__(self) -> None:
        explicit = (self.origin, self.latitude, self.longitude, self.depth_km)
        if self.eventid is None and any(v is None for v in explicit):
            raise ValueError(
                "an event needs either `eventid`, to resolve from the "
                "catalogue, or all of `origin`, `latitude`, `longitude` and "
                "`depth_km`"
            )

    def resolved(self) -> Event:
        """The :class:`~specmod.datasets.Event` these fields describe."""
        if self.origin is None:
            raise ValueError(
                "this event is declared by eventid and has not been resolved "
                "against a catalogue yet"
            )
        assert self.latitude is not None
        assert self.longitude is not None
        assert self.depth_km is not None
        return Event(
            origin=self.origin,
            latitude=self.latitude,
            longitude=self.longitude,
            depth_km=self.depth_km,
            catalogue_magnitude=self.catalogue_magnitude,
            catalogue_magnitude_type=self.catalogue_magnitude_type,
        )


@dataclass(frozen=True)
class StationSpec:
    """Which channels to ask for.

    The patterns are FDSN wildcards, so the config alone does not say what you
    got — which is why the manifest records the channel list after expansion.

    ``channel`` says what the data centre may offer; the priority lists say
    which of what it offers to keep. A station carrying both a broadband and an
    accelerometer matches ``HH*,HN*`` twice, and both records are the same
    ground motion measured by two instruments — one station, not two
    observations. ``channel_priorities = ["HH[ZNE]", "HN[ZNE]"]`` keeps the
    broadband where there is one and the accelerometer where there is not.

    The radii are **epicentral**. For a local array the depth is usually the
    larger term — every station of a 500 m array over a 2 km-deep event is at
    2 km hypocentral distance — so a small ``max_radius_km`` selects on map
    distance, not on how far the waves travelled.
    """

    network: str = "*"
    station: str = "*"
    location: str = "*"
    channel: str = "*"
    #: Kilometres from the epicentre, cut against the true WGS84 distance
    #: rather than against the degrees FDSN takes. ``None`` means no limit.
    max_radius_km: float | None = None
    min_radius_km: float | None = None
    #: Channel-code patterns in descending preference, applied per station:
    #: every channel matching the first pattern that matches anything is kept
    #: and the rest are dropped. ``None`` keeps every channel the query
    #: returned; ``["*"]`` does too, and says so deliberately.
    channel_priorities: tuple[str, ...] | None = None
    #: Location codes in descending preference, ``""`` for a blank one. Applied
    #: after ``channel_priorities``, in the same first-match-wins way.
    location_priorities: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("max_radius_km", "min_radius_km"):
            radius = getattr(self, name)
            if radius is not None and radius <= 0:
                raise ValueError(
                    f"`{name}` is {radius} km, which selects nothing. Omit it "
                    f"for no limit. Note the unit is kilometres: a local array "
                    f"is `max_radius_km = 0.5`, not 500."
                )
        if (
            self.max_radius_km is not None
            and self.min_radius_km is not None
            and self.min_radius_km >= self.max_radius_km
        ):
            raise ValueError(
                f"`min_radius_km` ({self.min_radius_km}) is not inside "
                f"`max_radius_km` ({self.max_radius_km}), so the annulus is "
                f"empty"
            )

        for name in ("channel_priorities", "location_priorities"):
            patterns = getattr(self, name)
            if patterns is None:
                continue
            if isinstance(patterns, str):
                raise ValueError(
                    f"`{name}` is a list of patterns in descending preference, "
                    f"not one pattern: write [{patterns!r}] rather than "
                    f"{patterns!r}"
                )
            patterns = tuple(str(p) for p in patterns)
            if not patterns:
                raise ValueError(
                    f"`{name}` is empty, which would select nothing. Omit it to "
                    f"keep every channel the station query returns."
                )
            object.__setattr__(self, name, patterns)


@dataclass(frozen=True)
class WindowSpec:
    """How much record to take, relative to the origin time."""

    before_origin_s: float = 10.0
    after_origin_s: float = 120.0

    def __post_init__(self) -> None:
        if self.before_origin_s + self.after_origin_s <= 0:
            raise ValueError("the window must have positive length")


@dataclass(frozen=True)
class AcquisitionConfig:
    """A complete, declarative description of one fetch."""

    name: str
    #: FDSN data centre, by short name (``"IRIS"``) or base URL. Recorded in
    #: the manifest because different centres serve different holdings for the
    #: same event.
    data_centre: str = "IRIS"
    event: EventSpec = field(default_factory=EventSpec)
    stations: StationSpec = field(default_factory=StationSpec)
    window: WindowSpec = field(default_factory=WindowSpec)
    #: The TOML this was parsed from, kept verbatim for the manifest.
    source_toml: str = ""

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, source_toml: str = ""
    ) -> AcquisitionConfig:
        known = {"name", "data_centre", "event", "stations", "window"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) in acquisition config: {sorted(unknown)}. "
                f"Known: {sorted(known)}"
            )
        if "name" not in data:
            raise ValueError("an acquisition config needs a `name`")
        return cls(
            name=str(data["name"]),
            data_centre=str(data.get("data_centre", "IRIS")),
            event=EventSpec(**data.get("event", {})),
            stations=StationSpec(**data.get("stations", {})),
            window=WindowSpec(**data.get("window", {})),
            source_toml=source_toml,
        )


def read_config(path: str | Path) -> AcquisitionConfig:
    """Parse an acquisition config, keeping the text for the manifest."""
    text = Path(path).read_text()
    return AcquisitionConfig.from_dict(tomllib.loads(text), source_toml=text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_client(data_centre: str) -> Any:
    from obspy.clients.fdsn import Client  # noqa: PLC0415

    return Client(data_centre)


def _resolve_event(
    config: AcquisitionConfig, client: Any, event_client: Any = None
) -> tuple[EventSpec, Any]:
    """Fill in the hypocentre from the catalogue when only an id was given.

    Returns the resolved spec **and the catalogue it came from**, so the caller
    can write the QuakeML rather than keeping only the handful of numbers this
    reads off it.
    """
    if config.event.eventid is None:
        return config.event, None

    if event_client is None:
        event_client = (
            client
            if config.event.catalogue in (None, config.data_centre)
            else _default_client(config.event.catalogue)
        )
    catalogue = event_client.get_events(eventid=config.event.eventid)
    if len(catalogue) != 1:
        raise ValueError(
            f"eventid {config.event.eventid!r} matched {len(catalogue)} events; "
            f"it must identify exactly one"
        )
    origin = catalogue[0].preferred_origin() or catalogue[0].origins[0]
    magnitude = catalogue[0].preferred_magnitude()
    magnitudes = catalogue[0].magnitudes
    if magnitude is None and magnitudes:
        magnitude = magnitudes[0]

    resolved = replace(
        config.event,
        origin=str(origin.time),
        latitude=float(origin.latitude),
        longitude=float(origin.longitude),
        depth_km=float(origin.depth) / 1000.0,
        catalogue_magnitude=None if magnitude is None else float(magnitude.mag),
        catalogue_magnitude_type=(
            None if magnitude is None else str(magnitude.magnitude_type)
        ),
    )
    return resolved, catalogue


#: Kilometres of WGS84 surface distance per degree of great-circle arc. A
#: degree is not a fixed number of kilometres — measured across latitude and
#: azimuth it runs from 110.574 km at the equator to 111.691 km at the poles, a
#: spread of 1.01% — so these bracket it rather than approximating it. The
#: station query is widened to the loose end of the bracket and the radius is
#: then cut exactly; see :func:`_within_radius`.
_KM_PER_DEGREE_MIN = 110.5
_KM_PER_DEGREE_MAX = 111.8


def _location(channel: Any) -> str:
    """A channel's location code, with a blank one as the empty string."""
    return channel.location_code or ""


def _pruned(inventory: Any, select: Any) -> Any:
    """A copy of the inventory holding only ``select(station)`` at each station.

    Stations left with no channel are dropped, and so are networks left with no
    station: the waveform request is built from the inventory, so pruning it is
    both how a selection reaches the download and how the archive is kept to
    describing exactly what it holds.
    """
    out = inventory.copy()
    for network in out:
        for station in network:
            station.channels = select(station)
        network.stations = [s for s in network.stations if s.channels]
    out.networks = [n for n in out.networks if n.stations]
    return out


def _within_radius(inventory: Any, config: AcquisitionConfig, event: Any) -> Any:
    """Drop stations outside the configured radius, measured on WGS84.

    FDSN takes a radius in degrees of arc, so a config in kilometres has to be
    converted, and no conversion is exact: a degree of arc is 110.574 km at the
    equator and 111.691 km at the poles. Converting with a single constant
    therefore moves the boundary by up to half a percent with latitude — ±0.3 km
    on a 50 km radius, ±2 km on 400 km.

    So the conversion is not relied on. The station query is widened by the
    spread, which costs a few extra stations in the margin, and the cut is made
    here against the true epicentral distance.
    """
    from obspy.geodetics import gps2dist_azimuth  # noqa: PLC0415

    spec = config.stations
    if spec.max_radius_km is None and spec.min_radius_km is None:
        return inventory

    def inside(station: Any) -> bool:
        metres, _, _ = gps2dist_azimuth(
            event.latitude, event.longitude, station.latitude, station.longitude
        )
        km = metres / 1000.0
        if spec.max_radius_km is not None and km > spec.max_radius_km:
            return False
        return not (spec.min_radius_km is not None and km < spec.min_radius_km)

    return _pruned(inventory, lambda s: list(s) if inside(s) else [])


def _channel_ids(inventory: Any) -> set[tuple[str, str, str, str]]:
    """``(network, station, location, channel)`` for each channel, epochs collapsed.

    An instrument replaced mid-window appears twice in the inventory and is one
    channel here, because it is one waveform request.
    """
    return {
        (network.code, station.code, _location(channel) or "--", channel.code)
        for network in inventory
        for station in network
        for channel in station
    }


def _by_priority(
    channels: list[Any], code: Any, patterns: tuple[str, ...]
) -> list[Any]:
    """Every channel matching the first pattern that matches anything.

    Once a pattern matches, the later ones are not considered — that is what
    makes the list a ranking rather than a filter. Channels matching no pattern
    at all are dropped, so a priority list is also a restriction.

    Patterns are matched case-sensitively against SEED codes, which are upper
    case.
    """
    for pattern in patterns:
        matched = [c for c in channels if fnmatch.fnmatchcase(code(c), pattern)]
        if matched:
            return matched
    return []


def _preferred_channels(station: Any, spec: StationSpec) -> list[Any]:
    """The channels at one station that survive its priority lists.

    Channel priorities rank within each location code, then location priorities
    rank across what is left. That is the order ObsPy's ``MassDownloader``
    applies them in, so a priority list carried over from it selects the same
    instrument here.
    """
    channels = list(station)
    if spec.channel_priorities is not None:
        ranked: list[Any] = []
        for _, group in itertools.groupby(sorted(channels, key=_location), _location):
            ranked += _by_priority(
                list(group), lambda c: c.code, spec.channel_priorities
            )
        channels = ranked
    if spec.location_priorities is not None:
        channels = _by_priority(channels, _location, spec.location_priorities)
    return channels


def _warn_about_co_sited_instruments(inventory: Any) -> None:
    """Say so when a station carries more than one instrument and nothing ranks them.

    Two instruments at one station are two recordings of the same ground
    motion. Downstream they are two entries in a ``SpectrumSet``, weighted as
    two stations by anything that averages over them.
    """
    crowded: dict[str, list[str]] = {}
    for network in inventory:
        for station in network:
            groups = sorted({f"{_location(c) or '--'}.{c.code[:2]}" for c in station})
            if len(groups) > 1:
                crowded[f"{network.code}.{station.code}"] = groups
    if not crowded:
        return

    examples = "; ".join(
        f"{name} has {', '.join(groups)}"
        for name, groups in sorted(crowded.items())[:3]
    )
    warnings.warn(
        f"{len(crowded)} station(s) carry more than one instrument, and every "
        f"channel of each will be fetched: {examples}. Rank them with "
        f"`channel_priorities` in [stations] — e.g. "
        f'["HH[ZNE]", "BH[ZNE]", "HN[ZNE]"] — or set it to ["*"] to keep them '
        f"all deliberately.",
        stacklevel=3,
    )


def _apply_priorities(inventory: Any, config: AcquisitionConfig) -> Any:
    """Prune the station query's answer to the instruments the config prefers.

    The waveform request is built from the inventory, so pruning it is what
    makes the priorities reach the download. The inventory is pruned rather
    than merely consulted so that the archive keeps describing exactly what it
    holds — a station with no selected channel is not written at all.
    """
    spec = config.stations
    if spec.channel_priorities is None and spec.location_priorities is None:
        _warn_about_co_sited_instruments(inventory)
        return inventory

    pruned = _pruned(inventory, lambda s: _preferred_channels(s, spec))

    available = len(_channel_ids(inventory))
    if available and not _channel_ids(pruned):
        named = ", ".join(
            f"{name}={list(value)}"
            for name, value in (
                ("channel_priorities", spec.channel_priorities),
                ("location_priorities", spec.location_priorities),
            )
            if value is not None
        )
        raise ValueError(
            f"the station query returned {available} channel(s) and {named} "
            f"matched none of them, so there are no waveforms to ask for. The "
            f"patterns are matched case-sensitively against SEED codes, which "
            f"are upper case."
        )
    return pruned


def _bulk_request(
    inventory: Any, config: AcquisitionConfig, start: Any, end: Any
) -> list[tuple[Any, ...]]:
    """One request line per channel the inventory holds.

    The waveforms are asked for by name, from the inventory the station query
    returned, rather than by repeating the config's wildcards. FDSN dataselect
    takes no geographic parameters, so a wildcard request is unbounded however
    the stations were selected: ``max_radius_km`` reaches ``get_stations`` and
    can never reach the waveform call. Asking channel by channel is what makes
    the radius apply to both, and it is also what guarantees every trace in the
    archive has metadata beside it.

    Channel epochs are collapsed. An instrument replaced mid-window appears
    twice in the inventory and is one request.
    """
    lines = _channel_ids(inventory)
    if not lines:
        raise ValueError(
            f"no channels matched {config.stations.network}."
            f"{config.stations.station}.{config.stations.location}."
            f"{config.stations.channel}"
            + (
                ""
                if config.stations.max_radius_km is None
                else f" within {config.stations.max_radius_km} km"
            )
            + f" at {config.data_centre}, so there are no waveforms to ask for."
        )
    return [(*line, start, end) for line in sorted(lines)]


def fetch(
    config: str | Path | AcquisitionConfig,
    out: str | Path,
    *,
    client: Any = None,
    event_client: Any = None,
) -> dict[str, Any]:
    """Fetch one event and write it as an :class:`EventDirectory`.

    Returns the manifest, which is also written to ``manifest.json`` beside the
    data. ``client`` accepts anything with the ObsPy FDSN client's
    ``get_events``, ``get_stations`` and ``get_waveforms_bulk`` methods; tests
    pass a fake so that no test touches the network.

    The waveform request is built from the station query's answer, not from the
    config's wildcards — see :func:`_bulk_request` — after
    ``[stations] channel_priorities`` has narrowed that answer to one instrument
    per station. With no priorities set, a station carrying two instruments is
    fetched twice and a warning says so.
    """
    import obspy  # noqa: PLC0415

    if not isinstance(config, AcquisitionConfig):
        config = read_config(config)
    if client is None:
        client = _default_client(config.data_centre)

    spec, catalogue = _resolve_event(config, client, event_client)
    event = spec.resolved()
    origin_time = obspy.UTCDateTime(event.origin)
    start = origin_time - config.window.before_origin_s
    end = origin_time + config.window.after_origin_s

    station_kwargs: dict[str, Any] = {
        "network": config.stations.network,
        "station": config.stations.station,
        "location": config.stations.location,
        "channel": config.stations.channel,
        "starttime": start,
        "endtime": end,
        "level": "response",
    }
    # Widened in both directions, because the exact cut is made against the
    # true epicentral distance below and this query must not pre-empt it.
    if config.stations.max_radius_km is not None:
        station_kwargs["latitude"] = event.latitude
        station_kwargs["longitude"] = event.longitude
        station_kwargs["maxradius"] = config.stations.max_radius_km / _KM_PER_DEGREE_MIN
    if config.stations.min_radius_km is not None:
        station_kwargs["latitude"] = event.latitude
        station_kwargs["longitude"] = event.longitude
        station_kwargs["minradius"] = config.stations.min_radius_km / _KM_PER_DEGREE_MAX

    inventory = client.get_stations(**station_kwargs)
    available = len(_channel_ids(inventory))
    inventory = _within_radius(inventory, config, event)
    inventory = _apply_priorities(inventory, config)
    bulk = _bulk_request(inventory, config, start, end)
    stream = client.get_waveforms_bulk(bulk)

    paths = EventDirectory(Path(out) / event.origin)
    paths.waveforms.mkdir(parents=True, exist_ok=True)
    paths.stations.mkdir(parents=True, exist_ok=True)
    paths.picks.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for trace in stream:
        # One file per channel, named as the shipped data is, so a fetched
        # event and a committed one are read by the same code.
        name = f"{trace.id}_{trace.stats.starttime}"
        target = paths.waveforms / name
        trace.write(str(target), format="MSEED")
        written[str(target.relative_to(Path(out)))] = _sha256(target)

    inventory.write(str(paths.inventory), format="STATIONXML")
    written[str(paths.inventory.relative_to(Path(out)))] = _sha256(paths.inventory)

    # The catalogue in full, not just the six numbers read off it above.
    # Origin uncertainties, every magnitude rather than the preferred one,
    # agency and evaluation status are all in here and nowhere else.
    if catalogue is not None:
        catalogue.write(str(paths.quakeml), format="QUAKEML")
        written[str(paths.quakeml.relative_to(Path(out)))] = _sha256(paths.quakeml)

    manifest = {
        "name": config.name,
        "fetched_at": datetime.now(UTC).isoformat(),
        "data_centre": config.data_centre,
        "event_catalogue": config.event.catalogue or config.data_centre,
        "specmod_version": __version__,
        "obspy_version": obspy.__version__,
        "config": config.source_toml,
        "quakeml": catalogue is not None,
        "resolved": {
            "origin": event.origin,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "depth_km": event.depth_km,
            "catalogue_magnitude": event.catalogue_magnitude,
            "catalogue_magnitude_type": event.catalogue_magnitude_type,
            "eventid": spec.eventid,
            # Three numbers, because they can differ: what the station query
            # offered, what the priority lists narrowed it to, and what the
            # archive actually served. The config alone says none of them.
            "channels_available": available,
            "channels_requested": len(bulk),
            "channels": sorted({trace.id for trace in stream}),
            "window": {"start": str(start), "end": str(end)},
        },
        "files": dict(sorted(written.items())),
    }
    (Path(out) / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def verify(out: str | Path) -> list[str]:
    """Re-hash what is on disk and report anything that no longer matches.

    Integrity only: it says whether the files changed since they were written,
    not whether the data centre has revised its holdings. That needs a re-fetch
    and a diff against the manifest, which is the fuller ``--verify`` §5.2.2
    describes and which needs the network.
    """
    root = Path(out)
    manifest = json.loads((root / "manifest.json").read_text())

    problems = []
    for name, digest in manifest["files"].items():
        path = root / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif _sha256(path) != digest:
            problems.append(f"changed: {name}")
    return problems
