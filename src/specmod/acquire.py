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

import hashlib
import json
import tomllib
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
    """

    network: str = "*"
    station: str = "*"
    location: str = "*"
    channel: str = "*"
    #: Kilometres from the epicentre. ``None`` means no limit.
    max_radius_km: float | None = None
    min_radius_km: float | None = None


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
) -> EventSpec:
    """Fill in the hypocentre from the catalogue when only an id was given."""
    if config.event.eventid is None:
        return config.event

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

    return replace(
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
    ``get_events``, ``get_stations`` and ``get_waveforms`` methods; tests pass a
    fake so that no test touches the network.
    """
    import obspy  # noqa: PLC0415

    if not isinstance(config, AcquisitionConfig):
        config = read_config(config)
    if client is None:
        client = _default_client(config.data_centre)

    spec = _resolve_event(config, client, event_client)
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
    if config.stations.max_radius_km is not None:
        station_kwargs["latitude"] = event.latitude
        station_kwargs["longitude"] = event.longitude
        station_kwargs["maxradius"] = config.stations.max_radius_km / 111.195
        if config.stations.min_radius_km is not None:
            station_kwargs["minradius"] = config.stations.min_radius_km / 111.195

    inventory = client.get_stations(**station_kwargs)
    stream = client.get_waveforms(
        network=config.stations.network,
        station=config.stations.station,
        location=config.stations.location,
        channel=config.stations.channel,
        starttime=start,
        endtime=end,
    )

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

    manifest = {
        "name": config.name,
        "fetched_at": datetime.now(UTC).isoformat(),
        "data_centre": config.data_centre,
        "event_catalogue": config.event.catalogue or config.data_centre,
        "specmod_version": __version__,
        "obspy_version": obspy.__version__,
        "config": config.source_toml,
        "resolved": {
            "origin": event.origin,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "depth_km": event.depth_km,
            "catalogue_magnitude": event.catalogue_magnitude,
            "catalogue_magnitude_type": event.catalogue_magnitude_type,
            "eventid": spec.eventid,
            # After wildcard expansion: the config alone does not say what the
            # request actually returned.
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
