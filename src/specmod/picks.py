"""Phase arrivals: what was picked, on which sensor, and which trace it reaches.

Reading a pick file is the easy half — ObsPy parses ten formats that carry
arrivals. This module is the other half: turning what a format supplied into an
arrival attached to the right trace, and failing loudly where that cannot be
decided.

Three rules do the work, each replacing a silent wrong answer:

* **Event selection.** A file may hold several events. One is chosen
  explicitly; ambiguity raises rather than merging them.
* **Sensor matching.** A pick carries as much identity as its format had. It
  matches a trace when every field it *specifies* agrees, with more than one
  match an error rather than a broadcast.
* **Duplicate resolution.** Several picks for one sensor and phase are reduced
  by a named policy, recorded in the resolution summary.

See §4.9 of ``docs/REFACTOR_PLAN.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import obspy
from obspy.core.event import Catalog

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Sequence
    from os import PathLike

    from obspy import UTCDateTime
    from obspy.core.event import Event as ObsPyEvent

__all__ = [
    "AmbiguousPolicy",
    "DuplicatePolicy",
    "Pick",
    "PickSet",
    "Resolution",
    "SensorID",
    "from_catalog",
    "read",
    "resolve",
    "select_event",
]

#: How an unset location code is spelled in a sensor key. Snuffler marker files
#: use it, and it keeps the key's field count fixed.
EMPTY_LOCATION = "--"

DuplicatePolicy = Literal["prefer_reviewed", "earliest", "highest_weight", "error"]
AmbiguousPolicy = Literal["error", "broadcast", "skip"]


@dataclass(frozen=True, order=True)
class SensorID:
    """A sensor identity, possibly partial.

    ``None`` means *the source did not say*, and matches anything. That is a
    different claim from an empty string, which means *stated, and empty*, and
    keeping them apart is what lets a pick carrying only a station code reach
    the right trace instead of no trace at all.

    The asymmetry between the two optional fields is deliberate. An empty
    network code is not a valid SEED network, so a format supplying one has
    said nothing and it is read as ``None``; an empty location code is the
    ordinary case for a single-sensor station, so it is kept as ``""``.
    """

    network: str | None
    station: str
    location: str | None

    @classmethod
    def parse(cls, text: str) -> SensorID:
        """Build from ``NET.STA.LOC``, reading ``--`` as an empty location."""
        parts = text.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"sensor id {text!r} is not NET.STA.LOC (got {len(parts)} fields)"
            )
        network, station, location = parts
        return cls(
            network=network or None,
            station=station,
            location="" if location == EMPTY_LOCATION else location,
        )

    def __str__(self) -> str:
        network = self.network if self.network is not None else "*"
        location = "*" if self.location is None else self.location or EMPTY_LOCATION
        return f"{network}.{self.station}.{location}"

    @property
    def is_complete(self) -> bool:
        """Whether every field is stated, so this names exactly one sensor."""
        return self.network is not None and self.location is not None

    def matches(self, other: SensorID) -> bool:
        """Whether this identity is consistent with ``other``.

        Every field this one specifies must agree. Fields left ``None`` do not
        constrain the match, so a partial identity can match several sensors —
        which is a condition :func:`resolve` reports rather than resolves.
        """
        return all(
            mine is None or theirs is None or mine == theirs
            for mine, theirs in (
                (self.station, other.station),
                (self.network, other.network),
                (self.location, other.location),
            )
        )


@dataclass(frozen=True)
class Pick:
    """One phase arrival, with whatever provenance its format carried.

    Only ``sensor``, ``phase`` and ``time`` are always present. The rest is
    absent wherever the source format has no field for it, and is what a
    :class:`DuplicatePolicy` uses to choose between competing picks.
    """

    sensor: SensorID
    phase: str
    time: UTCDateTime
    raw_phase: str | None = None
    uncertainty: float | None = None
    polarity: str | None = None
    weight: float | None = None
    automatic: bool | None = None
    reviewed: bool | None = None
    channel: str | None = None
    author: str | None = None


@dataclass(frozen=True)
class PickSet:
    """The picks of a single event."""

    picks: tuple[Pick, ...] = ()
    event_id: str | None = None
    origin: UTCDateTime | None = None

    def __len__(self) -> int:
        return len(self.picks)

    def __iter__(self) -> Any:
        return iter(self.picks)

    def sensors(self) -> tuple[SensorID, ...]:
        """The distinct sensor identities picked, in sorted order."""
        return tuple(sorted({pick.sensor for pick in self.picks}))

    def mapping(
        self, *, duplicates: DuplicatePolicy = "prefer_reviewed"
    ) -> dict[str, dict[str, UTCDateTime]]:
        """``{"NET.STA.LOC": {"P": UTCDateTime}}``, as the readers once returned.

        Groups on the pick's own identity without matching it against any
        stream, so a partial identity keys on ``*`` and will not equal a
        trace's key. :func:`resolve` is what reconciles the two.
        """
        out: dict[str, dict[str, UTCDateTime]] = {}
        for (sensor, phase), picks in _grouped(self.picks).items():
            chosen = _choose(picks, duplicates, sensor, phase)
            out.setdefault(str(sensor), {})[phase] = chosen.time
        return out


@dataclass(frozen=True)
class Resolution:
    """The outcome of matching a :class:`PickSet` against a set of sensors."""

    #: Attached picks, keyed by the *sensor's* complete id, then by phase.
    attached: dict[str, dict[str, Pick]] = field(default_factory=dict)
    #: Picks that matched no sensor.
    unused: tuple[Pick, ...] = ()
    #: Picks skipped because they matched several sensors.
    ambiguous: tuple[Pick, ...] = ()
    #: ``(sensor, phase)`` where a policy chose between competing picks.
    duplicated: tuple[tuple[str, str], ...] = ()

    @property
    def n_attached(self) -> int:
        return sum(len(phases) for phases in self.attached.values())

    def summary(self) -> str:
        """One line, suitable for printing or asserting on."""
        return (
            f"{self.n_attached} attached to {len(self.attached)} sensors, "
            f"{len(self.unused)} unused, {len(self.ambiguous)} ambiguous, "
            f"{len(self.duplicated)} resolved by policy"
        )


def _grouped(picks: Iterable[Pick]) -> dict[tuple[SensorID, str], list[Pick]]:
    grouped: dict[tuple[SensorID, str], list[Pick]] = {}
    for pick in picks:
        grouped.setdefault((pick.sensor, pick.phase), []).append(pick)
    return grouped


def _choose(
    picks: Sequence[Pick], policy: DuplicatePolicy, sensor: Any, phase: str
) -> Pick:
    """Reduce competing picks for one sensor and phase to one."""
    if len(picks) == 1:
        return picks[0]

    if policy == "error":
        times = ", ".join(str(pick.time) for pick in picks)
        raise ValueError(
            f"{sensor} has {len(picks)} {phase} picks ({times}) and "
            f"duplicates='error'. Choose a policy, or drop the extras."
        )
    if policy == "earliest":
        return min(picks, key=lambda pick: pick.time)
    if policy == "highest_weight":
        # An absent weight loses to any stated one, and ties break on time so
        # the choice does not depend on file order.
        return max(
            picks,
            key=lambda pick: (
                pick.weight if pick.weight is not None else float("-inf"),
                -float(pick.time.timestamp),
            ),
        )
    # prefer_reviewed: an analyst's pick beats an automatic one, then earliest.
    return min(
        picks,
        key=lambda pick: (
            0 if pick.reviewed else 1,
            0 if pick.automatic is False else 1,
            pick.time,
        ),
    )


def select_event(
    sets: Sequence[PickSet],
    *,
    event_id: str | None = None,
    near: UTCDateTime | None = None,
    tolerance_s: float = 60.0,
) -> PickSet:
    """Choose one :class:`PickSet` from a file that may hold several.

    With neither selector, the file must hold exactly one event. Merging them
    is never the answer: a bulletin holds unrelated earthquakes, and combining
    their arrivals produces a set that describes none of them.
    """
    if not sets:
        raise ValueError("no events in this source")

    if event_id is not None:
        matched = [s for s in sets if s.event_id == event_id]
        if len(matched) != 1:
            known = ", ".join(str(s.event_id) for s in sets)
            raise ValueError(
                f"event_id={event_id!r} matched {len(matched)} events. "
                f"Available: {known}."
            )
        return matched[0]

    if near is not None:
        timed = [s for s in sets if s.origin is not None]
        matched = [s for s in timed if abs(s.origin - near) <= tolerance_s]  # type: ignore[operator]
        if len(matched) != 1:
            raise ValueError(
                f"{len(matched)} of {len(sets)} events lie within "
                f"{tolerance_s} s of {near} ({len(sets) - len(timed)} carry no "
                f"origin time). Widen or narrow the tolerance, or use event_id."
            )
        return matched[0]

    if len(sets) > 1:
        known = ", ".join(f"{s.event_id or '<no id>'}@{s.origin}" for s in sets)
        raise ValueError(
            f"this source holds {len(sets)} events and no selector was given: "
            f"{known}. Pass event_id= or near=."
        )
    return sets[0]


def resolve(
    picks: PickSet,
    sensors: Iterable[SensorID],
    *,
    on_ambiguous: AmbiguousPolicy = "error",
    duplicates: DuplicatePolicy = "prefer_reviewed",
) -> Resolution:
    """Match a set of picks against the sensors actually present.

    ``sensors`` are complete identities, as built from a stream. A pick reaches
    exactly one of them, none — in which case it is unused — or several, which
    is ``on_ambiguous``:

    ``error``
        Raise, naming the candidates. The default, because two sensors at one
        site see genuinely different arrivals and giving one the other's pick
        is not recoverable downstream.
    ``skip``
        Leave the pick unattached and report it.
    ``broadcast``
        Attach to every match. Correct only where a site's sensors are known to
        be co-located.
    """
    # Deduplicated: a stream carries one trace per component, so the same
    # sensor arrives two or three times. Ambiguity is about distinct sensors —
    # counting components would make every ordinary stream ambiguous.
    targets = list(dict.fromkeys(sensors))
    attached: dict[str, dict[str, Pick]] = {}
    unused: list[Pick] = []
    ambiguous: list[Pick] = []
    duplicated: list[tuple[str, str]] = []

    for (sensor, phase), competing in _grouped(picks.picks).items():
        matched = [target for target in targets if sensor.matches(target)]

        if not matched:
            unused.extend(competing)
            continue

        if len(matched) > 1:
            if on_ambiguous == "error":
                names = ", ".join(str(target) for target in matched)
                missing = [
                    name
                    for name, value in (
                        ("network", sensor.network),
                        ("location code", sensor.location),
                    )
                    if value is None
                ]
                raise ValueError(
                    f"the {phase} pick for {sensor} matches {len(matched)} "
                    f"sensors ({names}). It states no "
                    f"{' and no '.join(missing)}, which is what would "
                    f"distinguish them. Pass on_ambiguous='broadcast' if the "
                    f"sensors are co-located, or 'skip' to drop it."
                )
            if on_ambiguous == "skip":
                ambiguous.extend(competing)
                continue

        chosen = _choose(competing, duplicates, sensor, phase)
        if len(competing) > 1:
            duplicated.append((str(sensor), phase))

        for target in matched:
            key = str(target)
            attached.setdefault(key, {})[phase] = replace(chosen, sensor=target)

    return Resolution(
        attached=attached,
        unused=tuple(unused),
        ambiguous=tuple(ambiguous),
        duplicated=tuple(duplicated),
    )


# ------------------------------------------------------------------ readers


#: Phase hints folded to their first letter. ``Pg``/``Pn``/``Pb`` and their S
#: counterparts are all direct-arrival branches this pipeline does not
#: distinguish; the original is kept on the pick as ``raw_phase``.
def _fold(hint: str | None) -> str | None:
    if not hint:
        return None
    first = hint.strip()[:1].upper()
    return first if first in ("P", "S") else None


def _sensor_from_waveform_id(waveform_id: Any) -> SensorID | None:
    station = (getattr(waveform_id, "station_code", None) or "").strip()
    if not station:
        return None
    network = (getattr(waveform_id, "network_code", None) or "").strip()
    location = getattr(waveform_id, "location_code", None)
    return SensorID(
        # An empty network code is not a valid SEED network, so it states
        # nothing; an empty location code is the normal single-sensor case.
        network=network or None,
        station=station,
        location=location if location is not None else None,
    )


def _pick_from_obspy(source: Any) -> Pick | None:
    phase = _fold(getattr(source, "phase_hint", None))
    if phase is None:
        return None
    if getattr(source, "evaluation_status", None) == "rejected":
        return None
    sensor = _sensor_from_waveform_id(getattr(source, "waveform_id", None))
    if sensor is None:
        return None

    mode = getattr(source, "evaluation_mode", None)
    status = getattr(source, "evaluation_status", None)
    creation = getattr(source, "creation_info", None)
    return Pick(
        sensor=sensor,
        phase=phase,
        time=source.time,
        raw_phase=(source.phase_hint or "").strip() or None,
        uncertainty=getattr(source, "time_errors", None)
        and getattr(source.time_errors, "uncertainty", None),
        polarity=getattr(source, "polarity", None),
        automatic=None if mode is None else mode == "automatic",
        reviewed=None if status is None else status in ("reviewed", "confirmed"),
        channel=getattr(getattr(source, "waveform_id", None), "channel_code", None),
        author=getattr(creation, "author", None) if creation is not None else None,
    )


def from_catalog(catalog: Catalog | ObsPyEvent) -> list[PickSet]:
    """Every event in an ObsPy catalogue, as one :class:`PickSet` each.

    The input may come from any format :func:`obspy.read_events` understands,
    which is what makes the standard roster — QuakeML, SC3ML, SEISAN Nordic,
    NonLinLoc, HypoDD, IMS/GSE bulletins — readable without a parser here.
    """
    events = list(catalog) if isinstance(catalog, Catalog) else [catalog]

    sets = []
    for event in events:
        picks = tuple(
            pick
            for pick in (_pick_from_obspy(raw) for raw in event.picks)
            if pick is not None
        )
        origin = event.preferred_origin() or (
            event.origins[0] if event.origins else None
        )
        resource_id = getattr(event, "resource_id", None)
        sets.append(
            PickSet(
                picks=picks,
                event_id=str(resource_id) if resource_id is not None else None,
                origin=origin.time if origin is not None else None,
            )
        )
    return sets


def read(source: str | PathLike[str] | Catalog) -> list[PickSet]:
    """Read picks from a Snuffler marker file or anything ObsPy can parse.

    Marker files are detected by suffix; everything else is handed to
    :func:`obspy.read_events`, which sniffs the format itself.
    """
    if isinstance(source, Catalog):
        return from_catalog(source)
    if str(source).endswith((".picks", ".markers")):
        from specmod.utils import read_pyrocko_picks  # noqa: PLC0415

        return [read_pyrocko_picks(source)]
    return from_catalog(obspy.read_events(str(source)))
