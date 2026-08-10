"""Phase arrivals, and matching them to the sensors a stream carries.

:func:`read` returns one :class:`PickSet` per event. :func:`select_event`
narrows a multi-event source to one. :func:`resolve` matches those picks
against a set of :class:`SensorID`, returning a :class:`Resolution`.

Three conditions have no single right answer and are therefore explicit: a
source holding several events, a pick whose identity fits several sensors, and
several picks for one sensor and phase. Each raises by default or takes a named
policy. Design notes are in §4.9 of ``docs/REFACTOR_PLAN.md``.
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

    ``None`` means the source did not state the field, and matches anything.
    ``""`` means stated and empty, and matches only an empty field.

    Readers apply that distinction asymmetrically: an empty network code is not
    a valid SEED network and becomes ``None``, while an empty location code is
    the ordinary single-sensor case and is kept as ``""``.
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
        """The picks as ``{"NET.STA.LOC": {"P": UTCDateTime}}``.

        Keyed on each pick's own identity, so an unstated field appears as
        ``*`` and matches no trace. Use :func:`resolve` to match against a
        stream.
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
    """Choose one :class:`PickSet` from a source that may hold several.

    Selects by ``event_id``, or by origin time within ``tolerance_s`` of
    ``near``. With neither, the source must hold exactly one event. Raises
    unless exactly one event is selected, naming those available.
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
        Raise, naming the candidates and the fields the pick left unstated.
        The default.
    ``skip``
        Leave the pick unattached, counted in :attr:`Resolution.ambiguous`.
    ``broadcast``
        Attach to every match.

    ``duplicates`` chooses between several picks for one sensor and phase:
    ``prefer_reviewed`` (then earliest), ``earliest``, ``highest_weight``, or
    ``error``.
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

    Phase hints are folded to ``P`` or ``S`` on their first letter, with the
    original kept as ``raw_phase``. Picks with no ``P``/``S`` hint, with an
    ``evaluation_status`` of ``rejected``, or with no station code are dropped.
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
