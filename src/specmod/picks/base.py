"""Record types and the reader contract.

A :class:`Pick` names an arrival on a :class:`SensorID`, which may be partial —
``None`` for a field the source did not state. :class:`PickSet` groups the picks
of one event; :class:`Resolution` is what :func:`specmod.picks.resolve` returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Sequence
    from os import PathLike

    from obspy import UTCDateTime

__all__ = [
    "EMPTY_LOCATION",
    "AmbiguousPolicy",
    "DuplicatePolicy",
    "Pick",
    "PickReader",
    "PickSet",
    "Resolution",
    "SensorID",
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


@runtime_checkable
class PickReader(Protocol):
    """One pick format.

    ``suffixes`` is a hint for error messages and nothing else: a suffix does
    not identify a format, so :func:`specmod.picks.detect_reader` selects on
    :meth:`can_read` alone.

    ``name`` and ``suffixes`` are read-only properties so that a frozen
    dataclass satisfies the protocol; mutable attributes would not.
    """

    @property
    def name(self) -> str: ...

    @property
    def suffixes(self) -> tuple[str, ...]: ...

    def can_read(self, source: str | PathLike[str]) -> bool:
        """Whether this reader recognises ``source``.

        Must not raise — on a missing, empty, truncated or binary file, and on
        every other format's files, it returns ``False``. Must be cheap: read a
        header, not the whole file.
        """
        ...

    def read(self, source: str | PathLike[str]) -> list[PickSet]:
        """Every event in ``source``, in file order."""
        ...
