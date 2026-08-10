"""The three rules: event selection, sensor matching, duplicate resolution."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .base import (
    AmbiguousPolicy,
    DuplicatePolicy,
    Pick,
    PickSet,
    Resolution,
    SensorID,
    _choose,
    _grouped,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Sequence

    from obspy import UTCDateTime

__all__ = ["resolve", "select_event"]


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
