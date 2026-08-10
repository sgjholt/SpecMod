"""Phase arrivals, and matching them to the sensors a stream carries.

:func:`read` returns one :class:`PickSet` per event, choosing a reader from
:data:`PICK_READERS`. :func:`select_event` narrows a multi-event source to one.
:func:`resolve` matches those picks against a set of :class:`SensorID`,
returning a :class:`Resolution`.

Three conditions have no single right answer and are therefore explicit: a
source holding several events, a pick whose identity fits several sensors, and
several picks for one sensor and phase. Each raises by default or takes a named
policy. Design notes are in §4.9 of ``docs/REFACTOR_PLAN.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from obspy.core.event import Catalog

from .base import (
    EMPTY_LOCATION,
    AmbiguousPolicy,
    DuplicatePolicy,
    Pick,
    PickReader,
    PickSet,
    Resolution,
    SensorID,
)
from .events import ObsPyEventsReader, from_catalog
from .resolution import resolve, select_event
from .snuffler import SnufflerReader

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike

__all__ = [
    "EMPTY_LOCATION",
    "PICK_READERS",
    "AmbiguousPolicy",
    "DuplicatePolicy",
    "ObsPyEventsReader",
    "Pick",
    "PickReader",
    "PickSet",
    "Resolution",
    "SensorID",
    "SnufflerReader",
    "detect_reader",
    "from_catalog",
    "get_reader",
    "read",
    "resolve",
    "select_event",
]

#: Registered readers, by the name ``read(format=...)`` refers to them by.
PICK_READERS: dict[str, PickReader] = {
    "snuffler": SnufflerReader(),
    "obspy_events": ObsPyEventsReader(),
}


def get_reader(name: str) -> PickReader:
    """Look a reader up by name."""
    try:
        return PICK_READERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown pick format {name!r}. Available: {sorted(PICK_READERS)}."
        ) from None


def detect_reader(source: str | PathLike[str]) -> PickReader:
    """The one registered reader that recognises ``source``.

    Every reader is offered the file and exactly one must claim it. Both no
    claim and several are errors: a tie means two readers sniff too loosely,
    which is a bug in them rather than something to settle by priority.
    """
    claims = [reader for reader in PICK_READERS.values() if reader.can_read(source)]
    if len(claims) == 1:
        return claims[0]

    tried = ", ".join(sorted(PICK_READERS))
    if not claims:
        raise ValueError(
            f"no reader recognises {source}. Tried: {tried}. Pass format= to "
            f"force one, or check the file is not empty or truncated."
        )
    raise ValueError(
        f"{len(claims)} readers claim {source} "
        f"({', '.join(sorted(reader.name for reader in claims))}). "
        f"Pass format= to choose; the overlap is a defect in their sniffing."
    )


def read(
    source: str | PathLike[str] | Catalog, *, format: str | None = None
) -> list[PickSet]:
    """Every event in ``source``, in file order.

    ``format`` names a reader in :data:`PICK_READERS` and skips detection.
    Without it, :func:`detect_reader` selects one. An ObsPy ``Catalog`` is
    converted directly.
    """
    if isinstance(source, Catalog):
        return from_catalog(source)
    reader = get_reader(format) if format is not None else detect_reader(source)
    return reader.read(source)
