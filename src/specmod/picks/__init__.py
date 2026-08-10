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

import warnings
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
from .csv import CSVPickReader
from .events import ObsPyEventsReader, from_catalog
from .resolution import resolve, select_event
from .snuffler import SnufflerReader

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike

__all__ = [
    "BUILTIN_READERS",
    "EMPTY_LOCATION",
    "ENTRY_POINT_GROUP",
    "PICK_READERS",
    "AmbiguousPolicy",
    "CSVPickReader",
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
    "load_plugins",
    "read",
    "register_reader",
    "resolve",
    "select_event",
]

#: Readers that ship with the package. Cannot be replaced by a plugin.
BUILTIN_READERS: dict[str, PickReader] = {
    "snuffler": SnufflerReader(),
    "obspy_events": ObsPyEventsReader(),
}

#: Registered readers, by the name ``read(format=...)`` refers to them by.
#: Plugins are added on first use; see :func:`load_plugins`.
PICK_READERS: dict[str, PickReader] = dict(BUILTIN_READERS)

#: Entry point group a third party registers a reader under. A format that is
#: an *event file* should register with ObsPy's ``obspy.plugin.event`` instead,
#: where it serves every ObsPy-based tool and reaches specmod through
#: :class:`ObsPyEventsReader` with no registration here at all.
ENTRY_POINT_GROUP = "specmod.pick_readers"

_plugins_loaded = False


def register_reader(reader: PickReader, *, replace: bool = False) -> None:
    """Add a reader to :data:`PICK_READERS`.

    For a reader defined in a notebook or a script, where there is no installed
    distribution to hang an entry point on. A name already registered raises
    unless ``replace``; a name in :data:`BUILTIN_READERS` raises regardless.
    """
    name = reader.name
    if name in BUILTIN_READERS:
        raise ValueError(f"{name!r} is a built-in reader and cannot be replaced.")
    if name in PICK_READERS and not replace:
        raise ValueError(f"{name!r} is already registered; pass replace=True.")
    PICK_READERS[name] = reader


def load_plugins() -> None:
    """Discover readers advertised under :data:`ENTRY_POINT_GROUP`.

    Called on first use of the registry, so a plugin costs nothing to a caller
    who never reads a pick. A plugin that fails to import, or that claims a
    built-in name, warns naming its distribution and is skipped: a broken
    third-party reader must not make the built-in formats unreadable.
    """
    global _plugins_loaded  # noqa: PLW0603
    if _plugins_loaded:
        return
    _plugins_loaded = True

    from importlib.metadata import entry_points  # noqa: PLC0415

    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        origin = getattr(getattr(entry_point, "dist", None), "name", "<unknown>")
        try:
            reader = entry_point.load()()
            register_reader(reader)
        except Exception as error:
            warnings.warn(
                f"pick reader plugin {entry_point.name!r} from {origin} could "
                f"not be registered and has been skipped: {error!r}",
                stacklevel=2,
            )


def get_reader(name: str) -> PickReader:
    """Look a reader up by name."""
    load_plugins()
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
    load_plugins()
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
