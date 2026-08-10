"""Every event format ObsPy parses, through one reader.

``obspy.read_events`` sniffs and parses nineteen formats, ten of which carry
arrivals, so this delegate is the whole standard roster. A format registered
with ObsPy's own ``obspy.plugin.event`` entry points is readable here without
being registered with specmod.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import obspy
from obspy.core.event import Catalog

from .base import Pick, PickSet, SensorID

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike

    from obspy.core.event import Event as ObsPyEvent

__all__ = ["ObsPyEventsReader", "from_catalog"]


def _obspy_claims(source: str | PathLike[str]) -> bool:
    """Whether any ObsPy event plugin recognises this file.

    Uses the same per-format ``isFormat`` functions ``read_events`` uses to
    auto-detect, so this agrees with what a subsequent read would do without
    parsing the file.
    """
    from obspy.core.util.base import (  # noqa: PLC0415
        ENTRY_POINTS,
        buffered_load_entry_point,
    )

    for entry_point in ENTRY_POINTS["event"].values():
        if entry_point.dist is None:
            continue
        try:
            is_format = buffered_load_entry_point(
                entry_point.dist.name,
                f"obspy.plugin.event.{entry_point.name}",
                "isFormat",
            )
            if is_format(str(source)):
                return True
        except Exception:
            # A detector that raises has not claimed the file. They are
            # third-party code reading untrusted bytes, and one of them
            # throwing must not stop the others being asked.
            continue
    return False


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


@dataclass(frozen=True)
class ObsPyEventsReader:
    """Anything :func:`obspy.read_events` parses."""

    @property
    def name(self) -> str:
        return "obspy_events"

    @property
    def suffixes(self) -> tuple[str, ...]:
        return (".xml", ".quakeml", ".scml", ".pha", ".hyp", ".nordic", ".out")

    def can_read(self, source: str | PathLike[str]) -> bool:
        return _obspy_claims(source)

    def read(self, source: str | PathLike[str]) -> list[PickSet]:
        return from_catalog(obspy.read_events(str(source)))
