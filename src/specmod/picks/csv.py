"""Delimited pick tables, with the columns named by the caller.

No vendor presets ship. PhaseNet, EQTransformer and SeisBench each write
several column layouts across versions, and a preset written from
documentation rather than from a real output file is a guess with a name on it.
Supply :class:`CSVPickReader` with your own mapping instead — see
``docs/pick-formats.md``.
"""

from __future__ import annotations

import csv as _csv
from dataclasses import dataclass
from typing import TYPE_CHECKING

import obspy

from .base import Pick, PickSet, SensorID

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping
    from os import PathLike

__all__ = ["CSVPickReader"]

#: Fields a mapping may name. ``station``, ``phase`` and ``time`` are required;
#: the rest populate the matching :class:`~specmod.picks.Pick` attributes.
FIELDS = (
    "station",
    "phase",
    "time",
    "network",
    "location",
    "channel",
    "weight",
    "uncertainty",
    "polarity",
    "author",
)
REQUIRED = ("station", "phase", "time")


@dataclass(frozen=True)
class CSVPickReader:
    """A delimited table of picks, one row per arrival.

    ``columns`` maps this reader's field names to the column headings in the
    file: ``{"station": "sta", "phase": "phase_type", "time": "arrival_time"}``.
    ``station``, ``phase`` and ``time`` are required.

    ``reader_name`` is what :func:`specmod.picks.read` refers to it by, and
    must be unique once registered. ``delimiter`` defaults to a comma.

    Detection is by header: :meth:`can_read` claims a file only when every
    mapped column is present, so two readers configured for different schemas
    do not collide. A reader whose columns are a subset of another's will
    collide, and that is the ambiguity :func:`specmod.picks.detect_reader`
    reports.
    """

    columns: Mapping[str, str]
    reader_name: str = "csv"
    delimiter: str = ","
    #: Rows whose phase does not fold to P or S are skipped rather than raising.
    skip_unknown_phases: bool = True

    def __post_init__(self) -> None:
        unknown = sorted(set(self.columns) - set(FIELDS))
        if unknown:
            raise ValueError(
                f"unknown pick fields {unknown}. Available: {list(FIELDS)}."
            )
        missing = [name for name in REQUIRED if name not in self.columns]
        if missing:
            raise ValueError(f"columns must map {missing}; got {sorted(self.columns)}.")

    @property
    def name(self) -> str:
        return self.reader_name

    @property
    def suffixes(self) -> tuple[str, ...]:
        return (".csv", ".tsv", ".txt")

    def _headings(self, source: str | PathLike[str]) -> list[str] | None:
        try:
            with open(source, newline="") as handle:
                row = next(_csv.reader(handle, delimiter=self.delimiter), None)
        except (OSError, UnicodeDecodeError, _csv.Error):
            return None
        return [cell.strip() for cell in row] if row else None

    def can_read(self, source: str | PathLike[str]) -> bool:
        headings = self._headings(source)
        if headings is None:
            return False
        return set(self.columns.values()) <= set(headings)

    def read(self, source: str | PathLike[str]) -> list[PickSet]:
        with open(source, newline="") as handle:
            rows = list(_csv.DictReader(handle, delimiter=self.delimiter))

        picks = []
        for row in rows:
            pick = self._pick(row)
            if pick is not None:
                picks.append(pick)
        return [PickSet(picks=tuple(picks))]

    def _value(self, row: Mapping[str, str], field_name: str) -> str | None:
        heading = self.columns.get(field_name)
        if heading is None:
            return None
        value = (row.get(heading) or "").strip()
        return value or None

    def _pick(self, row: Mapping[str, str]) -> Pick | None:
        raw_phase = self._value(row, "phase")
        phase = (raw_phase or "")[:1].upper()
        if phase not in ("P", "S"):
            if self.skip_unknown_phases:
                return None
            raise ValueError(f"phase {raw_phase!r} is neither P nor S")

        station = self._value(row, "station")
        time = self._value(row, "time")
        if station is None or time is None:
            return None

        location = self._value(row, "location")
        return Pick(
            sensor=SensorID(
                network=self._value(row, "network"),
                station=station,
                # A column that is present but blank states an empty location;
                # a column that is absent states nothing.
                location=(location or "") if "location" in self.columns else None,
            ),
            phase=phase,
            time=obspy.UTCDateTime(time),
            raw_phase=raw_phase,
            channel=self._value(row, "channel"),
            weight=_as_float(self._value(row, "weight")),
            uncertainty=_as_float(self._value(row, "uncertainty")),
            polarity=self._value(row, "polarity"),
            author=self._value(row, "author"),
        )


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
