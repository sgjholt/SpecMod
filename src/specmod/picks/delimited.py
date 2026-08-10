"""Delimited pick tables: one row per arrival, columns named by the caller.

:class:`DelimitedPickReader` is the general case, configured with a delimiter
and a column mapping. :class:`CSVPickReader`, :class:`TSVPickReader` and
:class:`WhitespacePickReader` are it with the delimiter and the plausible file
suffixes already set.

No vendor presets ship. PhaseNet, EQTransformer and SeisBench each write
several column layouts across versions, and a preset written from
documentation rather than from a real output file is a guess with a name on it.
Supply the mapping yourself — see ``docs/pick-formats.md``.
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

__all__ = [
    "CSVPickReader",
    "DelimitedPickReader",
    "TSVPickReader",
    "WhitespacePickReader",
]

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
class DelimitedPickReader:
    """A delimited table of picks, one row per arrival.

    ``columns`` maps this reader's field names to the column headings in the
    file: ``{"station": "sta", "phase": "phase_type", "time": "arrival_time"}``.
    ``station``, ``phase`` and ``time`` are required; :data:`FIELDS` lists what
    else may be mapped.

    ``delimiter`` is a single character, or ``None`` to split on runs of
    whitespace — the latter takes a separate parse path, since quoting has no
    meaning in a whitespace-aligned table.

    ``reader_name`` is what :func:`specmod.picks.read` refers to it by and must
    be unique once registered. ``file_suffixes`` is a hint for error messages
    and nothing else; detection is by header, so a reader claims a file only
    when every mapped column is present. Two readers configured for different
    schemas therefore do not collide, and one whose columns are a subset of
    another's collides visibly — the ambiguity
    :func:`specmod.picks.detect_reader` reports.
    """

    columns: Mapping[str, str]
    reader_name: str = "delimited"
    delimiter: str | None = ","
    file_suffixes: tuple[str, ...] = (".csv", ".tsv", ".txt", ".dat")
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
        if self.delimiter is not None and len(self.delimiter) != 1:
            raise ValueError(
                f"delimiter must be one character or None for whitespace; "
                f"got {self.delimiter!r}."
            )

    @property
    def name(self) -> str:
        return self.reader_name

    @property
    def suffixes(self) -> tuple[str, ...]:
        return self.file_suffixes

    def _rows(self, source: str | PathLike[str]) -> list[list[str]] | None:
        """Every line as a list of cells, or ``None`` if this is not a table."""
        try:
            with open(source, newline="") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            return None

        if self.delimiter is None:
            return [line.split() for line in text.splitlines() if line.strip()]
        try:
            return [
                [cell.strip() for cell in row]
                for row in _csv.reader(text.splitlines(), delimiter=self.delimiter)
                if row
            ]
        except _csv.Error:
            return None

    def can_read(self, source: str | PathLike[str]) -> bool:
        rows = self._rows(source)
        if not rows:
            return False
        return set(self.columns.values()) <= set(rows[0])

    def read(self, source: str | PathLike[str]) -> list[PickSet]:
        rows = self._rows(source)
        if not rows:
            return [PickSet()]

        headings, body = rows[0], rows[1:]
        picks = []
        for cells in body:
            row = dict(zip(headings, cells, strict=False))
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


@dataclass(frozen=True)
class CSVPickReader(DelimitedPickReader):
    """Comma-separated, with the quoting rules of :mod:`csv`."""

    reader_name: str = "csv"
    delimiter: str | None = ","
    file_suffixes: tuple[str, ...] = (".csv",)


@dataclass(frozen=True)
class TSVPickReader(DelimitedPickReader):
    """Tab-separated."""

    reader_name: str = "tsv"
    delimiter: str | None = "\t"
    file_suffixes: tuple[str, ...] = (".tsv", ".tab")


@dataclass(frozen=True)
class WhitespacePickReader(DelimitedPickReader):
    """Columns separated by runs of spaces or tabs.

    The shape most hand-written and Fortran-era arrival tables come in. Cells
    cannot contain spaces, and quoting is not honoured.

    This **subsumes** :class:`TSVPickReader` — splitting on whitespace splits
    on tabs — so registering both against the same column names makes every
    tab-separated file ambiguous. Register one, or pass ``format=``.
    """

    reader_name: str = "whitespace"
    delimiter: str | None = None
    file_suffixes: tuple[str, ...] = (".txt", ".dat", ".lst")


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
