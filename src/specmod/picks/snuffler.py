"""Snuffler (Pyrocko) marker files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import obspy

from .base import Pick, PickSet, SensorID

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike

__all__ = ["SnufflerReader"]

#: First line of every marker file Snuffler writes.
MAGIC = "# Snuffler Markers File Version"

#: Marker symbol to ``(phase, branch, first motion)``.
PHASES = {
    "^": ("P", "Pg", "u"),
    "v": ("P", "Pg", "d"),
    "P": ("P", "Pg", None),
    "S": ("S", "Sg", None),
}

#: Weights above this are how the format spells a rejected arrival.
MAX_WEIGHT = 3


@dataclass(frozen=True)
class SnufflerReader:
    """Marker files, as saved from Snuffler.

    Not an ObsPy format: markers are an editor's working notes rather than an
    event, and carry no origin.
    """

    @property
    def name(self) -> str:
        return "snuffler"

    @property
    def suffixes(self) -> tuple[str, ...]:
        return (".picks", ".markers")

    def can_read(self, source: str | PathLike[str]) -> bool:
        try:
            with open(source) as handle:
                return handle.readline().startswith(MAGIC)
        except (OSError, UnicodeDecodeError):
            return False

    def read(self, source: str | PathLike[str]) -> list[PickSet]:
        with open(source) as handle:
            lines = handle.readlines()[1:]

        picks = []
        for line in lines:
            fields = line.split()
            if not fields or fields[0] != "phase:":
                # Snuffler also writes plain time markers, which name no phase.
                continue
            identity = fields[4].replace("..", ".--.").split(".")
            phase, branch, motion = PHASES[fields[8]]
            weight = int(fields[3])
            if weight > MAX_WEIGHT:
                continue
            picks.append(
                Pick(
                    sensor=SensorID.parse(
                        ".".join(
                            (
                                identity[0],
                                identity[1],
                                identity[2] if len(identity) > 2 else "--",
                            )
                        )
                    ),
                    phase=phase,
                    time=obspy.UTCDateTime("T".join(fields[1:3])),
                    raw_phase=branch,
                    polarity=motion,
                    weight=float(weight),
                    channel=identity[3] if len(identity) > 3 else None,
                    # A marker file is what an analyst saved out of Snuffler.
                    automatic=False,
                    reviewed=True,
                )
            )
        return [PickSet(picks=tuple(picks))]
