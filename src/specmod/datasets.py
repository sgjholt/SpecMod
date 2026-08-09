"""Where an event's data lives on disk, and the events shipped with the repo.

The layout follows ObsPy's ``mass_downloader``, which writes ``waveforms/``
and ``stations/`` beneath a per-event directory. This adds the three things a
spectral workflow needs alongside them::

    tutorial/data/events/<origin>/
        waveforms/                 # one miniSEED file per channel
        stations/inventory.xml     # StationXML for those channels
        picks/*.picks              # Pyrocko/Snuffler markers
        spectra/*.h5               # computed spectra
        spectra/flatfiles/*.csv    # and their tabular export

:class:`EventDirectory` resolves those paths and :class:`Event` carries the
hypocentre needed to set source-station geometry. Tests, ``tools/`` and the
documentation notebooks all read the layout from here.

This is the read side; ``specmod.acquire`` (§5.2.3 of
``docs/REFACTOR_PLAN.md``) will write it.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

__all__ = ["EVENTS", "PNR_2019", "Event", "EventDirectory"]

#: Event directories, relative to the repository root.
EVENTS = Path("tutorial") / "data" / "events"


@dataclass(frozen=True)
class EventDirectory:
    """The paths beneath one event directory.

    Reads nothing on construction, so it is safe to build at import time.
    """

    root: Path

    @property
    def waveforms(self) -> Path:
        return self.root / "waveforms"

    @property
    def stations(self) -> Path:
        return self.root / "stations"

    @property
    def inventory(self) -> Path:
        """The StationXML covering the channels in :attr:`waveforms`."""
        return self.stations / "inventory.xml"

    @property
    def picks(self) -> Path:
        return self.root / "picks"

    @property
    def spectra(self) -> Path:
        return self.root / "spectra"

    @property
    def flatfiles(self) -> Path:
        return self.spectra / "flatfiles"

    def waveform_glob(self, pattern: str = "*") -> str:
        """A glob over :attr:`waveforms`, as a string for ``obspy.read``."""
        return str(self.waveforms / pattern)

    def picks_file(self) -> Path:
        """The single pick file for this event.

        Raises :class:`FileNotFoundError` when there is none.
        """
        found = sorted(glob.glob(str(self.picks / "*.picks")))
        if not found:
            raise FileNotFoundError(f"no *.picks file under {self.picks}")
        return Path(found[0])

    def is_present(self) -> bool:
        """Whether the waveforms and station metadata are both present.

        Does not check ``spectra/``, which is generated rather than shipped.
        """
        return self.waveforms.is_dir() and self.inventory.is_file()


@dataclass(frozen=True)
class Event:
    """An earthquake: where its data sits, and the hypocentre it happened at.

    ``origin``, ``latitude``, ``longitude`` and ``depth_km`` are the four
    values :func:`specmod.preprocess.set_stream_distance` takes.
    """

    origin: str
    latitude: float
    longitude: float
    depth_km: float
    #: The published magnitude and the scale it is on, e.g. ``2.9`` and
    #: ``"Mw"``. Kept as a pair: ML and Mw diverge below about magnitude 3, so
    #: a bare number cannot be compared against a computed one.
    catalogue_magnitude: float | None = None
    catalogue_magnitude_type: str | None = None

    def directory(self, project_root: Path | str) -> EventDirectory:
        """Locate this event beneath ``project_root``, the repository root."""
        return EventDirectory(Path(project_root) / EVENTS / self.origin)


#: Preston New Road, 26 August 2019 — the induced event the tutorial and both
#: golden references are built around, and the largest of the PNR-2 sequence.
#: The origin time doubles as the directory name.
#:
#: ``Mw 2.9`` is from the PNR-2 catalogue published with Cuadrilla's
#: hydraulic-fracture monitoring (NGDC, `709cbc2f-af5c-4d09-a4ea-6deb5aa8c5d8
#: <https://www2.bgs.ac.uk/nationalgeosciencedatacentre/citedData/catalogue/709cbc2f-af5c-4d09-a4ea-6deb5aa8c5d8.html>`_),
#: which gives ``surface_ML``, ``surface_Mw`` and ``corrected_Mw`` all as 2.9.
#:
#: The hypocentre is the one the published results were computed with. The
#: catalogue puts the event 60 m shallower, at 2.04 km; adopting that would
#: move every distance and so every golden reference, which is a change to make
#: deliberately rather than in passing.
PNR_2019 = Event(
    origin="2019-08-26T07:30:47.000000Z",
    latitude=53.784,
    longitude=-2.967,
    depth_km=2.1,
    catalogue_magnitude=2.9,
    catalogue_magnitude_type="Mw",
)
