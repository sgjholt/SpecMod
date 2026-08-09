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

Naming the layout once matters more than it looks. Before this module the same
six paths were restated in ``conftest.py``, two test modules and three scripts
under ``tools/`` and ``docs/``; when the directories were reorganised, three of
those copies were missed, and because every one of them guards a
``skipif``, the tests that depended on them **skipped silently** rather than
failing. A path constant that is wrong in a way that turns tests off is worse
than one that raises, so there is now a single definition to get wrong.

This is the read side. §5.2.3 of ``docs/REFACTOR_PLAN.md`` plans a
``specmod.acquire`` that fetches an event and writes exactly this shape; when
it lands it should produce :class:`EventDirectory`, not its own idea of where
things go.
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

    Holds no data and reads nothing on construction, so it is safe to build at
    import time — which is what lets a ``skipif`` marker consult
    :meth:`is_present` without paying for it.
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
        """A glob over :attr:`waveforms`, as ``obspy.read`` wants it.

        Returns a string rather than a ``Path`` because ObsPy globs internally
        and a ``Path`` would be taken as a single filename.
        """
        return str(self.waveforms / pattern)

    def picks_file(self) -> Path:
        """The single pick file for this event.

        Raises rather than returning ``None`` when there is no such file: every
        caller goes straight on to read it, so a ``None`` would surface as an
        unrelated ``TypeError`` further down.
        """
        found = sorted(glob.glob(str(self.picks / "*.picks")))
        if not found:
            raise FileNotFoundError(f"no *.picks file under {self.picks}")
        return Path(found[0])

    def is_present(self) -> bool:
        """Whether the waveforms and metadata needed to load this event exist.

        Deliberately does **not** check ``spectra/``, which is generated.
        """
        return self.waveforms.is_dir() and self.inventory.is_file()


@dataclass(frozen=True)
class Event:
    """An earthquake, and the hypocentre needed to set source-station distance.

    Carried together with the layout because every consumer that resolves the
    paths also passes these four values to
    :func:`specmod.preprocess.set_stream_distance` immediately afterwards, and
    splitting them was how the origin time and the directory name drifted apart
    once already.
    """

    origin: str
    latitude: float
    longitude: float
    depth_km: float

    def directory(self, project_root: Path | str) -> EventDirectory:
        """Locate this event beneath ``project_root``, the repository root."""
        return EventDirectory(Path(project_root) / EVENTS / self.origin)


#: The Preston New Road event the tutorial and both golden references are built
#: around. The origin time doubles as the directory name.
PNR_2019 = Event(
    origin="2019-08-26T07:49:24.200000Z",
    latitude=53.784,
    longitude=-2.967,
    depth_km=2.1,
)
