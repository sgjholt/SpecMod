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

Datasets that ship with the repository have their own loader —
:func:`load_pnr_2019` — and need no network. Published ones are fetched by
:func:`load`, cached by pooch and pinned by hash. :mod:`specmod.acquire`
produces both.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "EVENTS",
    "PNR_2019",
    "REGISTRY",
    "Dataset",
    "DatasetSpec",
    "Event",
    "EventDirectory",
    "data_dir",
    "load",
    "load_pnr_2019",
]

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
#: The hypocentre is the catalogue's, converted from its British National Grid
#: easting/northing (336135.0, 432515.0; EPSG:27700) to WGS84. The catalogue
#: gives depth as an elevation of -2040 m.
PNR_2019 = Event(
    origin="2019-08-26T07:30:47.000000Z",
    latitude=53.785021,
    longitude=-2.970780,
    depth_km=2.04,
    catalogue_magnitude=2.9,
    catalogue_magnitude_type="Mw",
)


# --------------------------------------------------------------- consuming
#
# Published datasets, fetched once and cached. `specmod.acquire` produces
# these; this half consumes them, and is offline after the first download.


@dataclass(frozen=True)
class DatasetSpec:
    """A published dataset: where to get it, and what it should hash to.

    The hash is what makes a regression test mean anything. A config records
    intent and makes a dataset regenerable, but FDSN is not content-addressed,
    so re-running the config is not guaranteed to return the same bytes — see
    §5.2.2 of ``docs/REFACTOR_PLAN.md``.

    Versioning is by name. ``magna_2020_v1`` and ``magna_2020_v2`` are separate
    entries, so a result pinned to v1 keeps fetching v1 after v2 exists.
    """

    name: str
    url: str
    #: ``sha256:...`` of the archive, as pooch expects it.
    sha256: str
    event: Event
    #: Path within the unpacked archive holding the event directory.
    member: str = ""


#: Published datasets by name. Local datasets are not listed: they ship with
#: the package and need no download.
REGISTRY: dict[str, DatasetSpec] = {}


def data_dir() -> Path:
    """Where downloaded datasets are cached.

    ``SPECMOD_DATA_DIR`` overrides the platform cache directory, which matters
    on a cluster where ``$HOME`` is small or not writable from a compute node.
    """
    override = os.environ.get("SPECMOD_DATA_DIR")
    if override:
        return Path(override)

    import pooch  # noqa: PLC0415

    return Path(pooch.os_cache("specmod"))


@dataclass(frozen=True)
class Dataset:
    """One event's data, wherever it came from.

    The readers are methods rather than eager attributes because a dataset is
    often opened for its metadata alone, and reading a stream costs real time.
    """

    event: Event
    paths: EventDirectory
    #: The acquisition manifest, where the dataset was produced by
    #: :mod:`specmod.acquire`. ``None`` for data that ships with the package.
    manifest: dict[str, Any] | None = None

    def stream(self, pattern: str = "*") -> Any:
        """Read the waveforms. Raw counts — the response is not removed."""
        import obspy  # noqa: PLC0415

        return obspy.read(self.paths.waveform_glob(pattern))

    def inventory(self) -> Any:
        """Read the station metadata, including responses."""
        import obspy  # noqa: PLC0415

        return obspy.read_inventory(str(self.paths.inventory))


def _repository_root() -> Path:
    """The checkout this package was imported from, if it is one.

    Data that ships with the repository is not inside the installed package, so
    it is only reachable from a source checkout or an editable install.
    """
    return Path(__file__).resolve().parent.parent.parent


def load_pnr_2019() -> Dataset:
    """The Preston New Road event committed to this repository.

    Needs no download and no network: the waveforms and inventory are in the
    checkout. Raises when they are not, rather than reaching for a URL, since
    there is no published artefact for this one.
    """
    paths = PNR_2019.directory(_repository_root())
    if not paths.is_present():
        raise FileNotFoundError(
            f"the PNR data is not at {paths.root}. It ships with the "
            f"repository rather than being downloaded, so this needs a source "
            f"checkout or an editable install."
        )
    return Dataset(event=PNR_2019, paths=paths)


def load(name: str, *, downloader: Any = None) -> Dataset:
    """Fetch a published dataset by name, from the cache after the first call.

    Downloads are hash-checked by pooch: a corrupted or substituted archive
    fails here rather than quietly becoming a new expected answer.

    ``downloader`` is passed through to :func:`pooch.retrieve`. It exists so
    the caching, hash check and unpacking can be exercised without a network —
    pooch has no ``file://`` support — and so an operator behind an
    authenticating proxy can supply their own.
    """
    try:
        spec = REGISTRY[name]
    except KeyError:
        known = sorted(REGISTRY) or ["<none published yet>"]
        raise ValueError(
            f"Unknown dataset {name!r}. Published: {known}. Data that ships "
            f"with the repository has its own loader, e.g. load_pnr_2019()."
        ) from None

    import pooch  # noqa: PLC0415

    unpacked = pooch.retrieve(
        url=spec.url,
        known_hash=spec.sha256,
        path=data_dir(),
        processor=pooch.Untar(),
        downloader=downloader,
    )
    root = Path(os.path.commonpath(unpacked)) if unpacked else data_dir()
    paths = EventDirectory(root / spec.member if spec.member else root)

    manifest_file = paths.root.parent / "manifest.json"
    manifest = (
        json.loads(manifest_file.read_text()) if manifest_file.is_file() else None
    )
    return Dataset(event=spec.event, paths=paths, manifest=manifest)
