"""Writing and reading whole files.

Paths, the format-version check, and walking one group per channel.
:mod:`specmod.io.layout` decides what goes inside a group; this module decides
where the file goes and what happens when it is not the file it claims to be.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core import BinnedSpectrum, SpectrumPair, SpectrumSet
from .backend import require_h5py
from .layout import FORMAT_VERSION, dataset, read_spectrum, write_spectrum

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

__all__ = ["load", "save"]


def save(path: str | Path, spectra: SpectrumSet) -> Path:
    """Write an event's spectra to ``path``.

    One group per trace id, keyed by the id itself — HDF5 group names may
    contain dots, so ``LV.L001..HHE`` needs no mangling and the file browses
    with the same names the pipeline uses.

    The parent directory is created if absent. The legacy version raised
    ``FileNotFoundError`` from inside ``open``, naming the file rather than the
    directory that did not exist.
    """
    h5py = require_h5py()

    path = Path(path)
    # Appended, never substituted. An event id is an ISO timestamp —
    # `2019-08-26T07:49:24.200000Z` — and `Path.with_suffix` reads `.200000Z`
    # as an extension and replaces it, silently truncating the name to
    # `2019-08-26T07:49:24.h5`. Two events in the same second would then
    # overwrite each other.
    if path.suffix != ".h5":
        path = path.with_name(path.name + ".h5")
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as handle:
        handle.attrs["specmod_format_version"] = FORMAT_VERSION
        handle.attrs["event"] = spectra.event
        handle.attrs["meta"] = json.dumps(dict(spectra.meta))

        for id in spectra.ids():
            pair = spectra[id]
            group = handle.create_group(id)
            # The two shared axes, written once each.
            dataset(group, "freq", pair.signal.freq)
            dataset(group, "binned_freq", pair.binned_signal.freq)

            write_spectrum(group, "signal", pair.signal)
            write_spectrum(group, "noise", pair.noise)
            dataset(group, "binned_signal_amp", pair.binned_signal.amp)
            dataset(group, "binned_noise_amp", pair.binned_noise.amp)
            dataset(group, "snr", pair.snr)
            group.attrs["resolution_floor"] = float(pair.resolution_floor)
            group.attrs["meta"] = json.dumps(dict(pair.meta))
            # `band` is absent rather than sentinel-valued when no band
            # survived. A NaN pair would read as a band, and "no usable
            # bandwidth" is a different claim from "a band at nowhere".
            if pair.band is not None:
                group.attrs["band"] = np.asarray(pair.band, dtype=np.float64)

    return path


def load(path: str | Path) -> SpectrumSet:
    """Read back what :func:`save` wrote.

    The arrays come back read-only, as they went in: a spectrum loaded from
    disk gives the same immutability guarantee as one just computed, which is
    what makes reload-and-refit safe to do in a loop.
    """
    h5py = require_h5py()
    path = Path(path)

    with h5py.File(path, "r") as handle:
        version = handle.attrs.get("specmod_format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} declares format version {version}, and this is "
                f"version {FORMAT_VERSION}. Re-run the pipeline, or read it "
                f"with the version of specmod that wrote it."
            )

        pairs: dict[str, SpectrumPair] = {}
        for id, group in handle.items():
            band = group.attrs.get("band")
            pairs[id] = SpectrumPair(
                signal=read_spectrum(group, "signal"),
                noise=read_spectrum(group, "noise"),
                binned_signal=BinnedSpectrum(
                    freq=np.asarray(group["binned_freq"]),
                    amp=np.asarray(group["binned_signal_amp"]),
                ),
                binned_noise=BinnedSpectrum(
                    freq=np.asarray(group["binned_freq"]),
                    amp=np.asarray(group["binned_noise_amp"]),
                ),
                snr=np.asarray(group["snr"]),
                resolution_floor=float(group.attrs["resolution_floor"]),
                band=None if band is None else (float(band[0]), float(band[1])),
                meta=json.loads(str(group.attrs["meta"])),
            )

        event = str(handle.attrs.get("event", ""))
        meta: Mapping[str, Any] = json.loads(str(handle.attrs.get("meta", "{}")))

    return SpectrumSet(pairs=pairs, event=event, meta=dict(meta))
