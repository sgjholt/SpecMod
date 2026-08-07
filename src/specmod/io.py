"""Writing an event's spectra to disk and reading them back.

Replaces ``spectral.Spectra.write_spectra``/``read_spectra``, which pickled the
container. Pickle stores the import path of every class it holds, so a stored
result stops loading the moment a class is renamed or moved — which is exactly
what happened to the shipped ``Tutorial/Spectra/*.spec``: unreadable since the
``Spectral.py`` → ``spectral.py`` rename, years before the classes were
deleted. **A format that breaks when you refactor is a cache, not a format.**

HDF5, per ``REFACTOR_PLAN`` §4.6. The four rules there are lessons from how
pickle failed rather than general good practice, and each is visible in the
layout below:

*Never store class identity.*
    Plain arrays and a documented layout. Nothing here names a Python type, so
    nothing here can be broken by renaming one.

*Every file carries a format version.*
    :data:`FORMAT_VERSION`, on the root group. A reader checks it and fails
    naming both versions. The absence of this is the whole problem with what
    came before.

*Self-describing units.*
    ``motion``, ``kind``, ``duration`` and ``sampling_rate`` are stored
    attributes, not conventions the reader has to know. This is the typing of
    :class:`~specmod.core.Spectrum` expressed on disk, and it is what stops a
    file being silently misread as displacement.

*One file per event, one group per channel.*
    Which matches how the science is done and how it is re-examined, and
    sidesteps HDF5's single-writer limitation if the workflow is ever
    parallelised across events.

**Nothing here can pickle.** ``h5py`` has no mechanism to; trace metadata is
stored as a JSON string attribute rather than as an object array, which is how
``numpy`` would otherwise smuggle pickling back in.

The tables — fit results — go to Parquet instead, in :mod:`specmod.tables`.
Arrays and tables are used in genuinely different ways (random access into one
event, versus a columnar scan over every event ever fitted), and one format for
both would be worse at each.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .core import (
    AmplitudeKind,
    BinnedSpectrum,
    Motion,
    Spectrum,
    SpectrumPair,
    SpectrumSet,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

__all__ = ["FORMAT_VERSION", "load", "save"]

#: Bumped when the layout changes in a way an older reader cannot handle.
FORMAT_VERSION = 1

#: Compress datasets at least this large, in bytes; store smaller ones raw.
#:
#: Measured, not guessed. Compression in HDF5 requires *chunked* storage, which
#: costs a chunk index per dataset. On this event the arrays are tiny — median
#: 91 float64, 728 bytes — and gzip took the file from 370 KB to **867 KB**
#: against a raw payload of 250 KB: the indices cost more than the data they
#: describe. Above the threshold the ratio flips and compression pays, which is
#: the case the plan's "chunked, compressed" argument was really about (a
#: scalogram is ~1 MB per trace).
_COMPRESS_ABOVE_BYTES = 16 * 1024


def _require_h5py() -> Any:
    """Import ``h5py``, or say what to install.

    Deferred rather than imported at module scope so that the message names the
    extra. A bare ``ModuleNotFoundError: h5py`` in the middle of a save tells a
    user nothing about how to fix it.
    """
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading and writing spectra needs h5py. Install it with "
            "`pip install specmod[io]`."
        ) from exc
    return h5py


def _dataset(group: Any, name: str, data: Any) -> None:
    """One array, compressed only where compression pays. See
    :data:`_COMPRESS_ABOVE_BYTES`."""
    data = np.asarray(data)
    if data.nbytes >= _COMPRESS_ABOVE_BYTES:
        group.create_dataset(name, data=data, compression="gzip")
    else:
        group.create_dataset(name, data=data)


def _write_spectrum(group: Any, name: str, spectrum: Spectrum) -> None:
    """One spectrum: two datasets, and the attributes that make them readable.

    The units go on as attributes rather than being implied, because a
    frequency and an amplitude array alone do not say whether they are a
    velocity magnitude or a displacement power spectral density — and the two
    differ by factors that would pass unnoticed.
    """
    # Only the amplitude. The frequency axis is written once per group, by
    # `save`: `compare` interpolates the noise onto the signal's axis and bins
    # both against the same edges, so the two spectra *share* an axis by
    # construction. Storing it twice would be waste, and worse, would make a
    # file expressible in which they disagree — a state the containers cannot
    # represent and no reader could act on.
    _dataset(group, f"{name}_amp", spectrum.amp)
    attrs = group.attrs
    attrs[f"{name}_motion"] = str(spectrum.motion)
    attrs[f"{name}_kind"] = str(spectrum.kind)
    attrs[f"{name}_duration"] = float(spectrum.duration)
    attrs[f"{name}_sampling_rate"] = float(spectrum.sampling_rate)
    attrs[f"{name}_meta"] = json.dumps(dict(spectrum.meta))


def _read_spectrum(group: Any, name: str) -> Spectrum:
    attrs = group.attrs
    return Spectrum(
        freq=np.asarray(group["freq"]),
        amp=np.asarray(group[f"{name}_amp"]),
        # Through the enums rather than as bare strings: constructing them is
        # what turns a corrupted or hand-edited attribute into an error naming
        # the value, instead of a spectrum that claims a unit nothing else
        # recognises.
        motion=Motion(str(attrs[f"{name}_motion"])),
        kind=AmplitudeKind(str(attrs[f"{name}_kind"])),
        duration=float(attrs[f"{name}_duration"]),
        sampling_rate=float(attrs[f"{name}_sampling_rate"]),
        meta=json.loads(str(attrs[f"{name}_meta"])),
    )


def save(path: str | Path, spectra: SpectrumSet) -> Path:
    """Write an event's spectra to ``path``.

    One group per trace id, keyed by the id itself — HDF5 group names may
    contain dots, so ``LV.L001..HHE`` needs no mangling and the file browses
    with the same names the pipeline uses.

    The parent directory is created if absent. The legacy version raised
    ``FileNotFoundError`` from inside ``open``, naming the file rather than the
    directory that did not exist.
    """
    h5py = _require_h5py()

    path = Path(path)
    if path.suffix != ".h5":
        path = path.with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as handle:
        handle.attrs["specmod_format_version"] = FORMAT_VERSION
        handle.attrs["event"] = spectra.event
        handle.attrs["meta"] = json.dumps(dict(spectra.meta))

        for id in spectra.ids():
            pair = spectra[id]
            group = handle.create_group(id)
            # The two shared axes, written once each.
            _dataset(group, "freq", pair.signal.freq)
            _dataset(group, "binned_freq", pair.binned_signal.freq)

            _write_spectrum(group, "signal", pair.signal)
            _write_spectrum(group, "noise", pair.noise)
            _dataset(group, "binned_signal_amp", pair.binned_signal.amp)
            _dataset(group, "binned_noise_amp", pair.binned_noise.amp)
            _dataset(group, "snr", pair.snr)
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
    h5py = _require_h5py()
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
                signal=_read_spectrum(group, "signal"),
                noise=_read_spectrum(group, "noise"),
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
