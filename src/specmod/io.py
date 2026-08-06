"""Writing an event's spectra to disk and reading them back.

Replaces ``spectral.Spectra.write_spectra``/``read_spectra``, which pickled the
container. Pickle stores the import path of every class it holds, so a stored
result stops loading the moment a class is renamed or moved — which is exactly
what happened to the shipped ``Tutorial/Spectra/*.spec``, unreadable since the
``Spectral.py`` → ``spectral.py`` rename and unreadable now for the better
reason that the classes are gone. A format that breaks when you refactor is not
a format, it is a cache.

**This is an interim format, deliberately.** ``REFACTOR_PLAN`` §4.6 specifies
HDF5 for arrays and Parquet for tables, which are the right answers — chunked,
partial-read, cross-language, queryable without loading. Both need a new
dependency (``h5py``, ``pyarrow``), and that is a decision to take on its
merits rather than to smuggle in behind a broken tutorial. So this uses
``numpy``'s compressed ``.npz``, which needs nothing new and is already typed,
compressed, and readable from anything that can read numpy.

What matters is that the *interface* is the one §4.6 wants: :func:`save` and
:func:`load` over a whole :class:`~specmod.core.SpectrumSet`, with the format
an implementation detail behind them. Adding an HDF5 backend later is a change
inside these two functions, not at every call site.

**Metadata is JSON inside the archive.** The alternative — numpy object arrays
— would reintroduce pickling by the back door, since that is how numpy stores
objects. JSON also means the metadata can be read by anything, and a
``.npz`` is a zip, so ``unzip -p spectra.npz meta.json`` inspects a run without
Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .core import BinnedSpectrum, Spectrum, SpectrumPair, SpectrumSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

__all__ = ["load", "save"]

#: Bumped when the layout changes in a way an older reader cannot handle.
#: Stored in the archive so a mismatch is an error naming both versions rather
#: than a ``KeyError`` on a renamed field.
FORMAT_VERSION = 1

#: What each pair contributes to the archive, as ``name -> (container, field)``.
_ARRAYS = (
    ("signal_freq", "signal", "freq"),
    ("signal_amp", "signal", "amp"),
    ("noise_freq", "noise", "freq"),
    ("noise_amp", "noise", "amp"),
    ("binned_signal_freq", "binned_signal", "freq"),
    ("binned_signal_amp", "binned_signal", "amp"),
    ("binned_noise_freq", "binned_noise", "freq"),
    ("binned_noise_amp", "binned_noise", "amp"),
)


def _spectrum_header(spectrum: Spectrum) -> dict[str, Any]:
    """Everything about a spectrum that is not its two arrays.

    ``duration`` and ``sampling_rate`` are carried explicitly rather than
    derived, because they cannot be recovered from ``len(freq)`` once padding
    is involved — the same reason :class:`~specmod.core.Spectrum` stores them.
    """
    return {
        "motion": str(spectrum.motion),
        "kind": str(spectrum.kind),
        "duration": spectrum.duration,
        "sampling_rate": spectrum.sampling_rate,
        "meta": dict(spectrum.meta),
    }


def save(path: str | Path, spectra: SpectrumSet) -> Path:
    """Write an event's spectra to ``path``.

    The parent directory is created if it does not exist — the legacy version
    raised ``FileNotFoundError`` from inside ``open``, naming the file rather
    than the missing directory.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    header: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "event": spectra.event,
        "meta": dict(spectra.meta),
        "pairs": {},
    }

    for index, id in enumerate(spectra.ids()):
        pair = spectra[id]
        for name, container, field in _ARRAYS:
            arrays[f"{index}/{name}"] = np.asarray(
                getattr(getattr(pair, container), field)
            )
        arrays[f"{index}/snr"] = np.asarray(pair.snr)
        header["pairs"][str(index)] = {
            "id": id,
            "signal": _spectrum_header(pair.signal),
            "noise": _spectrum_header(pair.noise),
            "resolution_floor": pair.resolution_floor,
            "band": list(pair.band) if pair.band is not None else None,
            "meta": dict(pair.meta),
        }

    # `allow_pickle=False` on the *write* as well as the read. Nothing here is
    # an object array, so it changes nothing today; it is what makes "this
    # format does not pickle" a property numpy enforces rather than one this
    # module merely intends. It also happens to type-check, where passing the
    # arrays alone does not — numpy types `**kwds` as `ArrayLike`, so an
    # unpacked dict alongside no explicit keyword leaves mypy trying to fit a
    # bool into it.
    np.savez_compressed(
        path,
        allow_pickle=False,
        header=np.array(json.dumps(header)),
        **arrays,
    )
    return path


def _spectrum(header: Mapping[str, Any], freq: Any, amp: Any) -> Spectrum:
    return Spectrum(
        freq=freq,
        amp=amp,
        motion=header["motion"],
        kind=header["kind"],
        duration=header["duration"],
        sampling_rate=header["sampling_rate"],
        meta=header["meta"],
    )


def load(path: str | Path) -> SpectrumSet:
    """Read back what :func:`save` wrote.

    The arrays come back read-only, as they went in: a spectrum loaded from
    disk gives the same immutability guarantee as one that was just computed,
    which is what makes "reload and refit" safe to do in a loop.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
        version = header.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} was written by format version {version}, and this is "
                f"version {FORMAT_VERSION}. Re-run the pipeline, or read it "
                f"with the version of specmod that wrote it."
            )

        pairs = {}
        for index, entry in header["pairs"].items():

            def array(name: str, index: str = index) -> Any:
                return archive[f"{index}/{name}"]

            band = entry["band"]
            pairs[entry["id"]] = SpectrumPair(
                signal=_spectrum(
                    entry["signal"], array("signal_freq"), array("signal_amp")
                ),
                noise=_spectrum(
                    entry["noise"], array("noise_freq"), array("noise_amp")
                ),
                binned_signal=BinnedSpectrum(
                    freq=array("binned_signal_freq"), amp=array("binned_signal_amp")
                ),
                binned_noise=BinnedSpectrum(
                    freq=array("binned_noise_freq"), amp=array("binned_noise_amp")
                ),
                snr=array("snr"),
                resolution_floor=entry["resolution_floor"],
                band=(float(band[0]), float(band[1])) if band else None,
                meta=entry["meta"],
            )

    return SpectrumSet(pairs=pairs, event=header["event"], meta=header["meta"])
