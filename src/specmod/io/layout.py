"""What is actually in the file.

The module to read to answer "what does this HDF5 file contain, and what do
the names mean". :mod:`specmod.io.hdf5` opens and walks files; this one
decides what a spectrum looks like once it is in one, and is where a format
change is made.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..core import AmplitudeKind, Motion, Spectrum

__all__ = ["FORMAT_VERSION"]

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
COMPRESS_ABOVE_BYTES = 16 * 1024


def dataset(group: Any, name: str, data: Any) -> None:
    """One array, compressed only where compression pays.

    See :data:`COMPRESS_ABOVE_BYTES`.
    """
    data = np.asarray(data)
    if data.nbytes >= COMPRESS_ABOVE_BYTES:
        group.create_dataset(name, data=data, compression="gzip")
    else:
        group.create_dataset(name, data=data)


def write_spectrum(group: Any, name: str, spectrum: Spectrum) -> None:
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
    dataset(group, f"{name}_amp", spectrum.amp)
    attrs = group.attrs
    attrs[f"{name}_motion"] = str(spectrum.motion)
    attrs[f"{name}_kind"] = str(spectrum.kind)
    attrs[f"{name}_duration"] = float(spectrum.duration)
    attrs[f"{name}_sampling_rate"] = float(spectrum.sampling_rate)
    attrs[f"{name}_meta"] = json.dumps(dict(spectrum.meta))


def read_spectrum(group: Any, name: str) -> Spectrum:
    """The inverse of :func:`write_spectrum`, for one spectrum in a group."""
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
