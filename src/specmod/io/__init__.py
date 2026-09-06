"""Writing an event's spectra to disk and reading them back.

Replaces ``spectral.Spectra.write_spectra``/``read_spectra``, which pickled the
container. Pickle stores the import path of every class it holds, so a stored
result stops loading the moment a class is renamed or moved — which is exactly
what happened to the shipped ``Tutorial/Spectra/*.spec``: unreadable since the
``Spectral.py`` → ``spectral.py`` rename, years before the classes were
deleted. **A format that breaks when you refactor is a cache, not a format.**

HDF5, per ``REFACTOR_PLAN`` §4.6. The four rules there are lessons from how
pickle failed rather than general good practice, and each is visible in the
layout:

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

The package is three modules, by what a change would be about:
:mod:`~specmod.io.layout` is what a file *contains*, :mod:`~specmod.io.hdf5`
is how one is written and read, and :mod:`~specmod.io.backend` is the single
place that knows the optional dependency's name.
"""

from __future__ import annotations

from .hdf5 import load, save
from .layout import FORMAT_VERSION

__all__ = ["FORMAT_VERSION", "load", "save"]
