"""What the fitting layer expects of whatever it is handed.

Structural rather than nominal: :class:`~specmod.fitting.FitSpectrum` reads
attributes off its input and does not care what class provides them, which is
what lets it take a :class:`~specmod.core.collection.FittableView`, a bare
spectrum, or something assembled by hand in a notebook.
"""

from __future__ import annotations

from typing import Any

from .. import config as cfg

__all__ = [
    "REQUIRED_SPECTRUM_ATTRIBUTES",
    "SpectraLike",
    "Spectrumish",
    "plot_columns",
]

#: What a spectrum-like object handed to :class:`FitSpectrum` looks like.
#: Structural rather than nominal on purpose — see :meth:`FitSpectrum.__check_input`.
Spectrumish = Any
#: A container mapping trace ids to paired spectra; see
#: :meth:`FitSpectra.__check_spectra`.
SpectraLike = Any


def plot_columns() -> int:
    """How many columns a multi-panel figure uses, from ``[viz]``.

    A function rather than a constant, and that is the whole point. This was
    ``PLOT_COLUMNS = cfg.load_config().config.viz.plot_columns`` evaluated at
    import time, so importing :mod:`specmod.fitting` resolved configuration
    against whatever directory the process happened to start in and froze the
    answer for the life of the interpreter. Measured: importing from a project
    whose ``specmod.toml`` says 5, then moving to one that resolves to 3, left
    the constant at 5 — a worker serving two projects would use the first
    one's layout for both.

    One home for the setting either way: it used to be defined in *both* the
    SPECTRAL and FITTING dicts, and the two copies could disagree.
    """
    return int(cfg.load_config().config.viz.plot_columns)


#: What :class:`FitSpectrum` reads off whatever it is given. Kept as data so
#: the requirement is stated once and can be asserted against.
REQUIRED_SPECTRUM_ATTRIBUTES = ("id", "meta", "freq", "amp", "bfreq", "bamp")
