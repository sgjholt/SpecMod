"""What the fitting layer expects of whatever it is handed.

Structural rather than nominal: :class:`~specmod.fitting.FitSpectrum` reads
attributes off its input and does not care what class provides them, which is
what lets it take a :class:`~specmod.core.collection.FittableView`, a bare
spectrum, or something assembled by hand in a notebook.
"""

from __future__ import annotations

from typing import Any

from .. import config as cfg

__all__ = ["PLOT_COLUMNS", "REQUIRED_SPECTRUM_ATTRIBUTES", "SpectraLike", "Spectrumish"]

#: What a spectrum-like object handed to :class:`FitSpectrum` looks like.
#: Structural rather than nominal on purpose — see :meth:`FitSpectrum.__check_input`.
Spectrumish = Any
#: A container mapping trace ids to paired spectra; see
#: :meth:`FitSpectra.__check_spectra`.
SpectraLike = Any

# One home for this: it used to be defined in *both* the SPECTRAL and FITTING
# dicts, and the two copies could disagree.
PLOT_COLUMNS = cfg.load_config().config.viz.plot_columns

#: What :class:`FitSpectrum` reads off whatever it is given. Kept as data so
#: the requirement is stated once and can be asserted against.
REQUIRED_SPECTRUM_ATTRIBUTES = ("id", "meta", "freq", "amp", "bfreq", "bamp")
