"""Fitting a source model to one spectrum, and to a whole event.

:func:`fittable_signal` decides what to fit, :func:`initial_guess` where to
start, :class:`FitSpectrum` fits one station and :class:`FitSpectra` an event.
"""

from __future__ import annotations

from .base import (
    PLOT_COLUMNS,
    REQUIRED_SPECTRUM_ATTRIBUTES,
    SpectraLike,
    Spectrumish,
)
from .event import FitSpectra
from .guess import fittable_signal, initial_guess, selected_band
from .spectrum import FitSpectrum

__all__ = [
    "PLOT_COLUMNS",
    "REQUIRED_SPECTRUM_ATTRIBUTES",
    "FitSpectra",
    "FitSpectrum",
    "SpectraLike",
    "Spectrumish",
    "fittable_signal",
    "initial_guess",
    "selected_band",
]
