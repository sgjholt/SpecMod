"""Fitting a source model to one spectrum, and to a whole event.

:func:`fittable_signal` decides what to fit, :func:`initial_guess` where to
start, :class:`FitSpectrum` fits one station and :class:`FitSpectra` an event.
"""

from __future__ import annotations

import warnings

from .base import (
    REQUIRED_SPECTRUM_ATTRIBUTES,
    SpectraLike,
    Spectrumish,
    plot_columns,
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
    "plot_columns",
    "selected_band",
]


def __getattr__(name: str) -> object:
    """Keep ``PLOT_COLUMNS`` importable, resolved at each access.

    It was a module-level constant evaluated at import time, which froze the
    configuration of whatever directory the process started in. Reading it now
    resolves configuration per access, so the value is at least correct — but
    a name that looks like a constant and performs a lookup is a poor bargain
    either way, hence the warning and :func:`plot_columns`.

    Deliberately *not* ``from .base import ...``: a from-import would run this
    once at import time and rebind the result, which is the frozen behaviour
    this replaced, reintroduced one level up.
    """
    if name == "PLOT_COLUMNS":
        warnings.warn(
            "specmod.fitting.PLOT_COLUMNS is deprecated and will be removed "
            "in 0.4.0; call specmod.fitting.plot_columns() instead, which "
            "resolves the current configuration rather than the one that "
            "happened to be in effect at import.",
            DeprecationWarning,
            stacklevel=2,
        )
        return plot_columns()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
