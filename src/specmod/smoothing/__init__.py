"""Spectral smoothing.

Separate from estimation, and lossy by design — see
:mod:`specmod.smoothing.base` for why a smoothed spectrum no longer satisfies
the Parseval contract that :mod:`specmod.transforms` guarantees.
"""

from __future__ import annotations

from ..core.spectrum import Spectrum

# `record_smoothing` is re-exported because anything implementing `Smoother`
# needs it to leave the same metadata trail the shipped smoothers do — it is
# to `Smoother` what `prepare_record` is to `SpectralEstimator`.
from .base import NoSmoothing, Smoother, record_smoothing
from .konno_ohmachi import KonnoOhmachi
from .log_bins import LogBinner
from .log_window import WINDOWS, LogWindow
from .savitzky_golay import SavitzkyGolay

#: Name -> smoother. This is what `[smoothing] method` resolves through, and
#: `tests/test_smoothing.py` asserts that every entry is reachable from the
#: configuration and survives the pipeline. The comment here used to claim the
#: resolution and no code performed it; see REFACTOR_PLAN §6.6.
SMOOTHERS: dict[str, type[Smoother]] = {
    "log_bins": LogBinner,
    "konno_ohmachi": KonnoOhmachi,
    "log_window": LogWindow,
    "savitzky_golay": SavitzkyGolay,
    "none": NoSmoothing,
}


def get_smoother(name: str, **kwargs: object) -> Smoother:
    """Construct a smoother by name."""
    try:
        cls = SMOOTHERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown smoother {name!r}. Available: {sorted(SMOOTHERS)}."
        ) from None
    smoother: Smoother = cls(**kwargs)
    return smoother


def is_smoothed(spectrum: Spectrum) -> bool:
    """Whether any smoother has been applied.

    Worth checking before comparing a spectrum's energy against a time-domain
    expectation: smoothing does not preserve it.
    """
    return bool(spectrum.meta.get("smoothing"))


__all__ = [
    "SMOOTHERS",
    "WINDOWS",
    "KonnoOhmachi",
    "LogBinner",
    "LogWindow",
    "NoSmoothing",
    "SavitzkyGolay",
    "Smoother",
    "get_smoother",
    "is_smoothed",
    "record_smoothing",
]
