"""Spectral smoothing.

Separate from estimation, and lossy by design — see
:mod:`specmod.smoothing.base` for why a smoothed spectrum no longer satisfies
the Parseval contract that :mod:`specmod.transforms` guarantees.
"""

from __future__ import annotations

from ..core.spectrum import Spectrum
from .base import Smoother
from .konno_ohmachi import KonnoOhmachi
from .log_bins import LogBinner

#: Name -> smoother, for resolving `SmoothingConfig.method`.
SMOOTHERS: dict[str, type[Smoother]] = {
    "log_bins": LogBinner,
    "konno_ohmachi": KonnoOhmachi,
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
    "KonnoOhmachi",
    "LogBinner",
    "Smoother",
    "get_smoother",
    "is_smoothed",
]
