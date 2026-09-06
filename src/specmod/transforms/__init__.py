"""Pluggable time-to-frequency estimators.

All backends satisfy the contract in :mod:`specmod.transforms.base`, so one
test suite pins every one of them.
"""

from __future__ import annotations

# Everything `base.__all__` names public is re-exported here, so no public
# name is reachable only through a submodule path. `specmod.api` publishes
# `make_window` and `window_correction`, and `docs/api.md` documents packages
# at the path you import from — a name missing here has nowhere to be
# documented, and its viewcode backlink points at an anchor that is never
# written.
from .base import (
    SpectralEstimator,
    TaperCorrection,
    make_window,
    prepare_record,
    window_correction,
)
from .cwt import CWTEstimator
from .fft import FFTEstimator, WelchEstimator
from .multitaper import MultitaperEstimator
from .prieto import PrietoMultitaperEstimator
from .quadratic import QuadraticMultitaperEstimator

#: Name -> estimator, for resolving `TransformConfig.estimator`.
ESTIMATORS: dict[str, type[SpectralEstimator]] = {
    "fft": FFTEstimator,
    "welch": WelchEstimator,
    "multitaper": MultitaperEstimator,
    "prieto": PrietoMultitaperEstimator,
    "cwt": CWTEstimator,
    "quadratic": QuadraticMultitaperEstimator,
}


def get_estimator(name: str, **kwargs: object) -> SpectralEstimator:
    """Construct an estimator by name."""
    try:
        cls = ESTIMATORS[name]
    except KeyError:
        raise ValueError(
            f"Unknown estimator {name!r}. Available: {sorted(ESTIMATORS)}."
        ) from None
    estimator: SpectralEstimator = cls(**kwargs)
    return estimator


__all__ = [
    "ESTIMATORS",
    "CWTEstimator",
    "FFTEstimator",
    "MultitaperEstimator",
    "PrietoMultitaperEstimator",
    "QuadraticMultitaperEstimator",
    "SpectralEstimator",
    "TaperCorrection",
    "WelchEstimator",
    "get_estimator",
    "make_window",
    "prepare_record",
    "window_correction",
]
