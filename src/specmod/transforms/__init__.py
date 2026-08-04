"""Pluggable time-to-frequency estimators.

All backends satisfy the contract in :mod:`specmod.transforms.base`, so one
test suite pins every one of them.
"""

from __future__ import annotations

from .base import SpectralEstimator, TaperCorrection
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
]
