"""Pluggable time-to-frequency estimators.

All backends satisfy the contract in :mod:`specmod.transforms.base`, so one
test suite pins every one of them.
"""

from __future__ import annotations

from .base import SpectralEstimator, TaperCorrection
from .fft import FFTEstimator, WelchEstimator
from .multitaper import MultitaperEstimator

#: Name -> estimator, for resolving `TransformConfig.estimator`.
ESTIMATORS: dict[str, type[SpectralEstimator]] = {
    "fft": FFTEstimator,
    "welch": WelchEstimator,
    "multitaper": MultitaperEstimator,
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
    "FFTEstimator",
    "MultitaperEstimator",
    "SpectralEstimator",
    "TaperCorrection",
    "WelchEstimator",
    "get_estimator",
]
