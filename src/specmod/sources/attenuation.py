"""Path attenuation, parameterised by ``t*``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ATTENUATION_MODELS",
    "AttenuationModel",
    "ConstantQ",
    "FrequencyDependentQ",
    "get_attenuation_model",
]


@runtime_checkable
class AttenuationModel(Protocol):
    """Anything that attenuates a source spectrum along the path."""

    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> tuple[str, ...]:
        """Free parameter names, in the order the fitter should take them."""
        ...

    def log10_decay(
        self, freq: NDArray[np.float64], *values: float
    ) -> NDArray[np.float64]:
        """``log10 D(f)``, given this model's parameters in order."""
        ...


@dataclass(frozen=True)
class ConstantQ:
    """Frequency-independent ``t*``.

    .. math:: \\log_{10} D(f) = -\\pi f t^{*} / \\ln 10

    The ``ln 10`` is the conversion into base-10 logs, which is the space the
    fit is performed in. Getting it wrong scales ``t*`` by 2.3 and leaves the
    spectrum looking plausible.
    """

    @property
    def name(self) -> str:
        return "constant_q"

    @property
    def parameters(self) -> tuple[str, ...]:
        return ("ts",)

    def log10_decay(
        self, freq: NDArray[np.float64], *values: float
    ) -> NDArray[np.float64]:
        (ts,) = values
        return np.asarray(
            -(np.pi * np.asarray(freq, dtype=np.float64) * ts / np.log(10))
        )


@dataclass(frozen=True)
class FrequencyDependentQ:
    """``t*`` with a power-law frequency dependence.

    .. math:: \\log_{10} D(f) = -\\pi f^{1-a} t^{*} / \\ln 10

    ``a = 0`` recovers :class:`ConstantQ`. The two are separate models rather
    than one with a switch because they have different free parameters, and a
    fitter needs to know that before it starts.
    """

    @property
    def name(self) -> str:
        return "frequency_dependent_q"

    @property
    def parameters(self) -> tuple[str, ...]:
        return ("ts", "a")

    def log10_decay(
        self, freq: NDArray[np.float64], *values: float
    ) -> NDArray[np.float64]:
        ts, a = values
        f = np.asarray(freq, dtype=np.float64)
        return np.asarray(-(np.pi * f ** (1.0 - a) * ts / np.log(10)))


#: Registered attenuation models.
ATTENUATION_MODELS: dict[str, type[ConstantQ] | type[FrequencyDependentQ]] = {
    "constant_q": ConstantQ,
    "frequency_dependent_q": FrequencyDependentQ,
}


def get_attenuation_model(name: str) -> AttenuationModel:
    """Resolve a registered attenuation model by name."""
    try:
        cls = ATTENUATION_MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown attenuation model {name!r}. "
            f"Available: {sorted(ATTENUATION_MODELS)}."
        ) from None
    return cls()
