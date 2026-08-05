"""Source spectral shapes, and what each implies about the source itself."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SOURCE_MODELS",
    "BoatwrightSource",
    "BruneSource",
    "SourceModel",
    "get_source_model",
]


@runtime_checkable
class SourceModel(Protocol):
    """A source spectral shape, plus what it implies about the source.

    The second half is the part that is easy to leave out. Two models can
    share a spectral shape exactly and still disagree by an order of magnitude
    on stress drop, because they disagree about what a given corner frequency
    says about the rupture dimension.
    """

    @property
    def name(self) -> str:
        """Short identifier, as configuration refers to it."""
        ...

    @property
    def corner_frequency_coefficient(self) -> tuple[float, float]:
        """``(k_P, k_S)`` in ``f_c = k * beta / r``.

        The bridge from a fitted corner frequency to a source radius, and
        therefore to stress drop. It belongs to the model rather than to
        whatever computes stress drop, because **models that share a spectral
        shape do not share this**.
        """
        ...

    def log10_shape(
        self, freq: NDArray[np.float64], log10_omega: float, f_c: float
    ) -> NDArray[np.float64]:
        """``log10 S(f)`` for a plateau ``log10_omega`` and corner ``f_c``."""
        ...


@dataclass(frozen=True)
class _GeneralisedSource:
    """The Boatwright family, of which Brune is the ``gamma = 1`` member.

    .. math::

        \\log_{10} S(f) = \\log_{10}\\Omega
            - \\frac{1}{\\gamma}\\log_{10}\\left[1 + (f/f_c)^{\\gamma n}\\right]

    ``n`` is the high-frequency falloff — 2 for an omega-squared model — and
    ``gamma`` controls how sharply the spectrum turns at the corner. Larger
    ``gamma`` is a sharper knee.
    """

    gamma: float
    n: float

    def log10_shape(
        self, freq: NDArray[np.float64], log10_omega: float, f_c: float
    ) -> NDArray[np.float64]:
        ratio = np.asarray(freq, dtype=np.float64) / f_c
        return np.asarray(
            log10_omega
            - (1.0 / self.gamma) * np.log10(1.0 + ratio ** (self.gamma * self.n))
        )


@dataclass(frozen=True)
class BruneSource(_GeneralisedSource):
    """Brune (1970), omega-squared with a smooth corner.

    The corner-frequency coefficients are Brune's own kinematic values. They
    are **not** interchangeable with Madariaga's — see the note in
    :mod:`specmod.sources`.
    """

    gamma: float = 1.0
    n: float = 2.0

    @property
    def name(self) -> str:
        return "brune"

    @property
    def corner_frequency_coefficient(self) -> tuple[float, float]:
        return (0.42, 0.37)


@dataclass(frozen=True)
class BoatwrightSource(_GeneralisedSource):
    """Boatwright (1980), omega-squared with a sharper corner than Brune.

    ``gamma = 2`` narrows the transition; the high-frequency falloff is the
    same ``f**-2``. The coefficients here are Brune's, because Boatwright's
    formulation shares the kinematic radius relation — recorded explicitly so
    that it is a stated choice rather than an omission.
    """

    gamma: float = 2.0
    n: float = 2.0

    @property
    def name(self) -> str:
        return "boatwright"

    @property
    def corner_frequency_coefficient(self) -> tuple[float, float]:
        return (0.42, 0.37)


#: Registered source models, by the name configuration refers to them by.
#:
#: **Madariaga belongs here next.** It is omega-squared and sits at the same
#: ``(gamma, n) = (1, 2)`` as Brune, so it registers as a shape identical to
#: Brune's with a *different*
#: :attr:`~SourceModel.corner_frequency_coefficient` — which is the entire
#: content of the difference and the reason that attribute exists. Take the
#: coefficients from the source paper rather than from any summary, including
#: this one: they differ between P and S and between authors, and they land
#: directly in published stress drops.
SOURCE_MODELS: dict[str, type[BruneSource] | type[BoatwrightSource]] = {
    "brune": BruneSource,
    "boatwright": BoatwrightSource,
}


def get_source_model(name: str) -> SourceModel:
    """Resolve a registered source model by name, with its defaults."""
    try:
        cls = SOURCE_MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown source model {name!r}. Available: {sorted(SOURCE_MODELS)}."
        ) from None
    return cls()
