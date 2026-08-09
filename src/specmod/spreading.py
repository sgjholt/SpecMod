"""Geometric spreading models, as a registry.

Each model returns the **dimensionless** amplitude ratio between the reference
distance and the site, so an observed plateau is corrected to the source by
dividing by it. Distances are in **kilometres**, as published spreading tables
are written.

:class:`PowerLaw` is the default at ``exponent=1``, the theoretical body-wave
value. :class:`Piecewise` takes the contiguous segments regional models are
published as, :class:`Tabulated` interpolates a supplied curve in log-log
space, and :data:`HOLT_2019_UTAH` is the refined Utah model of Holt (2019)
Table 2.1.

Register new models in :data:`SPREADING_MODELS`; build one by name with
:func:`get_spreading_model`. Why the default is not fitted, and why a single
event cannot measure an exponent, is §4.7 of ``docs/REFACTOR_PLAN.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "HOLT_2019_UTAH",
    "SPREADING_MODELS",
    "Piecewise",
    "PowerLaw",
    "SpreadingModel",
    "Tabulated",
    "get_spreading_model",
]


@runtime_checkable
class SpreadingModel(Protocol):
    """Amplitude decay from the reference distance to the site.

    ``name`` and ``reference_km`` are read-only so frozen dataclasses satisfy
    the protocol.
    """

    @property
    def name(self) -> str: ...  # pragma: no cover

    @property
    def reference_km(self) -> float: ...  # pragma: no cover

    def __call__(
        self, distance_km: ArrayLike
    ) -> NDArray[np.float64]: ...  # pragma: no cover


def _as_distance(distance_km: ArrayLike) -> NDArray[np.float64]:
    r = np.asarray(distance_km, dtype=np.float64)
    if np.any(r <= 0):
        raise ValueError(
            "geometric spreading is undefined at zero or negative distance; "
            f"got {np.min(r)}"
        )
    return r


@dataclass(frozen=True)
class PowerLaw:
    """``(R_0 / R) ** exponent`` — a single power law.

    ``exponent=1`` is body-wave amplitude decay in a homogeneous whole space
    and is the default; ``0.5`` is the surface-wave value, for distance ranges
    where an Lg phase dominates the window. Note that ``1/R**2`` is how
    *energy* decays, and a moment expression corrects an amplitude.
    """

    exponent: float = 1.0
    reference_km: float = 1.0
    name: str = "power_law"

    def __call__(self, distance_km: ArrayLike) -> NDArray[np.float64]:
        r = _as_distance(distance_km)
        out: NDArray[np.float64] = (self.reference_km / r) ** self.exponent
        return out


@dataclass(frozen=True)
class Piecewise:
    """A contiguous piecewise power law, the shape regional models are published in.

    ``segments`` is ``((exponent, upper_km), ...)`` in increasing distance.
    Each segment starts where the previous one ended, so the decay accumulates
    as a product and the curve is continuous across every hinge — which is what
    the published tables mean.

    Beyond the last boundary the final exponent continues rather than raising,
    so one far station cannot drop an event. The segment table records where
    the fitted evidence stopped.
    """

    segments: tuple[tuple[float, float], ...]
    reference_km: float = 1.0
    name: str = "piecewise"

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a piecewise spreading model needs at least one segment")
        edges = [upper for _, upper in self.segments]
        if any(b <= a for a, b in pairwise(edges)):
            raise ValueError(
                f"segment boundaries must increase with distance, got {edges}"
            )
        if edges[0] <= self.reference_km:
            raise ValueError(
                f"the first segment ends at {edges[0]} km, at or below the "
                f"reference distance {self.reference_km} km, so it spans nothing"
            )

    def __call__(self, distance_km: ArrayLike) -> NDArray[np.float64]:
        r = _as_distance(distance_km)
        log_g = np.zeros_like(r)
        lower = float(self.reference_km)
        for exponent, upper in self.segments:
            # How far into this segment each distance reaches. Distances short
            # of it contribute nothing; distances past it contribute the whole
            # segment, which is what makes the product contiguous.
            reach = np.clip(r, lower, upper)
            log_g -= exponent * np.log10(reach / lower)
            lower = float(upper)
        beyond = r > lower
        if np.any(beyond):
            tail = self.segments[-1][0]
            log_g[beyond] -= tail * np.log10(r[beyond] / lower)
        out: NDArray[np.float64] = np.power(10.0, log_g)
        return out


@dataclass(frozen=True)
class Tabulated:
    """A spreading curve supplied as data, interpolated in log-log space.

    For spreading that has no functional form — a non-parametric ``G(R)``
    inversion, for instance. Interpolation is linear in ``log10`` of both axes,
    the space such curves are inverted in.
    """

    distances_km: tuple[float, ...]
    values: tuple[float, ...]
    reference_km: float = 1.0
    name: str = "tabulated"

    def __post_init__(self) -> None:
        if len(self.distances_km) != len(self.values):
            raise ValueError(
                f"got {len(self.distances_km)} distances and "
                f"{len(self.values)} values; a table needs one of each"
            )
        if len(self.distances_km) < 2:
            raise ValueError("a tabulated model needs at least two points")
        d = np.asarray(self.distances_km, dtype=np.float64)
        if np.any(d[1:] <= d[:-1]):
            raise ValueError("tabulated distances must increase")
        if np.any(d <= 0) or np.any(np.asarray(self.values) <= 0):
            raise ValueError("tabulated distances and values must be positive")

    def __call__(self, distance_km: ArrayLike) -> NDArray[np.float64]:
        r = _as_distance(distance_km)
        out: NDArray[np.float64] = np.power(
            10.0,
            np.interp(
                np.log10(r),
                np.log10(np.asarray(self.distances_km, dtype=np.float64)),
                np.log10(np.asarray(self.values, dtype=np.float64)),
            ),
        )
        return out


#: Holt (2019) Table 2.1, the refined Utah model ``Holt et al. [R]``, which
#: §2.7 of that work recommends over the original ``[O]``. Bootstrap standard
#: deviations on the four slopes are 0.01, 0.07, 0.08 and 0.04.
#:
#: Regional and fitted for 1-400 km. It is not a default and says nothing about
#: the microseismic range below its first hinge.
HOLT_2019_UTAH = Piecewise(
    segments=((0.90, 43.0), (2.57, 76.0), (0.44, 136.0), (1.54, 400.0)),
    name="holt_2019_utah",
)


#: Registered models, by the name configuration refers to them by.
SPREADING_MODELS: dict[str, Any] = {
    "power_law": PowerLaw,
    "piecewise": Piecewise,
    "tabulated": Tabulated,
}


def get_spreading_model(name: str, **kwargs: Any) -> SpreadingModel:
    """Build a registered spreading model by name."""
    try:
        factory = SPREADING_MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown spreading model {name!r}. Available: {sorted(SPREADING_MODELS)}."
        ) from None
    model: SpreadingModel = factory(**kwargs)
    return model
