"""Geometric spreading, as a registry rather than a hardcoded exponent.

Spreading is the term that carries a spectral amplitude from the source to the
site, and it is the one input to :mod:`specmod.magnitude` least checkable by
any other means. Density and velocity are bounded by physics and by the
literature; a spreading function is bounded only by the inversion that produced
it. So it is a model an operator supplies, in the same shape as
:data:`specmod.core.noise.NOISE_MODELS` and
:data:`specmod.distance.DISTANCE_MEASURES`.

**Every model here returns a dimensionless factor**, the amplitude ratio
between the reference distance and the site. An observed plateau is corrected
to the source by *dividing* by it. Returning a ratio rather than a raw
``R**-n`` is what makes the reference distance explicit instead of an implied
unit convention: `1 km` here is the thesis's ``R_0 = 1000 m``, the distance at
which the source spectrum is defined.

Distances are in **kilometres**, which is how every published spreading model
tabulates them — Holt (2019) Table 2.1 says so outright, "R in all cells is
hypocentral distance in kilometres". The metres in the moment expression are a
different quantity; see :mod:`specmod.magnitude`.

The default is theoretical ``1/R``. That is defensible precisely because it is
not fitted: §4.7 of ``docs/REFACTOR_PLAN.md`` records what happens when a
single event is asked for its own spreading exponent — a bilinear fit to the 28
PNR channels buys 5% of rms for two extra parameters across one decade of
distance, which is a hinge finding scatter rather than a break. Spreading
separates from site response only across a dataset where each station sees many
distances, so one event cannot measure it and should not pretend to.
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

    ``name`` and ``reference_km`` are declared read-only so that the frozen
    dataclasses below satisfy the protocol. A bare annotation would demand a
    *settable* attribute, which no frozen implementation can offer.
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

    ``exponent=1`` is body-wave amplitude decay in a homogeneous whole space,
    and is the default. **It is 1 rather than 2, and that is worth stating
    because it is easy to say the other one**: energy decays as ``1/R**2``,
    amplitude as ``1/R``, and a moment expression corrects an amplitude.
    Measured on the 28 PNR windows the difference is not subtle — ``1/R**2``
    puts the event at Mw 5.39 against a catalogue 1.6, which is the distance
    term applied twice.

    ``exponent=0.5`` is the surface-wave value, for the distance ranges where
    an Lg phase dominates the window.
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

    ``segments`` is ``((exponent, upper_km), ...)`` in increasing distance. The
    model is continuous by construction: each segment starts where the previous
    one ended, so the decay accumulates as a product rather than each segment
    being anchored independently. That is what the published tables mean, and
    evaluating a segment in isolation would silently drop everything the wave
    lost getting there.

    Beyond the last boundary the final exponent continues, rather than raising.
    A model fitted to 400 km says nothing about 500 km, but refusing outright
    would make a single far station drop an event; the extrapolation is the
    lesser evil and the segment table records where the evidence stopped.
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

    Required rather than a nicety: the non-parametric ``G(R)`` inversion is not
    in this package (§5.2.5), so the Magna comparison has to be fed its
    spreading as a table. A registry that accepted only functional forms would
    make the better-constrained model the one it could not express.

    Interpolation is linear in ``log10`` of both axes because that is the space
    the curve is a straight line in, and the space it was inverted in.
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


#: Holt (2019) Table 2.1, the **refined** Utah model — ``Holt et al. [R]``.
#:
#: The thesis prefers this over the original ``[O]`` (§2.7) on lower
#: uncertainty across every slope, more events resolved, and the Mw-Mc relation
#: moving closer to Mw-ML. Slope uncertainties from bootstrapping, in order:
#: 0.01, 0.07, 0.08, 0.04.
#:
#: **Regional, and not a default.** It is fitted for 1-400 km, and every
#: spreading model in that thesis breaks at 40-50 km — so it says nothing about
#: the microseismic range, where the PNR data used throughout this repository
#: sits entirely (2.3 to 22.9 km, inside the first segment of all of them).
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
