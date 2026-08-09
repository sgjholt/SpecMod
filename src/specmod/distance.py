"""Source-to-site distance, as a registry rather than a stat name.

Which distance you mean is a modelling choice, and at short range it is not a
small one. On the PNR data the nearest station is **1.02 km epicentral against
2.30 km hypocentral** — a factor of 2.24 — while the farthest agree to 1.004.
Anything weighted by inverse distance, or corrected for geometric spreading,
therefore depends on the choice most strongly at exactly the station that
matters most.

Two are implemented here because they are the two a point source supports.
Both read a value :func:`specmod.preprocess.set_stream_distance` has already
computed:

``repi``
    Epicentral. Horizontal distance from the epicentre.
``rhyp``
    Hypocentral. Slant distance from the hypocentre.

**Epicentral is the honest choice when sensor depths are unknown**, and that
is more often than it sounds. ``rhyp`` is built from the source depth and the
station *elevation*, which silently assumes every sensor sits at the surface.
For a borehole deployment that is wrong by the burial depth, and nothing in
the metadata announces it — the PNR inventory records channel ``depth`` as
``123456.0``, a placeholder, so on that dataset ``rhyp`` is an assumption
wearing a measurement's name.

Finite-fault measures
---------------------
``Rrup`` (closest distance to the rupture surface) and ``Rjb`` (Joyner-Boore,
closest horizontal distance to the surface projection of the rupture) are the
measures ground-motion work generally wants, and they are **not implemented**
— deliberately, rather than by omission.

Both need a rupture *surface*: strike, dip, length, width and a hypocentre
position on it. SpecMod carries a point source, so there is nothing to compute
them from, and a version that quietly degenerated to ``rhyp`` and ``repi``
would be worse than an error — those are exactly what `Rrup` and `Rjb` reduce
to for a point source, so the substitution would be invisible in the output
and wrong for any event large enough to warrant asking.

They are registered all the same, raising with what they would need. A name
that resolves to a clear failure is a better extension point than a name that
does not resolve at all, and it puts the requirement where someone adding
finite-fault support will read it.

The registry is the same shape as :data:`specmod.transforms.ESTIMATORS`,
:data:`specmod.core.noise.NOISE_MODELS` and
:data:`specmod.staged.WEIGHT_MODELS`, so a study names a distance the way it
names anything else and the choice travels with the resolved configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .config import load_config

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "DISTANCE_MEASURES",
    "DistanceMeasure",
    "Epicentral",
    "FiniteFaultDistance",
    "Hypocentral",
    "get_distance_measure",
    "resolve_distance_measure",
]


@runtime_checkable
class DistanceMeasure(Protocol):
    """One distance per channel, in kilometres."""

    name: str

    def distances(
        self, spectra: Any, ids: Sequence[str]
    ) -> NDArray[np.float64]: ...  # pragma: no cover


@dataclass(frozen=True, slots=True)
class _FromMeta:
    """A distance already computed onto the trace metadata."""

    key: str
    name: str

    def distances(self, spectra: Any, ids: Sequence[str]) -> NDArray[np.float64]:
        out = np.empty(len(ids), dtype=np.float64)
        for i, id in enumerate(ids):
            meta = spectra[id].signal.meta
            if self.key not in meta:
                raise ValueError(
                    f"{id} carries no {self.key!r}, so its {self.name} distance "
                    f"is unknown. Set the geometry with "
                    f"specmod.preprocess.set_stream_distance."
                )
            value = float(meta[self.key])
            if value <= 0:
                raise ValueError(
                    f"{id} has {self.key}={value}, which is not a distance"
                )
            out[i] = value
        return out


@dataclass(frozen=True, slots=True)
class Epicentral(_FromMeta):
    key: str = "repi"
    name: str = "epicentral"


@dataclass(frozen=True, slots=True)
class Hypocentral(_FromMeta):
    key: str = "rhyp"
    name: str = "hypocentral"


@dataclass(frozen=True, slots=True)
class FiniteFaultDistance:
    """``Rrup`` and ``Rjb``: registered, and not implemented.

    Raising here rather than omitting the name is the point. For a point source
    these degenerate exactly to hypocentral and epicentral, so an
    implementation that silently fell back would produce plausible numbers that
    are wrong for any event big enough to justify asking for them.
    """

    name: str
    needs: str

    def distances(self, spectra: Any, ids: Sequence[str]) -> NDArray[np.float64]:
        raise NotImplementedError(
            f"{self.name} is not implemented. It needs {self.needs}, and "
            f"SpecMod carries a point source — there is no rupture surface to "
            f"measure from. For a point source {self.name} degenerates to "
            f"{'hypocentral' if self.name == 'rrup' else 'epicentral'}; name "
            f"that instead if it is what you mean, rather than getting it by "
            f"accident."
        )


#: Registered distance measures, resolved by name from configuration.
DISTANCE_MEASURES: dict[str, Any] = {
    "repi": Epicentral,
    "rhyp": Hypocentral,
    "rrup": lambda: FiniteFaultDistance(
        name="rrup", needs="a rupture surface — strike, dip, length and width"
    ),
    "rjb": lambda: FiniteFaultDistance(
        name="rjb",
        needs="the surface projection of a rupture — strike, dip, length and width",
    ),
}


def get_distance_measure(name: str) -> DistanceMeasure:
    """Resolve a registered measure by name."""
    try:
        factory = DISTANCE_MEASURES[name]
    except KeyError:
        raise ValueError(
            f"Unknown distance measure {name!r}. "
            f"Available: {sorted(DISTANCE_MEASURES)}."
        ) from None
    measure: DistanceMeasure = factory()
    return measure


def resolve_distance_measure(
    measure: str | DistanceMeasure | None = None,
) -> DistanceMeasure:
    """A measure from a name, an instance, or the configuration.

    ``None`` takes ``[geometry] distance_measure``, which is the project-wide
    choice. It lived in ``[windows]`` and had no reader at all until this
    module; cutting a window does not depend on how distance is measured.
    """
    if measure is None:
        measure = str(load_config().config.geometry.distance_measure)
    if isinstance(measure, str):
        return get_distance_measure(measure)
    return measure
