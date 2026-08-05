"""Converting a displacement source spectrum to the recorded motion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.units import Motion

__all__ = ["motion_scaling"]

#: Differentiation in the frequency domain multiplies by ``2 pi f`` per order.
_ORDERS: dict[Motion, int] = {
    Motion.DISPLACEMENT: 0,
    Motion.VELOCITY: 1,
    Motion.ACCELERATION: 2,
}


def motion_scaling(
    freq: NDArray[np.float64], motion: Motion | str
) -> NDArray[np.float64]:
    """``log10 G(f)``: the factor taking displacement to ``motion``.

    Source models are written for displacement, because that is where
    ``Omega`` and hence ``M0`` are defined. Records are usually velocity. Each
    order of differentiation multiplies by ``2 pi f``, so in log space it is
    that many copies of ``log10(2 pi f)``.

    The order comes from :class:`specmod.core.units.Motion` rather than a
    string comparison, so an unrecognised motion fails at the enum instead of
    silently returning zero — which would fit a displacement model to velocity
    data and put ``Omega`` out by a factor of ``2 pi f``.
    """
    order = _ORDERS[Motion(motion)]
    if order == 0:
        return np.zeros_like(np.asarray(freq, dtype=np.float64))
    return np.asarray(
        order * np.log10(2.0 * np.pi * np.asarray(freq, dtype=np.float64))
    )
