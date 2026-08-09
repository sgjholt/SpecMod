"""Seismic moment and moment magnitude from a fitted long-period plateau.

The final step of the Edwards et al. (2010) spectral method. The fit supplies
``Omega``, the long-period **displacement** plateau in ``m s`` — ``llpsp`` is
its base-10 logarithm — and each channel supplies a distance.

    M0 = 4 pi rho beta**3 R_0 (Omega / G(R)) / (Theta F)      [N m]
    Mw = (2/3) (log10 M0 - 9.1)

Units
-----

``rho`` in kg/m^3, ``beta`` in m/s, ``R_0`` in **metres**, ``Omega`` in ``m s``
give ``M0`` in newton-metres. ``G(R)`` is :mod:`specmod.spreading`, whose
distance is in **kilometres**. These are two different distances and both units
are correct; they cancel only at spreading exponent 1.

Constants
---------

Defaults are the S-wave values of Holt (2019) Ch. 1 §1.4 and Ch. 2 §2.2, all
taken **at the source**:

===================== ========== =============================================
``density``           2700       kg/m^3
``velocity``          3500       m/s, cubed here
``radiation_pattern`` 0.55       average SH pattern over the focal sphere
``free_surface``      2          vertically incident SH
``reference_dist...`` 1000       metres
===================== ========== =============================================

``0.55`` is the **SH** average and pairs with ``free_surface = 2``; the
textbook 0.63 is the RMS over total S, a different quantity. No partition
factor is applied.

Use :func:`seismic_moment` and :func:`moment_magnitude` for arrays,
:func:`station_moments` to append ``m0`` and ``mw`` to a fit table, and
:func:`event_magnitude` for an event value aggregated over stations.

The absolute calibration is unverified — see §4.7 of
``docs/REFACTOR_PLAN.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .spreading import PowerLaw, SpreadingModel

__all__ = [
    "MediumConstants",
    "MomentMagnitude",
    "event_magnitude",
    "moment_magnitude",
    "seismic_moment",
    "station_moments",
]

#: Hanks and Kanamori (1979), for ``M0`` in N m. Use 16.1 for dyne cm.
_HK79_CONSTANT = 9.1


@dataclass(frozen=True)
class MediumConstants:
    """Medium properties at the source, and the geometry of the measurement.

    Values are those at the rupture, not at the station and not a crustal
    average. ``velocity`` enters cubed.
    """

    #: kg/m^3.
    density: float = 2700.0
    #: m/s, shear-wave velocity at the source.
    velocity: float = 3500.0
    #: Average SH radiation pattern over the focal sphere (Boatwright, 1978).
    radiation_pattern: float = 0.55
    #: Free-surface factor; 2 for vertically incident SH.
    free_surface: float = 2.0
    #: Metres. The distance at which the source spectrum is defined.
    reference_distance_m: float = 1000.0

    def __post_init__(self) -> None:
        for name in (
            "density",
            "velocity",
            "radiation_pattern",
            "free_surface",
            "reference_distance_m",
        ):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.velocity < 100.0:
            raise ValueError(
                f"velocity={self.velocity} looks like km/s, not m/s. It is "
                f"cubed here, so the error would be a factor of 1e9 in M0 and "
                f"six magnitude units in Mw."
            )


def seismic_moment(
    omega: ArrayLike,
    distance_km: ArrayLike,
    *,
    constants: MediumConstants | None = None,
    spreading: SpreadingModel | None = None,
) -> NDArray[np.float64]:
    """Scalar seismic moment in **N m**, from plateaus and their distances.

    ``omega`` is the long-period displacement plateau in ``m s`` — the fit's
    ``llpsp`` is its base-10 logarithm. ``distance_km`` is source-to-site, in
    kilometres, one per plateau.
    """
    medium = constants if constants is not None else MediumConstants()
    model: SpreadingModel = spreading if spreading is not None else PowerLaw()

    plateau = np.asarray(omega, dtype=np.float64)
    r = np.asarray(distance_km, dtype=np.float64)
    if plateau.shape != r.shape:
        raise ValueError(
            f"got {plateau.size} plateaus and {r.size} distances; "
            f"each plateau needs its own distance"
        )
    if np.any(plateau <= 0):
        raise ValueError(
            "the plateau must be positive and linear, not logarithmic. The "
            "fit reports log10(Omega) as `llpsp`; raise it to the power of ten "
            "before passing it here."
        )

    # Correct the observed plateau back to the reference distance, then apply
    # the source term. The reference distance is in metres because it sits
    # beside kg/m^3 and m/s; the spreading model's distance is in kilometres.
    at_source = plateau / model(r)
    source_term = (
        4.0
        * math.pi
        * medium.density
        * medium.velocity**3
        * medium.reference_distance_m
    )
    out: NDArray[np.float64] = (
        source_term * at_source / (medium.radiation_pattern * medium.free_surface)
    )
    return out


def moment_magnitude(m0: ArrayLike) -> NDArray[np.float64]:
    """Hanks and Kanamori (1979), for ``m0`` in **N m**."""
    moment = np.asarray(m0, dtype=np.float64)
    if np.any(moment <= 0):
        raise ValueError("seismic moment must be positive to take its logarithm")
    out: NDArray[np.float64] = (2.0 / 3.0) * (np.log10(moment) - _HK79_CONSTANT)
    return out


@dataclass(frozen=True)
class MomentMagnitude:
    """An event magnitude and the station estimates behind it.

    Per-station values are kept so :meth:`spread` can report how well
    constrained the event value is.
    """

    value: float
    m0: float
    stations: tuple[str, ...]
    station_magnitudes: tuple[float, ...]
    excluded: dict[str, str] = field(default_factory=dict)
    constants: MediumConstants = field(default_factory=MediumConstants)
    spreading_model: str = "power_law"
    distance_measure: str = "rhyp"
    #: Always ``"Mw"``. Carried so the number cannot be read without its scale.
    unit: str = "Mw"

    def spread(self) -> dict[str, float]:
        """How much the stations disagreed, in magnitude units."""
        if not self.station_magnitudes:
            return {}
        values = np.asarray(self.station_magnitudes, dtype=np.float64)
        return {
            "n": float(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }


def station_moments(
    table: Any,
    *,
    constants: MediumConstants | None = None,
    spreading: SpreadingModel | None = None,
    distance_measure: str = "rhyp",
    plateau_column: str = "llpsp",
) -> Any:
    """Per-station ``M0`` and ``Mw`` from a fit table.

    Takes a table rather than a :class:`~specmod.staged.StagedFit`, so either
    stage or a flatfile read from disk all work. Returns a copy with ``m0`` and
    ``mw`` appended; the input is not modified.
    """
    import pandas as pd  # noqa: PLC0415

    for column in (plateau_column, distance_measure, "id"):
        if column not in table.columns:
            raise ValueError(
                f"the fit table has no {column!r} column, so a magnitude "
                f"cannot be computed from it. Available: "
                f"{sorted(table.columns)}"
            )

    out = table.copy()
    usable = out[plateau_column].notna() & out[distance_measure].notna()
    m0 = np.full(len(out), np.nan)
    if usable.any():
        m0[usable.to_numpy()] = seismic_moment(
            np.power(10.0, out.loc[usable, plateau_column].to_numpy(dtype=np.float64)),
            out.loc[usable, distance_measure].to_numpy(dtype=np.float64),
            constants=constants,
            spreading=spreading,
        )
    out["m0"] = m0
    finite = ~np.isnan(m0)
    mw = np.full(len(out), np.nan)
    if finite.any():
        mw[finite] = moment_magnitude(m0[finite])
    out["mw"] = mw
    return pd.DataFrame(out)


def event_magnitude(
    staged: Any,
    *,
    constants: MediumConstants | None = None,
    spreading: SpreadingModel | None = None,
    distance_measure: str = "rhyp",
    outlier_sigma: float = 2.5,
    min_stations: int = 3,
) -> MomentMagnitude:
    """Event ``Mw``: the mean of station estimates, after rejecting outliers.

    Follows Holt (2019) §2.2 — the sample mean of station magnitudes, excluding
    anything beyond ``outlier_sigma`` standard deviations, and no event value
    at all on fewer than ``min_stations``.

    Averaging in magnitude rather than in ``M0`` makes this a geometric mean of
    moments. Rejection is a single pass, not iterated to convergence.
    """
    medium = constants if constants is not None else MediumConstants()
    model: SpreadingModel = spreading if spreading is not None else PowerLaw()

    table = station_moments(
        staged.table,
        constants=medium,
        spreading=model,
        distance_measure=distance_measure,
    )
    contributing = set(getattr(staged, "contributing", ()) or ())
    if contributing:
        table = table[table["id"].isin(contributing)]
    table = table[table["mw"].notna()]

    ids = tuple(str(i) for i in table["id"])
    values = table["mw"].to_numpy(dtype=np.float64)
    if values.size < min_stations:
        raise ValueError(
            f"{values.size} station magnitude(s) available, below "
            f"min_stations={min_stations}. Fewer than three stations cannot "
            f"average out the radiation pattern, so the published method "
            f"declines to report an event value rather than reporting a poor "
            f"one."
        )

    excluded: dict[str, str] = {}
    keep = np.ones(values.size, dtype=bool)
    if values.size > 1 and outlier_sigma > 0:
        sigma = float(values.std(ddof=1))
        if sigma > 0:
            deviation = np.abs(values - values.mean())
            keep = deviation <= outlier_sigma * sigma
            for id, value, kept in zip(ids, values, keep, strict=True):
                if not kept:
                    excluded[id] = f"Mw {value:.2f}, beyond {outlier_sigma} sigma"

    if keep.sum() < min_stations:
        # Rejecting below the floor means the scatter, not a few bad stations,
        # is the problem. Report the unrejected set rather than nothing.
        keep = np.ones(values.size, dtype=bool)
        excluded = {}

    kept_ids = tuple(i for i, k in zip(ids, keep, strict=True) if k)
    kept_values = values[keep]
    mean_mw = float(kept_values.mean())
    m0 = float(np.power(10.0, 1.5 * mean_mw + _HK79_CONSTANT))

    return MomentMagnitude(
        value=mean_mw,
        m0=m0,
        stations=kept_ids,
        station_magnitudes=tuple(float(v) for v in kept_values),
        excluded=excluded,
        constants=medium,
        spreading_model=getattr(model, "name", "power_law"),
        distance_measure=distance_measure,
    )
