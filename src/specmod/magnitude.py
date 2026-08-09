"""Seismic moment and moment magnitude from a fitted long-period plateau.

This is the last step of the Edwards et al. (2010) spectral method: the fit has
already produced ``Omega``, the long-period displacement plateau, and this turns
it into ``M0`` and ``Mw``. Everything it needs is already computed —
``llpsp`` from the fit is ``log10(Omega)`` on **displacement** (source models
are written for displacement and :func:`specmod.sources.motion_scaling`
converts the model to whatever motion was recorded, so the fitted plateau is
``Omega`` itself rather than a velocity plateau needing correction), and a
distance is on every channel.

Constants and their sources
---------------------------

Read from Holt (2019), Ch. 1 §1.4 and Ch. 2 §2.2, which is the published form
of the method this package implements:

===================== ============================ ==============================
symbol                default                      what it is
===================== ============================ ==============================
``density``           2700 kg/m^3                  density **at the source**
``velocity``          3500 m/s                     S velocity **at the source**
``radiation_pattern`` 0.55                         average **SH** radiation pattern
``free_surface``      2                            free surface, vertically incident SH
``reference_dis...``  1000 m                       where the source spectrum sits
===================== ============================ ==============================

**0.55 is not a worse average of the same quantity than the textbook 0.63 — it
is a different quantity.** 0.63 is the RMS over total S; 0.55 is the SH
average, and it pairs with ``free_surface = 2`` for vertically incident SH.
Phase, component and radiation-pattern constant are one choice, made once: a
formulation using the SH radiation pattern wants the transverse component, and
`§4.7 <../docs/REFACTOR_PLAN.md>`_ tracks the rotation work that makes that
true here.

Ch. 2 enumerates the constants of its Eq. 2.7 **without** a partition factor,
though Ch. 1 lists one among the parts of the generic constant. So none is
folded in here. A study that wants one should say so rather than inherit it.

The two distances, which are not the same distance
--------------------------------------------------

This is the trap the whole module is arranged around, and two people can each
be certain about "the unit of R" and both be right:

* ``reference_distance_m`` is in **metres**, alongside density in kg/m^3 and
  velocity in m/s, which is what makes ``M0`` come out in newton-metres.
* The **spreading** model's distance is in **kilometres**, because that is how
  every published spreading table is written.

They cancel exactly for the default ``1/R``, which is why the short form
``M0 = 4 pi rho beta^3 R Omega / (Theta F)`` circulates and works. They do not
cancel for any other exponent, or for a piecewise or tabulated model, so they
are kept separate here rather than folded together.

What this does and does not establish
-------------------------------------

The spreading **exponent** is 1 on theory — amplitude decays as ``1/R``,
energy as ``1/R**2``, and a moment expression corrects an amplitude — and on
the thesis's own inverted near-field values of 0.88 and 0.90.

**The absolute calibration is not settled, and the size of the gap depends on
a question this repository has not answered.** With the shipped defaults the
PNR event comes out near Mw 3.1, and near Mw 2.75 with the constants §4.7
used. The catalogue value carried through this repository is **1.6, quoted as
"Mw" but nowhere sourced** — and BGS reports the Preston New Road sequence in
**local magnitude**, the well-known 26 August 2019 event being 2.9 ML. If the
1.6 is likewise ML, comparing it against a moment magnitude is a category
error, and the two are not as far apart as they look:

- Holt (2019) Table 2.2, the ``[R]`` relation for ``ML < 2.60``, gives
  ``Mw = 0.67 ML + 1.03``. **ML 1.6 predicts Mw 2.10**, against 2.75 here — a
  gap of 0.65 rather than 1.15.
- Read the other way, Mw 2.75 implies **ML 2.57**.

That relation is Utah's, fitted to tectonic events with Utah's own spreading,
so applying it to shallow UK induced seismicity is an extrapolation across
both region and source type. The 2/3 *slope* travels better than the
intercept: it has a theoretical basis at small magnitudes (Deichmann, 2017 —
attenuation makes observed pulse durations nearly constant, so ML falls away
faster than Mw), where the intercept is regional.

So: **do not read the residual gap as a known bias.** Establishing whether the
catalogue value is ML or Mw would change its size materially, and it is the
cheapest thing to check. Beyond that the candidates are the horizontals being
fitted as independent measurements rather than combined, phase constants not
matching the phase measured, and a microseismic spreading regime the
theoretical ``1/R`` does not describe. Treat the output as a self-consistent
computation whose scale is pending calibration against Magna (§5.2.5).
:func:`moment_magnitude` is exact; what feeds it is a model.
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

#: The Hanks and Kanamori (1979) constant, for ``M0`` in **N m**.
#:
#: The thesis quotes the relation in SI — "the equivalent relation in SI units
#: of Newton meters (N.m) where 1 N.m = 1x10^7 dyne.cm" — so this is the right
#: form for the units used throughout. The value itself is the standard one;
#: the thesis's own equation images did not survive its PDF export, so it is
#: taken from Hanks and Kanamori rather than re-read from there.
_HK79_CONSTANT = 9.1


@dataclass(frozen=True)
class MediumConstants:
    """Properties **at the source**, and the geometry of the measurement.

    "At the source" is not a formality. These are the values where the rupture
    is, not at the station and not a crustal average — the velocity enters
    cubed, so taking it from the wrong depth is a larger error than it looks.
    """

    #: kg/m^3.
    density: float = 2700.0
    #: m/s. Shear-wave velocity at the source, cubed in the moment expression.
    velocity: float = 3500.0
    #: Average SH radiation pattern over the focal sphere (Boatwright, 1978).
    radiation_pattern: float = 0.55
    #: Free-surface factor, 2 for vertically incident SH.
    free_surface: float = 2.0
    #: Metres. Where the source spectrum is defined.
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
                f"cubed here, so the mistake is a factor of 1e9 in M0 and six "
                f"magnitude units in Mw — which is why this refuses rather "
                f"than computing it."
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
    """An event magnitude, and the station estimates it was averaged from.

    The per-station values are kept rather than reduced away, for the same
    reason :class:`specmod.staged.StagedFit` keeps stage one: the spread across
    stations is the only evidence for how well constrained the event value is,
    and a magnitude with no spread beside it cannot be judged.
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

    Takes the table rather than the :class:`~specmod.staged.StagedFit` so that
    a stage-one table, a stage-two table or a flatfile read back from disk all
    work the same way. Returns a copy with ``m0`` and ``mw`` appended; the input
    is not modified.
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

    The aggregation follows the published method (Holt, 2019 §2.2) rather than
    being invented here: take the sample mean of the station magnitudes,
    exclude anything beyond ``outlier_sigma`` standard deviations, and refuse
    to report an event at all on fewer than ``min_stations``.

    Averaging in **magnitude** rather than in ``M0`` is deliberate and is what
    the method specifies. The two differ — a mean of logarithms is a geometric
    mean of moments — and for scattered station estimates the log-domain mean
    is the more robust of the two, which is the point of averaging across a
    network at all.

    Rejection is a single pass, not iterated to convergence. Repeating it until
    nothing moves will eventually trim a merely-broad distribution down to its
    core and report a spread far tighter than the data support.
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
