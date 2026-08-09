"""The two-stage event fit, and the channel selection that feeds it.

Fitting a source model to one spectrum is not a unique inversion. The source
corner and the path attenuation trade off against each other on the falling
limb, and two minimisers can reach the same reduced chi-squared at corner
frequencies differing by tens of percent — a factor of several in stress drop,
which scales as ``fc**3``. On the 28 PNR windows, Powell and ``leastsq`` land
at 21.26 Hz and 14.75 Hz on one station at redchi 0.0259 against 0.0254.

That is not resolvable from one spectrum by any minimiser, and the published
workflow does not try. ``f_c`` belongs to the **source**: every station sees
the same rupture, so there is one value of it for the event. ``t*`` belongs to
the **path**, and every station has a different one. So a station whose ``t*``
came out too high returns a corner that is too high, and the next station's
error does not point the same way. Averaging over the ensemble is not
cosmetic smoothing — it uses the fact that the quantity being averaged is
common to all of them while the contaminating one is not.

The two stages
--------------
1. ``Omega``, ``f_c`` and ``t*`` free at every station independently. The
   output is not the answer; it is N noisy estimates of one number plus N
   estimates of N different numbers.
2. The event ``f_c`` — a weighted mean over the stations that survived
   selection — is held fixed, and every station refits ``Omega`` and ``t*``
   against a corner it can no longer trade against.

Measured on the same 28 windows, with the same 28 channels contributing to
both: the two minimisers differ by a factor 1.44 in ``f_c`` at the worst
station and by **0.4%** on the event value. After stage two they agree to
0.23% on ``t*`` and 1.7e-3 log10 units on ``Omega``.

"With the same channels contributing" is load-bearing, and is the trap in this
module. See :attr:`ChannelSelection.require_pass` — comparing two minimisers
under the default selection compares two different ensembles, and gives 125%
rather than 0.4%.

Be clear about what that last number is not. Fixing ``f_c`` removes the
parameter the minimisers were disagreeing about, so of course they then agree.
What it shows is that the residual two-parameter problem is well conditioned:
once the corner is pinned, ``Omega`` and ``t*`` are determined by the spectrum
rather than negotiable. The judgement is concentrated into one number for the
whole event, and that number came from the ensemble.

Choosing which channels contribute
----------------------------------
Selection is the part that cannot be automated away, because it is where
quality control enters. A station with a bad instrument response, a clipped
record or a pick on the wrong phase produces a corner frequency that is
confidently wrong, and averaging it in moves the event value for every other
station.

So the ensemble is chosen by :class:`ChannelSelection`, which reads
``[fitting]`` and can be overridden per call. The order is fixed and each step
is recorded with a reason, so ``StagedFit.excluded`` says why any given
channel is not contributing:

1. anything not matching ``include`` (when ``include`` is non-empty),
2. anything matching ``exclude``,
3. anything whose stage-1 fit failed its bounds, when ``require_pass``,
4. anything left with no fit at all.

Patterns are shell globs, and they match at whichever level you write them.
A trace id is ``NET.STA.LOC.CHA``, and a pattern is tried against the whole
id, against ``NET.STA``, and against each component on its own — so all of
these do what they look like they do::

    "AQ07"              every channel of that station
    "UR"                every station of that network
    "UR.AQ07"           that station, spelled unambiguously
    "HHE"               every east component, at every station
    "HH?"               every high-gain broadband channel
    "UR.AQ07.00.HHE"    exactly that channel
    "LV.L00[123]..HH?"  a glob over the full id

Writing the station code alone is the common case after quality control —
a clipped record or a bad response is a property of the instrument, not of one
component — and needing ``"UR.AQ07.*"`` for it is the kind of detail that gets
mistyped as ``"AQ07"`` and silently matches nothing. Station and channel codes
do not collide in practice; where a pattern could match at two levels it
matches, and :attr:`StagedFit.excluded` records which level it hit.
"""

from __future__ import annotations

import contextlib
import fnmatch
import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .config import load_config
from .distance import DistanceMeasure, resolve_distance_measure
from .fitting import FitSpectra

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable, Mapping, Sequence

    from numpy.typing import NDArray

__all__ = [
    "WEIGHT_MODELS",
    "ChannelSelection",
    "StagedFit",
    "WeightModel",
    "fit_event",
    "get_weight_model",
]


@runtime_checkable
class WeightModel(Protocol):
    """How much each station's stage-1 estimate counts toward the event value.

    Given the stage-1 table and the spectra it was fitted to, return one weight
    per row. Weights need not sum to anything; they are normalised on use.
    """

    name: str

    def weights(
        self, table: Any, spectra: Any, ids: Sequence[str]
    ) -> NDArray[np.float64]: ...  # pragma: no cover


@dataclass(frozen=True, slots=True)
class InverseDistance:
    """Weight by ``1 / distance``, the published choice.

    The nearer station has less path between the source and the sensor, so
    less of its high-frequency falloff can be attenuation and its corner is the
    better constrained of the two. That is the argument for it; it is a
    modelling choice rather than a derivation, which is why this is a registry
    and not a hardcoded expression.

    **Which distance is itself a choice**, and at short range not a small one:
    see :mod:`specmod.distance`. ``measure=None`` takes the project-wide
    setting, so a study that has decided on epicentral does not have to say so
    again here.
    """

    #: ``None`` means "whatever the configuration says". Distance is needed by
    #: geometric spreading as well as by weighting, so the choice belongs in
    #: one place rather than being restated per consumer.
    measure: str | DistanceMeasure | None = None
    name: str = "inverse_distance"

    def weights(
        self, table: Any, spectra: Any, ids: Sequence[str]
    ) -> NDArray[np.float64]:
        distances = resolve_distance_measure(self.measure).distances(spectra, ids)
        weights: NDArray[np.float64] = 1.0 / distances
        return weights


@dataclass(frozen=True, slots=True)
class Uniform:
    """Every contributing station counts the same.

    The honest default when the geometry is unknown, and the right one when
    the stations are at comparable distances — where inverse-distance weighting
    adds variance without adding information.
    """

    name: str = "uniform"

    def weights(
        self, table: Any, spectra: Any, ids: Sequence[str]
    ) -> NDArray[np.float64]:
        return np.ones(len(ids), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class InverseVariance:
    """Weight by ``1 / stderr**2`` on the aggregated parameter.

    Statistically the right answer when the uncertainties are real, and
    unavailable under the shipped minimiser: Powell estimates no covariance
    matrix, so lmfit reports ``stderr`` as ``None`` for every parameter. This
    raises rather than falling back, because silently becoming uniform is the
    kind of substitution that ends up in a paper.
    """

    name: str = "inverse_variance"

    def weights(
        self, table: Any, spectra: Any, ids: Sequence[str]
    ) -> NDArray[np.float64]:
        indexed = table.set_index("id")
        column = f"{_aggregated_parameter()}-stderr"
        if column not in indexed:
            raise ValueError(f"the fit table has no {column!r} column")
        errors = indexed.loc[list(ids), column].to_numpy(dtype=np.float64)
        if not np.isfinite(errors).all():
            raise ValueError(
                "inverse-variance weighting needs an uncertainty on every "
                "contributing station, and some are missing. The configured "
                "minimiser is likely 'powell', which estimates no covariance "
                "matrix — use 'leastsq', or weight another way."
            )
        weights: NDArray[np.float64] = 1.0 / errors**2
        return weights


#: Registered weightings, resolved by name from ``[fitting] event_weighting``.
WEIGHT_MODELS: dict[str, Any] = {
    # Follows the configured distance measure, so a project-wide choice is
    # honoured in one place. The shipped default.
    "inverse_distance": InverseDistance,
    # And explicit spellings, for a study that wants to say which it used
    # regardless of what the rest of the configuration says.
    "inverse_hypocentral_distance": lambda: InverseDistance(measure="rhyp"),
    "inverse_epicentral_distance": lambda: InverseDistance(measure="repi"),
    "uniform": Uniform,
    "inverse_variance": InverseVariance,
}


def get_weight_model(name: str) -> WeightModel:
    """Resolve a registered weighting by name, with its defaults."""
    try:
        factory = WEIGHT_MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown weighting {name!r}. Available: {sorted(WEIGHT_MODELS)}."
        ) from None
    model: WeightModel = factory()
    return model


def _aggregated_parameter() -> str:
    return str(load_config().config.fitting.event_parameter)


@dataclass(frozen=True, slots=True)
class ChannelSelection:
    """Which channels contribute to the event value.

    Defaults come from ``[fitting]``; pass one of these to override per call.
    The point of it being a value rather than four arguments is that a
    selection can be written down, compared and stored — a run that dropped
    three stations should be able to say so afterwards.
    """

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    #: Drop a station whose stage-1 fit ended with a parameter against a bound.
    #:
    #: **This interacts with the minimiser, and not symmetrically.**
    #: ``pass_fitting`` asks whether ``value +/- stderr`` reaches a bound.
    #: Powell — the shipped default — estimates no covariance matrix, so
    #: ``stderr`` is ``None``, the spread is zero, and the test degenerates to
    #: "is the value exactly on the bound", which essentially never fires. On
    #: these 28 windows it drops **0** stations under Powell and **6** under
    #: ``leastsq``.
    #:
    #: So changing the minimiser changes *which stations vote*, not just how
    #: each one is fitted. Comparing two minimisers under the default
    #: selection therefore compares two different ensembles: 12.081 Hz from 28
    #: channels against 5.376 Hz from 22, a 125% difference that says almost
    #: nothing about the minimisers. Holding the ensemble fixed with
    #: ``require_pass=False`` gives 12.081 against 12.029 — 0.4%, which is the
    #: honest comparison and the number quoted in the module docstring.
    #:
    #: Left ``True`` because the flag is doing the right thing when it fires:
    #: a pinned parameter reports its bound rather than a measurement, and
    #: averaging that in averages in a constant. But a study comparing
    #: minimisers, or reporting a corner frequency alongside one fitted
    #: another way, has to set this ``False`` or account for it.
    require_pass: bool = True

    @classmethod
    def from_config(cls) -> ChannelSelection:
        fitting = load_config().config.fitting
        return cls(
            include=tuple(fitting.include),
            exclude=tuple(fitting.exclude),
            require_pass=bool(fitting.require_pass),
        )

    def choose(self, fit: FitSpectra) -> tuple[list[str], dict[str, str]]:
        """Split ``fit``'s stations into contributors and reasoned exclusions."""
        contributing: list[str] = []
        excluded: dict[str, str] = {}

        table = fit.table.set_index("id") if len(fit.table) else None
        # `ids()` on a SpectrumSet, plain iteration on anything else — the same
        # duck-typed contract the fitter itself accepts.
        spectra = fit.spectra
        ids = spectra.ids() if hasattr(spectra, "ids") else list(spectra)
        for id in ids:
            if id not in fit.models:
                excluded[id] = "no fit: the station did not pass the band gate"
                continue
            if self.include and _match(id, self.include) is None:
                excluded[id] = f"not matched by include={list(self.include)}"
                continue
            hit = _match(id, self.exclude)
            if hit is not None:
                level, pattern = hit
                excluded[id] = f"matched exclude={pattern!r} at {level}"
                continue
            if table is None or id not in table.index:
                excluded[id] = "no row in the stage-1 table: the fit did not run"
                continue
            if self.require_pass and not bool(table.loc[id, "pass_fitting"]):
                excluded[id] = (
                    "stage-1 fit failed: a parameter is pinned against a bound, "
                    "so the value reported is the bound rather than a measurement"
                )
                continue
            contributing.append(id)

        return contributing, excluded


#: The parts of ``NET.STA.LOC.CHA`` a pattern may be written against, in the
#: order a match is reported. Full id first so the most specific spelling wins
#: the explanation.
def _levels(id: str) -> list[tuple[str, str]]:
    parts = id.split(".")
    if len(parts) != 4:
        # Not a SEED id — match it whole and say nothing more about it.
        return [("id", id)]
    net, sta, loc, cha = parts
    return [
        ("id", id),
        ("station", f"{net}.{sta}"),
        ("network", net),
        ("station", sta),
        ("location", loc),
        ("channel", cha),
    ]


def _match(id: str, patterns: Iterable[str]) -> tuple[str, str] | None:
    """The (level, pattern) a channel matched at, or ``None``.

    Returned rather than a bool so an exclusion can say *why*: "matched
    exclude='AQ07' at station" is actionable where "excluded" is not.
    """
    for pattern in patterns:
        for level, value in _levels(id):
            if fnmatch.fnmatch(value, pattern):
                return level, pattern
    return None


@dataclass(frozen=True, slots=True)
class StagedFit:
    """The result of :func:`fit_event`: both stages and how they were joined."""

    #: Every station fitted independently. Kept because the spread across it is
    #: the evidence for how well constrained the event value is, and discarding
    #: it would leave only a number with no error on it.
    stage1: FitSpectra
    #: Every station refitted with the event parameter held fixed. ``None`` when
    #: no station survived selection, in which case there is nothing to fix it
    #: to and the honest result is stage one alone.
    stage2: FitSpectra | None
    #: The aggregated value and what produced it.
    parameter: str
    value: float
    weighting: str
    contributing: tuple[str, ...]
    excluded: Mapping[str, str] = field(default_factory=dict)

    @property
    def table(self) -> Any:
        """The stage-2 table where there is one, else stage one's."""
        return self.stage1.table if self.stage2 is None else self.stage2.table

    def spread(self) -> dict[str, float]:
        """How much the contributing stations disagreed, before aggregation.

        The number worth reporting beside the event value. A 2% spread and a
        60% spread give the same weighted mean and mean very different things.
        """
        if not self.contributing:
            return {}
        values = (
            self.stage1.table.set_index("id")
            .loc[list(self.contributing), self.parameter]
            .to_numpy(dtype=np.float64)
        )
        return {
            "n": float(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(np.median(values)),
            "weighted_mean": self.value,
            "relative_spread": float((values.max() - values.min()) / self.value)
            if self.value
            else float("nan"),
        }

    def describe(self) -> str:
        """One paragraph a caller can print or paste into a log."""
        lines = [
            f"{self.parameter} = {self.value:.4g} "
            f"from {len(self.contributing)} channels, weighted by "
            f"{self.weighting}",
        ]
        spread = self.spread()
        if spread:
            lines.append(
                f"  stage-1 range {spread['min']:.4g} to {spread['max']:.4g} "
                f"({100 * spread['relative_spread']:.0f}% of the event value)"
            )
        if self.excluded:
            lines.append(f"  {len(self.excluded)} excluded:")
            lines += [f"    {id}: {why}" for id, why in sorted(self.excluded.items())]
        return "\n".join(lines)


def fit_event(
    spectra: Any,
    *,
    model: Any = None,
    guess: Mapping[str, Mapping[str, float]] | None = None,
    fit_bins: bool | None = None,
    parameter: str | None = None,
    weighting: str | WeightModel | None = None,
    selection: ChannelSelection | None = None,
    quiet: bool = True,
    **fit_kwargs: Any,
) -> StagedFit:
    """Fit an event in two stages, with the ensemble deciding the source term.

    Every argument has a configured default, so ``fit_event(spectra)`` is the
    published workflow and the rest is there for when the default is wrong.

    Parameters
    ----------
    spectra
        A :class:`~specmod.core.SpectrumSet`, or anything mapping trace ids to
        paired spectra.
    model, guess, fit_bins
        Passed to :class:`~specmod.fitting.FitSpectra` for both stages, so the
        two are fitting the same thing.
    parameter
        Which parameter the ensemble determines and stage two holds fixed.
        ``[fitting] event_parameter``, default ``"fc"`` — the corner frequency
        is the term that belongs to the source. ``"ts"`` is the meaningful
        alternative, for a study with an independent handle on ``Q``.
    weighting
        A registered name or a :class:`WeightModel`. ``[fitting]
        event_weighting``, default inverse hypocentral distance.
    selection
        Which channels contribute. Defaults to :meth:`ChannelSelection.from_config`.
    quiet
        Suppress the fitter's per-station chatter, which is two full passes of
        it here. The failures are on :attr:`StagedFit.excluded` either way.
    **fit_kwargs
        Passed to :meth:`~specmod.fitting.FitSpectra.fit_spectra` for both
        stages — ``method=`` most usefully.

    Notes
    -----
    Stage two is skipped, with ``stage2=None``, when selection leaves nothing.
    Fixing the parameter to a mean of no stations would be inventing the very
    number the second stage exists to constrain, and a caller who asked for two
    stages and got one should be able to see that rather than read a value.
    """
    fitting = load_config().config.fitting
    parameter = parameter or str(fitting.event_parameter)
    if selection is None:
        selection = ChannelSelection.from_config()
    if weighting is None:
        weighting = str(fitting.event_weighting)
    model_ = get_weight_model(weighting) if isinstance(weighting, str) else weighting

    def run(constant: tuple[str, float] | None = None) -> FitSpectra:
        with contextlib.redirect_stdout(io.StringIO() if quiet else None):
            fit = FitSpectra(spectra, model=model, guess=guess, fit_bins=fit_bins)
            if constant is not None:
                fit.set_const(*constant)
            fit.fit_spectra(**fit_kwargs)
        return fit

    stage1 = run()
    contributing, excluded = selection.choose(stage1)

    if not contributing:
        return StagedFit(
            stage1=stage1,
            stage2=None,
            parameter=parameter,
            value=float("nan"),
            weighting=getattr(model_, "name", str(weighting)),
            contributing=(),
            excluded=excluded,
        )

    table = stage1.table.set_index("id")
    values = table.loc[contributing, parameter].to_numpy(dtype=np.float64)
    weights = np.asarray(
        model_.weights(stage1.table, spectra, contributing), dtype=np.float64
    )
    if weights.shape != values.shape:
        raise ValueError(
            f"{getattr(model_, 'name', weighting)} returned {weights.size} "
            f"weights for {values.size} channels"
        )
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(
            f"{getattr(model_, 'name', weighting)} gave every contributing "
            "channel zero weight, so there is no event value to compute"
        )
    value = float((values * weights).sum() / total)

    return StagedFit(
        stage1=stage1,
        stage2=run(constant=(parameter, value)),
        parameter=parameter,
        value=value,
        weighting=getattr(model_, "name", str(weighting)),
        contributing=tuple(contributing),
        excluded=excluded,
    )
