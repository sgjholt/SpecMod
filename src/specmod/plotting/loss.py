"""Where a fit sits in its misfit, and how well the misfit holds it there.

A fitted value and a goodness of fit say nothing about whether the value was
*constrained*. A corner frequency at the bottom of a tight bowl and one at an
arbitrary point along a flat valley report the same misfit, and only one of
them is a measurement. These draw the misfit around the best fit so the
difference can be seen.

What is drawn is the misfit as a multiple of the best fit's: 1 at the fitted
values, 2 where the sum of squared residuals has doubled. The contours are
those multiples. **They are not confidence regions** — that needs a noise
model this package does not assume — but they are comparable between stations
and between parameters, which is what telling a constrained fit from an
unconstrained one needs.

Three things are marked on every panel: the fitted value, the value the fit
started from, and any bound inside the range, with the region beyond it
shaded. A fitted value sitting on the shading, with the misfit still falling
into it, is a search space that needs opening.

**Slice or profile.** By default the parameters not on a panel's axes stay at
their fitted values, which is a slice through the best fit. It is exact and
fast, and it makes every valley look narrower than it is: on one PNR station
the region within 1.1 times the best misfit covers 0.9% of a corner-frequency
by ``t*`` slice and 3.2% of the profile. ``profile=True`` refits the others at
every point, which is the honest picture of a trade-off and costs a fit per
grid point. See :meth:`specmod.fitting.FitSpectrum.loss_scan`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from ..fitting import FitSpectrum

__all__ = ["plot_loss", "plot_loss_set"]

#: Multiples of the best misfit the contours are drawn at.
LEVELS = (1.1, 1.5, 3.0)

#: The colour scale every panel shares, so two stations can be compared: the
#: best fit to a hundred times it. Anything worse is the same dark colour.
_NORM_LIMITS = (1.0, 100.0)

#: How far either side of the fitted value a panel reaches when no range is
#: given: this factor for a positive parameter, drawn on a log axis, and this
#: many units for one that can be negative.
_SPAN_FACTOR = 10.0
_SPAN_UNITS = 1.0


def _is_positive(par: Any) -> bool:
    """Whether a parameter belongs on a log axis: positive, and bounded so."""
    return bool(par.value > 0 and par.min >= 0)


def _axis(
    par: Any, limits: tuple[float, float] | None, points: int
) -> tuple[NDArray[np.float64], str]:
    """The values to scan for one parameter, and the scale to draw them on."""
    if limits is not None:
        low, high = float(limits[0]), float(limits[1])
        if not low < high:
            raise ValueError(
                f"the range for {par.name} is ({low}, {high}); the first "
                f"value must be below the second"
            )
        if low > 0 and _is_positive(par):
            return np.geomspace(low, high, points), "log"
        return np.linspace(low, high, points), "linear"
    if _is_positive(par):
        return (
            np.geomspace(par.value / _SPAN_FACTOR, par.value * _SPAN_FACTOR, points),
            "log",
        )
    return (
        np.linspace(par.value - _SPAN_UNITS, par.value + _SPAN_UNITS, points),
        "linear",
    )


def _relative(fit: FitSpectrum, loss: NDArray[np.float64]) -> NDArray[np.float64]:
    """Misfit as a multiple of the best fit's, or unchanged if that is zero.

    A fit to noiseless synthetic data has a best misfit of nothing, and there
    is no multiple of nothing to draw.
    """
    best = float(fit.fitted.chisqr)
    return loss / best if best > 0 else loss


def _mark_bounds(ax: Axes, par: Any, *, along: str) -> None:
    """Shade what lies beyond a bound, where the bound is inside the panel."""
    low, high = ax.get_xlim() if along == "x" else ax.get_ylim()
    shade = ax.axvspan if along == "x" else ax.axhspan
    line = ax.axvline if along == "x" else ax.axhline
    if np.isfinite(par.min) and low < par.min < high:
        shade(low, par.min, color="tab:red", alpha=0.15, lw=0)
        line(par.min, color="tab:red", ls="--", lw=1)
    if np.isfinite(par.max) and low < par.max < high:
        shade(par.max, high, color="tab:red", alpha=0.15, lw=0)
        line(par.max, color="tab:red", ls="--", lw=1)
    # Shading must not move the limits it was drawn to.
    (ax.set_xlim if along == "x" else ax.set_ylim)(low, high)


def _draw_surface(
    ax: Axes,
    fit: FitSpectrum,
    x: str,
    y: str,
    *,
    ranges: Mapping[str, tuple[float, float]],
    points: int,
    profile: bool,
) -> Any:
    """One two-parameter panel. Returns the mesh, for a colour bar."""
    params = fit.fitted.params
    xs, xscale = _axis(params[x], ranges.get(x), points)
    ys, yscale = _axis(params[y], ranges.get(y), points)
    surface = _relative(fit, fit.loss_scan({x: xs, y: ys}, profile=profile)).T

    low, high = _NORM_LIMITS
    best = float(fit.fitted.chisqr)
    norm = LogNorm(low, high) if best > 0 else None
    shown = np.clip(surface, low, high) if best > 0 else surface
    mesh = ax.pcolormesh(xs, ys, shown, norm=norm, cmap="viridis_r", shading="auto")
    if best > 0:
        ax.contour(xs, ys, surface, levels=LEVELS, colors="k", linewidths=0.8)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    _mark_bounds(ax, params[x], along="x")
    _mark_bounds(ax, params[y], along="y")

    x0, y0 = params[x].init_value, params[y].init_value
    if x0 is not None and y0 is not None:
        ax.plot(x0, y0, "o", mfc="none", mec="w", ms=7, label="start")
    ax.plot(
        params[x].value, params[y].value, "+", color="w", ms=12, mew=2, label="fitted"
    )
    return mesh


def _draw_curve(
    ax: Axes,
    fit: FitSpectrum,
    name: str,
    *,
    ranges: Mapping[str, tuple[float, float]],
    points: int,
    profile: bool,
) -> None:
    """One single-parameter panel."""
    par = fit.fitted.params[name]
    xs, xscale = _axis(par, ranges.get(name), points)
    curve = _relative(fit, fit.loss_scan({name: xs}, profile=profile))

    ax.plot(xs, curve, "k", lw=1.5)
    ax.set_xscale(xscale)
    ax.set_yscale("log")
    ax.set_xlim(xs[0], xs[-1])
    if float(fit.fitted.chisqr) > 0:
        for level in LEVELS:
            ax.axhline(level, color="0.7", lw=0.8, zorder=0)
    if par.init_value is not None:
        ax.axvline(par.init_value, color="0.5", ls=":", lw=1, label="start")
    ax.axvline(par.value, color="tab:blue", lw=1.5, label="fitted")
    _mark_bounds(ax, par, along="x")


def _varying(fit: FitSpectrum) -> list[str]:
    return [name for name, par in fit.fitted.params.items() if par.vary]


def plot_loss(
    fit: FitSpectrum,
    *,
    parameters: Sequence[str] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
    points: int = 41,
    profile: bool = False,
) -> Figure:
    """Draw the misfit around one station's fit, parameter by parameter.

    A triangle of panels. Down the diagonal, the misfit along each parameter
    alone; below it, the misfit over each pair. A well-constrained parameter
    has a narrow dip on the diagonal and closed contours around the fitted
    value; a trade-off is a long diagonal valley; an unconstrained one is a
    curve that flattens out, or contours that run off the panel.

    Parameters
    ----------
    fit
        A fitted :class:`~specmod.fitting.FitSpectrum`, such as
        ``fits.models[id]``.
    parameters
        Which to draw, in order. Defaults to every parameter the fit varied,
        so a stage-two fit with the corner frequency held fixed draws the two
        that were left.
    ranges
        ``{name: (low, high)}`` for any parameter whose default range is not
        the one wanted. The default is a decade either side of the fitted
        value for a positive parameter and one unit either side otherwise,
        and is not clipped to the bounds: what lies beyond one is shaded
        rather than hidden.
    points
        Grid points per axis.
    profile
        Refit the parameters not on a panel's axes at every point, rather
        than holding them at their fitted values. See the module docstring;
        it is the honest picture of a trade-off and costs a fit per point.
    """
    names = list(parameters) if parameters is not None else _varying(fit)
    known = fit.fitted.params.valuesdict()
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"{', '.join(unknown)} is not a parameter of this model, which "
            f"takes {', '.join(known)}"
        )
    if not names:
        raise ValueError(
            "nothing to plot: this fit varied no parameter. Pass `parameters` "
            "to draw the misfit along ones that were held fixed."
        )
    ranges = dict(ranges or {})

    n = len(names)
    # Constrained layout, which makes room for the title and the colour bar;
    # `tight_layout` knows about neither and draws the title over the panels.
    figure, axes = plt.subplots(
        n, n, figsize=(3.6 * n + 1.0, 3.3 * n), squeeze=False, layout="constrained"
    )
    mesh = None
    for row, y in enumerate(names):
        for column, x in enumerate(names):
            ax = axes[row, column]
            if column > row:
                ax.set_visible(False)
                continue
            if column == row:
                _draw_curve(ax, fit, x, ranges=ranges, points=points, profile=profile)
                if row == 0:
                    ax.legend(fontsize="small")
                    ax.set_ylabel("misfit / best")
            else:
                mesh = _draw_surface(
                    ax, fit, x, y, ranges=ranges, points=points, profile=profile
                )
                if column == 0:
                    ax.set_ylabel(y)
            if row == n - 1:
                ax.set_xlabel(x)

    label = getattr(fit.sig, "id", "") or "fit"
    how = "profile" if profile else "slice through the best fit"
    title = f"{label} — misfit, {how}"
    near = fit.near_bounds()
    if near:
        title += "\n" + "; ".join(
            f"{item.parameter} against its {item.side} bound" for item in near
        )
    figure.suptitle(title)
    if mesh is not None:
        figure.colorbar(mesh, ax=axes, shrink=0.6, label="misfit / best")
    return figure


def _default_pair(fits: Mapping[str, FitSpectrum]) -> tuple[str, str]:
    """Corner frequency against ``t*`` where both varied, else the first two."""
    varying = _varying(next(iter(fits.values())))
    if "fc" in varying and "ts" in varying:
        return "fc", "ts"
    if len(varying) < 2:
        raise ValueError(
            f"these fits varied {varying or 'nothing'}, and a surface needs "
            f"two parameters. Name them with `x` and `y`, or use plot_loss "
            f"for the single-parameter curve."
        )
    return varying[0], varying[1]


def plot_loss_set(
    fits: Any,
    x: str | None = None,
    y: str | None = None,
    *,
    ids: Sequence[str] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
    points: int = 31,
    profile: bool = False,
    columns: int | None = None,
) -> Figure:
    """One pair of parameters, for every fitted station, on one figure.

    The quick look across an event: which stations hold the pair in a closed
    contour and which let it run. Every panel shares one colour scale and one
    set of contour levels, so they can be compared, and a station whose fit
    finished against a bound says so in its title.

    Parameters
    ----------
    fits
        A :class:`~specmod.fitting.FitSpectra`, or a mapping of id to
        :class:`~specmod.fitting.FitSpectrum`. Unfitted stations are skipped.
    x, y
        The two parameters. Defaults to corner frequency against ``t*``, the
        trade-off that decides most of what a spectral fit can and cannot
        say; otherwise the first two that varied.
    ids
        Which stations, in order. Defaults to all that were fitted.
    ranges, points, profile
        As in :func:`plot_loss`. A range applies to every panel; without one
        each panel is centred on its own fitted value.
    columns
        Defaults to ``[viz] plot_columns``.
    """
    from ..config import load_config  # noqa: PLC0415  (circular at module level)

    models = getattr(fits, "models", fits)
    fitted = {
        id: model
        for id, model in models.items()
        if getattr(model, "result", None) is not None
    }
    if ids is not None:
        missing = [id for id in ids if id not in fitted]
        if missing:
            raise ValueError(f"no fit for {', '.join(missing)}")
        fitted = {id: fitted[id] for id in ids}
    if not fitted:
        raise ValueError("nothing to plot: no station here has been fitted")

    if x is None or y is None:
        x, y = _default_pair(fitted)
    if columns is None:
        columns = load_config().config.viz.plot_columns
    ranges = dict(ranges or {})

    rows = int(np.ceil(len(fitted) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns + 1.0, 3.6 * rows),
        squeeze=False,
        layout="constrained",
    )
    flat = axes.ravel()
    mesh = None
    for ax, (id, model) in zip(flat, fitted.items(), strict=False):
        mesh = _draw_surface(
            ax, model, x, y, ranges=ranges, points=points, profile=profile
        )
        near = model.near_bounds()
        ax.set_title(
            f"{id} — {', '.join(item.parameter for item in near)} on a bound"
            if near
            else id,
            fontsize="small",
            color="tab:red" if near else "k",
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    for ax in flat[len(fitted) :]:
        ax.set_visible(False)

    how = "profile" if profile else "slice through each best fit"
    figure.suptitle(f"misfit over {x} and {y}, {how}")
    if mesh is not None:
        # Beside the first row only. Every panel shares the scale, and a bar
        # the height of a ten-row figure is mostly bar.
        figure.colorbar(mesh, ax=axes[0, :], label="misfit / best")
    return figure
