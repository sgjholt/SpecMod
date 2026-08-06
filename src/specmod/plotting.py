"""Looking at a spectrum, a pair, or a whole event.

Replaces ``spectral.SNP.quick_vis`` and ``spectral.Spectra.quick_vis``, which
were methods on the mutable containers and went with them. A spectral package
where you cannot look at a spectrum is not usable, so this is not optional
furniture.

**Functions over methods.** The legacy version was a method, so plotting a
spectrum required owning the container it lived in — which is why the fitter
grew its own near-duplicate rather than reusing it. These take a
:class:`~specmod.core.SpectrumPair` and draw it; anything that has one can
call them, and a caller supplying their own ``Axes`` gets full control of the
figure.

Nothing here mutates its argument, and nothing calls ``plt.show()``. Returning
the axes is what lets a caller compose these into a larger figure, annotate
them, or save without a window ever opening — which is also what makes them
usable from a script and from a notebook without behaving differently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, StrMethodFormatter

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .core import SpectrumPair, SpectrumSet

__all__ = ["plot_pair", "plot_set"]


def _tidy_frequency_axis(ax: Axes) -> None:
    """Plain decimal tick labels on a log axis.

    Matplotlib's default on a log scale is ``10^0``-style exponents, which for
    a band running 1 to 40 Hz is less readable than the numbers themselves.
    """
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_pair(
    pair: SpectrumPair,
    ax: Axes | None = None,
    *,
    id: str = "",
    fit: Any = None,
    show_binned: bool = False,
) -> Axes:
    """Draw one signal-and-noise pair, with the band it selected.

    Parameters
    ----------
    pair
        What to draw.
    ax
        Draw here; a new figure is made if omitted.
    id
        Station label. A frozen pair does not carry one, so the caller that
        knows the key supplies it — ``pair.signal.meta["id"]`` is used when it
        is there and this is not.
    fit
        A :class:`~specmod.fitting.FitSpectrum` whose model to overlay, if it
        has been fitted. Passed in rather than read off the spectrum: the
        containers are frozen precisely so a result cannot write itself back
        into its own input.
    show_binned
        Also draw the log-binned spectra the signal-to-noise ratio is actually
        computed on. Off by default because it doubles the lines, on when the
        question is why a band came out where it did.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)

    label = id or pair.signal.meta.get("id", "signal")

    ax.loglog(pair.noise.freq, pair.noise.amp, "b--", lw=1, label="noise")
    ax.loglog(pair.signal.freq, pair.signal.amp, "k", lw=1, label=label)

    if show_binned:
        ax.loglog(
            pair.binned_signal.freq,
            pair.binned_signal.amp,
            "o-",
            color="0.4",
            ms=3,
            lw=1,
            label="binned signal",
        )
        ax.loglog(
            pair.binned_noise.freq,
            pair.binned_noise.amp,
            "s--",
            color="steelblue",
            ms=3,
            lw=1,
            label="binned noise",
        )

    if fit is not None and getattr(fit, "result", None) is not None:
        ax.loglog(
            fit.mod_freq,
            10**fit.result.best_fit,
            color="green",
            lw=2,
            label="best fit",
        )

    if pair.band is None:
        ax.set_title(f"{label} — no usable band")
    else:
        low = min(float(pair.noise.amp.min()), float(pair.signal.amp.min()))
        high = max(float(pair.noise.amp.max()), float(pair.signal.amp.max()))
        for edge in pair.band:
            ax.vlines(edge, low, high, color="r", linestyles="dashed")
        ax.set_title(f"{label} — {pair.band[0]:.2f} to {pair.band[1]:.2f} Hz")

    # The floor is drawn because a band clamped to it looks arbitrary
    # otherwise: it is the lowest frequency the *shorter* of the two windows
    # can resolve, and it is why the band often does not open where the signal
    # first rises above the noise.
    if pair.resolution_floor > 0:
        ax.axvline(
            pair.resolution_floor,
            color="0.7",
            lw=1,
            zorder=0,
            label="resolution floor",
        )

    _tidy_frequency_axis(ax)
    ax.legend(fontsize="small")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(f"amplitude [{pair.signal.unit}]")
    return ax


def plot_set(
    spectra: SpectrumSet,
    *,
    fits: Any = None,
    columns: int | None = None,
    passing_only: bool = False,
    **kwargs: Any,
) -> Figure:
    """Draw every pair in an event on one grid.

    ``fits`` may be a :class:`~specmod.fitting.FitSpectra`, or any mapping from
    trace id to a fit; each pair gets its own model overlaid where one exists.

    ``columns`` defaults to ``viz.plot_columns`` in the configuration, which is
    the one place that number lives now — it used to be defined in both the
    ``SPECTRAL`` and ``FITTING`` dicts, where the two copies could disagree.
    """
    from .config import load_config  # noqa: PLC0415  (circular at module level)

    if columns is None:
        columns = load_config().config.viz.plot_columns

    ids = [k for k in spectra.ids() if not passing_only or spectra[k].passes]
    if not ids:
        raise ValueError("nothing to plot: no pair in this set has a usable band")

    models = getattr(fits, "models", fits) or {}
    rows = int(np.ceil(len(ids) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 4 * rows), squeeze=False
    )
    flat = axes.ravel()

    for ax, id in zip(flat, ids, strict=False):
        plot_pair(spectra[id], ax, id=id, fit=models.get(id), **kwargs)
    for ax in flat[len(ids) :]:
        ax.set_visible(False)

    figure.suptitle(spectra.event or "spectra")
    figure.tight_layout()
    return figure
