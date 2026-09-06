"""Every pair in an event, on one figure."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from .pair import plot_pair

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

    from ..core import SpectrumSet

__all__ = ["plot_set"]


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
    from ..config import load_config  # noqa: PLC0415  (circular at module level)

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
