"""One signal-and-noise pair, and the band it selected."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from .style import tidy_frequency_axis

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

    from ..core import SpectrumPair

__all__ = ["plot_pair"]


def _fits(fit: Any) -> list[tuple[str, Any]]:
    """Normalise ``fit`` to ``[(label, fit), ...]``, dropping anything unfitted.

    Accepts one fit or a mapping of label to fit, so a caller comparing two
    minimisers does not have to reach into the axes afterwards.
    """
    if fit is None:
        return []
    if hasattr(fit, "items"):
        return [
            (str(k), v)
            for k, v in fit.items()
            if getattr(v, "result", None) is not None
        ]
    return [("best fit", fit)] if getattr(fit, "result", None) is not None else []


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
        has been fitted — or a **mapping of label to fit**, to draw several at
        once. Passed in rather than read off the spectrum: the containers are
        frozen precisely so a result cannot write itself back into its own
        input.

        Several matters more than it sounds. Fitting a source model is not a
        unique inversion, and two minimisers can reach the same goodness of fit
        at corner frequencies differing by tens of percent — which is a factor
        of several in stress drop, since it scales as ``fc**3``. Drawing them
        together is how that stops being invisible.
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

    # Not `label` — that name holds the station id the title is built from.
    for name, one in _fits(fit):
        ax.loglog(one.mod_freq, 10**one.result.best_fit, lw=2, label=name)

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

    tidy_frequency_axis(ax)
    ax.legend(fontsize="small")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(f"amplitude [{pair.signal.unit}]")
    return ax
