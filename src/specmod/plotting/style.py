"""Presentation shared by every plot here.

Small on purpose. It exists so that a decision about how axes look is made
once, rather than drifting between the single-pair plot and the grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.ticker import NullFormatter, StrMethodFormatter

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes


def tidy_frequency_axis(ax: Axes) -> None:
    """Plain decimal tick labels on a log axis.

    Matplotlib's default on a log scale is ``10^0``-style exponents, which for
    a band running 1 to 40 Hz is less readable than the numbers themselves.
    """
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
