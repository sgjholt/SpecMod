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

The package splits by what is being drawn: :mod:`~specmod.plotting.pair` draws
one pair, :mod:`~specmod.plotting.grid` arranges many, and
:mod:`~specmod.plotting.style` holds the presentation both share, so an axis
decision is made once rather than twice.
"""

from __future__ import annotations

from .grid import plot_set
from .pair import plot_pair

__all__ = ["plot_pair", "plot_set"]
