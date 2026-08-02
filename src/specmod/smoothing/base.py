"""The smoother interface.

Smoothing is deliberately separate from estimation. The pre-refactor code baked
log-binning into ``Spectrum.__init__``, so every spectrum was binned on
construction with parameters no caller could reach, and the unbinned and binned
arrays travelled together as ``amp``/``bamp`` with nothing recording which was
which.

A smoother maps a :class:`~specmod.core.spectrum.Spectrum` to another one. The
kind, motion, duration and sampling rate are carried through unchanged; only the
frequency axis and amplitudes move.

Smoothing is lossy
------------------
None of these preserve energy — that is the point of them. A smoothed spectrum
therefore fails the Parseval contract that :mod:`specmod.transforms` guarantees,
and :meth:`Spectrum.energy` on one is not meaningful. Every smoother records
what it did under ``meta["smoothing"]`` so this is visible downstream rather
than inferred.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..core.spectrum import Spectrum

__all__ = ["Smoother", "record_smoothing"]


@runtime_checkable
class Smoother(Protocol):
    """Anything that maps a spectrum to a smoothed spectrum."""

    @property
    def name(self) -> str:
        """Short identifier, recorded in the spectrum's metadata."""
        ...

    def smooth(self, spectrum: Spectrum) -> Spectrum:
        """Return a smoothed copy of ``spectrum``."""
        ...


def record_smoothing(meta: Any, name: str, **params: Any) -> dict[str, Any]:
    """Merge a smoothing record into metadata.

    Appends rather than overwrites, so chaining smoothers leaves a trail. An
    energy check downstream can look for this key and refuse rather than
    silently comparing a smoothed spectrum against a Parseval expectation.
    """
    info = dict(meta)
    applied = list(info.get("smoothing", ()))
    applied.append({"method": name, **params})
    info["smoothing"] = applied
    return info
