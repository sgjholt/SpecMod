"""Konno & Ohmachi (1998) smoothing.

A constant-width window in log-frequency: narrow at low frequency, wide at
high. That is what makes it the convention in engineering seismology — it
avoids over-smoothing the low-frequency band that constrains the long-period
level, while still suppressing scatter in the high-frequency tail.

The window is ``[sin(b log10(f/fc)) / (b log10(f/fc))]^4``, with ``b`` the
bandwidth. Smaller ``b`` smooths harder; 40 is the conventional value.

ObsPy already implements this and is already a hard dependency, so this is a
thin adapter rather than a reimplementation. It keeps the frequency axis
unchanged, unlike :class:`~specmod.smoothing.log_bins.LogBinner`, which is
often what you want before fitting on the original grid.

References
----------
Konno, K. & Ohmachi, T. (1998). Ground-motion characteristics estimated from
spectral ratio between horizontal and vertical components of microtremor.
*BSSA* 88(1), 228-241.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing

from ..core.spectrum import Spectrum
from .base import record_smoothing

__all__ = ["KonnoOhmachi"]


@dataclass(frozen=True)
class KonnoOhmachi:
    """Konno-Ohmachi smoothing at fixed log-frequency bandwidth.

    Parameters
    ----------
    bandwidth
        The ``b`` coefficient. Smaller values smooth more aggressively.
    count
        Number of times to apply the window. More than one is occasionally
        used for very noisy spectra.
    normalize
        Normalise each window to unit area. Without this the smoothed
        amplitudes are biased low where the window is truncated at the edges of
        the frequency axis.
    """

    bandwidth: float = 40.0
    count: int = 1
    normalize: bool = True
    name: str = "konno_ohmachi"

    def __post_init__(self) -> None:
        if self.bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {self.bandwidth}")
        if self.count < 1:
            raise ValueError(f"count must be at least 1, got {self.count}")

    def smooth(self, spectrum: Spectrum) -> Spectrum:
        # ObsPy needs float32 or float64 and rejects anything else outright.
        smoothed = konno_ohmachi_smoothing(
            np.asarray(spectrum.amp, dtype=np.float64),
            np.asarray(spectrum.freq, dtype=np.float64),
            bandwidth=self.bandwidth,
            count=self.count,
            normalize=self.normalize,
        )
        return replace(
            spectrum,
            amp=np.asarray(smoothed, dtype=np.float64),
            meta=record_smoothing(
                spectrum.meta,
                self.name,
                bandwidth=self.bandwidth,
                count=self.count,
                normalize=self.normalize,
            ),
        )
