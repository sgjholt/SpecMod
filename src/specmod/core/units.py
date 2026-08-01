"""Physical units and domains, as types rather than conventions.

The pre-refactor code tracked none of this. A ``Spectrum`` did not know whether
it held power or amplitude, nor whether it was in displacement, velocity or
acceleration; that lived in ``Models.MOTION``, a module global read at import
time which the user had to keep in sync by hand with however many times they
had called ``.inte()`` or ``.diff()``. Getting it wrong returned a wrong seismic
moment with no error anywhere.

Making both a typed attribute turns those silent factor errors into exceptions.

Conventions
-----------
The canonical amplitude kind is the **one-sided Fourier amplitude spectrum**
(:attr:`AmplitudeKind.FAS`), in units of ``[signal] * s`` — metres for a
velocity record, since m/s times s is m. This is the quantity the source model
is written in: the long-period spectral level Omega is quoted in "displacement
units of m*s".

Relationships between the kinds, for a record of duration ``T``:

===========  ========================  ==================================
Kind         Units                     From FAS ``A``
===========  ========================  ==================================
``FAS``      ``[x] * s``               --
``PSD``      ``[x]^2 / Hz``            ``P = A^2 / (2 T)``
``ASD``      ``[x] / sqrt(Hz)``        ``D = A / sqrt(2 T)``
===========  ========================  ==================================

``T`` is the **physical record duration**, ``n_samples * dt``. It is never
inferred from the length of the frequency axis: zero-padding changes that length
while leaving the duration alone, which is precisely how the old
``psd_to_amp`` acquired a padding-dependent error.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["AmplitudeKind", "Motion"]


class Motion(StrEnum):
    """Ground-motion domain of a time series or spectrum."""

    DISPLACEMENT = "displacement"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"

    @property
    def derivative_order(self) -> int:
        """Order of time differentiation relative to displacement.

        Converting between domains multiplies the spectrum by ``(2*pi*f)`` per
        order, so the difference of two orders gives the exponent directly.
        """
        return {"displacement": 0, "velocity": 1, "acceleration": 2}[self.value]

    @property
    def unit(self) -> str:
        """SI unit of the time-domain signal."""
        return {"displacement": "m", "velocity": "m/s", "acceleration": "m/s^2"}[
            self.value
        ]


class AmplitudeKind(StrEnum):
    """What the amplitude axis of a spectrum represents."""

    #: One-sided Fourier amplitude spectrum, ``[x] * s``.
    FAS = "fas"
    #: One-sided power spectral density, ``[x]^2 / Hz``.
    PSD = "psd"
    #: One-sided amplitude spectral density, ``[x] / sqrt(Hz)``.
    ASD = "asd"

    def unit(self, motion: Motion) -> str:
        """Full unit string for this kind in a given motion domain."""
        base = motion.unit
        return {
            "fas": f"{base}*s",
            "psd": f"({base})^2/Hz",
            "asd": f"{base}/sqrt(Hz)",
        }[self.value]
