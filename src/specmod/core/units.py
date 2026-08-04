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
The canonical amplitude kind is the **folded** one-sided Fourier amplitude
spectrum (:attr:`AmplitudeKind.FAS`), in units of ``[signal] * s``. "Folded"
means the negative-frequency half has been added in, so ``FAS = 2|X|`` where
``X`` is the Fourier transform. That is what makes energy recoverable by
integrating over non-negative frequencies alone, and it is why every estimator
here can be held to one Parseval check.

**It is not the quantity the source model is written in.** Omega, the
long-period spectral level, is the plateau of ``|X|`` — at zero frequency
``|X(0)| = |integral u dt|``, which is what ``M0`` is proportional to. Reading
``FAS`` as Omega puts ``M0`` out by two, which is 0.2 magnitude units. Use
:attr:`AmplitudeKind.MAGNITUDE` for that, and let :meth:`Spectrum.to_kind`
apply the factor rather than doing it by hand.

Relationships between the kinds, for a record of duration ``T``:

=============  ========================  ==================================
Kind           Units                     From FAS ``A``
=============  ========================  ==================================
``FAS``        ``[x] * s``               --
``MAGNITUDE``  ``[x] * s``               ``|X| = A / 2``
``PSD``        ``[x]^2 / Hz``            ``P = A^2 / (2 T)``
``ASD``        ``[x] / sqrt(Hz)``        ``D = A / sqrt(2 T)``
=============  ========================  ==================================

Parseval takes a different form in each amplitude convention, which is the
whole reason both are named here rather than left to the caller::

    E = integral A**2 / 2 df        (FAS, folded)
    E = 2 * integral |X|**2 df      (MAGNITUDE, unfolded)

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

    #: Folded one-sided Fourier amplitude spectrum, ``2|X|``, in ``[x] * s``.
    #: Energy is ``integral(FAS**2 / 2) df``. The default, because it is the
    #: convention in which one Parseval check covers every estimator.
    FAS = "fas"
    #: Unfolded Fourier transform magnitude, ``|X| = |rfft(x)| * dt``, in
    #: ``[x] * s``. **This is the one Omega is defined in**, and the one to
    #: read a long-period spectral level off. Energy is
    #: ``2 * integral(|X|**2) df``.
    MAGNITUDE = "magnitude"
    #: One-sided power spectral density, ``[x]^2 / Hz``.
    PSD = "psd"
    #: One-sided amplitude spectral density, ``[x] / sqrt(Hz)``.
    ASD = "asd"

    @property
    def is_amplitude(self) -> bool:
        """Whether this kind scales linearly with the record.

        The distinction that matters for :meth:`Spectrum.to_motion`: applying
        a ``2*pi*f`` factor to a squared quantity is wrong by ``2*pi*f`` again.
        """
        return self in (AmplitudeKind.FAS, AmplitudeKind.MAGNITUDE, AmplitudeKind.ASD)

    def unit(self, motion: Motion) -> str:
        """Full unit string for this kind in a given motion domain."""
        base = motion.unit
        return {
            "fas": f"{base}*s",
            "magnitude": f"{base}*s",
            "psd": f"({base})^2/Hz",
            "asd": f"{base}/sqrt(Hz)",
        }[self.value]
