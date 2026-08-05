"""Models for the noise level a signal is judged against.

A recorded noise window understates the noise underneath a strong signal: it
is a sample of the same process, but taken where the signal is not. Comparing
a signal against it unmodified selects a band wider than the data supports,
particularly at the low end where ``Omega`` is read.

How to correct for that is a modelling choice, not a fact, so this module
holds a **set** of methods rather than one. They share a signature — given the
frequencies, the noise and the signal, return a multiplicative factor to apply
to the noise — which is what lets the band search stay indifferent to which
was used. :data:`NOISE_MODELS` maps names to implementations and
:func:`get_noise_model` resolves one, mirroring how
:mod:`specmod.transforms` handles estimators.

The factor is returned rather than the corrected array because it has to be
applied to two things: the binned noise and the unbinned noise, the latter by
interpolation onto the finer axis. Returning the factor keeps those two from
drifting apart.

Currently implemented
---------------------
``boost``
    The default, described below. Lifts the low and high tails independently
    until each touches the signal.

Not yet ported
--------------
``rotate``
    ``utils.rotate_noise_full`` — the legacy ``ROT_METHOD = 1``. Rotates the
    spectrum in log-log space through an angle found by iteration, forwards
    and backwards, and splices the two. It still lives in ``utils`` and still
    prints diagnostics; when it moves here it registers alongside ``boost``
    and the integer ``ROT_METHOD`` global goes away.

Anything added here should say what it assumes about the noise, because that
assumption is the whole content of the method — and it propagates into every
bandwidth, and so into every ``Omega``.


The boost method
----------------

Noise rotation exists because a raw noise spectrum understates the noise at the
edges of the band. The recorded noise window is a sample of a process, and at
frequencies where the signal is strong the *same* process is present underneath
it — so a band chosen against the unmodified noise level runs wider than the
data supports, particularly at the low end where ``Omega`` is read.

``ROT_METHOD = 2`` in the legacy module, the shipped default, and the one the
Magna study uses. It raises the low and high tails independently until each
touches the signal, then keeps the larger of the two at every frequency.

**It assumes** the noise beneath the signal follows the same spectral shape as
the recorded noise, scaled by a power of a frequency ramp — so it can be
corrected by a smooth monotone lift anchored at the point where noise first
meets signal. That is an assumption about the noise process, and a different
one would give a different band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "NOISE_MODELS",
    "BoostNoise",
    "NoiseModel",
    "boost_noise",
    "centroid_frequency",
    "get_noise_model",
]


def centroid_frequency(freq: NDArray[np.float64], amp: NDArray[np.float64]) -> float:
    """Amplitude-weighted mean frequency.

    Used to split the spectrum into a low and a high half. It is taken from the
    **signal**, not the noise — the split should follow where the energy
    actually is, and the noise has no reason to share that shape.
    """
    return float(np.sum(freq * amp) / np.sum(amp))


def boost_noise(
    freq: NDArray[np.float64],
    noise_amp: NDArray[np.float64],
    signal_amp: NDArray[np.float64],
    *,
    inc: float = 0.05,
    space: tuple[float, float] = (0.001, 1.001),
    max_iter: int = 1000,
) -> NDArray[np.float64]:
    """Multiplicative factor lifting ``noise_amp`` toward ``signal_amp``.

    Returns the *factor*, not the boosted array, because it has to be applied
    to two things — the binned noise and the unbinned noise, the latter by
    interpolation onto the finer axis. Returning the factor keeps the two
    applications from drifting apart.

    Each frequency is mapped onto ``space``, which runs from near zero at the
    low end to just above one at the high end. Dividing by ``sample ** n`` with
    ``sample < 1`` therefore *raises* the low frequencies fastest, and reversing
    the mapping does the same for the high ones. The exponent grows by ``inc``
    until any point in the half being lifted reaches the signal.
    """
    if noise_amp.shape != signal_amp.shape or noise_amp.shape != freq.shape:
        raise ValueError(
            f"freq {freq.shape}, noise {noise_amp.shape} and signal "
            f"{signal_amp.shape} must all match"
        )

    centroid = centroid_frequency(freq, signal_amp)
    low = freq <= centroid
    high = ~low

    sample = np.interp(freq, [freq.min(), freq.max()], space)

    lifted_low = _lift(noise_amp, signal_amp, sample, low, inc=inc, max_iter=max_iter)
    lifted_high = _lift(
        noise_amp, signal_amp, sample[::-1], high, inc=inc, max_iter=max_iter
    )

    return np.asarray(np.maximum(lifted_low, lifted_high) / noise_amp)


def _lift(
    noise_amp: NDArray[np.float64],
    signal_amp: NDArray[np.float64],
    sample: NDArray[np.float64],
    where: NDArray[np.bool_],
    *,
    inc: float,
    max_iter: int,
) -> NDArray[np.float64]:
    """Raise ``noise_amp`` until any point in ``where`` reaches the signal.

    Solved rather than searched. The legacy code stepped the exponent by
    ``inc`` up to a thousand times and stopped at the first trial that touched
    the signal; the exponent it lands on has a closed form. For a bin to touch,

    .. code-block:: text

        noise * sample ** -n  >=  signal
        n  >=  ln(signal / noise) / -ln(sample)

    so the first exponent at which *any* bin in ``where`` touches is the
    minimum of that expression over the bins the lift actually raises — those
    with ``sample < 1``, since dividing by a larger ``sample`` lowers them.

    **The exponent is used exactly, not rounded up to a multiple of ``inc``.**
    That is the change that makes this reproducible. The old loop stepped and
    stopped at the first multiple past the touching point, so the result was a
    step function of its input: two machines differing in the last bit could
    land on either side of a step and move the noise by ``1.41x``. Solving for
    the touching point directly makes the exponent a continuous function of the
    input, so a last-bit difference in gives a last-bit difference out.

    It also removes a bias. Rounding was always *upward*, so the old code
    consistently overshot and overstated the lifted noise — a median ``1.18x``
    and up to ``1.41x`` across 39 lifts on the 28 PNR windows. That made the
    signal-to-noise pessimistic at exactly the band edges the ratio is read
    from. The exact touching point is what the algorithm was always trying to
    compute; the stepping was a crude search for it.

    ``inc`` is therefore no longer used, and is accepted only so that callers
    and stored configurations do not have to change.
    """
    del max_iter, inc  # the search they parameterised is gone

    noise, signal, scale = noise_amp[where], signal_amp[where], sample[where]
    if noise.size == 0:
        return noise_amp

    log_scale = np.log(scale)
    rises = log_scale < 0.0
    if not np.any(rises):
        # Nothing in this half can be raised: dividing by a `sample` above one
        # lowers it. The loop would have exhausted `max_iter` and returned the
        # record unchanged.
        return noise_amp

    # A bin that *already* touches needs a negative exponent to get there, so
    # clamping the minimum at zero is what "no lift needed" means. Written this
    # way rather than as an `if np.any(noise >= signal)` guard on purpose: the
    # guard is a branch, and a branch is a step. As a bin crosses from below
    # the signal to above it, `needed` passes smoothly through zero and the
    # clamp keeps the result continuous. The guard instead jumped, which is why
    # four CWT stations still disagreed across machines after the other three
    # discontinuities were fixed.
    needed = np.log(signal[rises] / noise[rises]) / -log_scale[rises]
    exponent = max(0.0, float(np.min(needed)))
    return np.asarray(noise_amp / sample**exponent)


@runtime_checkable
class NoiseModel(Protocol):
    """Anything that produces a multiplicative correction to a noise spectrum.

    Implementations are frozen dataclasses carrying their own parameters, so a
    configured model can be stored, compared and recorded in provenance.
    """

    @property
    def name(self) -> str:
        """Short identifier, recorded alongside the result."""
        ...

    def factor(
        self,
        freq: NDArray[np.float64],
        noise_amp: NDArray[np.float64],
        signal_amp: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Multiply the noise by this to get the level to judge against."""
        ...


@dataclass(frozen=True)
class BoostNoise:
    """The default: lift each tail until it touches the signal.

    See the module docstring for the assumption this encodes and
    :func:`boost_noise` for the derivation of the exponent.
    """

    space: tuple[float, float] = (0.001, 1.001)

    @property
    def name(self) -> str:
        return "boost"

    def factor(
        self,
        freq: NDArray[np.float64],
        noise_amp: NDArray[np.float64],
        signal_amp: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return boost_noise(freq, noise_amp, signal_amp, space=self.space)


@dataclass(frozen=True)
class NoNoiseModel:
    """Use the recorded noise as measured.

    Not a placeholder — it is the honest choice when the noise window is
    genuinely representative, and it is what a run needs in order to show what
    the correction is doing. Every other model here should be compared against
    it before being trusted.
    """

    @property
    def name(self) -> str:
        return "none"

    def factor(
        self,
        freq: NDArray[np.float64],
        noise_amp: NDArray[np.float64],
        signal_amp: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del freq, signal_amp
        return np.ones_like(noise_amp)


#: Registered noise models, by the name configuration refers to them by.
NOISE_MODELS: dict[str, type[BoostNoise] | type[NoNoiseModel]] = {
    "boost": BoostNoise,
    "none": NoNoiseModel,
}


def get_noise_model(name: str) -> NoiseModel:
    """Resolve a registered model by name, with its defaults."""
    try:
        cls = NOISE_MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown noise model {name!r}. Available: {sorted(NOISE_MODELS)}."
        ) from None
    return cls()
