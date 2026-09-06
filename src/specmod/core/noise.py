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
``rotate``
    The legacy ``ROT_METHOD = 1``, described below. Rotates the log-log
    spectrum about the origin instead of scaling it.
``none``
    The recorded noise, uncorrected. The control the other two are read
    against.

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


The rotate method
-----------------

``ROT_METHOD = 1``, described in the legacy source as "actual rotation, quite
aggressive". In log-log space it rotates the noise spectrum about the origin,

.. code-block:: text

    Y'(theta) = X sin(theta) + Y cos(theta) + Y[0] * theta

with ``X = log10(f)`` and ``Y = log10(noise)``, the trailing term holding the
low-frequency end in place so the rotation pivots there rather than about the
axis origin. As with ``boost``, one angle is found for the low half and one for
the high, and the larger of the two results is kept at every frequency.

**It assumes** the discrepancy between recorded and underlying noise is a *tilt*
— the recorded window has the right level somewhere in the middle and the wrong
slope — rather than the level offset ``boost`` assumes. That is a genuinely
different claim about the noise process, which is why both are kept: comparing
the two bands is the only way to see how much of a result is the method.

A single rotation is not monotone — tilting the spectrum raises one end and
lowers the other — so it is not obvious that the spliced result can only raise
the noise, the way ``boost`` provably can. There is no proof of it here, but it
was not observed to fail: over 4000 randomised spectra spanning three decades
of frequency, spectral slopes from ``f**-3`` to ``f**1``, and noise between
0.1% and 99% of signal, the smallest factor returned was ``1 - 2e-15``. The
registry-wide test asserts the property; if some real spectrum ever violates
it, that test is where it will show up, and the honest fix is to relax the
claim rather than to clamp the method.

Two deliberate departures from the legacy implementation, neither of which can
move a published number — ``ROT_METHOD = 1`` was commented out on ``master``,
so it has never produced one:

* **The angle is solved, not stepped.** The legacy loop advanced ``theta`` by a
  fixed ``inc`` and stopped at the first trial past the touching point, which
  made the result a step function of its input — the same defect that made
  ``boost`` machine-dependent. Here the touching angle is bracketed and then
  bisected to a tolerance, so it is continuous in the input.
* **The split between halves is taken from the signal.** The legacy version
  used the *noise* centroid here and the *signal* centroid in the boost path.
  The signal is the defensible one, for the reason given at
  :func:`centroid_frequency`, and using it in both makes the two bands
  comparable — which is the point of having a registry rather than a flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "NOISE_MODELS",
    "BoostNoise",
    "NoNoiseModel",
    "NoiseModel",
    "RotateNoise",
    "boost_noise",
    "centroid_frequency",
    "get_noise_model",
    "rotate_log_spectrum",
    "rotate_noise",
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


#: Angles beyond this are not a correction to a noise level. At a quarter turn
#: the frequency and amplitude axes have swapped, and past it the rotated
#: spectrum runs backwards in frequency. The legacy loop would have kept going
#: to 250 radians — forty full turns — before giving up and returning zero;
#: anything needing more than this has already stopped meaning anything.
MAX_ROTATION = np.pi / 2

#: Coarse steps used to bracket the touching angle before bisection. Only the
#: bracket depends on this; the answer inside it does not, which is what keeps
#: the result continuous as the root crosses a step boundary.
_BRACKET_STEPS = 128


def rotate_log_spectrum(
    log_freq: NDArray[np.float64], log_amp: NDArray[np.float64], theta: float
) -> NDArray[np.float64]:
    """Rotate a log-log spectrum through ``theta`` radians.

    The trailing ``log_amp[0] * theta`` term is what makes this a rotation
    *about the low-frequency end* rather than about the axis origin: without it
    the whole curve translates as well as tilts, and the correction stops being
    anchored to the one part of the noise record that is least contaminated.
    """
    return np.asarray(
        log_freq * np.sin(theta) + log_amp * np.cos(theta) + log_amp[0] * theta
    )


def _touching_angle(
    log_freq: NDArray[np.float64],
    log_noise: NDArray[np.float64],
    log_signal: NDArray[np.float64],
    where: NDArray[np.bool_],
    *,
    backwards: bool,
    tol: float = 1e-12,
) -> float:
    """Smallest rotation at which any point in ``where`` reaches the signal.

    Bracketed then bisected rather than stepped. The margin

    .. code-block:: text

        g(t) = max over `where` of [ rotate(t)  -  log_signal ]

    is continuous in ``t`` and negative at ``t = 0`` unless the two already
    touch, so the first ``t`` with ``g(t) >= 0`` is a root of a continuous
    function and moves continuously with the data. Stepping to a multiple of a
    fixed increment does not, which is what made the legacy version — and
    ``boost`` before it was solved — differ between machines on last-bit
    input differences.

    Returns ``0.0`` when the halves already touch, and when no rotation within
    :data:`MAX_ROTATION` brings them together. Both are the legacy fallbacks;
    the difference is that the second no longer takes five thousand iterations
    to discover.
    """
    if not np.any(where):
        return 0.0

    sign = -1.0 if backwards else 1.0
    target = log_signal[where]

    def margin(t: float) -> float:
        rotated = rotate_log_spectrum(log_freq, log_noise, sign * t)[where]
        return float(np.max(rotated - target))

    if margin(0.0) >= 0.0:
        return 0.0

    step = MAX_ROTATION / _BRACKET_STEPS
    lo = 0.0
    for i in range(1, _BRACKET_STEPS + 1):
        hi = i * step
        if margin(hi) >= 0.0:
            break
        lo = hi
    else:
        return 0.0

    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if margin(mid) >= 0.0:
            hi = mid
        else:
            lo = mid
    return sign * hi


def rotate_noise(
    freq: NDArray[np.float64],
    noise_amp: NDArray[np.float64],
    signal_amp: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Multiplicative factor from rotating the noise toward the signal.

    See the module docstring for what this assumes and how it differs from
    :func:`boost_noise`. Returns the factor, for the same reason
    :func:`boost_noise` does.
    """
    if noise_amp.shape != signal_amp.shape or noise_amp.shape != freq.shape:
        raise ValueError(
            f"freq {freq.shape}, noise {noise_amp.shape} and signal "
            f"{signal_amp.shape} must all match"
        )

    log_freq = np.log10(freq)
    log_noise = np.log10(noise_amp)
    log_signal = np.log10(signal_amp)

    centroid = centroid_frequency(freq, signal_amp)
    low = freq <= centroid

    back = _touching_angle(log_freq, log_noise, log_signal, low, backwards=True)
    forward = _touching_angle(log_freq, log_noise, log_signal, ~low, backwards=False)

    lifted = np.maximum(
        10.0 ** rotate_log_spectrum(log_freq, log_noise, back),
        10.0 ** rotate_log_spectrum(log_freq, log_noise, forward),
    )
    return np.asarray(lifted / noise_amp)


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
class RotateNoise:
    """Tilt the noise toward the signal rather than scaling it.

    The legacy ``ROT_METHOD = 1``. See the module docstring for the assumption
    this encodes, how it differs from :class:`BoostNoise`, and the two
    departures from the legacy implementation.
    """

    @property
    def name(self) -> str:
        return "rotate"

    def factor(
        self,
        freq: NDArray[np.float64],
        noise_amp: NDArray[np.float64],
        signal_amp: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return rotate_noise(freq, noise_amp, signal_amp)


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
NOISE_MODELS: dict[str, type[BoostNoise] | type[RotateNoise] | type[NoNoiseModel]] = {
    "boost": BoostNoise,
    "rotate": RotateNoise,
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
