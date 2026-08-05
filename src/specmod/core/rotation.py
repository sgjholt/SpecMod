"""Lifting a noise spectrum toward the signal before the ratio is taken.

Noise rotation exists because a raw noise spectrum understates the noise at the
edges of the band. The recorded noise window is a sample of a process, and at
frequencies where the signal is strong the *same* process is present underneath
it — so a band chosen against the unmodified noise level runs wider than the
data supports, particularly at the low end where ``Omega`` is read.

The scheme here is the "boost" method (``ROT_METHOD = 2`` in the legacy
module), which is the shipped default and the one the Magna study uses. It
raises the low and high tails independently until each touches the signal,
then keeps the larger of the two at every frequency.

Ported verbatim from ``utils.non_lin_boost_noise_func``, including the
iteration order — the increment happens *after* the trial array is computed, so
the array kept on break corresponds to the previous exponent, not the one the
counter has just reached. That off-by-one is behaviour, not an accident to fix
here: changing it moves the selected band on real data.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["boost_noise", "centroid_frequency"]


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

    lifted_low = _lift(noise_amp, signal_amp, sample, low, inc, max_iter)
    lifted_high = _lift(noise_amp, signal_amp, sample[::-1], high, inc, max_iter)

    return np.asarray(np.maximum(lifted_low, lifted_high) / noise_amp)


def _lift(
    noise_amp: NDArray[np.float64],
    signal_amp: NDArray[np.float64],
    sample: NDArray[np.float64],
    where: NDArray[np.bool_],
    inc: float,
    max_iter: int,
) -> NDArray[np.float64]:
    """Raise ``noise_amp`` until any point in ``where`` reaches the signal.

    The trial is computed before the exponent advances, matching the legacy
    loop: on break, the array returned is the one that first satisfied the
    condition, while the counter has already moved past it.
    """
    trial = noise_amp
    exponent = 0.0
    for _ in range(max_iter):
        trial = noise_amp / sample**exponent
        exponent += inc
        if np.any(trial[where] >= signal_amp[where]):
            break
    return trial
