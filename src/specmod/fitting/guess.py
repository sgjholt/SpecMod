"""Choosing what to fit, and where the fit starts from."""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np

from .. import config as cfg
from .. import sources
from ..core.units import Motion
from .base import SpectraLike, Spectrumish

__all__ = ["fittable_signal", "initial_guess", "selected_band"]


def fittable_signal(pair: Any, id: str = "") -> Spectrumish | None:
    """The signal to fit from a paired spectrum, or ``None`` to skip it.

    Skipping is a decision the container should not have to spell out at every
    call site: a pair is unfittable when the signal-to-noise gate rejected it.

    What comes back for a :class:`~specmod.core.SpectrumPair` is its
    :class:`~specmod.core.collection.FittableView`, not its ``signal``. The
    pair keeps the unbinned and binned spectra as separate objects, which is
    right for the comparison and wrong for a fitter that wants ``freq``,
    ``amp``, ``bfreq`` and ``bamp`` side by side; the view is what puts them
    there. ``id`` names the station on it, since a frozen pair does not carry
    one of its own.

    The ``getattr`` fallback below is what a spectrum-like object that is not
    a pair takes — a bare view, or anything else presenting the same
    attributes. It is not a legacy shim; it is what lets the fitter be given
    something constructed by hand.
    """
    view = getattr(pair, "for_fitting", None)
    if view is not None:
        return pair.for_fitting(id) if pair.passes else None

    signal = getattr(pair, "signal", pair)
    passes = getattr(pair, "passes", None)
    if passes is None:
        passes = getattr(signal, "pass_snr", True)
    return signal if passes else None


def _warn_if_peak_is_meaningless(signal: Spectrumish, id: str) -> None:
    """Warn when the peak-as-``fc`` guess is being read off the wrong domain.

    A no-op for velocity, and for a spectrum that does not say what motion it
    carries — something assembled by hand is the caller's business.
    """
    motion = getattr(signal, "motion", None)
    if motion is None or Motion(motion) is Motion.VELOCITY:
        return
    warnings.warn(
        f"{id or 'this spectrum'} is in {Motion(motion).value}, and the "
        f"initial guess for fc is the frequency of the spectral peak — which "
        f"is the corner only in velocity. A {Motion(motion).value} spectrum "
        f"falls monotonically across the band, so the guess will be a band "
        f"edge and the fit will settle near it. Fit the velocity spectrum; "
        f"`llpsp` is the displacement plateau either way.",
        stacklevel=3,
    )


def initial_guess(
    spectra: SpectraLike, model: Any = None
) -> dict[str, dict[str, float]]:
    """Starting parameters for every fittable spectrum in ``spectra``.

    Replaces ``model_guess.create_simple_guess`` and its ``_fdep`` twin, which
    were two near-identical functions differing only in whether they added an
    ``a`` for frequency-dependent Q — so adding a third model meant writing a
    third guess function, and picking the wrong one gave lmfit a parameter the
    model did not take.

    **Which parameters are needed is asked of the model, not assumed.** The
    fitted callable declares them in its signature, so a model gets exactly the
    guesses it takes and nothing else. Values that cannot be read off the
    spectrum come from ``[fitting]`` in the configuration.

    The two that *are* read off the spectrum:

    ``fc``
        the frequency of the largest amplitude inside the selected band.
    ``llpsp``
        the plateau that puts the model through that peak, with every other
        parameter at its starting value — see :func:`_plateau_through`.

    Both assume a **velocity** spectrum, which is where a fit belongs anyway:
    the model carries a motion factor, so ``llpsp`` is the displacement plateau
    whichever domain is fitted, but converting first is not a neutral change of
    view — integrating implicitly low-passes and differentiating amplifies
    high-frequency noise, so the record to fit is the one the sensor recorded.

    In velocity the peak is not merely near the corner, it *is* the corner, for
    any omega-squared source: the stationary point of
    ``f * [1 + (f/fc)**(gamma*n)]**(-1/gamma)`` sits at ``f = fc`` whenever
    ``n == 2``, whatever the corner sharpness. In displacement and acceleration
    the spectrum is monotonic across the band, so the peak is whichever band
    edge it was handed and the guess is meaningless. Handed one of those, this
    warns rather than proceeding quietly.

    Stations with no band are omitted rather than given ``None`` guesses. The
    old version emitted ``{"llpsp": None, "fc": None, "ts": None}`` on
    ``IndexError``, which lmfit cannot use — the failure simply moved to the
    fit call.
    """
    if model is None:
        model = sources.from_config()
    callable_ = (
        model.as_callable() if isinstance(model, sources.SpectralModel) else model
    )
    wanted = set(inspect.signature(callable_).parameters) - {"f"}

    fitting = cfg.load_config().config.fitting
    #: Parameters no spectrum can suggest a value for.
    defaults = {
        "ts": fitting.initial_t_star,
        "a": fitting.initial_alpha,
    }

    guesses: dict[str, dict[str, float]] = {}
    for id in spectra:
        signal = fittable_signal(spectra[id], id)
        if signal is None:
            continue
        band = selected_band(signal)
        if band is None:
            continue
        inside = (signal.freq >= band[0]) & (signal.freq <= band[1])
        if not inside.any():
            continue

        _warn_if_peak_is_meaningless(signal, id)

        amp, freq = signal.amp[inside], signal.freq[inside]
        peak = int(amp.argmax())
        available = {
            "llpsp": float(np.log10(amp[peak])),
            "fc": float(freq[peak]),
            **defaults,
        }
        if "llpsp" in wanted:
            available["llpsp"] = _plateau_through(
                callable_,
                float(freq[peak]),
                float(np.log10(amp[peak])),
                {k: v for k, v in available.items() if k in wanted},
            )
        missing = wanted - set(available)
        if missing:
            raise ValueError(
                f"no initial guess is defined for {sorted(missing)}, which "
                f"{getattr(model, 'describe', lambda: callable_.__name__)()} "
                f"takes. Add it to specmod.config.FittingConfig and to "
                f"`initial_guess`, or pass explicit guesses."
            )
        guesses[id] = {k: v for k, v in available.items() if k in wanted}

    return guesses


def _plateau_through(
    model: Any, freq: float, log10_amp: float, start: dict[str, float]
) -> float:
    """The ``llpsp`` at which ``model`` passes through one observed point.

    ``llpsp`` is the displacement plateau, and the spectrum being fitted is
    usually not displacement. In velocity the peak sits at
    ``2 pi fc Omega / 2**(1/gamma)``, less whatever attenuation has taken by
    then, so seeding ``llpsp`` with the peak amplitude itself started every
    fit about ``log10(pi fc)`` too high: a median 1.08 dex on the 28 PNR
    channels, a factor of twelve.

    Asking the model avoids writing that formula per source, per attenuation
    law and per motion. ``llpsp`` enters as an additive constant in
    ``log10 A(f)``, so the model's value at ``llpsp = 0`` is exactly the
    offset to remove.

    That holds for every :class:`~specmod.sources.SpectralModel`. A bare
    callable may use the name for something else, so it is checked rather
    than assumed: unless raising ``llpsp`` by one raises the model by one,
    the observed amplitude is returned unchanged, which is what the guess
    was before.
    """
    others = {k: v for k, v in start.items() if k != "llpsp"}
    at = np.array([freq], dtype=np.float64)
    try:
        zero = float(np.asarray(model(at, llpsp=0.0, **others))[0])
        one = float(np.asarray(model(at, llpsp=1.0, **others))[0])
    except Exception:
        return log10_amp
    if not np.isfinite(zero) or not np.isclose(one - zero, 1.0):
        return log10_amp
    return log10_amp - zero


def selected_band(spectrum: Any) -> tuple[float, float] | None:
    """The band to fit over, or ``None`` to fit everything available.

    ``None`` rather than an empty array, because "no band survived" and "a band
    from 0 to 0" are different claims and the legacy spelling — an empty
    ``ubfreqs`` — could be read as either.
    """
    band = getattr(spectrum, "band", None)
    if band is None:
        return None
    return (float(band[0]), float(band[1]))
