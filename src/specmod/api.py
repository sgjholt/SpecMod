"""Stable public surface for downstream packages.

Anything not exported here is internal and may change without notice. Anything
exported here follows the deprecation policy in ``CONTRIBUTING.md``: one minor
version of ``DeprecationWarning`` before a removal or a signature change, even
while SpecMod is ``0.x``.

The point of the module is containment. SpecMod's internals are still being
refactored and its own documentation warns of breaking changes at every ``0.x``
release; downstream packages import *this* and nothing else, so an internal
rename costs a line here instead of a release there.

Five properties hold for everything below, and they are what make the surface
usable from a service that owns its own IO and has to be able to replay a job:

1. **Path-free.** Every function takes in-memory data — arrays, or ObsPy
   objects. None of them opens a file. Convenience wrappers that take paths
   live elsewhere in the package.
2. **Deterministic.** The same inputs and the same explicit arguments produce
   the same outputs. Nothing here reads the working directory or the
   environment, and nothing draws random numbers. See the caveat on
   :func:`fit_spectrum`.
3. **Non-mutating.** Inputs are left as they were found; results are new
   objects.
4. **Quiet.** Nothing prints. Diagnostics go through :mod:`logging` and
   :mod:`warnings`.
5. **Typed errors.** Failures are :class:`~specmod.exceptions.SpecModError`
   subclasses — see :mod:`specmod.exceptions` for which of the three, and why
   the distinction is the useful part.

Examples
--------
>>> import numpy as np
>>> from specmod import api
>>> rng = np.random.default_rng(0)
>>> signal = api.estimate_spectrum(rng.normal(size=2048), 0.01,
...                                estimator="multitaper")
>>> noise = api.estimate_spectrum(rng.normal(size=1024), 0.01,
...                               estimator="multitaper")
>>> pair = api.compare_spectra(signal, noise)
>>> pair.snr.shape == pair.binned_signal.freq.shape
True
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import __version__
from .config import Config, ResolvedConfig, config_hash, load_config
from .config.serialize import to_toml as _to_toml
from .core.collection import SpectrumPair
from .core.spectrum import Spectrum
from .core.units import AmplitudeKind, Motion
from .exceptions import (
    InternalError,
    InvalidInputError,
    MissingBackendError,
    SpecModError,
)
from .fitting import FitSpectrum, fittable_signal, initial_guess
from .transforms import (
    ESTIMATORS,
    get_estimator,
    make_window,
    window_correction,
)

__all__ = [
    "AmplitudeKind",
    "Config",
    "InternalError",
    "InvalidInputError",
    "MissingBackendError",
    "Motion",
    "ResolvedConfig",
    "SpecModError",
    "Spectrum",
    "SpectrumFit",
    "SpectrumPair",
    "__version__",
    "available_estimators",
    "compare_spectra",
    "config_hash",
    "config_to_toml",
    "estimate_spectrum",
    "fit_spectrum",
    "load_config",
    "make_window",
    "window_correction",
]

#: Which distribution each estimator needs beyond a default install.
#:
#: Measured rather than inferred, by constructing every registered estimator
#: and running it in an environment with none of the extras present: `cwt` and
#: `quadratic` are implemented against numpy and scipy and work without their
#: nominal extras, and only `prieto` actually requires one. Guessing from the
#: extras table in ``pyproject.toml`` would have marked three unavailable.
_ESTIMATOR_REQUIRES: Mapping[str, str | None] = {
    "fft": None,
    "welch": None,
    "multitaper": None,
    "quadratic": None,
    "cwt": None,
    "prieto": "multitaper",
}


@contextmanager
def _typed_errors() -> Iterator[None]:
    """Translate the builtins internals raise into the documented hierarchy.

    At the boundary rather than inside, because the internals are still moving
    and this module is the thing that is supposed to stay still.
    """
    try:
        yield
    except SpecModError:
        raise
    except ImportError as error:
        raise MissingBackendError(str(error)) from error
    except (ValueError, TypeError, KeyError) as error:
        raise InvalidInputError(str(error)) from error


def available_estimators() -> tuple[str, ...]:
    """The estimators that can actually run in this environment, sorted.

    SpecMod installs without its optional backends, so the registry is not the
    same question as what will work. Ask this before offering a choice to a
    user, rather than discovering the answer as a failed job.

    Returns
    -------
    tuple of str
        Names accepted by ``estimator=`` on :func:`estimate_spectrum`.

    Examples
    --------
    >>> "fft" in available_estimators()
    True
    """
    available = []
    for name, requires in sorted(_ESTIMATOR_REQUIRES.items()):
        if name not in ESTIMATORS:  # pragma: no cover - registry drift
            continue
        if requires is None:
            available.append(name)
            continue
        try:
            found = importlib.util.find_spec(requires) is not None
        except (ImportError, ValueError):
            # A blocked or broken module. `--without-optional-extras` installs
            # a finder that raises ModuleNotFoundError from `find_spec`, which
            # is exactly the "not available" answer.
            found = False
        if found:
            available.append(name)
    return tuple(available)


def estimate_spectrum(
    data: ArrayLike,
    dt: float,
    *,
    estimator: str,
    motion: Motion | str = Motion.VELOCITY,
    meta: Mapping[str, Any] | None = None,
    **options: Any,
) -> Spectrum:
    """Estimate the amplitude spectrum of one in-memory record.

    Parameters
    ----------
    data
        The record, as a 1-D array of samples. Not modified.
    dt
        Sample interval in seconds.
    estimator
        Which backend, from :func:`available_estimators`. Required rather than
        defaulted: the configured default is a property of a study, and a
        service that resolves it silently cannot replay a job it did not
        record.
    motion
        The ground-motion domain the record is in. Carried on the result, and
        what makes converting between domains a typed operation later.
    meta
        Extra metadata to attach to the spectrum. Copied, not held.
    **options
        Passed to the estimator's constructor — ``n_tapers``,
        ``time_bandwidth`` and so on. Backend-specific.

    Returns
    -------
    Spectrum
        Frequency axis, amplitude, and the metadata needed to interpret both.

    Raises
    ------
    InvalidInputError
        The record is empty, not 1-D, contains non-finite values, or the
        estimator name is not known.
    MissingBackendError
        The estimator needs an optional extra that is not installed.
    """
    with _typed_errors():
        backend = get_estimator(estimator, **options)
        return backend.estimate(
            np.asarray(data), dt, motion=motion, meta=dict(meta) if meta else None
        )


def compare_spectra(
    signal: Spectrum,
    noise: Spectrum,
    **settings: Any,
) -> SpectrumPair:
    """Judge a signal spectrum against its noise window.

    Returns the pair, **including the per-bin signal-to-noise curve** rather
    than only the band derived from it: ``pair.snr`` is an array aligned with
    ``pair.binned_signal.freq``, and ``pair.band`` is one summary of it. A
    consumer that needs a different threshold, or that admits data bin by bin
    rather than over a contiguous interval, needs the curve — and a curve
    cannot be recovered from a stored interval.

    Parameters
    ----------
    signal, noise
        Spectra from :func:`estimate_spectrum`. Neither is modified.
    **settings
        ``threshold``, ``f_min``, ``f_max``, ``n_bins``, ``noise_model``,
        ``bandwidth`` and the rest of
        :meth:`specmod.core.collection.SpectrumPair.compare`. All have explicit
        defaults; none is read from configuration.

    Returns
    -------
    SpectrumPair
        With ``binned_signal``, ``binned_noise``, ``snr``, ``band`` and
        ``resolution_floor``.

    Raises
    ------
    InvalidInputError
        The two spectra do not describe the same record geometry — a frequency
        axis above its own Nyquist, most often from pairing windows that came
        from different sampling rates.
    """
    with _typed_errors():
        return SpectrumPair.compare(signal, noise, **settings)


@dataclass(frozen=True, slots=True)
class SpectrumFit:
    """The result of fitting a source model to one spectrum.

    Frozen, and holding plain numbers rather than the fitter's own objects, so
    it can be serialised and compared without depending on lmfit's API.

    Attributes
    ----------
    params
        Fitted values, keyed by name — ``llpsp`` (the long-period spectral
        level Ω₀, as its base-10 logarithm), ``fc``, ``ts`` (t\\*).
    stderr
        One standard error per parameter, where the fitter could estimate one.
        **Empty under some minimisers** — see the note on
        :func:`fit_spectrum`. Absent means not measured, and is left absent
        rather than filled with a zero that would read as "certain".
    covariance
        The covariance matrix, with ``names`` giving its row and column order,
        or ``None`` when the minimiser produced none. The ``fc``-``t*``
        correlation lives here, and reporting either parameter without it
        overstates both.
    chisqr, redchi
        Misfit, and misfit per degree of freedom.
    n_points
        How many spectral samples the fit actually used.
    success
        Whether the minimiser reported convergence.
    """

    params: Mapping[str, float]
    stderr: Mapping[str, float]
    covariance: NDArray[np.float64] | None
    names: tuple[str, ...]
    chisqr: float
    redchi: float
    n_points: int
    success: bool

    def correlation(self, a: str, b: str) -> float | None:
        """Correlation between two fitted parameters, or ``None``.

        ``None`` when there is no covariance matrix, or when either parameter
        has no variance to correlate — not zero, which would read as
        "independent" rather than "not measured".
        """
        if self.covariance is None or a not in self.names or b not in self.names:
            return None
        i, j = self.names.index(a), self.names.index(b)
        denominator = np.sqrt(self.covariance[i, i] * self.covariance[j, j])
        if not np.isfinite(denominator) or denominator == 0:
            return None
        return float(self.covariance[i, j] / denominator)


def fit_spectrum(
    pair: SpectrumPair,
    *,
    id: str = "",
    model: Any = None,
    guess: Mapping[str, float] | None = None,
    fit_bins: bool = False,
    method: str | None = None,
    weight_method: str | None = None,
    **fit_options: Any,
) -> SpectrumFit:
    """Fit a source model to one spectrum, with its uncertainty.

    This is the **per-spectrum** fit. SpecMod does not do a joint per-event
    inversion: :class:`specmod.fitting.FitSpectra` loops over stations and fits
    each independently, sharing no parameters between them. A joint solver
    belongs to whoever needs one, on top of this.

    Parameters
    ----------
    pair
        From :func:`compare_spectra`. Its selected band is what gets fitted.
        Not modified.
    id
        Label carried into the result's metadata.
    model
        A model object, or ``None`` for the configured default.
    guess
        Starting values for the fitted parameters. ``None`` derives them from
        the spectrum with :func:`specmod.fitting.initial_guess`, which is what
        :class:`specmod.fitting.FitSpectra` does. Do not skip it: without a
        starting corner frequency the minimiser walks ``fc`` to zero and the
        model evaluates to NaN, so an unguessed fit does not merely fit worse,
        it raises.
    fit_bins
        Fit the log-binned spectrum rather than the full-resolution one.
    method
        Minimiser name, passed to lmfit. ``None`` takes ``[fitting] method``
        from configuration, which is what :class:`specmod.fitting.FitSpectra`
        does — so a single-spectrum fit here matches the same station's fit in
        an event run. Naming it explicitly is what makes the call reproducible
        somewhere else, and the default matters: on the 28 PNR windows lmfit's
        own default returns a negative corner frequency on one station where
        the configured ``powell`` does not.
    weight_method
        ``"log"`` weights residuals by ``1/f``; ``"none"`` does not. ``None``
        takes ``[fitting] weight_method`` from configuration.
    **fit_options
        Anything else lmfit's ``fit`` accepts.

    Returns
    -------
    SpectrumFit
        Point estimates *and* their errors and covariance. Frozen.

    Raises
    ------
    InvalidInputError
        The pair has no usable band, or the spectrum is missing an attribute
        the model needs.

    Notes
    -----
    **Uncertainty depends on the minimiser, and the configured default does
    not provide it.** Only the least-squares family produces a covariance
    matrix. Measured on one synthetic station, all four agree on the corner
    frequency and only two report an error for it:

    ========== ======== ============ ===================
    ``method`` ``fc``   ``fc`` error ``fc``-``t*`` corr.
    ========== ======== ============ ===================
    powell     7.925    --           --
    nelder     7.925    --           --
    leastsq    7.925    0.129        0.837
    ========== ======== ============ ===================

    ``[fitting] method`` ships as ``powell``, so a default fit returns point
    estimates with an empty ``stderr`` and no covariance. Pass
    ``method="leastsq"`` when the uncertainty is the point. That correlation is
    not incidental: 0.84 between ``fc`` and ``t*`` is why neither should be
    quoted alone.

    **One determinism caveat, and it is the only one on this surface.** The
    initial guess and the default minimiser are read from configuration by
    internals, through :func:`specmod.config.load_config`, which resolves
    against the current working directory and the environment. Two runs in the
    same process with the same working directory agree exactly; two runs in
    different directories may not, if a ``specmod.toml`` differs between them.

    Pass ``model`` and the minimiser options explicitly to close that gap, and
    record :func:`config_hash` alongside any result you intend to replay.
    """
    with _typed_errors():
        if method is None or weight_method is None:
            fitting = load_config().config.fitting
            method = fitting.method if method is None else method
            weight_method = (
                fitting.weight_method if weight_method is None else weight_method
            )
        if weight_method not in ("log", "none"):
            raise InvalidInputError(
                f"Unknown weight_method {weight_method!r}; expected 'log' or 'none'."
            )

        signal = fittable_signal(pair, id)
        if signal is None:
            raise InvalidInputError(
                f"{id or 'this pair'} has no usable band, so there is nothing "
                "to fit. `SpectrumPair.passes` reports that before you get here."
            )
        if guess is None:
            guess = initial_guess({id: pair}, model).get(id, {})

        fitter = FitSpectrum(signal, model, fit_bins, **guess)
        if weight_method == "log":
            fit_options["weights"] = 1 / fitter.mod_freq
        fitter.fit_mod(method=method, **fit_options)
        return _as_fit(fitter)


def _as_fit(fitter: FitSpectrum) -> SpectrumFit:
    """Extract the numbers from lmfit's result object.

    Nothing is computed here that the fitter did not already produce; this is
    a projection, so that the public type does not have lmfit in it.
    """
    result = fitter.result
    if result is None:  # pragma: no cover - fit_mod always assigns
        raise InternalError("fit produced no result object")

    names = tuple(result.params)
    params = {name: float(result.params[name].value) for name in names}

    stderr = {}
    for name in names:
        error = result.params[name].stderr
        if error is not None:
            stderr[name] = float(error)

    covariance = None
    if result.covar is not None:
        covariance = np.asarray(result.covar, dtype=np.float64)
        # lmfit's covariance covers only the parameters it varied, in their
        # order — not every parameter in `params`, which may include fixed
        # ones. Using `names` for its axes would mislabel the matrix.
        names = tuple(name for name in names if result.params[name].vary)

    return SpectrumFit(
        params=params,
        stderr=stderr,
        covariance=covariance,
        names=names,
        chisqr=float(result.chisqr),
        redchi=float(result.redchi),
        n_points=int(result.ndata),
        success=bool(result.success),
    )


def config_to_toml(config: Config, *, header: str | None = None) -> str:
    """Serialise a configuration to TOML, as ``specmod config freeze`` does.

    Returns the text rather than writing it, so the caller decides where it
    goes — which for anything but a local filesystem is the only workable
    arrangement.
    """
    with _typed_errors():
        return _to_toml(config, header=header)
