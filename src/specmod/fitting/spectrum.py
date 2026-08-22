"""Fitting a source model to a single spectrum.

The lmfit surface used here is declared in ``stubs/lmfit``; see
``stubs/README.md``. lmfit ships no annotations, so without those a
``ModelResult`` is `Any` and nothing checks that ``result.redchi`` exists or
that ``Parameter.stderr`` can be `None` — which it is under every minimiser
that estimates no covariance matrix, including the shipped default.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, StrMethodFormatter

from .. import config as cfg
from .. import sources
from .base import REQUIRED_SPECTRUM_ATTRIBUTES, Spectrumish
from .guess import selected_band

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = ["FitSpectrum"]


class FitSpectrum:
    """Fit a source model to one spectrum with lmfit.

    Takes anything carrying :data:`REQUIRED_SPECTRUM_ATTRIBUTES` — in practice
    a :class:`~specmod.core.collection.FittableView` from
    :func:`fittable_signal`.
    """

    #: Declarations, not defaults. These were class attributes carrying `None`
    #: and `{}`, which meant two things at once: every read had to cope with a
    #: `None` that `__init__` had in fact replaced, and `meta = {}` was one
    #: dictionary shared by every instance ever constructed. `__init__` assigns
    #: all of them, so the type is what it is after construction — and the
    #: shared-mutable-default hazard is gone rather than merely unreached.
    sig: Spectrumish
    mod: lm.Model
    params: lm.Parameters
    #: `None` until :meth:`fit_mod` runs. This one really is optional, and
    #: callers test it — see :func:`specmod.plotting.plot_pair`.
    result: lm.ModelResult | None
    mod_freq: NDArray[np.float64]
    mod_amp: NDArray[np.float64]
    pass_fitting: bool
    fit_bins: bool
    meta: dict[str, Any]
    #: The :class:`specmod.sources.SpectralModel` behind the fit, when there is
    #: one. ``None`` if a bare callable was supplied.
    spectral_model: sources.SpectralModel | None

    def __init__(
        self,
        signal: Spectrumish,
        model: Any = None,
        fit_bins: bool = False,
        **params: float,
    ) -> None:
        self.result = None
        self.pass_fitting = True
        self.meta = {}
        self.spectral_model = None
        self.mod_freq = np.array([])
        self.mod_amp = np.array([])
        self.fit_bins = fit_bins
        self.set_signal(signal)
        self.set_model(model, **params)

    def fit_mod(self, **kwargs: Any) -> None:
        """Fit, judge the result, then record it — in that order.

        The judgement used to be made *after* the recording, so the
        ``pass_fitting`` column of every flat file held the value from before
        the fit ran — ``True``, the class default, on a fresh `FitSpectrum`.
        The attribute and the table disagreed, and the table is what gets
        written out and regressed on.
        """
        self.result = self.mod.fit(self.mod_amp, self.params, f=self.mod_freq, **kwargs)
        self.__determine_pass_or_fail()
        self.__set_results_to_meta()

    def set_signal(self, signal: Spectrumish) -> None:
        if self.__check_input(signal):
            self.sig = signal
            self.__set_meta(signal.meta)
        # if setting a new signal - assess and adjust the freq bounds
        self.__set_mod_amp_freq()

    def set_model(self, model: Any = None, **params: float) -> None:
        """Set the model to fit.

        Accepts a :class:`specmod.sources.SpectralModel`, a bare callable, or
        ``None`` — in which case the model is whatever ``[model]`` in the
        configuration asks for. That default is the point: before it existed,
        ``config.model.source`` was read by nothing and the caller had to pass
        the right function by hand, so a study file saying
        ``source = "boatwright"`` silently got Brune.

        A bare callable still works, because fitting an ad-hoc shape is a
        legitimate thing to want. It simply carries no provenance:
        :attr:`spectral_model` is ``None`` and nothing can report what was fitted.
        """
        if model is None:
            model = sources.from_config()

        if isinstance(model, sources.SpectralModel):
            self.spectral_model = model
            model = model.as_callable()
        else:
            self.spectral_model = None

        self.mod = lm.Model(model)
        # whenever a model is set the inital params must be set also
        self.__init_params(**params)

    @property
    def fitted(self) -> lm.ModelResult:
        """The fit result, or a message saying it has not been fitted.

        Every private reader below went straight through ``self.result``,
        which is ``None`` until :meth:`fit_mod` runs — so calling
        :meth:`quick_vis` on an unfitted spectrum raised ``AttributeError:
        'NoneType' object has no attribute 'best_fit'``, from a line that
        names neither the station nor the missing step.
        """
        if self.result is None:
            raise RuntimeError(
                f"{getattr(self.sig, 'id', 'this spectrum')} has not been "
                "fitted yet; call fit_mod() first"
            )
        return self.result

    def describe_model(self) -> str | None:
        """What is being fitted, or ``None`` for a bare callable."""
        return None if self.spectral_model is None else self.spectral_model.describe()

    def set_const(self, pname: str, value: float) -> None:
        self.params[pname].value = value
        self.params[pname].vary = False

    def set_bounds(
        self, pname: str, min: float | None = None, max: float | None = None
    ) -> None:
        if min is not None:
            self.params[pname].min = min
        if max is not None:
            self.params[pname].max = max

    def __set_meta(self, meta: Mapping[str, Any]) -> None:
        self.meta = deepcopy(dict(meta))

    def __init_params(self, **params: Any) -> None:
        """Seed the parameters, and floor ``t*`` where the configuration says.

        ``fitting.t_star_min`` existed and was read by nothing. The tutorial
        did ``fits.set_bounds("ts", min=0.0001)`` by hand and the config value
        is 1e-4 — the same number — so the setting was a written-down record of
        something every caller had to remember. Applied here, forgetting it is
        no longer possible.

        The same applies to ``fc``, and the legacy code knew it — the line
        ``# self.set_bounds('fc', min=0)`` sat commented out here. It is not a
        poor fit but an unphysical one: a negative ``t*`` says the wave gained
        energy travelling, and a corner frequency below zero says nothing at
        all. lmfit returns either if the misfit surface leans that way, and
        with the shipped multitaper default it returned ``fc = -4.45 Hz`` on
        one PNR station while ``pass_fitting`` reported success — because a
        parameter with no bound cannot be *at* its bound.
        """
        # `**params: Any` rather than `float`, because lmfit's `make_params`
        # takes a leading `verbose` argument: a model with a parameter of that
        # name would have its seed swallowed as a flag. Not a hazard for any
        # source model here, and not one this package can fix.
        self.params = self.mod.make_params(**params)
        fitting = cfg.load_config().config.fitting
        for name, floor in (
            ("ts", fitting.t_star_min),
            ("fc", fitting.corner_frequency_min),
        ):
            if name in self.params and floor is not None:
                self.set_bounds(name, min=floor)

    def reset(self) -> None:
        for par in self.params.values():
            par.vary = True
            par.min = -np.inf
            par.max = np.inf

    def __check_input(self, signal: Spectrumish) -> bool:
        """Accept anything carrying what the fit reads, not one named class.

        This used to be ``isinstance(signal, spectral.Signal)``, which is the
        coupling that kept the container holding the legacy pair — nothing
        could be handed to the fitter unless it *was* that class. What the fit
        actually needs is the six attributes below, so that is what is checked.

        Named explicitly rather than left to fail at first use: a missing
        ``bamp`` should say so here, not surface as an AttributeError from
        inside a band selection three calls later.
        """
        missing = [
            name for name in REQUIRED_SPECTRUM_ATTRIBUTES if not hasattr(signal, name)
        ]
        if missing:
            raise ValueError(
                f"{type(signal).__name__} cannot be fitted: missing "
                f"{', '.join(missing)}. A fittable spectrum needs "
                f"{', '.join(REQUIRED_SPECTRUM_ATTRIBUTES)}."
            )
        return True

    def __set_mod_amp_freq(self) -> None:
        """
        Only fit between signal limits if they are specified.
        """

        if self.fit_bins:
            freq = self.sig.bfreq
            amp = self.sig.bamp
        else:
            freq = self.sig.freq
            amp = self.sig.amp

        band = selected_band(self.sig)
        if band is not None:
            inds = np.where((freq >= band[0]) & (freq <= band[1]))
            self.mod_freq = freq[inds]
            self.mod_amp = amp[inds]
        else:
            self.mod_freq = freq
            self.mod_amp = amp

        self.mod_amp = np.log10(self.mod_amp)

    def __param_string(self) -> str:
        """``name: value+/-2sigma`` per parameter, or ``name: value`` alone.

        The old version computed ``2 * k.stderr`` unconditionally inside a
        bare ``except Exception``. ``stderr`` is ``None`` whenever the
        minimiser estimated no covariance matrix — which Powell, the shipped
        default, never does — so this raised ``TypeError`` on every fit made
        with the default configuration, swallowed it, and titled the plot
        ``NaN``. A missing uncertainty is a property of the method, not a
        failed fit, so the value is still worth printing.
        """
        parts = []
        for k in self.fitted.params.values():
            if k.stderr is None:
                parts.append(f"{k.name}: {k.value:.3f}")
            else:
                parts.append(f"{k.name}: {k.value:.3f}+/-{2 * k.stderr:.3f}")
        return ", ".join(parts)

    def quick_vis(self, ax: Axes | None = None) -> Axes:
        if ax is None:
            _fig, ax = plt.subplots(1, 1)

        ax.loglog(self.mod_freq, 10**self.mod_amp, color="grey", label=self.sig.id)
        ax.loglog(self.mod_freq, 10**self.fitted.best_fit, "k--", label="model")
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(self.__param_string())
        ax.set_xlabel("freq [Hz]")
        ax.set_ylabel("spectral amp")
        ax.legend()
        return ax

    def __get_pars(self) -> dict[str, Any]:
        p: dict[str, Any] = {}
        for k in self.fitted.params.values():
            p.update({k.name: k.value})
            p.update({k.name + "-stderr": k.stderr})
        return p

    def __get_fit_stats(self) -> dict[str, float]:
        res = self.fitted
        s: dict[str, float] = {}
        s.update({"aic": res.aic})
        s.update({"bic": res.bic})
        s.update({"chisqr": res.chisqr})
        s.update({"redchi": res.redchi})
        return s

    def __get_test_results(self) -> dict[str, bool]:
        t: dict[str, bool] = {}
        t.update({"pass_fitting": self.pass_fitting})
        return t

    def __set_results_to_meta(self) -> None:
        self.meta.update(self.__get_pars())
        self.meta.update(self.__get_fit_stats())
        self.meta.update(self.__get_test_results())

    def __determine_pass_or_fail(self) -> None:
        """A fit fails when a parameter is pinned against one of its bounds.

        Which is the useful question: a corner frequency resting on its floor
        is the minimiser saying "lower, if you would let me", and the value it
        reports is the bound rather than a measurement.

        Reset first. ``pass_fitting`` starts as a class attribute and was only
        ever set *False*, so a `FitSpectrum` that failed once could never pass
        again however many times it was refitted.

        **Where there is no uncertainty, the value itself is compared.** The
        old version treated a missing ``stderr`` as a failure, which would mark
        every fit failed under the shipped configuration: Powell does not
        estimate a covariance matrix, so lmfit has no uncertainties to report.
        That is a property of the minimiser, not a fault in the fit. Asking
        whether the value sits on the bound is the same question with the
        error bar removed.
        """
        self.pass_fitting = True
        for _par, vals in self.fitted.params.items():
            if not vals.vary:
                continue
            spread = vals.stderr if vals.stderr is not None else 0.0
            if (vals.value - spread <= vals.min) or (vals.value + spread >= vals.max):
                self.pass_fitting = False
