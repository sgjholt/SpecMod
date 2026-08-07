from copy import deepcopy

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, StrMethodFormatter

from . import config as cfg
from . import sources
from .tables import read_table, write_table

# global variables
# One home for this: it used to be defined in *both* the SPECTRAL and FITTING
# dicts, and the two copies could disagree.
PLOT_COLUMNS = cfg.load_config().config.viz.plot_columns

#: What :class:`FitSpectrum` reads off whatever it is given. Kept as data so
#: the requirement is stated once and can be asserted against.
REQUIRED_SPECTRUM_ATTRIBUTES = ("id", "meta", "freq", "amp", "bfreq", "bamp")


def fittable_signal(pair, id=""):
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


def initial_guess(spectra, model=None):
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

    ``llpsp``
        ``log10`` of the largest amplitude inside the selected band — the
        long-period plateau, which is what ``Omega`` is.
    ``fc``
        the frequency at which that maximum falls.

    Both assume a **velocity** spectrum, where the peak sits near the corner.
    On a displacement spectrum the peak is at the low-frequency end and ``fc``
    would start at the bottom of the band; that is the pre-existing assumption,
    made explicit here rather than left in a function name.

    Stations with no band are omitted rather than given ``None`` guesses. The
    old version emitted ``{"llpsp": None, "fc": None, "ts": None}`` on
    ``IndexError``, which lmfit cannot use — the failure simply moved to the
    fit call.
    """
    import inspect  # noqa: PLC0415

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

    guesses = {}
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

        amp, freq = signal.amp[inside], signal.freq[inside]
        peak = int(amp.argmax())
        available = {
            "llpsp": float(np.log10(amp[peak])),
            "fc": float(freq[peak]),
            **defaults,
        }
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


def selected_band(spectrum):
    """The band to fit over, or ``None`` to fit everything available.

    ``None`` rather than an empty array, because "no band survived" and "a band
    from 0 to 0" are different claims and the legacy spelling — an empty
    ``ubfreqs`` — could be read as either.
    """
    band = getattr(spectrum, "band", None)
    if band is None:
        return None
    return (float(band[0]), float(band[1]))


class FitSpectrum:
    """Fit a source model to one spectrum with lmfit.

    Takes anything carrying :data:`REQUIRED_SPECTRUM_ATTRIBUTES` — in practice
    a :class:`~specmod.core.collection.FittableView` from
    :func:`fittable_signal`.
    """

    sig = None
    mod = None
    params = None
    result = None
    mod_freq = np.array([])
    mod_amp = np.array([])
    pass_fitting = True
    fit_bins = False
    meta = {}
    #: The :class:`specmod.sources.SpectralModel` behind the fit, when there is
    #: one. ``None`` if a bare callable was supplied.
    spectral_model = None

    def __init__(self, signal, model=None, fit_bins=False, **params):
        self.fit_bins = fit_bins
        self.set_signal(signal)
        self.set_model(model, **params)

    def fit_mod(self, **kwargs):
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

    def set_signal(self, signal):
        if self.__check_input(signal):
            self.sig = signal
            self.__set_meta(signal.meta)
        # if setting a new signal - assess and adjust the freq bounds
        self.__set_mod_amp_freq()

    def set_model(self, model=None, **params):
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

    def describe_model(self):
        """What is being fitted, or ``None`` for a bare callable."""
        return None if self.spectral_model is None else self.spectral_model.describe()

    def set_const(self, pname, value):
        self.params[pname].value = value
        self.params[pname].vary = False

    def set_bounds(self, pname, min=None, max=None):
        if min is not None:
            self.params[pname].min = min
        if max is not None:
            self.params[pname].max = max

    def __set_meta(self, meta):
        self.meta = deepcopy(meta)

    def __init_params(self, **params):
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
        self.params = self.mod.make_params(**params)
        fitting = cfg.load_config().config.fitting
        for name, floor in (
            ("ts", fitting.t_star_min),
            ("fc", fitting.corner_frequency_min),
        ):
            if name in self.params and floor is not None:
                self.set_bounds(name, min=floor)

    def reset(self):
        for par in self.params.values():
            par.vary = True
            par.min = -np.inf
            par.max = np.inf

    def __check_input(self, signal):
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

    def __set_mod_amp_freq(self):
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

    def __param_string(self):
        try:
            pars = [
                [k.name, k.value, 2 * k.stderr] for k in self.result.params.values()
            ]
            return ", ".join(["{}: {:.3f}+/-{:.3f}" for _ in pars]).format(
                *[val for sublist in pars for val in sublist]
            )
        except Exception as msg:
            print(msg)
            return "NaN"

    def quick_vis(self, ax=None):
        if ax is None:
            _fig, ax = plt.subplots(1, 1)

        ax.loglog(self.mod_freq, 10**self.mod_amp, color="grey", label=self.sig.id)
        ax.loglog(self.mod_freq, 10**self.result.best_fit, "k--", label="model")
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(self.__param_string())
        ax.set_xlabel("freq [Hz]")
        ax.set_ylabel("spectral amp")
        ax.legend()

        if ax is not None:
            return ax

    def __get_pars(self):
        p = {}
        for k in self.result.params.values():
            p.update({k.name: k.value})
            p.update({k.name + "-stderr": k.stderr})
        return p

    def __get_fit_stats(self):
        res = self.result
        s = {}
        s.update({"aic": res.aic})
        s.update({"bic": res.bic})
        s.update({"chisqr": res.chisqr})
        s.update({"redchi": res.redchi})
        return s

    def __get_test_results(self):
        t = {}
        t.update({"pass_fitting": self.pass_fitting})
        return t

    def __set_results_to_meta(self):
        self.meta.update(self.__get_pars())
        self.meta.update(self.__get_fit_stats())
        self.meta.update(self.__get_test_results())

    def __determine_pass_or_fail(self):
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
        for _par, vals in self.result.params.items():
            if not vals.vary:
                continue
            spread = vals.stderr if vals.stderr is not None else 0.0
            if (vals.value - spread <= vals.min) or (vals.value + spread >= vals.max):
                self.pass_fitting = False


class FitSpectra:
    spectra = None
    models = {}
    guess = {}
    table = pd.DataFrame([])

    def __init__(self, spectra, model=None, guess=None, fit_bins=None):
        """``guess=None`` derives one, rather than fitting nothing.

        It used to skip `init_fitting` entirely, so `FitSpectra(spectra)` built
        an object with no models and `fit_spectra()` silently did nothing and
        produced an empty table. There is a sensible guess available — see
        :func:`initial_guess` — so that is now the default and an explicit
        ``guess={}`` is how you say "none".
        """
        self.set_spectra(spectra)
        if fit_bins is None:
            fit_bins = cfg.load_config().config.fitting.fit_bins
        if guess is None:
            guess = initial_guess(spectra, model)
        self.init_fitting(model, guess, fit_bins)

    def __len__(self):
        return len(self.models)

    def set_spectra(self, spectra):
        if self.__check_spectra(spectra):
            self.spectra = spectra

    def get_spectra(self):
        return self.spectra

    def get_fit(self, id):
        if id.upper() in self.models.keys():
            return self.models[id.upper()]
        else:
            print(f"WARNING: {id.upper()} not in group of available fits.")

    def fit_spectra(self, weight_method=None, **kwargs):
        """Fit every station, with the configured minimiser unless told otherwise.

        ``method`` and ``weight_method`` both come from ``[fitting]`` when not
        given. Neither used to: `fit_spectra()` fell through to lmfit's default
        minimiser, so a study file saying ``method = "powell"`` was ignored and
        the caller had to remember ``fit_spectra(method="powell")`` — which the
        tutorial does and nothing enforced.

        It matters. On the 28 PNR windows lmfit's default returns a **negative
        corner frequency** on one station where Powell does not; a corner
        frequency below zero is not a degraded measurement but a meaningless
        one, and nothing downstream rejects it.
        """
        fitting = cfg.load_config().config.fitting
        if weight_method is None:
            weight_method = fitting.weight_method
        kwargs.setdefault("method", fitting.method)
        wm = self.__check_wm(weight_method)
        for name, mod in self.models.items():
            try:
                if wm == "log":
                    mod.fit_mod(weights=1 / mod.mod_freq, **kwargs)
                else:
                    mod.fit_mod(**kwargs)
            except ValueError as msg:
                print(msg)
                print("-" * 40)
                print(f"Skipping {name}")

        self.__set_fit_models_to_spectrum()
        self.__generate_group_fit_table()

    def init_fitting(self, model, guess, fit_bins):
        """Build a fit per passing station.

        ``model=None`` resolves through the configuration once per station,
        which is cheap and keeps every fit in a run agreeing on what it is
        fitting.
        """
        # Iterate the container rather than reaching into `.group`. `Spectra`
        # and `core.SpectrumSet` both present this interface, which is what
        # lets the container be swapped underneath without touching the fitter.
        # A station is fitted when it passed the gate *and* has a guess.
        # Indexing `guess[id]` unconditionally made a partial guess dict a
        # `KeyError` naming a station, rather than a way to fit a subset —
        # and made `guess={}` a crash instead of "fit nothing".
        tmp = {}
        for id in self.spectra:
            signal = fittable_signal(self.spectra[id], id)
            if signal is None or id not in guess:
                continue
            tmp[id] = FitSpectrum(signal, model, **guess[id], fit_bins=fit_bins)
        self.models = tmp

    def set_const(self, pname, value, id=None):
        if id is None:
            for mod in self.models.values():
                mod.set_const(pname, value)
        else:
            if id in self.models.keys():
                self.models[id].set_const(pname, value)

    def set_bounds(self, pname, min=None, max=None):
        for mod in self.models.values():
            mod.set_bounds(pname, min, max)

    def reset(self, name="all"):
        if name.upper() == "ALL":
            for mod in self.models.values():
                mod.reset()
        else:
            if name.upper() in self.models.keys():
                self.models[name].reset()
            else:
                print(f"WARNING: {name.upper()} not in available channels.")

    def quick_vis(self, save=None):
        l = self.__num_rows()
        fig, axes = plt.subplots(l, PLOT_COLUMNS, figsize=(17, int(l * 5)))
        axes = axes.flatten()
        for ax, mod in zip(axes, self.models.values()):
            if mod.result is None or not mod.pass_fitting:
                ax.set_title(f"Fitting Failed for {mod.sig.id}")
            else:
                ax = mod.quick_vis(ax)

        if save is not None:
            if type(save) is str:
                fig.savefig(save)
            else:
                raise ValueError("Must provide valid path as str.")

    @staticmethod
    def write_flatfile(path, fits):
        """Write the group fit table, in the format ``path``'s suffix names.

        ``.parquet`` is typed, compressed and queryable without loading;
        ``.csv`` is what journal supplements want. See :mod:`specmod.tables`.

        The previous implementation was ``os.makedirs(os.path.join(
        *path.split("/")[:-1]))``, which raised ``TypeError: join() missing 1
        required positional argument`` for any path without a directory
        component — ``write_flatfile("out.csv", fits)`` could not work. It also
        split on ``/`` literally, so it did nothing useful on Windows.
        """
        return write_table(path, fits.table)

    @staticmethod
    def read_flatfile(path):
        """Read a fit table back. Format follows the suffix."""
        return read_table(path)

    def __check_wm(self, wm):
        if wm not in ["log", "none"]:
            print(f"WARNING: did not recognise weight method {wm}.")
            print("Setting to none...")
            wm = "none"
        return wm

    def __generate_group_fit_table(self):
        ds = [m.meta for m in self.models.values()]
        df1 = pd.DataFrame([])
        for i, d in enumerate(ds):
            df1 = pd.concat(
                [df1, pd.DataFrame(d, index=[i])], ignore_index=True, sort=False
            )
        self.table = df1

    def __set_fit_models_to_spectrum(self):
        """Hand each fit back to the spectrum it came from, where that is possible.

        The legacy `Signal` carries its own fit so that plotting and
        serialisation can reach it from the spectrum. `core.SpectrumPair` is
        frozen and cannot, by design — a result writing itself back into its
        own input is how a container stops being trustworthy.

        Nothing is lost by skipping it: `self.models` is the source of truth
        either way, and the write-back was only ever a convenience. So this
        writes where the container accepts it and moves on where it does not,
        rather than requiring every container to be mutable.
        """
        for id, mod in self.models.items():
            spectrum = self.spectra[id]
            signal = getattr(spectrum, "signal", spectrum)
            setter = getattr(signal, "set_model", None)
            if setter is not None:
                setter(mod)

    def __check_spectra(self, spectra):
        """Accept anything that maps trace ids to paired spectra.

        Was ``isinstance(spectra, spectral.Spectra)``, which is why the fitter
        could not be handed a :class:`~specmod.core.SpectrumSet` even though it
        only ever iterates and indexes. Requiring one concrete class was the
        last thing tying the fitter to the legacy module.
        """
        required = ("__iter__", "__getitem__", "__len__")
        missing = [name for name in required if not hasattr(spectra, name)]
        if missing:
            raise ValueError(
                f"{type(spectra).__name__} cannot be fitted: it must map trace "
                f"ids to paired spectra, and is missing {', '.join(missing)}. "
                f"Use specmod.pipeline.spectrum_set_from_streams."
            )
        return True

    def __num_rows(self):
        l = self.__len__()
        cols = PLOT_COLUMNS
        if l % cols > 0:
            return int((cols * (int(l / cols) + 1)) / cols)
        else:
            return int(l / cols)
