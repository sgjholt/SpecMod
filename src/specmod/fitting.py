import os
from copy import deepcopy

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter, StrMethodFormatter

from . import config as cfg
from . import sources
from . import spectral as sp

# global variables
PLOT_COLUMNS = cfg.FITTING["PLOT_COLUMNS"]


class FitSpectrum:
    """
    Takes an Spectral.Signal and fits an arbitrary model to the signal spectrum
    using the lmfit package.
    """

    sig = sp.Signal()
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
        self.result = self.mod.fit(self.mod_amp, self.params, f=self.mod_freq, **kwargs)
        self.__set_results_to_meta()
        self.__determine_pass_or_fail()

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
        self.params = self.mod.make_params(**params)
        # self.set_bounds('fc', min=0)

    def reset(self):
        for par in self.params.values():
            par.vary = True
            par.min = -np.inf
            par.max = np.inf

    def __check_input(self, signal):
        if not isinstance(signal, sp.Signal):
            raise ValueError(f"Must be a signal object not {type(signal)}")
        else:
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

        if self.sig.ubfreqs.size > 0:
            inds = np.where(
                (freq >= self.sig.ubfreqs[0]) & (freq <= self.sig.ubfreqs[1])
            )
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
        for par, vals in self.result.params.items():
            try:
                if (vals.value - vals.stderr <= vals.min) or (
                    vals.value + vals.stderr >= vals.max
                ):
                    # print(par, vals)
                    self.pass_fitting = False
            except TypeError:
                # print("std err is none")
                # print(par, vals)
                self.pass_fitting = False


class FitSpectra:
    spectra = sp.Spectra()
    models = {}
    guess = {}
    table = pd.DataFrame([])

    def __init__(self, spectra, model=None, guess=None, fit_bins=False):
        self.set_spectra(spectra)
        if guess is not None:
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

    def fit_spectra(self, weight_method="none", **kwargs):
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
        tmp = {}
        for id, spec in self.spectra.group.items():
            if spec.signal.pass_snr:
                fit = FitSpectrum(spec.signal, model, **guess[id], fit_bins=fit_bins)
                tmp.update({id: fit})
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
        os.makedirs(os.path.join(*path.split("/")[:-1]), exist_ok=True)
        fits.table.to_csv(path, index=False)

    @staticmethod
    def read_flatfile(path):
        return pd.read_csv(path)

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
        for id, mod in self.models.items():
            tmp = self.spectra.get_spectra(id)
            tmp.signal.set_model(mod)

    def __check_spectra(self, spectra):
        if not isinstance(spectra, sp.Spectra):
            raise ValueError(f"Must be a spectra object not {type(spectra)}")
        else:
            return True

    def __num_rows(self):
        l = self.__len__()
        cols = PLOT_COLUMNS
        if l % cols > 0:
            return int((cols * (int(l / cols) + 1)) / cols)
        else:
            return int(l / cols)
