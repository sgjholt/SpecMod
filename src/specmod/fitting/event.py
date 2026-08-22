"""Fitting every passing station in an event, and the table that comes out."""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd

from .. import config as cfg
from ..tables import read_table, write_table
from .base import SpectraLike, plot_columns
from .guess import fittable_signal, initial_guess
from .spectrum import FitSpectrum

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping
    from pathlib import Path

__all__ = ["FitSpectra"]


#: Per-station progress goes here rather than to stdout. A library must not
#: configure logging for its host, so there is no `basicConfig` anywhere in
#: this package: a caller that wants to see these calls `logging.basicConfig`
#: itself, and one that does not is not written to by surprise.
logger = logging.getLogger(__name__)


class FitSpectra:
    """Fit every passing station in an event."""

    #: Declarations, as on :class:`FitSpectrum`. `models = {}` at class level
    #: was one dictionary shared by every `FitSpectra` ever built; `__init__`
    #: rebinds it, so nothing reached the shared copy, but nothing prevented it
    #: either. `guess = {}` was never assigned anywhere at all — a class
    #: attribute recording a constructor argument that is not kept.
    spectra: SpectraLike
    models: dict[str, FitSpectrum]
    table: pd.DataFrame

    def __init__(
        self,
        spectra: SpectraLike,
        model: Any = None,
        guess: Mapping[str, Mapping[str, float]] | None = None,
        fit_bins: bool | None = None,
    ) -> None:
        """``guess=None`` derives one, rather than fitting nothing.

        It used to skip `init_fitting` entirely, so `FitSpectra(spectra)` built
        an object with no models and `fit_spectra()` silently did nothing and
        produced an empty table. There is a sensible guess available — see
        :func:`initial_guess` — so that is now the default and an explicit
        ``guess={}`` is how you say "none".
        """
        self.models = {}
        self.table = pd.DataFrame([])
        self.set_spectra(spectra)
        if fit_bins is None:
            fit_bins = cfg.load_config().config.fitting.fit_bins
        if guess is None:
            guess = initial_guess(spectra, model)
        self.init_fitting(model, guess, fit_bins)

    def __len__(self) -> int:
        return len(self.models)

    def set_spectra(self, spectra: SpectraLike) -> None:
        if self.__check_spectra(spectra):
            self.spectra = spectra

    def get_spectra(self) -> SpectraLike:
        return self.spectra

    def get_fit(self, id: str) -> FitSpectrum | None:
        if id.upper() in self.models:
            return self.models[id.upper()]
        warnings.warn(
            f"{id.upper()} is not among the fitted stations "
            f"({', '.join(sorted(self.models)) or 'none'}); returning None.",
            stacklevel=2,
        )
        return None

    def fit_spectra(self, weight_method: str | None = None, **kwargs: Any) -> None:
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
            except ValueError as error:
                # `logging`, not `warnings`, and the difference matters here:
                # warnings are deduplicated per code location by default, so a
                # run that skipped twenty stations would report one. Each skip
                # is a station missing from the results and has to be visible.
                logger.warning("skipping %s: %s", name, error)

        self.__set_fit_models_to_spectrum()
        self.__generate_group_fit_table()

    def init_fitting(
        self,
        model: Any,
        guess: Mapping[str, Mapping[str, float]],
        fit_bins: bool,
    ) -> None:
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
        tmp: dict[str, FitSpectrum] = {}
        for id in self.spectra:
            signal = fittable_signal(self.spectra[id], id)
            if signal is None or id not in guess:
                continue
            tmp[id] = FitSpectrum(signal, model, **guess[id], fit_bins=fit_bins)
        self.models = tmp

    def set_const(self, pname: str, value: float, id: str | None = None) -> None:
        if id is None:
            for mod in self.models.values():
                mod.set_const(pname, value)
        elif id in self.models:
            self.models[id].set_const(pname, value)

    def set_bounds(
        self, pname: str, min: float | None = None, max: float | None = None
    ) -> None:
        for mod in self.models.values():
            mod.set_bounds(pname, min, max)

    def reset(self, name: str = "all") -> None:
        """Unbind every parameter, on one station or all of them.

        The lookup tested ``name.upper()`` for membership and then indexed with
        ``name``, so any id not already upper-case passed the check and raised
        ``KeyError`` on the next line. Station ids are upper-case in practice,
        which is why it never fired.
        """
        if name.upper() == "ALL":
            for mod in self.models.values():
                mod.reset()
            return

        id = name.upper()
        if id in self.models:
            self.models[id].reset()
        else:
            warnings.warn(
                f"{id} is not among the fitted stations; nothing was reset.",
                stacklevel=2,
            )

    def quick_vis(self, save: str | None = None) -> None:
        rows = self.__num_rows()
        fig, axes = plt.subplots(rows, plot_columns(), figsize=(17, int(rows * 5)))
        # `strict=False`: the grid is rounded up to whole rows, so there are
        # more axes than models by construction.
        for ax, mod in zip(axes.flatten(), self.models.values(), strict=False):
            if mod.result is None or not mod.pass_fitting:
                ax.set_title(f"Fitting Failed for {mod.sig.id}")
            else:
                mod.quick_vis(ax)

        if save is not None:
            if type(save) is str:
                fig.savefig(save)
            else:
                raise ValueError("Must provide valid path as str.")

    @staticmethod
    def write_flatfile(path: str | Path, fits: FitSpectra) -> Path:
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
    def read_flatfile(path: str | Path) -> pd.DataFrame:
        """Read a fit table back. Format follows the suffix."""
        return read_table(path)

    def __check_wm(self, wm: str) -> str:
        if wm not in ["log", "none"]:
            warnings.warn(
                f"Unknown weight method {wm!r}; expected 'log' or 'none'. "
                "Falling back to 'none'.",
                stacklevel=3,
            )
            wm = "none"
        return wm

    def __generate_group_fit_table(self) -> None:
        ds = [m.meta for m in self.models.values()]
        df1 = pd.DataFrame([])
        for i, d in enumerate(ds):
            df1 = pd.concat(
                [df1, pd.DataFrame(d, index=[i])], ignore_index=True, sort=False
            )
        self.table = df1

    def __set_fit_models_to_spectrum(self) -> None:
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

    def __check_spectra(self, spectra: SpectraLike) -> bool:
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

    def __num_rows(self) -> int:
        count = len(self)
        cols = plot_columns()
        if count % cols > 0:
            return int((cols * (int(count / cols) + 1)) / cols)
        return int(count / cols)
