"""The ``[fitting]`` configuration is read, and the defaults are usable.

Four of the six values in :class:`~specmod.config.FittingConfig` were read by
nothing: ``method``, ``fit_bins``, ``weight_method`` and ``t_star_min``. A
study file setting any of them was silently ignored, and the caller had to
remember an incantation instead — which the tutorial does, in a comment, and
nothing enforced.

That is the same defect class as ``config.model.source`` before the
``sources`` package existed, and as ``snr.bandwidth_method`` naming a strategy
the registry would not accept. It recurs because a config field and its reader
are in different files and nothing ties them together, so these tests are the
tie.
"""

from __future__ import annotations

import contextlib
import functools
import io
from typing import Any

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

from specmod.config import load_config  # noqa: E402
from specmod.fitting import FitSpectra, initial_guess  # noqa: E402
from specmod.pipeline import spectrum_set_from_streams  # noqa: E402


@functools.cache
def _spectra(windows: Any) -> Any:
    signal, noise = windows()
    return spectrum_set_from_streams(signal, noise, estimator="fft")


def _fit(spectra: Any, **kwargs: Any) -> Any:
    bounds = kwargs.pop("bounds", None)
    fit_kwargs = kwargs.pop("fit", {})
    with contextlib.redirect_stdout(io.StringIO()):
        fit = FitSpectra(spectra, **kwargs)
        for name, limits in (bounds or {}).items():
            fit.set_bounds(name, **limits)
        fit.fit_spectra(**fit_kwargs)
    return fit


class TestDefaults:
    def test_a_bare_fit_spectra_actually_fits(self, pnr_windows: Any) -> None:
        """``FitSpectra(spectra)`` used to build nothing at all.

        ``guess=None`` skipped ``init_fitting`` entirely, so ``fit_spectra()``
        iterated an empty dict, said nothing, and produced an empty table. A
        sensible guess is derivable, so that is the default now.
        """
        fit = _fit(_spectra(pnr_windows))
        assert len(fit.models) == 28
        assert len(fit.table) == 28

    def test_an_explicit_empty_guess_still_means_none(self, pnr_windows: Any) -> None:
        """The old behaviour is reachable, it is just no longer the default."""
        fit = _fit(_spectra(pnr_windows), guess={})
        assert len(fit.models) == 0

    def test_the_defaults_match_what_the_tutorial_did_by_hand(
        self, pnr_windows: Any
    ) -> None:
        """The binding test for this whole change.

        The tutorial wrote ``fits.set_bounds("ts", min=0.0001)`` and
        ``fits.fit_spectra(method="powell")``. Both values are in the shipped
        configuration — ``t_star_min = 1e-4``, ``method = "powell"`` — and were
        read by nothing, so the settings were a written-down record of
        something every caller had to remember. Reading them has to reproduce
        the remembered version exactly, or it is a different change.
        """
        spectra = _spectra(pnr_windows)
        automatic = _fit(spectra).table
        by_hand = _fit(
            spectra, bounds={"ts": {"min": 1e-4}}, fit={"method": "powell"}
        ).table

        for column in ("fc", "llpsp", "ts", "chisqr"):
            assert automatic[column].to_numpy() == pytest.approx(
                by_hand[column].to_numpy(), rel=1e-9
            ), column


class TestTheSettingsBite:
    def test_the_minimiser_comes_from_configuration(self, pnr_windows: Any) -> None:
        """And it is not cosmetic: the default minimiser gives a nonsense answer.

        lmfit's default returns a **negative corner frequency** on one of the
        28 PNR stations where Powell does not. A corner frequency below zero is
        not a poor measurement, it is a meaningless one — nothing downstream
        rejects it, and ``pass_fitting`` does not catch it.
        """
        spectra = _spectra(pnr_windows)
        assert load_config().config.fitting.method == "powell"

        configured = _fit(spectra).table
        assert (configured["fc"] > 0).all()

        # The path the caller used to fall into by omitting `method`.
        default = _fit(spectra, fit={"method": "leastsq"}).table
        assert (default["fc"] <= 0).any(), (
            "the unbounded minimiser no longer produces a negative corner; "
            "if that is a real improvement this test should say so instead"
        )

    def test_t_star_is_floored_at_the_configured_minimum(
        self, pnr_windows: Any
    ) -> None:
        """A negative ``t*`` says the wave gained energy travelling."""
        floor = load_config().config.fitting.t_star_min
        table = _fit(_spectra(pnr_windows)).table
        assert (table["ts"] >= floor - 1e-18).all()

    def test_fit_bins_comes_from_configuration(self, pnr_windows: Any) -> None:
        """Fitting the binned spectrum uses far fewer points than the raw one."""
        spectra = _spectra(pnr_windows)
        raw = _fit(spectra, fit_bins=False)
        binned = _fit(spectra, fit_bins=True)

        id = next(iter(raw.models))
        assert binned.models[id].mod_freq.size < raw.models[id].mod_freq.size
        # The shipped default is the unbinned spectrum.
        assert load_config().config.fitting.fit_bins is False
        assert _fit(spectra).models[id].mod_freq.size == raw.models[id].mod_freq.size


class TestInitialGuess:
    def test_it_asks_the_model_which_parameters_it_takes(
        self, pnr_windows: Any
    ) -> None:
        """One function, not one per model.

        ``model_guess`` had ``create_simple_guess`` and
        ``create_simple_guess_fdep``, differing only in whether they added an
        ``a``. A third model meant a third function, and choosing the wrong one
        handed lmfit a parameter the model does not take.
        """
        from specmod import sources  # noqa: PLC0415

        spectra = _spectra(pnr_windows)
        constant_q = initial_guess(spectra)
        frequency_dependent = initial_guess(
            spectra,
            sources.build_model(frequency_dependent_attenuation=True),
        )

        id = next(iter(constant_q))
        assert set(constant_q[id]) == {"llpsp", "fc", "ts"}
        assert set(frequency_dependent[id]) == {"llpsp", "fc", "ts", "a"}

    def test_it_reads_the_plateau_and_corner_off_the_band(
        self, pnr_windows: Any
    ) -> None:
        spectra = _spectra(pnr_windows)
        guesses = initial_guess(spectra)
        for id, guess in guesses.items():
            band = spectra[id].band
            assert band is not None
            assert band[0] <= guess["fc"] <= band[1], id
            assert np.isfinite(guess["llpsp"]), id

    def test_the_scalar_seeds_come_from_configuration(self, pnr_windows: Any) -> None:
        fitting = load_config().config.fitting
        guesses = initial_guess(_spectra(pnr_windows))
        for guess in guesses.values():
            assert guess["ts"] == fitting.initial_t_star

    def test_stations_without_a_band_are_omitted_not_given_none(self) -> None:
        """The old version emitted ``{"llpsp": None, ...}`` on ``IndexError``.

        lmfit cannot use ``None``, so that did not handle the failure — it
        moved it to the fit call, where the message names a parameter rather
        than the station that had no bandwidth.
        """

        class Rejected:
            passes = False
            band = None

        assert initial_guess({"XX.TEST": Rejected()}) == {}
