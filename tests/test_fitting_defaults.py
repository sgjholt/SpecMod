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
        """And it changes the answer, so a study file naming one must be read.

        An earlier version of this asserted that lmfit's default returned a
        **negative** corner frequency where Powell did not — which was true,
        and is no longer, because ``fc`` now carries a lower bound. The bound
        is what prevents the nonsense; the minimiser is what decides which
        minimum is found.

        That distinction is worth keeping straight. The two still disagree by
        more than 1% on **5 of 28** stations and reach the same median
        goodness-of-fit, which is the signature of a shallow surface with
        several local minima rather than of one method being wrong.
        """
        spectra = _spectra(pnr_windows)
        assert load_config().config.fitting.method == "powell"

        configured = _fit(spectra).table
        alternative = _fit(spectra, fit={"method": "leastsq"}).table

        differing = np.abs(
            configured["fc"].to_numpy() / alternative["fc"].to_numpy() - 1.0
        )
        assert (differing > 0.01).sum() >= 1, (
            "the two minimisers now agree everywhere; if that is real, this "
            "test should assert the agreement instead"
        )

    def test_t_star_is_floored_at_the_configured_minimum(
        self, pnr_windows: Any
    ) -> None:
        """A negative ``t*`` says the wave gained energy travelling."""
        floor = load_config().config.fitting.t_star_min
        table = _fit(_spectra(pnr_windows)).table
        assert (table["ts"] >= floor - 1e-18).all()

    @pytest.mark.parametrize("estimator", ["fft", "multitaper", "cwt"])
    def test_the_corner_frequency_cannot_come_back_negative(
        self, estimator: str, pnr_windows: Any
    ) -> None:
        """Across every estimator, not just the one that was checked first.

        Choosing the minimiser from configuration fixed a negative corner on
        the ``fft`` path, and an earlier version of this suite asserted exactly
        that — on ``fft`` alone. The shipped default is ``multitaper``, and
        there the same station still came back at **-4.45 Hz** with
        ``pass_fitting`` reporting success, because a parameter with no bound
        cannot be *at* its bound.

        ``fc`` now carries the lower bound the legacy code had commented out.
        Parametrised over the estimators so the next one added is covered by
        construction rather than by someone remembering.
        """
        signal, noise = pnr_windows()
        spectra = spectrum_set_from_streams(signal, noise, estimator=estimator)
        table = _fit(spectra).table
        assert (table["fc"] > 0).all(), (
            f"{estimator}: {table.loc[table['fc'] <= 0, 'fc'].tolist()}"
        )

    def test_a_parameter_resting_on_its_bound_fails_the_fit(self) -> None:
        """Why the floor is zero rather than a small positive number.

        lmfit flags a parameter sitting on its bound, and ``pass_fitting``
        reads that. So a fit that wants a corner frequency of nothing is
        *rejected* rather than reported — which is the outcome wanted, and is
        only available if the bound is reachable.
        """
        assert load_config().config.fitting.corner_frequency_min == 0.0

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


class TestPassFitting:
    """The flag that says whether a fitted parameter is worth reading.

    A parameter pinned against one of its bounds is the minimiser saying
    "further, if you would let me", and the value reported is the bound rather
    than a measurement. Three separate defects meant the column recording that
    said nothing at all.
    """

    def test_the_table_agrees_with_the_object(self, pnr_windows: Any) -> None:
        """The judgement used to be made *after* it was recorded.

        ``fit_mod`` called ``__set_results_to_meta()`` — which copies
        ``pass_fitting`` into the metadata — and only then
        ``__determine_pass_or_fail()``. So every flat file held the value from
        before the fit ran: ``True``, the class default. The attribute and the
        table disagreed, and the table is the thing written out and regressed
        on.
        """
        fit = _fit(_spectra(pnr_windows))
        for row in fit.table.itertuples():
            assert fit.models[row.id].pass_fitting == row.pass_fitting, row.id

    def test_a_refit_can_pass_again(self, pnr_windows: Any) -> None:
        """``pass_fitting`` was only ever set *False*.

        It starts as a class attribute and nothing reset it, so a
        `FitSpectrum` that failed once could never pass again however many
        times it was refitted — including after the caller relaxed the bound
        that failed it.
        """
        fit = _fit(_spectra(pnr_windows))
        model = next(iter(fit.models.values()))

        model.pass_fitting = False
        model.fit_mod(method="powell")
        assert model.pass_fitting is True

    def test_a_missing_uncertainty_is_not_a_failure(self, pnr_windows: Any) -> None:
        """Powell does not estimate a covariance matrix, so lmfit reports no
        uncertainties — and the shipped configuration uses Powell.

        The old check treated a missing ``stderr`` as a failed fit, so once the
        ordering above was corrected *every* fit under the default
        configuration would have been marked failed. That is a property of the
        minimiser, not a fault in the fit.
        """
        fit = _fit(_spectra(pnr_windows))
        assert fit.table["fc-stderr"].isna().all(), (
            "Powell now reports uncertainties; this test should assert them "
            "instead of asserting their absence is tolerated"
        )
        assert fit.table["pass_fitting"].any()

    def test_it_still_rejects_a_parameter_against_its_bound(
        self, pnr_windows: Any
    ) -> None:
        """The flag has to reject something, or it is decoration.

        ``leastsq`` does report uncertainties, and on these 28 windows six
        stations have a ``value +/- stderr`` that reaches a bound. Those are
        the poorly-constrained fits the flag exists to mark.
        """
        table = _fit(_spectra(pnr_windows), fit={"method": "leastsq"}).table
        assert table["fc-stderr"].notna().all()
        assert 0 < (~table["pass_fitting"]).sum() < len(table)


class TestWhatTypingTheModuleFound:
    """Three defects that annotating ``fitting.py`` against the lmfit stubs
    made visible. All three were reachable; none had a test."""

    def test_an_unfitted_spectrum_says_so_rather_than_raising_attributeerror(
        self, pnr_windows: Any
    ) -> None:
        """``self.result`` is ``None`` until ``fit_mod`` runs.

        Every private reader went straight through it, so ``quick_vis`` on a
        built-but-unfitted model raised ``AttributeError: 'NoneType' object has
        no attribute 'best_fit'`` — from a line naming neither the station nor
        the step that was skipped.
        """
        from specmod.fitting import FitSpectrum  # noqa: PLC0415

        spectra = _spectra(pnr_windows)
        id = spectra.ids()[0]
        unfitted = FitSpectrum(spectra[id].for_fitting(id), llpsp=-7.0, fc=4.0, ts=0.02)

        assert unfitted.result is None
        with pytest.raises(RuntimeError, match="has not been fitted"):
            unfitted.quick_vis()

    def test_the_plot_title_survives_a_minimiser_without_uncertainties(
        self, pnr_windows: Any
    ) -> None:
        """It used to read ``NaN`` under the shipped configuration.

        ``__param_string`` computed ``2 * k.stderr`` unconditionally inside a
        bare ``except Exception``. Powell reports no ``stderr``, so this raised
        ``TypeError`` on *every* fit made with the default settings, swallowed
        it, and titled the plot ``NaN``. A missing uncertainty is a property of
        the method; the value is still worth printing.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fit = _fit(_spectra(pnr_windows))
        model = next(iter(fit.models.values()))
        assert all(p.stderr is None for p in model.result.params.values())

        title = model.quick_vis().get_title()
        plt.close("all")
        assert title != "NaN"
        for name in ("llpsp", "fc", "ts"):
            assert name in title
        # No uncertainty to show, so none is claimed.
        assert "+/-" not in title

    def test_the_title_shows_uncertainties_when_there_are_some(
        self, pnr_windows: Any
    ) -> None:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fit = _fit(_spectra(pnr_windows), fit={"method": "leastsq"})
        title = next(iter(fit.models.values())).quick_vis().get_title()
        plt.close("all")
        assert "+/-" in title

    def test_reset_finds_a_station_named_in_lower_case(self, pnr_windows: Any) -> None:
        """The membership test upper-cased the name and the lookup did not.

        Any id not already upper-case passed the ``in`` check and raised
        ``KeyError`` on the next line. Station ids are upper-case in practice,
        which is the only reason it never fired.
        """
        fit = _fit(_spectra(pnr_windows))
        id = next(iter(fit.models))
        model = fit.models[id]
        model.set_bounds("fc", min=1.0, max=2.0)

        fit.reset(id.lower())
        assert model.params["fc"].min == -np.inf
        assert model.params["fc"].max == np.inf
