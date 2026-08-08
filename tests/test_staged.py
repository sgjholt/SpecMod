"""The two-stage event fit, and the selection that decides who votes.

The science this encodes is in :mod:`specmod.staged`. What these tests hold is
that the default path is the published workflow, that every departure from it
is deliberate and recorded, and that the aggregate cannot be computed from
nothing.
"""

from __future__ import annotations

import contextlib
import functools
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

from specmod.config import load_config  # noqa: E402
from specmod.distance import (  # noqa: E402
    Epicentral,
    Hypocentral,
    get_distance_measure,
    resolve_distance_measure,
)
from specmod.fitting import FitSpectra  # noqa: E402
from specmod.pipeline import spectrum_set_from_streams  # noqa: E402
from specmod.staged import (  # noqa: E402
    WEIGHT_MODELS,
    ChannelSelection,
    InverseDistance,
    Uniform,
    fit_event,
    get_weight_model,
)


@functools.cache
def _spectra(windows: Any) -> Any:
    signal, noise = windows()
    with contextlib.redirect_stdout(io.StringIO()):
        return spectrum_set_from_streams(signal, noise)


@functools.cache
def _event(windows: Any) -> Any:
    return fit_event(_spectra(windows))


class TestTheDefaultPath:
    def test_it_reproduces_the_weighted_mean_computed_by_hand(
        self, pnr_windows: Any
    ) -> None:
        """The whole point of the module is that nobody writes this again.

        So it has to give what writing it by hand gives — the inverse
        hypocentral distance weighted mean of the stage-1 corner frequencies.
        """
        spectra = _spectra(pnr_windows)
        staged = _event(pnr_windows)

        with contextlib.redirect_stdout(io.StringIO()):
            stage1 = FitSpectra(spectra)
            stage1.fit_spectra()
        table = stage1.table.set_index("id")
        # `repi`, because that is what `[geometry] distance_measure` says and
        # the weighting now reads it. It used to hardcode `rhyp`.
        weights = np.array(
            [1.0 / float(spectra[id].signal.meta["repi"]) for id in table.index]
        )
        by_hand = float((table["fc"].to_numpy() * weights).sum() / weights.sum())

        assert staged.value == pytest.approx(by_hand, rel=1e-9)

    def test_stage_two_holds_the_event_value_fixed_everywhere(
        self, pnr_windows: Any
    ) -> None:
        """Fixed, not merely seeded. A parameter that is still free has not
        been constrained by the ensemble, it has only been given a hint."""
        staged = _event(pnr_windows)
        assert staged.stage2 is not None
        for id, model in staged.stage2.models.items():
            assert model.params["fc"].vary is False, id
            assert model.params["fc"].value == pytest.approx(staged.value), id

    def test_the_two_stages_are_both_kept(self, pnr_windows: Any) -> None:
        """The stage-1 spread is the evidence for how well constrained the
        event value is. Returning only stage two would leave a number with no
        error on it."""
        staged = _event(pnr_windows)
        assert len(staged.stage1.models) == 28
        assert staged.stage2 is not None
        assert len(staged.stage2.models) == 28
        assert staged.table is staged.stage2.table

    def test_it_reads_the_configured_parameter_and_weighting(self) -> None:
        fitting = load_config().config.fitting
        assert fitting.event_parameter == "fc"
        assert fitting.event_weighting == "inverse_distance"
        assert fitting.include == ()
        assert fitting.exclude == ()
        assert fitting.require_pass is True


class TestSelection:
    """Patterns match at whichever level they are written."""

    @pytest.mark.parametrize(
        ("pattern", "expected"),
        [
            ("AQ04", 2),  # a station, by its bare code
            ("UR.AQ04", 2),  # the same station, spelled out
            ("UR.AQ04.00.HHE", 1),  # one channel
            ("HHE", 14),  # one component, everywhere
            ("UR", 16),  # a whole network (LV has the other 12)
            ("HH?", 28),  # a glob over the channel code
            ("AQ0[45]", 4),  # a glob over the station code
        ],
    )
    def test_a_pattern_excludes_what_it_names(
        self, pattern: str, expected: int, pnr_windows: Any
    ) -> None:
        spectra = _spectra(pnr_windows)
        staged = fit_event(spectra, selection=ChannelSelection(exclude=(pattern,)))
        dropped = [
            id for id, why in staged.excluded.items() if "matched exclude" in why
        ]
        assert len(dropped) == expected, sorted(dropped)

    def test_include_keeps_only_what_it_names(self, pnr_windows: Any) -> None:
        staged = fit_event(
            _spectra(pnr_windows), selection=ChannelSelection(include=("HHE",))
        )
        assert len(staged.contributing) == 14
        assert all(id.endswith("HHE") for id in staged.contributing)

    def test_an_exclusion_says_why_and_at_which_level(self, pnr_windows: Any) -> None:
        """ "Excluded" is not actionable; "matched exclude='AQ04' at station" is.

        It also disambiguates the case the docstring warns about — a pattern
        that could match at two levels reports the one it hit.
        """
        staged = fit_event(
            _spectra(pnr_windows), selection=ChannelSelection(exclude=("AQ04",))
        )
        reasons = {id: why for id, why in staged.excluded.items() if "matched" in why}
        assert reasons == {
            "UR.AQ04.00.HHE": "matched exclude='AQ04' at station",
            "UR.AQ04.00.HHN": "matched exclude='AQ04' at station",
        }

    def test_selection_changes_the_event_value_it_is_meant_to_change(
        self, pnr_windows: Any
    ) -> None:
        """Otherwise the whole feature is decoration.

        The nearest station carries the most weight by construction, so
        dropping it is the single largest lever selection has — and this is
        why the option has to exist rather than being an average over whatever
        happened to be recorded.
        """
        spectra = _spectra(pnr_windows)
        everything = fit_event(spectra)
        without = fit_event(spectra, selection=ChannelSelection(exclude=("AQ04",)))

        assert len(without.contributing) == len(everything.contributing) - 2
        moved = abs(without.value / everything.value - 1)
        assert moved > 0.1, (
            f"dropping the nearest station moved the event value by only "
            f"{100 * moved:.1f}%; if the weighting has genuinely become that "
            "insensitive this test should say so instead"
        )

    def test_require_pass_is_nearly_inert_under_the_shipped_minimiser(
        self, pnr_windows: Any
    ) -> None:
        """The trap, pinned so it cannot drift into being a surprise.

        `pass_fitting` asks whether ``value +/- stderr`` reaches a bound.
        Powell estimates no covariance matrix, so the spread is zero and the
        test degenerates to "is the value exactly on the bound" — which
        essentially never fires. `leastsq` reports uncertainties and fails six.

        The consequence is that changing the minimiser changes *which stations
        vote*, not just how each is fitted, and a naive comparison of two
        minimisers compares two ensembles. That is worth an assertion rather
        than a docstring, because the failure is a plausible-looking number.
        """
        spectra = _spectra(pnr_windows)
        assert len(fit_event(spectra, method="powell").contributing) == 28
        assert len(fit_event(spectra, method="leastsq").contributing) == 22

    def test_holding_the_ensemble_fixed_is_what_makes_minimisers_comparable(
        self, pnr_windows: Any
    ) -> None:
        """The module docstring's 0.6% is only true like-for-like.

        Under the default selection the same comparison gives 144%, and the
        difference is entirely the six stations `leastsq` drops and Powell
        keeps — not a property of either minimiser.
        """
        spectra = _spectra(pnr_windows)
        same = ChannelSelection(require_pass=False)
        powell = fit_event(spectra, method="powell", selection=same)
        leastsq = fit_event(spectra, method="leastsq", selection=same)

        assert len(powell.contributing) == len(leastsq.contributing) == 28
        assert abs(powell.value / leastsq.value - 1) < 0.01

        default = abs(
            fit_event(spectra, method="powell").value
            / fit_event(spectra, method="leastsq").value
            - 1
        )
        assert default > 1.0, (
            "the two ensembles no longer disagree wildly; if selection has "
            "changed so that this is no longer a trap, say so here instead"
        )

    def test_require_pass_drops_a_fit_resting_on_its_bound(
        self, pnr_windows: Any
    ) -> None:
        """A pinned parameter reports the bound, not a measurement, so
        averaging it in averages in a constant.

        ``leastsq`` reports uncertainties and so fails several stations on the
        bound check, which is what makes it the minimiser to test this with.
        """
        spectra = _spectra(pnr_windows)
        strict = fit_event(spectra, method="leastsq")
        loose = fit_event(
            spectra, method="leastsq", selection=ChannelSelection(require_pass=False)
        )

        assert len(strict.contributing) < len(loose.contributing)
        assert any("stage-1 fit failed" in why for why in strict.excluded.values())

    def test_a_station_with_no_band_is_excluded_with_that_reason(self) -> None:
        """Not silently absent. A station that never got a fit and a station
        that was deselected are different outcomes, and the caller needs to be
        able to tell them apart without re-deriving either."""
        import pandas as pd  # noqa: PLC0415

        @dataclass
        class Fit:
            """Just enough of `FitSpectra` for selection to read."""

            spectra: dict[str, object]
            models: dict[str, object]
            table: Any

        fit = Fit(spectra={"XX.TEST..HHZ": object()}, models={}, table=pd.DataFrame([]))
        contributing, excluded = ChannelSelection().choose(fit)  # type: ignore[arg-type]
        assert contributing == []
        assert "no fit" in excluded["XX.TEST..HHZ"]


class TestWhenNobodyVotes:
    def test_stage_two_is_skipped_rather_than_invented(self, pnr_windows: Any) -> None:
        """Fixing the corner to a mean of no stations would be inventing the
        number the second stage exists to constrain."""
        staged = fit_event(
            _spectra(pnr_windows), selection=ChannelSelection(include=("NOTHING",))
        )
        assert staged.stage2 is None
        assert np.isnan(staged.value)
        assert staged.contributing == ()
        assert len(staged.excluded) == 28
        # Stage one is still there, so the caller has something to look at.
        assert len(staged.stage1.models) == 28
        assert staged.table is staged.stage1.table

    def test_describe_still_says_something_useful(self, pnr_windows: Any) -> None:
        staged = fit_event(
            _spectra(pnr_windows), selection=ChannelSelection(include=("NOTHING",))
        )
        text = staged.describe()
        assert "from 0 channels" in text
        assert "28 excluded" in text


class TestWeighting:
    def test_the_registry_resolves_and_rejects_by_name(self) -> None:
        assert isinstance(get_weight_model("uniform"), Uniform)
        assert isinstance(get_weight_model("inverse_distance"), InverseDistance)
        assert isinstance(
            get_weight_model("inverse_hypocentral_distance"), InverseDistance
        )
        with pytest.raises(ValueError, match="Unknown weighting"):
            get_weight_model("nope")

    def test_every_registered_weighting_is_usable(self, pnr_windows: Any) -> None:
        """A registry entry that cannot be selected is decoration.

        ``inverse_variance`` is exercised separately: it needs uncertainties,
        which the configured minimiser does not produce.
        """
        spectra = _spectra(pnr_windows)
        for name in WEIGHT_MODELS:
            if name == "inverse_variance":
                continue
            assert fit_event(spectra, weighting=name).value > 0, name

    def test_the_choice_of_weighting_moves_the_answer(self, pnr_windows: Any) -> None:
        """Which is why it is a choice and not a constant."""
        spectra = _spectra(pnr_windows)
        by_distance = fit_event(spectra, weighting="inverse_hypocentral_distance").value
        uniform = fit_event(spectra, weighting="uniform").value
        assert abs(by_distance / uniform - 1) > 0.05

    def test_inverse_variance_refuses_rather_than_falling_back(
        self, pnr_windows: Any
    ) -> None:
        """Powell estimates no covariance matrix, so there is nothing to weight
        by. Silently becoming uniform is the kind of substitution that ends up
        in a paper."""
        with pytest.raises(ValueError, match="uncertainty on every"):
            fit_event(_spectra(pnr_windows), weighting="inverse_variance")

    def test_inverse_variance_works_where_the_uncertainties_exist(
        self, pnr_windows: Any
    ) -> None:
        value = fit_event(
            _spectra(pnr_windows), weighting="inverse_variance", method="leastsq"
        ).value
        assert value > 0

    def test_distance_weighting_names_the_missing_geometry(self) -> None:
        """Rather than weighting it zero, which would quietly drop a station."""

        @dataclass
        class Signal:
            meta: dict[str, Any]

        @dataclass
        class Pair:
            signal: Signal

        spectra = {"XX.A..HHZ": Pair(signal=Signal(meta={}))}
        with pytest.raises(ValueError, match="set_stream_distance"):
            InverseDistance().weights(None, spectra, ["XX.A..HHZ"])

    def test_the_configured_distance_measure_is_what_gets_used(
        self, pnr_windows: Any
    ) -> None:
        """`[geometry] distance_measure` had no reader before this.

        It matters at short range. Hypocentral and epicentral converge far from
        the source and diverge near it — on these windows the nearest station
        is 0.89 km epicentral against 2.30 km hypocentral — and since this
        weighting is by *inverse* distance the disagreement lands hardest on
        the station carrying the most weight.
        """
        spectra = _spectra(pnr_windows)
        assert load_config().config.geometry.distance_measure == "repi"

        configured = fit_event(spectra).value
        epicentral = fit_event(spectra, weighting="inverse_epicentral_distance").value
        hypocentral = fit_event(spectra, weighting="inverse_hypocentral_distance").value

        assert configured == pytest.approx(epicentral, rel=1e-12)
        assert abs(hypocentral / epicentral - 1) > 0.05, (
            "the two distance measures now agree; if the geometry has changed "
            "so that this no longer matters, say so here instead"
        )


class TestDistanceMeasures:
    def test_the_registry_resolves_and_rejects(self) -> None:
        assert isinstance(get_distance_measure("repi"), Epicentral)
        assert isinstance(get_distance_measure("rhyp"), Hypocentral)
        with pytest.raises(ValueError, match="Unknown distance measure"):
            get_distance_measure("nope")

    @pytest.mark.parametrize("name", ["rrup", "rjb"])
    def test_finite_fault_measures_refuse_rather_than_degenerate(
        self, name: str
    ) -> None:
        """For a point source these reduce exactly to hypocentral and
        epicentral, so a silent fallback would give plausible numbers that are
        wrong for any event large enough to justify asking for them."""
        with pytest.raises(NotImplementedError, match="rupture surface"):
            get_distance_measure(name).distances(None, ["XX.A..HHZ"])

    def test_resolve_takes_a_name_an_instance_or_the_configuration(self) -> None:
        assert resolve_distance_measure("rhyp").name == "hypocentral"
        assert resolve_distance_measure(Epicentral()).name == "epicentral"
        assert resolve_distance_measure().name == "epicentral"  # configured


class TestReporting:
    def test_spread_reports_what_the_mean_hides(self, pnr_windows: Any) -> None:
        """A 2% spread and a 300% spread give the same weighted mean and mean
        very different things."""
        spread = _event(pnr_windows).spread()
        assert spread["n"] == 28
        assert spread["min"] < spread["median"] < spread["max"]
        assert spread["relative_spread"] > 1.0

    def test_describe_names_the_weighting_and_the_count(self, pnr_windows: Any) -> None:
        text = _event(pnr_windows).describe()
        assert "from 28 channels" in text
        assert "inverse_distance" in text
        assert "stage-1 range" in text
