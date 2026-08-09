"""Moment, magnitude, and the spreading models that feed them."""

from __future__ import annotations

import math

import numpy as np
import pytest

from specmod.magnitude import (
    MediumConstants,
    event_magnitude,
    moment_magnitude,
    seismic_moment,
    station_moments,
)
from specmod.spreading import (
    HOLT_2019_UTAH,
    Piecewise,
    PowerLaw,
    Tabulated,
    get_spreading_model,
)


class TestSpreading:
    def test_power_law_is_the_ratio_to_the_reference_distance(self) -> None:
        model = PowerLaw(exponent=1.0, reference_km=1.0)
        assert model(1.0) == pytest.approx(1.0)
        assert model(10.0) == pytest.approx(0.1)
        assert model(100.0) == pytest.approx(0.01)

    def test_the_exponent_is_on_amplitude_not_energy(self) -> None:
        """``1/R**2`` is energy; a moment expression corrects an amplitude."""
        assert PowerLaw(exponent=2.0)(10.0) == pytest.approx(
            PowerLaw(exponent=1.0)(10.0) ** 2
        )

    def test_piecewise_is_continuous_across_its_hinges(self) -> None:
        """The product accumulates, so no segment boundary is a step.

        This is the property that distinguishes a contiguous piecewise model
        from segments anchored independently at the source — the latter would
        jump at every hinge and silently drop the decay accumulated before it.
        """
        for edge in (43.0, 76.0, 136.0):
            below = float(HOLT_2019_UTAH(edge - 1e-6)[()])
            above = float(HOLT_2019_UTAH(edge + 1e-6)[()])
            assert below == pytest.approx(above, rel=1e-5)

    def test_piecewise_matches_a_power_law_inside_the_first_segment(self) -> None:
        """Below the first hinge the Utah model is its first exponent, alone."""
        single = PowerLaw(exponent=0.90)
        for r in (2.0, 10.0, 22.9, 42.0):
            assert HOLT_2019_UTAH(r) == pytest.approx(single(r), rel=1e-12)

    def test_piecewise_beyond_the_table_continues_the_last_exponent(self) -> None:
        far = float(HOLT_2019_UTAH(800.0)[()])
        edge = float(HOLT_2019_UTAH(400.0)[()])
        assert far == pytest.approx(edge * (400.0 / 800.0) ** 1.54, rel=1e-12)

    def test_piecewise_rejects_unordered_segments(self) -> None:
        with pytest.raises(ValueError, match="increase with distance"):
            Piecewise(segments=((1.0, 50.0), (1.0, 20.0)))

    def test_tabulated_recovers_its_own_points(self) -> None:
        table = Tabulated(distances_km=(1.0, 10.0, 100.0), values=(1.0, 0.1, 0.001))
        for r, expected in zip((1.0, 10.0, 100.0), (1.0, 0.1, 0.001), strict=True):
            assert table(r) == pytest.approx(expected)

    def test_tabulated_interpolates_in_log_log(self) -> None:
        table = Tabulated(distances_km=(1.0, 100.0), values=(1.0, 0.01))
        # Halfway in log-distance is halfway in log-amplitude: a straight line.
        assert table(10.0) == pytest.approx(0.1)

    def test_tabulated_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="one of each"):
            Tabulated(distances_km=(1.0, 10.0), values=(1.0,))

    def test_zero_distance_is_refused(self) -> None:
        with pytest.raises(ValueError, match="undefined at zero"):
            PowerLaw()(0.0)

    def test_the_registry_builds_by_name(self) -> None:
        model = get_spreading_model("power_law", exponent=0.5)
        assert model(4.0) == pytest.approx(0.5)
        with pytest.raises(ValueError, match="Unknown spreading model"):
            get_spreading_model("inverse_vibes")


class TestMoment:
    def test_the_moment_expression_is_what_it_claims(self) -> None:
        """Hand-computed against ``4 pi rho beta^3 R Omega / (Theta F)``.

        For the default ``1/R`` the reference distance cancels, leaving the
        short form with ``R`` in **metres** — which is the identity worth
        pinning, because it is where a kilometre would hide.
        """
        c = MediumConstants(
            density=2700.0,
            velocity=3500.0,
            radiation_pattern=0.55,
            free_surface=2.0,
            reference_distance_m=1000.0,
        )
        omega, r_km = 1e-6, 10.0
        expected = (
            4.0 * math.pi * 2700.0 * 3500.0**3 * (r_km * 1000.0) * omega / (0.55 * 2.0)
        )
        assert seismic_moment(omega, r_km, constants=c) == pytest.approx(expected)

    def test_moment_magnitude_round_trips(self) -> None:
        """Mw 6 is 1.259e18 N m under Hanks and Kanamori."""
        assert moment_magnitude(1.2589e18) == pytest.approx(6.0, abs=1e-4)
        assert moment_magnitude(np.array([1e13, 1e16])) == pytest.approx(
            [(2 / 3) * (13 - 9.1), (2 / 3) * (16 - 9.1)]
        )

    def test_a_velocity_in_km_per_second_is_refused(self) -> None:
        """The factor-of-1e9 trap this whole module exists to make impossible."""
        with pytest.raises(ValueError, match="looks like km/s"):
            MediumConstants(velocity=3.5)

    def test_a_logarithmic_plateau_is_refused(self) -> None:
        """``llpsp`` is log10(Omega); passing it raw is negative, so it raises."""
        with pytest.raises(ValueError, match="not logarithmic"):
            seismic_moment(-5.0, 10.0)

    def test_plateau_and_distance_must_correspond(self) -> None:
        with pytest.raises(ValueError, match="own distance"):
            seismic_moment(np.array([1e-6, 2e-6]), np.array([10.0]))


PLAN_CONSTANTS = MediumConstants(
    density=2500.0, velocity=2500.0, radiation_pattern=0.63, free_surface=2.0
)


@pytest.mark.usefixtures("pnr_windows")
class TestOnTheRealEvent:
    """Measured on the 28 PNR channels through the two-stage fit.

    These pin a computation, not a physical truth. The absolute calibration is
    unsettled, and how far off it is depends on an open question: the
    catalogue value carried through this repository is 1.6, labelled "Mw" but
    nowhere sourced, while BGS reports this sequence in ML. Under the thesis's
    own ML-Mw relation an ML 1.6 predicts Mw 2.10, not 1.6. So the value of
    these assertions is that a change to the constants, the units or the
    spreading moves the number visibly — not that the number is right. See the
    module docstring of ``specmod.magnitude``.
    """

    @staticmethod
    def _staged(windows):  # type: ignore[no-untyped-def]
        from specmod.pipeline import spectrum_set_from_streams  # noqa: PLC0415
        from specmod.staged import fit_event  # noqa: PLC0415

        signal, noise = windows()
        return fit_event(spectrum_set_from_streams(signal, noise))

    def test_the_event_magnitude_is_unchanged(self, pnr_windows) -> None:  # type: ignore[no-untyped-def]
        event = event_magnitude(self._staged(pnr_windows))
        assert event.value == pytest.approx(3.097, abs=0.01)
        assert event.unit == "Mw"
        assert len(event.stations) == 28
        # M0 and Mw must describe the same event.
        assert moment_magnitude(event.m0) == pytest.approx(event.value, abs=1e-9)

    def test_it_reproduces_the_plan_s_measured_value(self, pnr_windows) -> None:  # type: ignore[no-untyped-def]
        """§4.7 recorded Mw +2.75 for ``1/r`` with these constants."""
        event = event_magnitude(
            self._staged(pnr_windows),
            constants=PLAN_CONSTANTS,
            spreading=PowerLaw(exponent=1.0),
        )
        assert event.value == pytest.approx(2.75, abs=0.01)

    def test_squaring_the_exponent_costs_one_reference_distance(
        self,
        pnr_windows,  # type: ignore[no-untyped-def]
    ) -> None:
        """``1/R**2`` is wrong, but by 0.68 magnitude units rather than 3.8.

        §4.7 originally recorded Mw 5.39 here. That figure came from applying
        ``r**n`` with ``r`` in metres, which is dimensionally inconsistent for
        any ``n != 1`` — it multiplies in an extra reference distance. The
        difference is exactly ``(2/3) * log10(1000) = 2.0`` magnitude units,
        which is what this pins.

        The conclusion that the exponent is 1 is unaffected; its evidence is
        now theory plus the thesis's own inverted 0.88-0.90, rather than a
        dramatic mismatch that was partly an arithmetic artefact.
        """
        staged = self._staged(pnr_windows)
        two = event_magnitude(
            staged, constants=PLAN_CONSTANTS, spreading=PowerLaw(exponent=2.0)
        )
        assert two.value == pytest.approx(3.42, abs=0.01)

        # Exact by construction, with rejection off so both average the same
        # stations — a wider exponent widens the spread, which moves which
        # stations survive 2.5 sigma and would otherwise compare two sets.
        raw = {
            n: event_magnitude(
                staged,
                constants=PLAN_CONSTANTS,
                spreading=PowerLaw(exponent=n),
                outlier_sigma=0.0,
            ).value
            for n in (1.0, 2.0)
        }
        distances = station_moments(staged.table).loc[lambda d: d["mw"].notna(), "rhyp"]
        expected = (2 / 3) * float(np.mean(np.log10(distances.to_numpy() / 1.0)))
        assert raw[2.0] - raw[1.0] == pytest.approx(expected, abs=1e-9)

        # And the gap to the figure §4.7 recorded is one reference distance.
        # Loose, because that figure was a median where this is a mean.
        assert 5.39 - two.value == pytest.approx((2 / 3) * math.log10(1000.0), abs=0.05)

    def test_the_station_spread_is_reported_beside_the_value(self, pnr_windows) -> None:  # type: ignore[no-untyped-def]
        event = event_magnitude(self._staged(pnr_windows))
        spread = event.spread()
        assert spread["n"] == 28
        assert spread["std"] < 0.3
        assert spread["min"] < event.value < spread["max"]

    def test_the_regional_model_is_not_the_default(self, pnr_windows) -> None:  # type: ignore[no-untyped-def]
        """Utah's table is fitted for 1-400 km; PNR sits at 2-23 km.

        It should give a *different* answer here, and the difference is the
        reason a regional model is something an operator supplies rather than
        something that ships switched on.
        """
        staged = self._staged(pnr_windows)
        default = event_magnitude(staged, constants=PLAN_CONSTANTS)
        utah = event_magnitude(
            staged, constants=PLAN_CONSTANTS, spreading=HOLT_2019_UTAH
        )
        assert utah.spreading_model == "holt_2019_utah"
        assert abs(utah.value - default.value) > 0.02

    def test_station_moments_leaves_the_input_alone(self, pnr_windows) -> None:  # type: ignore[no-untyped-def]
        staged = self._staged(pnr_windows)
        before = list(staged.table.columns)
        table = station_moments(staged.table)
        assert "mw" in table.columns
        assert "m0" in table.columns
        assert list(staged.table.columns) == before

    def test_too_few_stations_refuses_rather_than_reporting(self, pnr_windows) -> None:  # type: ignore[no-untyped-def]
        staged = self._staged(pnr_windows)
        with pytest.raises(ValueError, match="min_stations"):
            event_magnitude(staged, min_stations=99)

    def test_a_missing_column_names_itself(self, pnr_windows) -> None:  # type: ignore[no-untyped-def]
        staged = self._staged(pnr_windows)
        with pytest.raises(ValueError, match="rjb"):
            station_moments(staged.table, distance_measure="rjb")
