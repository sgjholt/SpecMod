"""Source, attenuation and motion models, and the join to configuration.

The join is the point. ``ModelConfig.source`` was a ``Literal["brune",
"boatwright"]`` that nothing read — ``FitSpectra.set_model`` took the model
*function* as an argument, so the caller passed Brune or Boatwright by hand and
the configured value was decorative. These tests pin that it is now connected,
and that the pieces compose without a global.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

import specmod.models as legacy
from specmod.fitting import FitSpectrum
from specmod.sources import (
    ATTENUATION_MODELS,
    SOURCE_MODELS,
    AttenuationModel,
    SourceModel,
    build_model,
    from_config,
    get_attenuation_model,
    get_source_model,
    motion_scaling,
)
from specmod.spectral import Spectra

FREQ = np.logspace(-1.0, 2.0, 60)
OMEGA, FC, TSTAR = -7.0, 4.0, 0.02


# ------------------------------------------------------- against the legacy


def test_the_composite_reproduces_the_legacy_model_exactly() -> None:
    """A restructuring, not a change of physics — so the numbers must not move."""
    mine = build_model(source="brune", motion="velocity").evaluate(
        FREQ, OMEGA, FC, TSTAR
    )
    assert mine == pytest.approx(legacy.simple_model(FREQ, OMEGA, FC, TSTAR), rel=1e-12)


def test_the_frequency_dependent_variant_reproduces_the_legacy_too() -> None:
    model = build_model(
        source="brune", motion="velocity", frequency_dependent_attenuation=True
    )
    assert model.parameters == ("llpsp", "fc", "ts", "a")
    mine = model.evaluate(FREQ, OMEGA, FC, TSTAR, 0.3)
    expected = legacy.simple_model_fdep(FREQ, OMEGA, FC, TSTAR, 0.3)
    assert mine == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------------ source shapes


def test_the_plateau_is_omega_and_the_corner_is_three_db_down() -> None:
    """Two properties that pin the shape without reproducing its algebra.

    At ``f -> 0`` a source model is its plateau. At ``f = fc`` a Brune spectrum
    is down by ``log10(2) = 0.301``, which is the definition of the corner.
    """
    brune = get_source_model("brune")
    assert brune.log10_shape(np.array([1e-6]), OMEGA, FC)[0] == pytest.approx(
        OMEGA, abs=1e-9
    )
    assert brune.log10_shape(np.array([FC]), OMEGA, FC)[0] == pytest.approx(
        OMEGA - np.log10(2.0)
    )


def test_both_sources_fall_off_as_omega_squared() -> None:
    """Brune and Boatwright differ at the knee, not in the tail.

    Well above the corner both go as ``f**-2``, so the log-log slope is -2.
    That is what makes the corner-frequency coefficient — not the shape — the
    thing that separates models like Madariaga from Brune.
    """
    high = np.logspace(2.0, 3.0, 40)
    for name in SOURCE_MODELS:
        shape = get_source_model(name).log10_shape(high, OMEGA, FC)
        slope = np.polyfit(np.log10(high), shape, 1)[0]
        assert slope == pytest.approx(-2.0, abs=0.02), f"{name} slope {slope}"


def test_boatwright_has_a_sharper_knee_than_brune() -> None:
    """The whole difference between them, and it lives near the corner."""
    near = np.logspace(np.log10(FC) - 1, np.log10(FC) + 1, 200)
    brune = get_source_model("brune").log10_shape(near, OMEGA, FC)
    boat = get_source_model("boatwright").log10_shape(near, OMEGA, FC)

    difference = np.abs(boat - brune)
    assert difference.max() > 0.1, "the two shapes are indistinguishable"
    # And the difference is concentrated at the corner, not in the tail.
    assert near[int(np.argmax(difference))] == pytest.approx(FC, rel=0.3)


def test_every_source_carries_its_corner_frequency_coefficient() -> None:
    """The attribute Madariaga will need, asserted so it cannot be dropped.

    A model that shares Brune's shape but not its radius relation changes no
    fitted parameter and every derived stress drop. Requiring the coefficient
    on every registered model is what keeps that difference expressible.
    """
    for name in SOURCE_MODELS:
        model = get_source_model(name)
        assert isinstance(model, SourceModel), name
        k_p, k_s = model.corner_frequency_coefficient
        assert 0.0 < k_s < 1.0, f"{name} k_S = {k_s}"
        assert 0.0 < k_p < 1.0, f"{name} k_P = {k_p}"


# -------------------------------------------------------------- attenuation


def test_constant_q_is_the_zero_exponent_case_of_the_frequency_dependent_one() -> None:
    constant = get_attenuation_model("constant_q").log10_decay(FREQ, TSTAR)
    equivalent = get_attenuation_model("frequency_dependent_q").log10_decay(
        FREQ, TSTAR, 0.0
    )
    assert constant == pytest.approx(equivalent, rel=1e-12)


def test_attenuation_is_a_decay() -> None:
    for name in ATTENUATION_MODELS:
        model = get_attenuation_model(name)
        assert isinstance(model, AttenuationModel), name
        values = (TSTAR,) if model.parameters == ("ts",) else (TSTAR, 0.3)
        decay = model.log10_decay(FREQ, *values)
        assert (decay <= 0).all(), f"{name} amplified the spectrum"
        assert np.all(np.diff(decay) < 0), f"{name} is not monotonic"


# ------------------------------------------------------------------ motion


def test_motion_scaling_is_one_factor_of_two_pi_f_per_order() -> None:
    assert motion_scaling(FREQ, "displacement") == pytest.approx(np.zeros_like(FREQ))
    assert motion_scaling(FREQ, "velocity") == pytest.approx(np.log10(2 * np.pi * FREQ))
    assert motion_scaling(FREQ, "acceleration") == pytest.approx(
        2 * np.log10(2 * np.pi * FREQ)
    )


def test_an_unknown_motion_fails_rather_than_scaling_by_nothing() -> None:
    """Returning zero would fit a displacement model to velocity data."""
    with pytest.raises(ValueError, match="jerk"):
        motion_scaling(FREQ, "jerk")


# ------------------------------------------------------ the registries and join


@pytest.mark.parametrize(
    ("resolve", "bad"),
    [(get_source_model, "madariaga"), (get_attenuation_model, "constant")],
)
def test_unknown_names_list_what_is_available(resolve, bad: str) -> None:
    with pytest.raises(ValueError, match="Available:"):
        resolve(bad)


def test_configuration_now_selects_the_model() -> None:
    """The gap this package closes.

    Before, ``config.model.source`` was read by nothing at all.
    """
    model = from_config()
    assert model.source.name == "brune"
    assert model.describe() == "brune+constant_q in velocity"


def test_two_models_can_exist_at_once() -> None:
    """Impossible before, and the reason it matters.

    The legacy bound the choice at import time — ``MODEL = which_model(...)``
    in ``models.py`` — so a Brune and a Boatwright could not be fitted in the
    same session without reimporting the module. They are values now.
    """
    brune = build_model(source="brune")
    boatwright = build_model(source="boatwright")

    assert brune.source.name == "brune"
    assert boatwright.source.name == "boatwright"
    assert not np.allclose(
        brune.evaluate(FREQ, OMEGA, FC, TSTAR),
        boatwright.evaluate(FREQ, OMEGA, FC, TSTAR),
    )


def test_the_model_reports_its_own_parameters_to_lmfit() -> None:
    """``lmfit`` reads names off the signature, so it has to be a real one.

    The legacy passed a bare function and let lmfit introspect it, which meant
    the model and the parameter list could disagree. Here the model is the
    source of both.
    """
    lm = pytest.importorskip("lmfit")

    model = build_model(frequency_dependent_attenuation=True)
    wrapped = lm.Model(model.as_callable())
    assert wrapped.param_names == list(model.parameters)
    assert wrapped.independent_vars == ["f"]


def test_a_wrong_number_of_parameters_is_refused() -> None:
    model = build_model()
    with pytest.raises(TypeError, match="takes 3 parameters"):
        model.evaluate(FREQ, OMEGA, FC)


def test_the_model_can_actually_be_fitted() -> None:
    """End to end: build from config, fit synthetic data, recover the truth."""
    lm = pytest.importorskip("lmfit")

    model = build_model()
    freq = np.logspace(-1.0, 1.6, 80)
    truth = model.evaluate(freq, OMEGA, FC, TSTAR)
    noisy = truth + np.random.default_rng(0).normal(0.0, 0.02, freq.size)

    wrapped = lm.Model(model.as_callable())
    result = wrapped.fit(
        noisy,
        wrapped.make_params(llpsp=-6.0, fc=2.0, ts=0.05),
        f=freq,
        method="powell",
    )
    assert result.params["llpsp"].value == pytest.approx(OMEGA, abs=0.05)
    assert result.params["fc"].value == pytest.approx(FC, rel=0.1)
    assert result.params["ts"].value == pytest.approx(TSTAR, abs=0.005)


# ------------------------------------------------------------- the fitter


@pytest.fixture(scope="module")
def real_signal(pnr_windows):
    """One passing station from the committed PNR waveforms."""
    signal, noise = pnr_windows()
    with contextlib.redirect_stdout(io.StringIO()):
        spectra = Spectra.from_streams(signal, noise)
    for snp in spectra.group.values():
        if snp.signal.pass_snr:
            return snp.signal
    pytest.skip("no station passed the signal-to-noise gate")
    return None


def test_the_fitter_takes_its_model_from_configuration(real_signal) -> None:
    """The end of the chain the `sources` package exists to complete.

    ``FitSpectrum`` used to *require* the model function as an argument. Now
    omitting it resolves through configuration, so `[model] source = ...` in a
    study file decides what is fitted.
    """
    fit = FitSpectrum(real_signal, llpsp=-7.0, fc=4.0, ts=0.02)
    assert fit.describe_model() == "brune+constant_q in velocity"
    assert fit.mod.param_names == ["llpsp", "fc", "ts"]


def test_the_fitter_reports_what_it_fitted(real_signal) -> None:
    """Provenance the legacy could not carry: it only ever had a raw callable."""
    boatwright = build_model(source="boatwright")
    fit = FitSpectrum(real_signal, boatwright, llpsp=-7.0, fc=4.0, ts=0.02)
    assert fit.spectral_model is boatwright
    assert fit.describe_model() == "boatwright+constant_q in velocity"


def test_a_bare_callable_still_works_but_carries_no_provenance(real_signal) -> None:
    """Fitting an ad-hoc shape stays possible — it just cannot describe itself."""
    fit = FitSpectrum(real_signal, legacy.simple_model, llpsp=-7.0, fc=4.0, ts=0.02)
    assert fit.spectral_model is None
    assert fit.describe_model() is None
    assert fit.mod.param_names == ["llpsp", "fc", "ts"]


def test_the_configured_model_fits_a_real_spectrum(real_signal) -> None:
    """End to end on real data, with the band the pipeline selected."""
    fit = FitSpectrum(real_signal, llpsp=-7.0, fc=4.0, ts=0.02)
    fit.fit_mod(method="powell")

    assert fit.result.success
    # Omega and the corner should land somewhere physical rather than at a bound.
    assert np.isfinite(fit.result.params["llpsp"].value)
    assert 0.0 < fit.result.params["fc"].value < real_signal.freq.max()
    assert fit.result.params["ts"].value > 0.0


def test_changing_the_configured_source_changes_the_fit(real_signal) -> None:
    """The join, demonstrated rather than asserted structurally.

    Two models, same data, different results — which is only possible because
    the choice is a value now instead of an import-time global.
    """
    results = {}
    for name in ("brune", "boatwright"):
        fit = FitSpectrum(
            real_signal, build_model(source=name), llpsp=-7.0, fc=4.0, ts=0.02
        )
        fit.fit_mod(method="powell")
        results[name] = fit.result.params["fc"].value

    assert results["brune"] != results["boatwright"]
