"""The promises :mod:`specmod.api` makes to downstream packages.

The surface exists so that SpecMod's internals can keep moving while
`specmod-studio`, `specmod-model` and `specmod-git` import one thing that does
not. That is only true if the promises are checked, so each one here is a test
rather than a sentence in a docstring:

- the export list is frozen, so adding or removing one shows up in review;
- every export is documented and annotated;
- the same input twice gives the same output;
- nothing touches the filesystem;
- :func:`~specmod.api.available_estimators` answers for the environment it is
  actually in, including under ``--without-optional-extras``.
"""

from __future__ import annotations

import builtins
import inspect
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from specmod import api
from specmod.exceptions import InvalidInputError, MissingBackendError, SpecModError

from .test_end_to_end import WINDOW_S, _trace

#: Every name the surface promises. Adding to this list is a compatibility
#: obligation under the policy in CONTRIBUTING.md; removing one needs a
#: deprecation cycle first. The literal is duplicated here on purpose —
#: comparing `api.__all__` against itself would assert nothing.
EXPECTED_EXPORTS = (
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
)


@pytest.fixture(scope="module")
def pair() -> Any:
    """A real signal/noise pair, through the public surface only."""
    import specmod.preprocess as pre  # noqa: PLC0415

    import obspy  # noqa: PLC0415, isort: skip

    stream = obspy.Stream([_trace("S00", seed=11)])
    signal = pre.get_signal(
        stream, pre.cut_s, rafp=0.0, tafs=WINDOW_S, time_after="absolute_time"
    )
    noise = pre.get_noise_p(stream, signal)
    sig, noi = signal[0], noise[0]
    return api.compare_spectra(
        api.estimate_spectrum(sig.data, float(sig.stats.delta), estimator="multitaper"),
        api.estimate_spectrum(noi.data, float(noi.stats.delta), estimator="multitaper"),
    )


class TestTheExportList:
    def test_it_is_exactly_what_was_agreed(self) -> None:
        assert tuple(sorted(api.__all__)) == EXPECTED_EXPORTS

    @pytest.mark.parametrize("name", EXPECTED_EXPORTS)
    def test_every_export_exists(self, name: str) -> None:
        assert hasattr(api, name), f"{name} is promised and missing"

    @pytest.mark.parametrize("name", EXPECTED_EXPORTS)
    def test_every_export_is_documented(self, name: str) -> None:
        obj = getattr(api, name)
        if name == "__version__":
            pytest.skip("a string, not a documentable object")
        assert obj.__doc__, f"{name} has no docstring"

    @pytest.mark.parametrize("name", EXPECTED_EXPORTS)
    def test_every_exported_function_is_annotated(self, name: str) -> None:
        obj = getattr(api, name)
        if not inspect.isfunction(obj):
            pytest.skip("not a function")
        signature = inspect.signature(obj)
        assert signature.return_annotation is not inspect.Signature.empty, name
        for parameter in signature.parameters.values():
            assert parameter.annotation is not inspect.Signature.empty, (
                f"{name}({parameter.name}) is not annotated"
            )


class TestCapabilities:
    def test_it_reports_what_actually_runs(self) -> None:
        """The point of the function: no name it returns may fail to run.

        This is the invariant under both installs. With the extras present
        `prieto` is in the list and works; under `--without-optional-extras`
        it is absent, and the ones that remain still work.
        """
        data = np.random.default_rng(0).normal(size=512)
        for name in api.available_estimators():
            spectrum = api.estimate_spectrum(data, 0.01, estimator=name)
            assert spectrum.freq.size > 0, name

    def test_the_backends_needing_nothing_are_always_there(self) -> None:
        assert {"fft", "welch", "multitaper"} <= set(api.available_estimators())

    def test_it_is_sorted_and_hashable(self) -> None:
        names = api.available_estimators()
        assert isinstance(names, tuple)
        assert list(names) == sorted(names)

    def test_an_unavailable_backend_says_so_in_the_type(self) -> None:
        """`prieto` needs an extra. Absent, it must raise the typed error and
        not a bare ImportError, so a caller can tell 'install something' from
        'your input is wrong'."""
        if "prieto" in api.available_estimators():
            pytest.skip("the multitaper extra is installed here")
        with pytest.raises(MissingBackendError):
            api.estimate_spectrum(np.zeros(512) + 1.0, 0.01, estimator="prieto")


class TestDeterminism:
    def test_estimation_repeats_exactly(self) -> None:
        data = np.random.default_rng(7).normal(size=1024)
        first = api.estimate_spectrum(data, 0.01, estimator="multitaper")
        second = api.estimate_spectrum(data, 0.01, estimator="multitaper")
        assert np.array_equal(first.amp, second.amp)
        assert np.array_equal(first.freq, second.freq)

    def test_comparison_repeats_exactly(self, pair: Any) -> None:
        again = api.compare_spectra(pair.signal, pair.noise)
        assert np.array_equal(again.snr, pair.snr)
        assert again.band == pair.band

    def test_fitting_repeats_exactly(self, pair: Any) -> None:
        first = api.fit_spectrum(pair, id="XX.S00..HHN")
        second = api.fit_spectrum(pair, id="XX.S00..HHN")
        assert first.params == second.params
        assert first.chisqr == second.chisqr

    def test_the_config_hash_is_stable(self) -> None:
        config = api.load_config(use_local=False, use_env=False).config
        assert api.config_hash(config) == api.config_hash(config)


class TestItDoesNotTouchTheFilesystem:
    """Studio owns its IO, so that projects can live on S3, Azure or GCS.

    A core function that opens a path itself defeats that, and the failure is
    silent on a workstation — it only shows up in a deployment where the path
    does not exist. So the check is mechanical: make opening a file an error
    and call the surface.
    """

    @pytest.fixture
    def no_open(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # An empty directory, so `load_config` finds no config to legitimately
        # read, and warm caches before the ban so a first-use import inside
        # numpy or lmfit is not blamed on the surface.
        monkeypatch.chdir(tmp_path)
        api.estimate_spectrum(np.arange(64.0), 0.01, estimator="fft")

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"opened a file: {args[0]!r}")

        monkeypatch.setattr(builtins, "open", refuse)
        monkeypatch.setattr(io, "open", refuse)

    def test_estimation(self, no_open: None) -> None:
        data = np.random.default_rng(1).normal(size=512)
        for name in api.available_estimators():
            api.estimate_spectrum(data, 0.01, estimator=name)

    def test_comparison_and_fitting(self, no_open: None, pair: Any) -> None:
        again = api.compare_spectra(pair.signal, pair.noise)
        fit = api.fit_spectrum(again, id="XX.S00..HHN")
        assert fit.n_points > 0

    def test_config_and_capabilities(self, no_open: None) -> None:
        config = api.load_config(use_local=False, use_env=False).config
        api.config_hash(config)
        api.config_to_toml(config)
        api.available_estimators()

    def test_tapers(self, no_open: None) -> None:
        assert api.make_window("tukey", 128).size == 128
        assert api.window_correction(api.make_window("tukey", 128), "energy") > 0


class TestItDoesNotMutateItsInputs:
    def test_estimation_leaves_the_record_alone(self) -> None:
        data = np.random.default_rng(3).normal(size=512)
        before = data.copy()
        api.estimate_spectrum(data, 0.01, estimator="multitaper")
        assert np.array_equal(data, before)

    def test_comparison_leaves_the_spectra_alone(self, pair: Any) -> None:
        signal_amp = pair.signal.amp.copy()
        noise_amp = pair.noise.amp.copy()
        api.compare_spectra(pair.signal, pair.noise)
        assert np.array_equal(pair.signal.amp, signal_amp)
        assert np.array_equal(pair.noise.amp, noise_amp)


class TestTheFitCarriesItsUncertainty:
    """A point estimate without a bound is not a measurement, and the f_c-t*
    correlation is the one downstream consumers are told to display.

    Which is available depends on the minimiser, and the configured default is
    `powell`, which produces no covariance. Both halves are pinned here: asking
    for it gets it, and not asking gets an honest absence rather than a
    fabricated zero.
    """

    def test_the_fit_finds_the_parameters(self, pair: Any) -> None:
        fit = api.fit_spectrum(pair, id="XX.S00..HHN")
        assert fit.success
        assert set(fit.params) >= {"llpsp", "fc", "ts"}
        assert fit.n_points > 0

    def test_least_squares_reports_errors_and_a_covariance(self, pair: Any) -> None:
        fit = api.fit_spectrum(pair, id="XX.S00..HHN", method="leastsq")
        assert set(fit.stderr) >= {"llpsp", "fc", "ts"}
        assert fit.covariance is not None
        assert fit.covariance.shape == (len(fit.names), len(fit.names))

    def test_the_correlation_is_reachable(self, pair: Any) -> None:
        fit = api.fit_spectrum(pair, id="XX.S00..HHN", method="leastsq")
        correlation = fit.correlation("fc", "ts")
        assert correlation is not None
        assert -1.0 <= correlation <= 1.0

    def test_a_minimiser_without_a_covariance_says_so(self, pair: Any) -> None:
        """`powell` gives no covariance. The result must show that, not
        invent one — a zero error is a claim, and the wrong one."""
        fit = api.fit_spectrum(pair, id="XX.S00..HHN", method="powell")
        assert fit.stderr == {}
        assert fit.covariance is None
        assert fit.correlation("fc", "ts") is None

    def test_an_unmeasured_correlation_is_none_not_zero(self, pair: Any) -> None:
        """Zero would read as 'independent' rather than 'not measured'."""
        fit = api.fit_spectrum(pair, id="XX.S00..HHN", method="leastsq")
        assert fit.correlation("fc", "not_a_parameter") is None

    def test_the_result_is_frozen(self, pair: Any) -> None:
        fit = api.fit_spectrum(pair, id="XX.S00..HHN")
        with pytest.raises((AttributeError, TypeError)):
            fit.chisqr = 0.0  # type: ignore[misc]


class TestTheSnrIsPerBin:
    """§3.5 of the Studio design: store the curve, derive the intervals. A
    scalar band cannot be un-collapsed later."""

    def test_the_curve_is_aligned_with_the_frequency_axis(self, pair: Any) -> None:
        assert pair.snr.shape == pair.binned_signal.freq.shape
        assert pair.snr.size > 1

    def test_the_noise_spectrum_comes_back_too(self, pair: Any) -> None:
        assert pair.binned_noise.amp.shape == pair.binned_signal.amp.shape

    def test_a_band_is_derivable_at_any_threshold(self, pair: Any) -> None:
        """What a consumer does with the curve, and cannot do with a band."""
        for threshold in (2.0, 3.0, 5.0):
            admitted = pair.snr >= threshold
            assert admitted.dtype == bool


class TestErrorsAreTyped:
    @pytest.mark.parametrize(
        ("data", "reason"),
        [
            (np.array([np.nan, 1.0, 2.0]), "non-finite"),
            (np.array([1.0]), "too short"),
            (np.zeros((4, 4)), "not 1-D"),
        ],
    )
    def test_a_bad_record_is_invalid_input(self, data: np.ndarray, reason: str) -> None:
        with pytest.raises(InvalidInputError):
            api.estimate_spectrum(data, 0.01, estimator="fft")

    def test_an_unknown_estimator_is_invalid_input(self) -> None:
        with pytest.raises(InvalidInputError):
            api.estimate_spectrum(np.arange(64.0), 0.01, estimator="nope")

    def test_an_unknown_weight_method_is_invalid_input(self, pair: Any) -> None:
        with pytest.raises(InvalidInputError):
            api.fit_spectrum(pair, weight_method="sideways")

    def test_every_typed_error_is_a_specmod_error(self) -> None:
        """One `except` clause has to be able to catch all of them."""
        assert issubclass(InvalidInputError, SpecModError)
        assert issubclass(MissingBackendError, SpecModError)
        assert issubclass(api.InternalError, SpecModError)

    def test_they_still_look_like_the_builtins_they_replace(self) -> None:
        """Existing callers catch ValueError and ImportError. They keep working
        rather than being broken by the introduction of the hierarchy."""
        assert issubclass(InvalidInputError, ValueError)
        assert issubclass(MissingBackendError, ImportError)
