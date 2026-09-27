"""Tests for the layered configuration (docs/REFACTOR_PLAN.md §4.7)."""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path
from typing import get_args, get_type_hints
from unittest import mock

import pytest

import specmod
from specmod import config
from specmod.config import (
    Config,
    SmoothingConfig,
    SnrConfig,
    config_hash,
    load_config,
)
from specmod.config.provenance import Provenance
from specmod.config.serialize import to_toml
from specmod.core.bandwidth import FixedBandwidth
from specmod.exceptions import InvalidInputError
from specmod.pipeline import _compare_settings
from specmod.smoothing import SMOOTHERS

STUDIES = Path(__file__).resolve().parents[1] / "studies"


@pytest.fixture
def isolated(tmp_path: Path) -> Path:
    """A directory with no config files, so defaults are what resolve."""
    return tmp_path


# ------------------------------------------------------------------ defaults


def test_defaults_preserve_pre_refactor_behaviour(isolated: Path) -> None:
    """Upgrading must not silently change anyone's numbers.

    These are the shipped values, deliberately *not* the published Magna ones
    (§5.2.5); those live in a study config.
    """
    cfg = load_config(isolated, use_local=False, use_env=False).config
    assert cfg.windows.s_velocity == 2.9
    assert cfg.windows.noise_shift == 0.2
    assert cfg.snr.assert_bandwidths is False
    assert cfg.snr.rotate_noise is True
    assert cfg.snr.tolerance == 3.0
    # mtspec weighted adaptively by default, so this matches it. It shipped
    # off for a while during the refactor, when our own adaptive routine was
    # collapsing for off-centre transients; that was a units bug in Eq. 5.1b's
    # regularisation term, now fixed and validated against Prieto's package.
    assert cfg.transform.adaptive is True
    assert cfg.model.source == "brune"
    assert cfg.model.motion == "velocity"


def test_plot_columns_has_exactly_one_home() -> None:
    """PLOT_COLUMNS used to exist in both SPECTRAL and FITTING."""
    data = Config().to_dict()
    holders = [s for s, v in data.items() if "plot_columns" in v]
    assert holders == ["viz"]


def test_time_bandwidth_is_reachable() -> None:
    """It was the literal 3 passed positionally to mtspec, unconfigurable."""
    assert Config().transform.time_bandwidth == 3.0


# ------------------------------------------------------------------- layering


def test_layers_apply_in_precedence_order(isolated: Path, monkeypatch) -> None:
    (isolated / "specmod.toml").write_text("[snr]\ntolerance = 4.0\nmin_points = 7\n")
    (isolated / "specmod.local.toml").write_text("[snr]\ntolerance = 5.0\n")
    monkeypatch.setenv("SPECMOD_SNR__MIN_POINTS", "9")

    resolved = load_config(isolated)
    assert resolved.config.snr.tolerance == 5.0  # local beats project
    assert resolved.config.snr.min_points == 9  # env beats both
    assert resolved.config.snr.bands == SnrConfig().bands  # untouched -> default


def test_arguments_win_over_every_file(isolated: Path) -> None:
    (isolated / "specmod.toml").write_text("[snr]\ntolerance = 4.0\n")
    resolved = load_config(isolated, snr={"tolerance": 99.0})
    assert resolved.config.snr.tolerance == 99.0
    assert resolved.source_of("snr.tolerance") == "arguments"


def test_layers_can_be_disabled_for_reproducibility(
    isolated: Path, monkeypatch
) -> None:
    """Tests must not inherit a developer's machine."""
    (isolated / "specmod.local.toml").write_text("[snr]\ntolerance = 5.0\n")
    monkeypatch.setenv("SPECMOD_SNR__TOLERANCE", "6.0")
    cfg = load_config(isolated, use_local=False, use_env=False).config
    assert cfg.snr.tolerance == 3.0


def test_sources_record_the_originating_layer(isolated: Path) -> None:
    (isolated / "specmod.toml").write_text("[windows]\ns_velocity = 3.4\n")
    (isolated / "specmod.local.toml").write_text("[snr]\ntolerance = 5.0\n")
    resolved = load_config(isolated, use_env=False)
    assert resolved.source_of("windows.s_velocity") == "specmod.toml"
    assert resolved.source_of("snr.tolerance") == "specmod.local.toml"
    assert resolved.source_of("snr.min_points") == "default"
    assert "<- specmod.local.toml" in resolved.explain()


def test_config_search_does_not_walk_upwards(tmp_path: Path) -> None:
    """An implicit parent search makes it unclear which file a run used."""
    (tmp_path / "specmod.toml").write_text("[snr]\ntolerance = 42.0\n")
    child = tmp_path / "nested"
    child.mkdir()
    assert load_config(child, use_env=False).config.snr.tolerance == 3.0


# ------------------------------------------------------------------ validation


def test_unknown_section_is_rejected(isolated: Path) -> None:
    (isolated / "specmod.toml").write_text("[spectral]\ntolerance = 4.0\n")
    with pytest.raises(ValueError, match="Unknown configuration section"):
        load_config(isolated, use_env=False)


def test_unknown_key_is_rejected(isolated: Path) -> None:
    """A silently ignored typo is a reproducibility bug: it looks configured."""
    (isolated / "specmod.toml").write_text("[snr]\ntolerence = 4.0\n")
    with pytest.raises(ValueError, match=r"Unknown key\(s\) in \[snr\]"):
        load_config(isolated, use_env=False)


def test_sections_are_immutable() -> None:
    cfg = Config()
    with pytest.raises(AttributeError):
        cfg.snr.tolerance = 9.0  # type: ignore[misc]


# ----------------------------------------------------------------- provenance


def test_config_hash_is_stable_and_sensitive() -> None:
    a = Config()
    b = load_config(use_local=False, use_env=False, snr={"tolerance": 4.0}).config
    assert config_hash(a) == config_hash(Config())
    assert config_hash(a) != config_hash(b)


def test_provenance_records_version_and_config() -> None:
    prov = Provenance.capture(Config())
    assert prov.specmod_version == specmod.__version__
    assert prov.config_hash == config_hash(Config())
    assert "windows" in prov.config
    assert "created_at" in prov.to_dict()


# ------------------------------------------------------------------- freezing


def test_freeze_round_trips(isolated: Path) -> None:
    """A frozen file must reproduce the config it was frozen from."""
    original = load_config(
        isolated,
        use_local=False,
        use_env=False,
        snr={"tolerance": 4.5, "assert_bandwidths": True},
        windows={"s_velocity": 3.4, "refine_percentiles": [2.0, 98.0]},
    ).config

    path = isolated / "frozen.toml"
    path.write_text(to_toml(original))
    restored = Config.from_dict(tomllib.loads(path.read_text()))

    assert restored == original
    assert config_hash(restored) == config_hash(original)


def test_freeze_emits_valid_toml_for_defaults() -> None:
    text = to_toml(Config(), header="test")
    parsed = tomllib.loads(text)
    assert Config.from_dict(parsed) == Config()


# -------------------------------------------------------------- study configs


def test_magna_study_config_matches_the_published_workflow() -> None:
    """studies/magna_2020_paper.toml pins the values in the SRL manuscript."""
    cfg = load_config(
        project_file=STUDIES / "magna_2020_paper.toml",
        use_local=False,
        use_env=False,
    ).config
    assert cfg.windows.s_velocity == 3.4  # Pechmann et al. 2007
    assert cfg.windows.p_velocity == 5.9
    assert cfg.windows.s_start_ratio == 0.8  # window opens at 80% of Pg-Sg
    assert cfg.windows.s_length == 20.0
    assert cfg.windows.refine_percentiles == (1.0, 99.0)
    assert cfg.windows.noise_shift == 0.5  # noise ends 0.5 s before Pg
    assert cfg.snr.tolerance == 3.0
    assert cfg.snr.assert_bandwidths is True  # the paper's selection criterion
    assert cfg.snr.bands == ((2.0, 4.0), (4.0, 6.0), (6.0, 8.0))
    assert cfg.model.source == "brune"
    assert cfg.model.motion == "velocity"
    assert cfg.fitting.method == "powell"
    assert cfg.acquire.max_radius_km == 400.0
    # Pinned in the file, not inherited: the package default is now False.
    assert cfg.transform.adaptive is True


def test_magna_study_config_differs_from_defaults() -> None:
    """If these ever coincide, the §5.2.5 warning has quietly become wrong."""
    study = load_config(
        project_file=STUDIES / "magna_2020_paper.toml", use_local=False, use_env=False
    ).config
    assert config_hash(study) != config_hash(Config())
    assert study.windows.s_velocity != Config().windows.s_velocity
    assert study.snr.assert_bandwidths != Config().snr.assert_bandwidths


# ---------------------------------------- the pipeline reads the real config


def test_the_comparison_settings_come_from_the_typed_config() -> None:
    """``pipeline`` resolves every setting from :class:`Config`, per call.

    This replaces the pair of tests that pinned ``spectral``'s module-level
    globals — ``BINNING_PARAMS``, ``ROT_METHOD`` and nine more, bound once at
    import from a hand-maintained dict of literals. That module is deleted and
    the dict with it; the settings now reach ``SpectrumPair.compare`` as
    arguments, which is what makes a study's TOML layer able to change them and
    what lets two configurations coexist in one session.

    Asserted rather than trusted because the mapping is one-way: nothing else
    would notice if ``n_bins`` stopped tracking ``smoothing.n_bins``.
    """
    from specmod.pipeline import _compare_settings  # noqa: PLC0415

    resolved = load_config().config
    settings = _compare_settings()

    assert settings["threshold"] == resolved.snr.tolerance
    assert settings["f_min"] == resolved.smoothing.f_min
    assert settings["f_max"] == resolved.smoothing.f_max
    assert settings["n_bins"] == resolved.smoothing.n_bins
    assert settings["scale_parseval"] is resolved.snr.scale_parseval
    assert settings["rotate_noise"] is resolved.snr.rotate_noise
    assert settings["noise_model"] == resolved.snr.rotation_method
    assert settings["resolution_floor"] is resolved.snr.resolution_floor
    assert settings["bandwidth"] == resolved.snr.bandwidth_method
    assert settings["rotation_space"] == resolved.snr.rotation_space


def test_an_override_reaches_the_comparison() -> None:
    """The point of the whole change. A module-level global could not do this."""
    from specmod.pipeline import _compare_settings  # noqa: PLC0415

    assert _compare_settings({"n_bins": 41})["n_bins"] == 41
    assert _compare_settings()["n_bins"] == load_config().config.smoothing.n_bins


class TestEverySmoothingMethodIsWiredUp:
    """``[smoothing] method`` has to select something, and nothing checked it.

    It selected nothing at all until this was written: the registry existed,
    the key validated against its ``Literal``, and no code read it. All three
    values then accepted produced bit-identical output on the 28 PNR windows,
    ``none`` included — which reads as "do not smooth my spectra" and left
    every one of them binned. Same shape as the three defects in
    ``docs/REFACTOR_PLAN.md`` §6.6: a setting, a claim about it, and nothing
    joining the two.

    So the joining is what is asserted here, per method rather than in general:
    the name resolves, the section's parameters reach the object, and adding a
    value to the ``Literal`` without wiring it fails rather than going quiet.
    """

    #: Every value `[smoothing] method` accepts, read off the annotation so a
    #: new one cannot be added without appearing here.
    METHODS = sorted(get_args(get_type_hints(SmoothingConfig)["method"]))

    def test_the_literal_and_the_registry_agree(self) -> None:
        """`log_bins` is the one name that does not come from the registry --
        it is `core.collection.log_bin`, which is not `LogBinner`."""
        assert set(self.METHODS) - {"log_bins"} <= set(SMOOTHERS)

    @pytest.mark.parametrize("method", METHODS)
    def test_each_method_resolves(
        self, isolated: Path, monkeypatch: pytest.MonkeyPatch, method: str
    ) -> None:
        from specmod.pipeline import _compare_settings  # noqa: PLC0415

        (isolated / "specmod.toml").write_text(f'[smoothing]\nmethod = "{method}"\n')
        monkeypatch.chdir(isolated)

        smoother = _compare_settings()["smoother"]
        if method == "log_bins":
            assert smoother is None
        else:
            assert isinstance(smoother, SMOOTHERS[method])

    def test_the_shipped_default_smooths_rather_than_bins(self) -> None:
        """What a user gets with no configuration at all.

        This used to assert the opposite, on the premise that the default was
        "the path every committed number came from". That premise is gone:
        the committed numbers now come from `tests/golden/reference.toml`,
        pinned, so the default is free to be the one that measures better
        rather than the one the references happen to hold.
        """
        from specmod.pipeline import _compare_settings  # noqa: PLC0415

        assert isinstance(_compare_settings()["smoother"], SMOOTHERS["konno_ohmachi"])

    def test_the_pinned_reference_config_still_bins(self) -> None:
        """And the references keep the path they were captured under."""
        from specmod import config as cfg  # noqa: PLC0415
        from specmod.pipeline import _compare_settings  # noqa: PLC0415

        reference = Path(__file__).parent / "golden" / "reference.toml"
        with cfg.using(reference):
            settings = _compare_settings()
            assert settings["smoother"] is None
            assert settings["n_bins"] == cfg.load_config().config.smoothing.n_bins

    def test_the_section_parameters_reach_the_smoother(
        self, isolated: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The half that a registry lookup alone would not give: a bandwidth
        or a window set in the file has to arrive on the object."""
        from specmod.pipeline import _compare_settings  # noqa: PLC0415

        (isolated / "specmod.toml").write_text(
            '[smoothing]\nmethod = "log_window"\n'
            'window = "bartlett"\noctave_fraction = 0.5\n'
        )
        monkeypatch.chdir(isolated)

        smoother = _compare_settings()["smoother"]
        assert smoother.window == "bartlett"
        assert smoother.octave_fraction == 0.5

    def test_konno_ohmachi_takes_its_bandwidth_from_the_section(
        self, isolated: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from specmod.pipeline import _compare_settings  # noqa: PLC0415

        (isolated / "specmod.toml").write_text(
            '[smoothing]\nmethod = "konno_ohmachi"\nkonno_ohmachi_bandwidth = 20.0\n'
        )
        monkeypatch.chdir(isolated)

        assert _compare_settings()["smoother"].bandwidth == 20.0

    def test_a_registered_method_with_no_parameters_mapped_is_refused(self) -> None:
        """The failure mode this replaces: a name that resolves and then
        silently runs with defaults the configuration meant to set."""
        from specmod.pipeline import _configured_smoother  # noqa: PLC0415

        config = load_config(use_local=False, use_env=False).config
        rogue = replace(config, smoothing=replace(config.smoothing, method="invented"))

        with pytest.raises(ValueError, match="no parameters mapped"):
            _configured_smoother(rogue)


def test_the_configured_names_resolve_in_their_registries() -> None:
    """``BW_METHOD`` and ``ROT_METHOD`` were integers naming a branch.

    They are names now, and each has to resolve in the registry that owns it —
    the thing the integers could not do, and why ``ROT_METHOD = 1`` sat
    commented out and unrunnable for years.
    """
    from specmod.core.bandwidth import BANDWIDTH_SELECTORS  # noqa: PLC0415
    from specmod.core.noise import NOISE_MODELS  # noqa: PLC0415

    resolved = load_config().config
    assert resolved.snr.bandwidth_method in BANDWIDTH_SELECTORS
    assert resolved.snr.rotation_method in NOISE_MODELS


def test_the_legacy_config_module_is_gone() -> None:
    assert not (Path(specmod.__file__).parent / "_config_legacy.py").exists()
    assert not hasattr(config, "SPECTRAL")
    assert not hasattr(config, "FITTING")


# ------------------------------------------------------- bandwidth overrides


def test_the_shipped_defaults_impose_no_band(isolated: Path) -> None:
    """Both overrides are opt-in. The right ceiling is a property of the
    instrument, not of this package, so nothing is imposed by default."""
    cfg = load_config(isolated, use_local=False, use_env=False).config
    assert cfg.snr.max_nyquist_fraction is None
    assert cfg.snr.fixed_band is None
    assert cfg.snr.bandwidth_method == "peak"


def test_a_configured_cap_reaches_the_comparison(isolated: Path) -> None:
    (isolated / "specmod.toml").write_text("[snr]\nmax_nyquist_fraction = 0.8\n")
    with mock.patch("specmod.pipeline.load_config", return_value=load_config(isolated)):
        assert _compare_settings()["max_nyquist_fraction"] == 0.8


def test_a_configured_fixed_band_becomes_a_selector(isolated: Path) -> None:
    (isolated / "specmod.toml").write_text(
        '[snr]\nbandwidth_method = "fixed"\nfixed_band = [1.5, 25.0]\n'
    )
    with mock.patch("specmod.pipeline.load_config", return_value=load_config(isolated)):
        selector = _compare_settings()["bandwidth"]
    assert selector == FixedBandwidth(1.5, 25.0)


def test_fixed_without_a_band_says_which_setting_is_missing(isolated: Path) -> None:
    """The failure has to name `fixed_band`. Raised where configuration is
    read, so it happens once per run rather than once per pair."""
    (isolated / "specmod.toml").write_text('[snr]\nbandwidth_method = "fixed"\n')
    with (
        mock.patch("specmod.pipeline.load_config", return_value=load_config(isolated)),
        pytest.raises(InvalidInputError, match="fixed_band"),
    ):
        _compare_settings()
