"""Tests for the layered configuration (docs/REFACTOR_PLAN.md §4.7)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from specmod.config import Config, SnrConfig, config_hash, load_config
from specmod.config.provenance import Provenance
from specmod.config.serialize import to_toml

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
    import specmod

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
