"""Holding a configuration across the ambient reads the pipeline makes.

`REFACTOR_PLAN` §6.6 records the claim this closes:

    "Regression tests pin an explicit config file, never the defaults, so
    changing a default cannot silently move a golden test."

The mechanism was absent. `load_config()` takes a `project_file`, but the
pipeline does not call it that way — `spectrum_set_from_streams` and the
functions under it read settings ambiently, from call sites that accept no
configuration argument. So a test could pin a config for itself and the code
under test would still read the shipped defaults, and bumping
`smoothing.n_bins` from 151 to 158 moved nine golden tests.
"""

from __future__ import annotations

import concurrent.futures
import textwrap
from pathlib import Path

import pytest

from specmod import config as cfg


@pytest.fixture
def pinned_file(tmp_path: Path) -> Path:
    path = tmp_path / "pinned.toml"
    path.write_text(
        textwrap.dedent("""
        [transform]
        estimator = "welch"

        [smoothing]
        n_bins = 77
        """)
    )
    return path


class TestThePinIsSeenByAnAmbientRead:
    def test_a_read_with_no_arguments_returns_the_pinned_config(
        self, pinned_file: Path
    ) -> None:
        """The property the whole thing exists for."""
        with cfg.using(pinned_file):
            assert cfg.load_config().config.transform.estimator == "welch"
            assert cfg.load_config().config.smoothing.n_bins == 77

    def test_the_default_is_back_after_the_block(self, pinned_file: Path) -> None:
        before = cfg.load_config().config.transform.estimator
        with cfg.using(pinned_file):
            pass
        assert cfg.load_config().config.transform.estimator == before

    def test_it_is_restored_even_when_the_block_raises(self, pinned_file: Path) -> None:
        before = cfg.load_config().config.transform.estimator
        with pytest.raises(RuntimeError), cfg.using(pinned_file):
            raise RuntimeError("boom")
        assert cfg.load_config().config.transform.estimator == before

    def test_pins_nest(self, pinned_file: Path, tmp_path: Path) -> None:
        inner = tmp_path / "inner.toml"
        inner.write_text('[transform]\nestimator = "cwt"\n')
        with cfg.using(pinned_file):
            with cfg.using(inner):
                assert cfg.load_config().config.transform.estimator == "cwt"
            assert cfg.load_config().config.transform.estimator == "welch"


class TestWhatThePinDoesNotCapture:
    def test_an_explicit_file_still_wins(
        self, pinned_file: Path, tmp_path: Path
    ) -> None:
        """A caller asking for a particular file gets it, pin or no pin.

        The pin replaces the ambient answer. Anything else would make
        `load_config(project_file=...)` lie inside a pinned block, which is
        where a study config is read.
        """
        other = tmp_path / "other.toml"
        other.write_text('[transform]\nestimator = "prieto"\n')
        with cfg.using(pinned_file):
            resolved = cfg.load_config(project_file=other)
        assert resolved.config.transform.estimator == "prieto"

    def test_explicit_overrides_still_win(self, pinned_file: Path) -> None:
        with cfg.using(pinned_file):
            resolved = cfg.load_config(transform={"estimator": "quadratic"})
        assert resolved.config.transform.estimator == "quadratic"

    def test_the_environment_cannot_reach_inside_a_pin(
        self, pinned_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pin a developer's shell could move would not be a pin."""
        monkeypatch.setenv("SPECMOD_TRANSFORM__ESTIMATOR", "cwt")
        with cfg.using(pinned_file):
            assert cfg.load_config().config.transform.estimator == "welch"

    def test_a_pin_is_not_visible_from_another_thread(self, pinned_file: Path) -> None:
        """`ContextVar`, not a module global.

        A pin that leaked across threads would make a parallel test run depend
        on which test happened to be inside a block.
        """

        def read() -> str:
            return str(cfg.load_config().config.transform.estimator)

        with (
            cfg.using(pinned_file),
            concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
        ):
            elsewhere = pool.submit(read).result()
        assert elsewhere != "welch"


class TestTheBlockReportsWhatItPinned:
    def test_it_yields_the_resolved_configuration(self, pinned_file: Path) -> None:
        with cfg.using(pinned_file) as resolved:
            assert resolved.config.transform.estimator == "welch"
            assert resolved is cfg.load_config()

    def test_pinned_says_whether_one_is_held(self, pinned_file: Path) -> None:
        assert cfg.pinned() is None
        with cfg.using(pinned_file) as resolved:
            assert cfg.pinned() is resolved
        assert cfg.pinned() is None

    def test_a_pin_with_no_file_is_the_defaults_plus_overrides(self) -> None:
        """Pinning the defaults is a real request: it excludes the local file
        and the environment, which an ambient read would include."""
        with cfg.using(transform={"estimator": "welch"}) as resolved:
            assert resolved.config.transform.estimator == "welch"
            assert cfg.load_config().config.transform.estimator == "welch"


def test_a_function_that_takes_no_configuration_sees_the_pin(tmp_path: Path) -> None:
    """The end the claim is about, and the shape that made it false.

    `plot_columns()` takes no arguments and reads its setting ambiently. There
    is no parameter to thread a configuration through, which is true of most
    of the settings the pipeline consults; before the pin, such a reader could
    only ever see the shipped defaults.
    """
    from specmod.fitting.base import plot_columns  # noqa: PLC0415

    default = plot_columns()
    path = tmp_path / "viz.toml"
    path.write_text(f"[viz]\nplot_columns = {default + 4}\n")

    with cfg.using(path):
        assert plot_columns() == default + 4
    assert plot_columns() == default
