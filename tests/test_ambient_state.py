"""Two properties the package has to hold everywhere, not just where fixed.

Both were found by auditing for what a long-lived process would do with this
library — see ``docs/notes/api-audit.md`` — and both are the kind of defect
that reappears one file at a time. So each is checked by walking the package
rather than by testing the one site that had it.

1. **No configuration is read at import time.** ``load_config()`` resolves
   against the current working directory and the environment. At module level
   that answer is frozen for the life of the interpreter, so a worker serving
   two projects uses the first one's settings for both.
2. **Nothing prints.** A service capturing logs per job gets nothing from
   ``print``, and a CLI writing to a pipe gets its output corrupted.
   Diagnostics go through :mod:`warnings` or :mod:`logging`.
"""

from __future__ import annotations

import ast
import logging
import warnings
from pathlib import Path
from typing import Any

import pytest

from specmod import fitting, utils

SOURCE = Path(fitting.__file__).resolve().parent.parent
#: Every module in the installed package.
MODULES = sorted(path for path in SOURCE.rglob("*.py") if "_vendor" not in path.parts)


def _module_level_calls(node: ast.AST) -> list[ast.Call]:
    """Calls evaluated when the module is imported.

    Recursion stops at a function body, which runs when something calls it —
    that is the whole point of moving a config read into one. It does *not*
    stop at a class body, which executes at import like any other statement,
    nor at a function's default arguments and decorators, which are evaluated
    at definition time and are a favourite hiding place for exactly this.

    Written as an explicit walk rather than `ast.walk`, because `ast.walk`
    descends into method bodies and would flag every call-time read in the
    package. It did, on the first run of this test.
    """
    found: list[ast.Call] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            for default in [*child.args.defaults, *child.args.kw_defaults]:
                if default is not None:
                    found.extend(_calls_in(default))
            for decorator in getattr(child, "decorator_list", []):
                found.extend(_calls_in(decorator))
            continue
        if isinstance(child, ast.Call):
            found.append(child)
        found.extend(_module_level_calls(child))
    return found


def _calls_in(node: ast.AST) -> list[ast.Call]:
    """Every call inside an expression that runs at import."""
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _callee(call: ast.Call) -> str:
    """The dotted name being called, as written."""
    node: Any = call.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
class TestTheModuleIsInert:
    def test_it_reads_no_configuration_at_import(self, path: Path) -> None:
        """`fitting/base.py` did this, and froze `[viz] plot_columns` at
        whatever the importing directory said. Measured before the fix:
        importing from a project whose `specmod.toml` said 5, then moving to
        one resolving to 3, left it at 5."""
        tree = ast.parse(path.read_text(), filename=str(path))
        offenders = [
            _callee(call)
            for call in _module_level_calls(tree)
            if _callee(call).endswith("load_config")
        ]
        assert not offenders, (
            f"{path.name} resolves configuration at import time "
            f"({', '.join(offenders)}). Read it inside the function that needs "
            "it, so the value follows the caller rather than the importer."
        )

    def test_it_does_not_print(self, path: Path) -> None:
        tree = ast.parse(path.read_text(), filename=str(path))
        printed = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _callee(node) == "print"
        ]
        assert not printed, (
            f"{path.name} calls print() at line(s) "
            f"{', '.join(map(str, printed))}. Use `warnings.warn` for something "
            "the caller should act on, or the module logger for progress."
        )


class TestPlotColumnsFollowsTheConfiguration:
    def test_it_reads_the_current_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "specmod.toml").write_text("[viz]\nplot_columns = 7\n")
        monkeypatch.chdir(tmp_path)
        assert fitting.plot_columns() == 7

    def test_it_changes_when_the_directory_does(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The defect itself: the value used to be whatever the *first*
        resolution said, forever."""
        first, second = tmp_path / "a", tmp_path / "b"
        for directory, columns in ((first, 5), (second, 2)):
            directory.mkdir()
            (directory / "specmod.toml").write_text(
                f"[viz]\nplot_columns = {columns}\n"
            )

        monkeypatch.chdir(first)
        assert fitting.plot_columns() == 5
        monkeypatch.chdir(second)
        assert fitting.plot_columns() == 2

    def test_the_old_name_still_works_and_says_it_is_going(self) -> None:
        with pytest.warns(DeprecationWarning, match="plot_columns"):
            value = fitting.PLOT_COLUMNS
        assert value == fitting.plot_columns()

    def test_an_unknown_attribute_is_still_an_attribute_error(self) -> None:
        """The `__getattr__` must not swallow typos into something else."""
        with pytest.raises(AttributeError):
            fitting.NOT_A_REAL_NAME  # noqa: B018


class TestDiagnosticsAreAudible:
    """Each path that used to print. What replaced it is chosen per site:
    `warnings` for something the caller should act on, `logging` for per-item
    progress — because warnings are deduplicated by code location, so a run
    skipping twenty stations would report one."""

    def test_an_unknown_weight_method_warns_and_falls_back(self) -> None:
        from specmod.fitting.event import FitSpectra  # noqa: PLC0415

        checker = FitSpectra.__dict__["_FitSpectra__check_wm"]
        with pytest.warns(UserWarning, match="Unknown weight method"):
            assert checker(None, "sideways") == "none"

    def test_a_missing_distance_warns_rather_than_printing(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import obspy  # noqa: PLC0415

        stream = obspy.Stream([obspy.Trace()])
        with pytest.warns(UserWarning, match="unsorted"):
            utils.stream_distance_sort(stream)
        assert capsys.readouterr().out == ""

    def test_the_module_loggers_exist_and_are_not_configured(self) -> None:
        """A library must not call `basicConfig` for its host: that decides
        formatting and destination for the whole process."""
        from specmod.fitting import event  # noqa: PLC0415

        for module in (event, utils):
            assert isinstance(module.logger, logging.Logger)
            assert module.logger.name.startswith("specmod")
            assert not module.logger.handlers

    def test_every_failed_station_is_reported_not_just_the_first(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason this site is logging rather than a warning.

        Driven through `fit_spectra` itself with two stations that both fail:
        a station that could not be fitted is missing from the results, and
        one line covering both would be a report that hides the second.
        """
        from specmod.fitting.event import FitSpectra  # noqa: PLC0415

        class _Fails:
            def fit_mod(self, **kwargs: Any) -> None:
                raise ValueError("no usable band")

        fitter = object.__new__(FitSpectra)
        fitter.models = {"XX.A..HHZ": _Fails(), "XX.B..HHZ": _Fails()}
        # The two bookkeeping steps after the loop need real models.
        monkeypatch.setattr(
            FitSpectra, "_FitSpectra__set_fit_models_to_spectrum", lambda self: None
        )
        monkeypatch.setattr(
            FitSpectra, "_FitSpectra__generate_group_fit_table", lambda self: None
        )

        with caplog.at_level(logging.WARNING, logger="specmod.fitting.event"):
            fitter.fit_spectra(weight_method="none")

        reported = [record.getMessage() for record in caplog.records]
        assert len(reported) == 2, reported
        assert any("XX.A..HHZ" in line for line in reported)
        assert any("XX.B..HHZ" in line for line in reported)
        assert all("no usable band" in line for line in reported)

    def test_warnings_really_would_have_collapsed_that(self) -> None:
        """Not an assumption about `warnings`; the behaviour it is avoiding."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(3):
                warnings.warn("same site, same message", stacklevel=1)
        assert len(caught) == 1
