"""The tutorial notebook refers only to API that exists.

It is the project's documented entry point, and it broke twice without anyone
noticing: once when ``Spectral.py`` was renamed to ``spectral.py`` — leaving
``import specmod.PreProcess`` failing at the second cell — and again when
``spectral.py`` was deleted, taking ``Spectra``, ``read_spectra`` and
``quick_vis`` with it.

Both times the notebook was the last thing to be updated and the first thing a
new user would run. This is the cheap check that stops it drifting again:
every name the notebook imports from ``specmod`` must resolve.

Two levels of check. The cheap one, on every run, is that every name the
notebook imports resolves. The thorough one, :func:`test_it_runs_end_to_end`,
executes it — that costs ~40 seconds and a Jupyter kernel, so it is marked
``notebook``, skips where those are not installed, and runs as its own CI job
rather than in every matrix cell.

Executing it used to be a release step. It was pulled forward because the
third break was not an import: renaming the event directory left every name
resolving and the first ``obspy.read`` raising, which only running it can
catch.
"""

from __future__ import annotations

import ast
import importlib
import json
import shutil
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "tutorial" / "SpecModTutorial.ipynb"


def _code() -> str:
    if not NOTEBOOK.is_file():
        pytest.skip("tutorial notebook not present")
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def _imports() -> list[tuple[str, str | None]]:
    """``(module, name)`` for everything the notebook imports from specmod."""
    found = []
    for node in ast.walk(ast.parse(_code())):
        if isinstance(node, ast.Import):
            found += [
                (a.name, None) for a in node.names if a.name.startswith("specmod")
            ]
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("specmod")
        ):
            found += [(node.module, a.name) for a in node.names]
    return found


def test_the_notebook_imports_something_from_specmod() -> None:
    """A guard on the guard: an empty list would make every check below
    vacuously pass."""
    assert len(_imports()) >= 5


@pytest.mark.parametrize(("module", "name"), _imports())
def test_every_imported_name_exists(module: str, name: str | None) -> None:
    imported = importlib.import_module(module)
    if name is None:
        return
    if hasattr(imported, name):
        return
    # `from specmod import sources` binds a *submodule*, which is not an
    # attribute of the package until something imports it. Importing it is the
    # same question asked the other way round.
    try:
        importlib.import_module(f"{module}.{name}")
    except ImportError:
        pytest.fail(f"{module}.{name} does not exist")


def test_it_parses_as_python() -> None:
    """Catches a cell left mid-edit, which nothing else would."""
    ast.parse(_code())


def test_it_does_not_reference_the_deleted_modules() -> None:
    """By name, including in prose.

    ``spectral``, ``models`` and ``model_guess`` are gone; a notebook that
    still talks about them misleads a reader even where the code no longer
    calls them.
    """
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    text = "\n".join("".join(c["source"]) for c in cells)
    for gone in (
        "specmod.spectral",
        "specmod.PreProcess",
        "specmod.Spectral",
        "model_guess",
        "create_simple_guess",
        "simple_model",
        "read_spectra",
        "write_spectra",
        "quick_vis",
    ):
        assert gone not in text, f"the notebook still refers to {gone}"


def test_it_does_not_write_a_pickle() -> None:
    """The tutorial is where a user learns what the output format is."""
    text = _code()
    assert "pickle" not in text.lower()
    assert ".spec" not in text


@pytest.mark.notebook
def test_it_runs_end_to_end(tmp_path: Path) -> None:
    """Execute the notebook, which is the only check that its paths still exist.

    The import checks above pass on a notebook that cannot run: they read the
    source without evaluating it. Both breaks this file was written for were
    import-level, but the next one was not — renaming the event directory left
    every name resolving and the first ``obspy.read`` raising, and nothing
    caught it until the notebook was executed by hand.

    Marked ``notebook`` and excluded from the matrix runs, so the ~40s and the
    Jupyter kernel are paid once in CI rather than six times.

    Runs against a copy so that the artefacts the notebook writes land in
    ``tmp_path`` rather than in the working tree. The kernel's working
    directory is the notebook's own, which is what lets the notebook use paths
    relative to itself and no ``os.chdir``.
    """
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")

    if not NOTEBOOK.is_file():
        pytest.skip("tutorial notebook not present")

    workdir = tmp_path / "tutorial"
    shutil.copytree(NOTEBOOK.parent, workdir)

    notebook = nbformat.read(workdir / NOTEBOOK.name, as_version=4)
    # `allow_errors` defaults to False, so any raising cell fails the test.
    nbclient.NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
    ).execute()
