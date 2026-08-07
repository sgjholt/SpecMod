"""The tutorial notebook refers only to API that exists.

It is the project's documented entry point, and it broke twice without anyone
noticing: once when ``Spectral.py`` was renamed to ``spectral.py`` — leaving
``import specmod.PreProcess`` failing at the second cell — and again when
``spectral.py`` was deleted, taking ``Spectra``, ``read_spectra`` and
``quick_vis`` with it.

Both times the notebook was the last thing to be updated and the first thing a
new user would run. This is the cheap check that stops it drifting again:
every name the notebook imports from ``specmod`` must resolve.

It deliberately does **not** execute the notebook. Doing so takes ~40 seconds
and needs a Jupyter kernel, which is a heavier dependency than the value
justifies for every run. Executing it is a release step; not referring to
deleted API is a per-commit one.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parent.parent / "Tutorial" / "SpecModTutorial.ipynb"


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
