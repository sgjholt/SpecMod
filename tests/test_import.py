"""Smoke tests: the package imports and every module in it is reachable.

These exist because they would not have passed before this refactor. Nearly
every module was unimportable on ``master`` — ``utils`` carried a syntax error
dating to the initial commit (2020-02-27), which took ``preprocess`` with it,
``spectral`` imported ``mtspec`` eagerly, and ``fitting`` imported a module
under its pre-rename name.

**The module list is discovered, not written down.** It used to be a literal,
and a hand-maintained list of modules is a trap that has now sprung twice in
this refactor: once when ``models`` was deleted and once when ``spectral``
was, each time leaving a stale name behind that failed somewhere unrelated to
where the change was made. Walking the package means a module cannot be
forgotten when it is added, and cannot be left behind when it is removed.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import specmod


def _modules() -> list[str]:
    """Every importable module in the package, found by walking it.

    ``_vendor`` is skipped: it is third-party source kept close to upstream,
    and whether it imports is upstream's business rather than a claim this
    suite should be making.
    """
    return sorted(
        info.name
        for info in pkgutil.walk_packages(specmod.__path__, prefix="specmod.")
        if "._vendor" not in info.name
    )


@pytest.mark.parametrize("name", _modules())
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_the_walk_finds_the_modules_that_matter() -> None:
    """A guard on the guard.

    ``walk_packages`` silently returning nothing would make every test above
    vacuous — zero parametrised cases, all passing. Naming a few modules that
    must be there is what stops that failing open.
    """
    found = set(_modules())
    for name in (
        "specmod.config",
        "specmod.core",
        "specmod.pipeline",
        "specmod.preprocess",
        "specmod.transforms",
        "specmod.utils",
    ):
        assert name in found, f"{name} was not discovered"
    assert len(found) > 15, found


def test_package_exposes_version() -> None:
    assert isinstance(specmod.__version__, str)
    assert specmod.__version__


def test_importable_without_mtspec() -> None:
    """The pipeline must import even though mtspec cannot build here.

    mtspec is Fortran source with no wheels; an eager import made the whole
    package uninstallable without a Fortran compiler. There is no longer any
    import of it to be eager — the ``_mtspec`` shim went with ``spectral`` —
    so what this now guards is that nothing reintroduces one.
    """
    # Local on purpose: the import succeeding *is* the assertion, so it has to
    # happen inside the test. At module scope a regression would surface as a
    # collection error for the whole file instead of a failure here.
    from specmod import pipeline  # noqa: PLC0415

    assert hasattr(pipeline, "spectrum_set_from_streams")


def test_asking_for_mtspec_says_what_to_use_instead() -> None:
    """It is not wired in, and the error has to be more than a KeyError.

    The pre-refactor backend is the reason half this package exists; someone
    reproducing an old run will reach for it by name, and being told
    ``'mtspec'`` is not in a dict does not tell them that ``prieto`` is the
    same lineage without a compiler.
    """
    import numpy as np  # noqa: PLC0415

    from specmod.pipeline import estimate_spectrum  # noqa: PLC0415

    with pytest.raises(ValueError, match="pre-refactor Fortran backend"):
        estimate_spectrum(np.zeros(64), 0.01, estimator="mtspec")
