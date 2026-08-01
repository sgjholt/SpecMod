"""Smoke tests: the package imports and its modules are reachable.

These exist because they would not have passed before this refactor. Every
module below was unimportable on ``master`` — ``utils`` carried a syntax error
dating to the initial commit (2020-02-27), which took ``preprocess`` with it,
``spectral`` imported ``mtspec`` eagerly, and ``fitting`` imported a module
under its pre-rename name.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "specmod.config",
    "specmod.fitting",
    "specmod.model_guess",
    "specmod.models",
    "specmod.preprocess",
    "specmod.ratios",
    "specmod.spectral",
    "specmod.utils",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_package_exposes_version() -> None:
    import specmod

    assert isinstance(specmod.__version__, str)
    assert specmod.__version__


def test_importable_without_mtspec() -> None:
    """``spectral`` must import even though mtspec cannot build here.

    mtspec is Fortran source with no wheels; an eager import made the whole
    package uninstallable without a Fortran compiler.
    """
    import specmod.spectral as sp

    assert hasattr(sp, "Spectrum")


def test_mtspec_shim_raises_a_useful_error() -> None:
    pytest.importorskip
    import specmod.spectral as sp

    try:
        import mtspec  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="mtspec backend is not installed"):
            sp._mtspec([0.0, 1.0], 0.01, 3)
