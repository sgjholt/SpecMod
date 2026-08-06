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

import specmod

MODULES = [
    "specmod.config",
    "specmod.fitting",
    "specmod.preprocess",
    "specmod.spectral",
    "specmod.utils",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_package_exposes_version() -> None:
    assert isinstance(specmod.__version__, str)
    assert specmod.__version__


def test_importable_without_mtspec() -> None:
    """``spectral`` must import even though mtspec cannot build here.

    mtspec is Fortran source with no wheels; an eager import made the whole
    package uninstallable without a Fortran compiler.
    """
    # Local on purpose: the import succeeding *is* the assertion, so it has to
    # happen inside the test. At module scope a regression would surface as a
    # collection error for the whole file instead of a failure here.
    import specmod.spectral as sp  # noqa: PLC0415

    assert hasattr(sp, "Spectrum")


def test_mtspec_shim_raises_a_useful_error() -> None:
    import specmod.spectral as sp  # noqa: PLC0415

    try:
        # Probing whether the optional extra is present; it cannot be a
        # module-scope import because the branch below depends on the answer.
        import mtspec  # noqa: F401, PLC0415
    except ImportError:
        with pytest.raises(ImportError, match="mtspec backend is not installed"):
            sp._mtspec([0.0, 1.0], 0.01, 3)
