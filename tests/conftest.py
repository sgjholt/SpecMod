"""Shared pytest configuration.

Provides ``--without-optional-extras``, which runs the suite as a default
install sees it.

The reason it exists: this development environment has ``specmod[multitaper]``
installed and CI does not, so a test that quietly depends on the extra passes
here and fails there. That has happened twice — once for a registry test that
assumed every backend was constructible, once for a parametrised list that
happened to include ``prieto``. Both were found by CI rather than locally,
which is the wrong way round for a five-minute feedback loop.

Usage::

    pytest --without-optional-extras

Anything genuinely optional should ``pytest.importorskip`` or carry a
``skipif``, and this flag is how you find out whether it does.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Sequence
from typing import Any

import pytest

#: Distributions provided by extras in pyproject.toml, and therefore absent
#: from a plain ``pip install specmod``.
OPTIONAL_DISTRIBUTIONS = ("multitaper", "mtspec", "pywt", "emcee")


class _BlockOptionalExtras(importlib.abc.MetaPathFinder):
    """Make the optional distributions genuinely unimportable.

    Raises :class:`ModuleNotFoundError` rather than returning ``None``, because
    returning ``None`` merely defers to the next finder — which would find the
    installed package. ``ModuleNotFoundError`` is what a truly absent module
    produces, and it is what ``pytest.importorskip`` looks for; a plain
    ``ImportError`` is treated differently and would not skip cleanly.
    """

    def __init__(self, names: Sequence[str]) -> None:
        self._names = tuple(names)

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> importlib.machinery.ModuleSpec | None:
        root = fullname.split(".", maxsplit=1)[0]
        if root in self._names:
            raise ModuleNotFoundError(
                f"No module named {fullname!r} (blocked by --without-optional-extras)",
                name=fullname,
            )
        return None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--without-optional-extras",
        action="store_true",
        default=False,
        help=(
            "Hide packages provided by optional extras, reproducing what a "
            "default install and CI see."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Install the blocker before collection.

    Ordering matters: test modules import their dependencies at collection
    time, so a blocker installed later would have nothing left to block.
    Already-imported modules are evicted for the same reason.
    """
    if not config.getoption("--without-optional-extras"):
        return

    sys.meta_path.insert(0, _BlockOptionalExtras(OPTIONAL_DISTRIBUTIONS))
    for name in list(sys.modules):
        if name.split(".")[0] in OPTIONAL_DISTRIBUTIONS:
            del sys.modules[name]
