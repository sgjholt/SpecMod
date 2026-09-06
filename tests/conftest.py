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
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

#: Distributions provided by extras in pyproject.toml, and therefore absent
#: from a plain ``pip install specmod``.
OPTIONAL_DISTRIBUTIONS = ("multitaper", "mtspec", "pywt", "emcee")

#: Not listed above on purpose. `specmod[io]` is optional to *install* but not
#: optional to *test*: `--without-optional-extras` exists to reproduce what CI
#: sees, and CI installs `[dev]`, which pulls h5py and pyarrow in so the
#: persistence suite runs. Blocking them here would silently skip it.
_IO_EXTRAS_ARE_TESTED = ("h5py", "pyarrow")


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


# --------------------------------------------------------- real waveforms

#: The Preston New Road event the tutorial is built around.
_ROOT = Path(__file__).resolve().parent.parent


def _pnr():
    """The tutorial event and its directory, resolved on demand.

    Imported inside the function for the same reason the fixtures below defer
    ``specmod.preprocess``: this module must import even when specmod is not
    installed, so that a missing package skips rather than erroring out of
    collection entirely.
    """
    from specmod.datasets import PNR_2019  # noqa: PLC0415

    return PNR_2019, PNR_2019.directory(_ROOT)


@pytest.fixture(scope="session")
def pnr_stream():
    """The prepared, *uncut* stream: metadata set, picks read, response removed.

    This is the input to :mod:`specmod.preprocess`'s windowing functions, which
    is what makes it useful separately from ``pnr_windows``. The golden
    spectral reference starts from cut windows, so a change to how windows are
    chosen *moves* that reference rather than failing against it; tests that
    want to pin the windowing itself need the stream from before the cut.

    Returns a callable, because every consumer mutates what it is given —
    ``s_window`` trims in place — and the expensive part (response removal) should
    happen once per session.
    """
    obspy = pytest.importorskip("obspy")
    event, paths = _pnr()
    if not paths.is_present():
        pytest.skip("tutorial waveforms not present")

    # Deferred: this module must import without specmod present, so the
    # optional-extras blocker above can be installed before anything loads.
    import specmod.preprocess as pre  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inventory = obspy.read_inventory(str(paths.inventory))
        stream = obspy.read(paths.waveform_glob("*HH[EN]*"))
        stream = pre.with_distance(
            stream,
            event.latitude,
            event.longitude,
            event.depth_km,
            obspy.UTCDateTime(event.origin),
            inventory=inventory,
            dtype="mseed",
        )
        stream = pre.with_picks(stream, str(paths.picks_file()))
        stream = obspy.Stream([tr for tr in stream if "s_time" in tr.stats])
        stream.detrend("linear")
        stream.detrend("demean")
        stream.taper(0.05)
        stream.remove_response(inventory, output="VEL")

    return stream.copy


@pytest.fixture(scope="session")
def pnr_windows(pnr_stream):
    """Signal and noise streams cut with the published Magna workflow.

    Session-scoped and copied on handout: response removal and window
    refinement are the slow part of the suite, every module that wants real
    data wants the same 28 windows, and ``Spectra.from_streams`` mutates what
    it is given.

    Lives here rather than in a test module because a fixture imported *across*
    test modules depends on pytest's path insertion having already happened,
    which is not guaranteed and fails quietly into a skip.
    """
    import specmod.preprocess as pre  # noqa: PLC0415

    stream = pnr_stream()
    signal = pre.s_window(
        stream, rafp=0.8, tafs=20, time_after="absolute_time", refine_window=True
    )
    noise = pre.get_noise_p(stream, signal)

    def cut():
        return signal.copy(), noise.copy()

    return cut
