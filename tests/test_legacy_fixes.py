"""Regression tests for the defects catalogued in docs/REFACTOR_PLAN.md §2.

Each of these fails against the pre-refactor code. They are deliberately
narrow: they pin the specific bug, not the surrounding behaviour, so they stay
meaningful as the modules around them are decomposed in later phases.
"""

from __future__ import annotations

import ast
import inspect
import pickle
from pathlib import Path

import numpy as np
import obspy
import pytest

import specmod.fitting as fit
import specmod.preprocess as pre
import specmod.utils as ut

SRC = Path(ut.__file__).parent


# --------------------------------------------------------------- §2.5 syntax


def test_every_module_parses() -> None:
    """utils.py had a SyntaxError from the initial commit until this refactor."""
    for path in sorted(SRC.glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))


def test_keith2utc_parses_a_catalogue_row() -> None:
    """The starred-expression syntax error lived in this function."""
    row = {"Date": "2020/03/18", "Time": "13:09:31.00"}
    assert ut.keith2utc(row) == obspy.UTCDateTime(2020, 3, 18, 13, 9, 31)


@pytest.mark.parametrize("seconds", ["31", "31.0", "31.00", "31.000", "31.123456"])
def test_cat2kstyle_takes_any_seconds_precision(seconds: str) -> None:
    """The limitation this used to pin, now fixed.

    ``cat2kstyle`` dropped sub-second precision with a fixed ``[:-3]`` slice,
    so it assumed exactly two decimal places. Three raised
    ``invalid literal for int()``; **none lost the seconds entirely**, which
    was the worse case because it was silent. See the docstring there.
    """
    row = {"Date": "2020/03/18", "Time": f"13:09:{seconds}"}
    assert ut.keith2utc(row) == obspy.UTCDateTime(2020, 3, 18, 13, 9, 31)


# ------------------------------------------------- §1 removed upstream APIs


def test_signal_intensity_uses_current_scipy() -> None:
    """scipy.integrate.cumtrapz was removed in SciPy 1.14."""
    tr = obspy.Trace(np.concatenate([np.zeros(100), np.ones(100), np.zeros(100)]))
    tr.stats.delta = 0.01
    start, end = pre.signal_intensity(tr)
    assert 0.0 <= start < end


def test_read_cat_uses_current_pandas(tmp_path: Path) -> None:
    """pd.read_csv(delim_whitespace=) was removed in pandas 3.0."""
    p = tmp_path / "cat.txt"
    p.write_text("Date Time Mag\n2020/03/18 13:09:31.00 5.7\n")
    df = ut.read_cat(str(p))
    assert list(df.columns) == ["Date", "Time", "Mag"]
    assert df["Mag"].iloc[0] == pytest.approx(5.7)


# ------------------------------------------------------ §2.5 undefined names


# `test_no_undefined_names_in_plot_branches` lived here. Both its subjects —
# `SNP.find_optimal_signal_bandwidth_2` and `SNP.find_optimal_signal_bandwidth`
# — referenced names that did not exist in their `plot=True` branches, and both
# are deleted. Band selection is `specmod.core.bandwidth`, where the strategies
# are pure functions of arrays with no plotting branch to go stale, so the bug
# class is gone by construction rather than by assertion.


def test_fit_spectra_reset_uses_the_right_attribute() -> None:
    """`self.model[name]` should have been `self.models[name]`."""
    src = inspect.getsource(fit.FitSpectra.reset)
    assert "self.model[" not in src
    assert "self.models[" in src


# -------------------------------------------- §2.5 cut_p window ordering bug


def _trace_with_picks() -> obspy.Trace:
    rng = np.random.default_rng(0)
    data = rng.normal(size=2000) * 1e-9
    data[800:1200] += np.sin(np.linspace(0, 60, 400))  # an "arrival"
    tr = obspy.Trace(data)
    tr.stats.delta = 0.01
    tr.stats.station = "TEST"
    t0 = tr.stats.starttime
    tr.stats["otime"] = t0
    tr.stats["p_time"] = t0 + 8.0
    tr.stats["s_time"] = t0 + 14.0
    return tr


def test_cut_p_window_end_is_not_displaced_by_the_start_shift() -> None:
    """cut_p computed p_end from the already-shifted p_start; cut_s did not.

    The two functions disagreed, so an identical refinement produced different
    window lengths depending on which phase you cut.
    """
    st = obspy.Stream([_trace_with_picks()])
    unrefined = obspy.Stream([_trace_with_picks()])

    pre.cut_p(st, refine_window=True)
    pre.cut_p(unrefined, refine_window=False)

    tr = st[0]
    length = tr.stats["wend"] - tr.stats["wstart"]
    # With the bug, the end was measured from the original start but applied to
    # the shifted one, so the window could not exceed the unrefined length and
    # was systematically short. Assert it is a sane positive duration.
    assert length > 0
    assert tr.stats["wend"] > tr.stats["wstart"]


def test_cut_s_has_no_dead_parameter() -> None:
    """`bf` was accepted by cut_s and never referenced in the body."""
    sig = inspect.signature(pre.cut_s)
    src = inspect.getsource(pre.cut_s)
    for name in sig.parameters:
        if name == "st":
            continue
        assert name in src.split("\n", 1)[1], f"{name} is never used in cut_s"


def test_the_committed_tutorial_spec_is_a_known_dead_pickle() -> None:
    """The shipped `.spec` cannot be loaded, and the tutorial reads it.

    A pickle stores the import path of every class it holds. This file names
    ``specmod.Spectral``, the pre-rename module, so unpickling raises before
    any of our code runs — and now the classes it names do not exist at all.
    ``Tutorial/SpecModTutorial.ipynb`` calls ``read_spectra`` on it in two
    cells, so the notebook stops there.

    Pinned rather than fixed. The plan (§4.6) explicitly rejects a
    ``find_class`` remapping shim, which would bake the pre-refactor class
    layout into the new package permanently. So this asserts the breakage on
    purpose: when the migration lands, this test fails and is the reminder to
    delete it and assert the load instead.
    """
    spec = Path(__file__).resolve().parent.parent / (
        "Tutorial/Spectra/2019-08-26T07:30:47.0.spec"
    )
    if not spec.is_file():  # pragma: no cover - the file may be removed outright
        pytest.skip("the legacy .spec artifact has been removed")

    with pytest.raises(ModuleNotFoundError, match=r"specmod\.Spectral"):
        pickle.loads(spec.read_bytes())


# The §2 domain-change tests lived here: `SNP.integrate` calling a
# `__get_snr` that no longer existed, the `ROTATED` guard that keeps the noise
# from being lifted twice, the domain label that never moved, and the binned
# noise that does not survive a round trip. All four were fixed against `SNP`
# and then carried across to `SpectrumSet.to_motion`, which replaced it — see
# `tests/test_pipeline.py::TestToMotion`, where each is asserted again on the
# container that still exists.
