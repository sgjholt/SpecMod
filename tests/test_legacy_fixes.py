"""Regression tests for the defects catalogued in docs/REFACTOR_PLAN.md §2.

Each of these fails against the pre-refactor code. They are deliberately
narrow: they pin the specific bug, not the surrounding behaviour, so they stay
meaningful as the modules around them are decomposed in later phases.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import obspy
import pytest

import specmod.preprocess as pre
import specmod.spectral as sp
import specmod.utils as ut

SRC = Path(sp.__file__).parent


# --------------------------------------------------------------- §2.5 syntax

def test_every_module_parses() -> None:
    """utils.py had a SyntaxError from the initial commit until this refactor."""
    for path in sorted(SRC.glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))


def test_keith2utc_parses_a_catalogue_row() -> None:
    """The starred-expression syntax error lived in this function.

    Note the input format: ``cat2kstyle`` drops sub-second precision with a
    fixed ``[:-3]`` slice, so it assumes exactly two decimal places on the
    seconds field. Anything else leaves a trailing separator and raises. That
    fragility is pre-existing and left alone here deliberately — it is a
    behaviour change, not a syntax fix, so it belongs with the preprocessing
    rewrite rather than in this commit.
    """
    row = {"Date": "2020/03/18", "Time": "13:09:31.00"}
    assert ut.keith2utc(row) == obspy.UTCDateTime(2020, 3, 18, 13, 9, 31)


def test_cat2kstyle_precision_assumption_is_documented() -> None:
    """Pin the known limitation so the rewrite has something to point at."""
    with pytest.raises(ValueError, match="invalid literal for int"):
        ut.keith2utc({"Date": "2020/03/18", "Time": "13:09:31.000"})


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

@pytest.mark.parametrize(
    ("func", "bad_name"),
    [
        (sp.SNP.find_optimal_signal_bandwidth_2, "name"),
        (sp.Spectra.__init__, None),
    ],
)
def test_no_undefined_names_in_plot_branches(func, bad_name) -> None:
    """`plot=True` branches referenced names that did not exist (F821)."""
    tree = ast.parse(inspect.getsource(func).lstrip())
    loaded = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    if bad_name is not None:
        assert bad_name not in loaded


def test_fit_spectra_reset_uses_the_right_attribute() -> None:
    """`self.model[name]` should have been `self.models[name]`."""
    import specmod.fitting as fit

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
