"""The decomposition must not move anyone's numbers without saying so.

There is no legacy Docker image to compare against — the 0.1.0 environment
cannot be rebuilt, and `master` is a reading reference rather than a runnable
one. So the reference for the rewrite is this code, captured at the point the
decomposition started, in ``tests/golden/pipeline_reference.json``.

This is the safety net the whole rewrite leans on. Every stage that moves a
class from ``spectral.py`` into ``core/`` is checked here, on 28 real windows
across 5 estimators. A refactor that is genuinely behaviour-preserving passes
untouched; one that is not fails loudly and names the station.

**When this fails,** the question is not "how do I make it pass" but "did I
mean to change this". If the change is intended, regenerate with
``python tools/make_golden.py`` and say what moved and why in the commit
message. Regenerating without that explanation destroys the only check
standing between the rewrite and a silently different Omega.

Why tolerances rather than exact digests
----------------------------------------
The first version hashed raw float64 bytes and was verified to catch a
1-part-in-1e12 perturbation. It also failed on every CI runner: a different
numpy/scipy build returns last-bit differences on identical input, and those
propagate. A reference that only holds on the machine that wrote it is not a
reference, so the comparison is now a relative tolerance over a distributional
summary — still far tighter than any real change in level or shape.

What is and is not reproducible across machines
------------------------------------------------
The signal spectra are. The noise, the signal-to-noise and the selected band
are not, and the reason is three discontinuities in the legacy pipeline that
turn a last-bit difference into a large one:

1. **The rotation's break condition.** ``boost_noise`` raises the noise by
   ``inc = 0.05`` in the exponent per iteration and stops when any point
   reaches the signal. Landing one iteration differently changes the noise by
   a factor of ``0.001 ** -0.05 = 1.41`` at the low-frequency end. The
   differences seen on CI — 41% and 82% — are exactly one and two steps.
2. **Empty-bin dropping.** ``log_bin`` keeps a bin if any sample falls within
   its edges. A sample sitting on an edge can fall either side depending on
   the last bit of ``np.logspace``, changing the surviving bin count by one
   and hence the length of ``bsnr``.
3. **The band search**, below.

So the strict noise and signal-to-noise checks run only where the recorded
environment matches. That is not a workaround for a flaky test — it is the
honest statement that **a 41% change in noise level can follow from a
last-bit difference**, which is a defect in the algorithm and is tracked in
``docs/REFACTOR_PLAN.md`` §4.5.1. The signal amplitudes carry no such
discontinuity and are checked everywhere, unconditionally.

The band is the exception, and for a reason worth knowing
---------------------------------------------------------
``find_optimal_signal_bandwidth`` thresholds with ``sign(bsnr - tolerance)``
and reads the band off percentiles of the integral, with a retry loop when the
edges cross. All three steps are discontinuous, so an input difference far
below anything visible can move an edge by *many* bins — on ``LV.L001..HHN``
the low edge moved by about 13 bins between two library versions, and the
narrower band sat entirely inside the wider one.

**The selected band is therefore not reproducible across platforms**, and the
band is what constrains ``Omega``. That is a property of the search, not of
this test and not of the refactor. It is checked here by containment rather
than equality, and it is the strongest argument for the band-search rework
tracked in ``docs/REFACTOR_PLAN.md`` §4.5.1.
"""

from __future__ import annotations

import contextlib
import functools
import glob
import io
import json
import os
import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy

obspy = pytest.importorskip("obspy")

import specmod.preprocess as pre  # noqa: E402
from specmod.spectral import Spectra  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Tutorial" / "Data" / "2019-08-26T07:30:47.0"
INVENTORY = ROOT / "Tutorial" / "MetaData" / "pnr_inventory.xml"
REFERENCE = ROOT / "tests" / "golden" / "pipeline_reference.json"

ORIGIN = "2019-08-26T07:49:24.2"
LATITUDE, LONGITUDE, DEPTH_KM = 53.784, -2.967, 2.1

pytestmark = pytest.mark.skipif(
    not DATA.is_dir() or not INVENTORY.is_file() or not REFERENCE.is_file(),
    reason="tutorial waveforms or the golden reference are not present",
)


#: Loose enough to survive a different numpy/scipy build, tight enough that no
#: change to what the code computes can hide under it.
#:
#: Calibrated rather than guessed: scaling the FFT amplitudes by ``1 + 1e-5``
#: fails, ``1 + 1e-7`` passes. Every change this suite exists to catch — a
#: factor of two from a fold, a normalisation keyed off the wrong length, a
#: different taper — is orders of magnitude larger than that.
#:
#: On the cross-platform side the evidence is weaker and worth stating as
#: such: in the CI run that failed the byte-exact version, the reported median
#: and max agreed with the reference to all 7 printed significant figures, so
#: the true spread is somewhere below 5e-7 and most likely at the last bit.
#: If a runner ever fails on tolerance alone, that bound is what to revisit.
RTOL = 1e-6

QUANTILES = np.linspace(0.0, 1.0, 33)


def _summary(a: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "max": float(a.max()),
        "sum": float(a.sum()),
        "quantiles": [float(q) for q in np.quantile(a, QUANTILES)],
    }


def _compare(got: dict[str, Any], want: dict[str, Any], where: str) -> list[str]:
    """Every way two summaries can disagree, reported with the size of it."""
    problems = []
    if got["n"] != want["n"]:
        problems.append(f"{where}: length {want['n']} -> {got['n']}")
        return problems
    for key in ("median", "max", "sum"):
        if got[key] != pytest.approx(want[key], rel=RTOL):
            rel = abs(got[key] - want[key]) / max(abs(want[key]), 1e-300)
            problems.append(
                f"{where}: {key} {want[key]:.6e} -> {got[key]:.6e} (rel {rel:.2e})"
            )
    a, b = np.array(got["quantiles"]), np.array(want["quantiles"])
    if a != pytest.approx(b, rel=RTOL):
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
        problems.append(f"{where}: quantile profile moved (max rel {rel:.2e})")
    return problems


@functools.cache
def _build_windows() -> tuple[Any, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inventory = obspy.read_inventory(str(INVENTORY))
        stream = obspy.read(os.path.join(str(DATA), "*HH[EN]*"))
        pre.set_stream_distance(
            stream,
            LATITUDE,
            LONGITUDE,
            DEPTH_KM,
            obspy.UTCDateTime(ORIGIN),
            inventory=inventory,
            dtype="mseed",
        )
        pre.set_picks_from_pyrocko(
            stream, glob.glob(os.path.join(str(DATA), "*.picks"))[0]
        )
        stream = obspy.Stream([tr for tr in stream if "s_time" in tr.stats])
        stream.detrend("linear")
        stream.detrend("demean")
        stream.taper(0.05)
        stream.remove_response(inventory, output="VEL")
        signal = pre.get_signal(
            stream,
            pre.cut_s,
            rafp=0.8,
            tafs=20,
            time_after="absolute_time",
            refine_window=True,
        )
        return signal, pre.get_noise_p(stream, signal)


@functools.cache
def _reference() -> dict[str, Any]:
    return json.loads(REFERENCE.read_text())


@functools.cache
def _run(estimator: str) -> Any:
    signal, noise = _build_windows()
    with contextlib.redirect_stdout(io.StringIO()):
        return Spectra.from_streams(signal.copy(), noise.copy(), estimator=estimator)


ESTIMATORS = ["fft", "welch", "multitaper", "quadratic", "cwt"]


def _environment() -> dict[str, str]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": ".".join(platform.python_version_tuple()[:2]),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _environment_matches() -> tuple[bool, str]:
    # Evaluated at import time to build the skip mark, which runs before the
    # module-level skipif can spare it — so a missing reference must not raise.
    if not REFERENCE.is_file():
        return False, "no reference file"
    recorded = _reference().get("_environment")
    if recorded is None:
        return False, "the reference records no environment"
    here = _environment()
    differing = [
        f"{k}: {recorded[k]} vs {here[k]}" for k in here if recorded.get(k) != here[k]
    ]
    return not differing, ", ".join(differing)


#: Skips the checks the pipeline's discontinuities make machine-specific.
#: Deliberately not applied to the signal amplitudes.
only_on_the_recorded_environment = pytest.mark.skipif(
    not _environment_matches()[0],
    reason=(
        "noise and SNR are not reproducible across builds — "
        f"{_environment_matches()[1]}; see the module docstring"
    ),
)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_same_windows_are_still_produced(estimator: str) -> None:
    assert sorted(_run(estimator).group) == sorted(_reference()[estimator])


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_amplitudes_are_unchanged(estimator: str) -> None:
    """The strict check. A behaviour-preserving refactor passes untouched."""
    expected = _reference()[estimator]
    problems: list[str] = []
    for name, snp in sorted(_run(estimator).group.items()):
        want = expected[name]
        problems += _compare(_summary(snp.signal.amp), want["amp"], f"{name} amp")
        problems += _compare(_summary(snp.signal.freq), want["freq"], f"{name} freq")
    assert not problems, "\n".join(
        [f"{len(problems)} difference(s) under {estimator!r}:", *problems]
    )


@only_on_the_recorded_environment
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_noise_and_snr_are_unchanged(estimator: str) -> None:
    """Separate from the amplitudes because it fails for different reasons.

    The noise passes through the Parseval rescale, the boost rotation and the
    interpolation onto the signal's axis. A rewrite of any of those can move
    ``bsnr`` while leaving the signal amplitudes untouched.
    """
    expected = _reference()[estimator]
    problems: list[str] = []
    for name, snp in sorted(_run(estimator).group.items()):
        want = expected[name]
        problems += _compare(
            _summary(snp.noise.amp), want["noise_amp"], f"{name} noise"
        )
        problems += _compare(_summary(snp.bsnr), want["bsnr"], f"{name} bsnr")
    assert not problems, "\n".join(
        [f"{len(problems)} difference(s) under {estimator!r}:", *problems]
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_selected_bandwidth_still_covers_the_same_measurement(
    estimator: str,
) -> None:
    """Containment, not equality — see the module docstring for why.

    The band search is discontinuous in three places, so an edge can jump many
    bins on a difference far too small to see in the spectrum. What must hold
    is that the band still describes the same measurement: the narrower of the
    two lies inside the wider, rather than the search having wandered somewhere
    else entirely.

    A band appearing or disappearing outright is always a failure — that is a
    station changing from measurable to not, which no tolerance should absorb.
    """
    expected = _reference()[estimator]
    problems = []
    for name, snp in sorted(_run(estimator).group.items()):
        want, band = expected[name], getattr(snp, "ubfreqs", None)
        has_band = band is not None and len(band) == 2
        got = [float(band[0]), float(band[1])] if has_band else None

        if (got is None) != (want["band"] is None):
            problems.append(f"{name}: {want['band']} -> {got}")
            continue
        if got is None:
            continue

        lo, hi = max(got[0], want["band"][0]), min(got[1], want["band"][1])
        narrower = min(got[1] - got[0], want["band"][1] - want["band"][0])
        if hi <= lo or (hi - lo) < 0.98 * narrower:
            problems.append(
                f"{name}: {want['band']} -> {got} — the narrower band is not "
                f"contained in the wider one"
            )
        if float(snp.resolution_floor) != pytest.approx(
            want["resolution_floor"], rel=RTOL
        ):
            problems.append(f"{name}: resolution floor moved")
    assert not problems, f"under {estimator!r}: " + "; ".join(problems)
