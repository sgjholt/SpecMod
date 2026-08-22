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
1-part-in-1e12 perturbation. It also failed on every CI runner: identical
input returns last-bit differences on different hardware, and those propagate.
A reference that only holds on the machine that wrote it is not a reference,
so the comparison is now a relative tolerance over a distributional summary —
still far tighter than any real change in level or shape.

Reproducibility, and how it was won
------------------------------------
This suite once could not run its own noise comparison on CI. Identical code
on a runner matching this machine exactly — same system, arch, Python, numpy
and scipy — produced noise levels 41% and 82% apart. Not floating-point
sensitivity: the pipeline was *piecewise constant*, and three decision points
turned a last-bit difference into a large one.

1. **The rotation stepped its exponent** by ``inc = 0.05`` and stopped at the
   first step past the touching point, so landing one iteration either side
   changed the noise by ``0.001 ** -0.05 = 1.41``. The exponent now has a
   closed form and is used exactly.
2. **The binning tested closed intervals** against each edge, so a sample on
   an edge belonged to two bins and the surviving count depended on the last
   bit of ``np.logspace``. Membership is now computed from position.
3. **The band search** took percentiles of an integrated sign function with a
   retry loop, moving an edge by up to 13 bins. It now takes the widest
   contiguous passing run.

Measured after the fix, end to end over all 28 windows: perturbing the input
by 1e-15 moves the noise by 1.8e-11 and **no band edge at all**. The response
is linear, so machine-level differences land far below ``RTOL``. That is why
the exact comparison runs unconditionally again.

The band is still compared by containment rather than equality — one bin
genuinely sitting on the threshold can still flip, which now moves an edge by
one bin instead of thirteen.
"""

from __future__ import annotations

import functools
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

import specmod.preprocess as pre  # noqa: E402
from specmod.datasets import PNR_2019  # noqa: E402
from specmod.pipeline import spectrum_set_from_streams  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent

_REFERENCE = _ROOT / "tests" / "golden" / "pipeline_reference.json"

# The event and its layout come from `specmod.datasets`, which is also what
# `conftest.py` and `tools/make_golden.py` read. This module cannot import the
# constants from `conftest.py` directly — a conftest is not an importable
# module — so the shared definition lives in the package instead, which has the
# side benefit that the script regenerating this reference reads the same one.
_EVENT = PNR_2019
_PATHS = PNR_2019.directory(_ROOT)

pytestmark = pytest.mark.skipif(
    not _PATHS.is_present() or not _REFERENCE.is_file(),
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

#: ``cwt`` was held at 5e-2 for a residual disagreement on CI that nothing
#: could explain. A round of measurement (below) failed to reproduce it and
#: ruled out every mechanism proposed for it, so the tolerance is being
#: tightened by four orders of magnitude to find out whether it still exists.
#:
#: **What was measured**, on a Linux box matching the reference environment
#: exactly — same system, arch, Python 3.11, numpy 2.4.6, scipy 1.17.1:
#:
#: - ``cwt`` reproduces the committed reference to **3.8e-16**, machine
#:   epsilon. The disagreement is not "Linux versus macOS"; it is that runner
#:   versus this one.
#: - The ``cwt`` noise path is **linear** in the input: a 1e-13 perturbation
#:   moves the noise by 8.7e-14, with no step, on all 28 windows. So a 1-2%
#:   output difference needs a 1-2% *input* difference — which is not
#:   something last-bit floating point can produce.
#: - Not bin-edge fragility: the closest ``cwt`` sample sits 5.7e-4 of a bin
#:   from an interior edge, against 1.4e-6 for ``fft``.
#: - Not a differing window. One sample fewer moves ``fft``'s noise by 8.5%
#:   and ``cwt``'s by 3.6%, so a window that differed on CI would show up in
#:   ``fft`` *first* — and ``fft`` agrees exactly.
#: - Not quantile fragility from ``cwt``'s shorter arrays (51 samples against
#:   109): a 1e-15 perturbation moves its worst quantile by 2.4e-14, better
#:   than ``fft``'s 5.1e-13.
#: - Not PyWavelets, which the estimator does not use, and not threading:
#:   the transform is a batched ``numpy.fft.ifft``.
#:
#: So the CWT is as numerically stable here as every other estimator, and
#: 5e-2 was 12 orders of magnitude looser than anything measurable. Holding it
#: there hides whatever the real difference was rather than describing it.
#:
#: **1e-3 was an experiment**, not a calibration, and it has now returned its
#: answer: the residual is real. What one CI run showed, across six test jobs
#: on the same commit:
#:
#: - It fails on **ubuntu 3.11 and 3.13** with the *same* 8 differences, the
#:   same three windows, and the same magnitudes to three significant figures.
#:   Deterministic, not flaky.
#: - It **passes on ubuntu 3.12** and on macOS 3.11, 3.12 and 3.13, in that
#:   same run. So it is not the OS, not the Python version, and not a package
#:   version that tracks the Python version — it is which machine the job
#:   landed on, which is what "that runner, not Linux" above suspected and
#:   this is the same-run control for.
#: - On the machine that disagrees, ``fft``, ``welch``, ``multitaper`` and
#:   ``quadratic`` all still reproduce the reference exactly. Whatever it is,
#:   it is in the CWT path and not in that machine's arithmetic generally.
#: - Worst observed: 1.44e-2, on ``UR.AQ10.00.HHN bsnr``. Three of 28 windows
#:   move at all (``LV.L007..HHN``, ``UR.AQ01.00.HHE``, ``UR.AQ10.00.HHN``).
#:
#: **2e-2 is a bound on that, not an explanation of it.** It is the worst
#: observed difference with about 40% of headroom, and 2.5 times tighter than
#: the 5e-2 it replaces — which is the most that can honestly be claimed while
#: the mechanism is still unidentified and unreproducible on any box available
#: to work on. A real regression in the CWT noise path smaller than 2% would
#: pass here; the four estimators held at 1e-6 are what covers the pipeline
#: those windows share.
#:
#: To take it further, the next measurement needs the failing machine: run the
#: cwt path on it against a passing one, on the three named windows, and diff
#: the noise arrays before binning rather than after.
RTOL_BY_ESTIMATOR = {"cwt": 2e-2}

QUANTILES = np.linspace(0.0, 1.0, 33)


def _summary(a: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "max": float(np.max(a)),
        "sum": float(a.sum()),
        "quantiles": [float(q) for q in np.quantile(a, QUANTILES)],
    }


def _compare(
    got: dict[str, Any], want: dict[str, Any], where: str, rtol: float = RTOL
) -> list[str]:
    """Every way two summaries can disagree, reported with the size of it."""
    problems = []
    if got["n"] != want["n"]:
        problems.append(f"{where}: length {want['n']} -> {got['n']}")
        return problems
    for key in ("median", "max", "sum"):
        if got[key] != pytest.approx(want[key], rel=rtol):
            rel = abs(got[key] - want[key]) / max(abs(want[key]), 1e-300)
            problems.append(
                f"{where}: {key} {want[key]:.6e} -> {got[key]:.6e} (rel {rel:.2e})"
            )
    a, b = np.array(got["quantiles"]), np.array(want["quantiles"])
    if a != pytest.approx(b, rel=rtol):
        rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
        problems.append(f"{where}: quantile profile moved (max rel {rel:.2e})")
    return problems


@functools.cache
def _build_windows() -> tuple[Any, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inventory = obspy.read_inventory(str(_PATHS.inventory))
        stream = obspy.read(_PATHS.waveform_glob("*HH[EN]*"))
        pre.set_stream_distance(
            stream,
            _EVENT.latitude,
            _EVENT.longitude,
            _EVENT.depth_km,
            obspy.UTCDateTime(_EVENT.origin),
            inventory=inventory,
            dtype="mseed",
        )
        pre.set_picks(stream, str(_PATHS.picks_file()))
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
    return json.loads(_REFERENCE.read_text())


@functools.cache
def _run(estimator: str) -> Any:
    """The pipeline's output, built by :mod:`specmod.pipeline`.

    No legacy object is involved. **The reference file was generated by the
    legacy path and has not been regenerated**, which is the point: these
    tests passing is a verified reproduction of the recorded numbers, not a
    new baseline agreeing with itself. Measured, the deviation from the
    committed summaries is 1e-15 — machine epsilon — across all five
    estimators.
    """
    signal, noise = _build_windows()
    return spectrum_set_from_streams(signal.copy(), noise.copy(), estimator=estimator)


ESTIMATORS = ["fft", "welch", "multitaper", "quadratic", "cwt"]


# The exact noise comparison used to be opt-in, because the three
# discontinuities below made it fail on hardware differences rather than on
# changes to the code. They are fixed, so it runs everywhere again.


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_same_windows_are_still_produced(estimator: str) -> None:
    assert _run(estimator).ids() == sorted(_reference()[estimator])


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_amplitudes_are_unchanged(estimator: str) -> None:
    """The strict check. A behaviour-preserving refactor passes untouched."""
    expected = _reference()[estimator]
    problems: list[str] = []
    for name, pair in sorted(_run(estimator).pairs.items()):
        want = expected[name]
        problems += _compare(_summary(pair.signal.amp), want["amp"], f"{name} amp")
        problems += _compare(_summary(pair.signal.freq), want["freq"], f"{name} freq")
    assert not problems, "\n".join(
        [f"{len(problems)} difference(s) under {estimator!r}:", *problems]
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_noise_and_snr_are_structurally_sound(estimator: str) -> None:
    """What can be asserted everywhere, given the discontinuities.

    Not a weakened version of the exact check — a different one. It targets
    the errors a refactor actually makes: a factor of two, a normalisation
    keyed off the wrong length, a units slip, an array that comes back empty
    or full of NaN. All of those move things by orders of magnitude, and none
    of them can hide inside the one bin and the 3x band allowed here.

    The bound is set by the machine variation it has to tolerate, and that
    is uncomfortably close to the error it most wants to catch: the worst
    noise median seen across runners moved by 1.82x, and a factor-of-two
    mistake is 2.0x. **This check cannot reliably separate those**, so it is
    honestly a backstop against gross breakage rather than a precision
    instrument. A 2x error in the noise is caught instead by the tests that
    compare against the legacy implementations directly and do not depend on
    two machines agreeing — ``boost_noise`` bit-for-bit over 200 randomised
    cases, ``parseval_scale`` on its own, and ``log_bin`` to 1e-12 on all 28
    windows, all in ``tests/test_collection.py``. Those are where the noise
    path is actually pinned.
    """
    expected = _reference()[estimator]
    problems = []
    for name, pair in sorted(_run(estimator).pairs.items()):
        want = expected[name]
        for label, got, ref in (
            ("noise", np.asarray(pair.noise.amp), want["noise_amp"]),
            ("bsnr", np.asarray(pair.snr), want["bsnr"]),
        ):
            where = f"{name} {label}"
            if not np.isfinite(got).all():
                problems.append(f"{where}: non-finite values")
                continue
            if not (got > 0).all():
                problems.append(f"{where}: non-positive values")
            if abs(got.size - ref["n"]) > 1:
                problems.append(f"{where}: length {ref['n']} -> {got.size}")
                continue
            ratio = float(np.median(got)) / ref["median"]
            if not 1 / 1.9 < ratio < 1.9:
                problems.append(
                    f"{where}: median moved by {ratio:.2f}x "
                    f"({ref['median']:.6e} -> {np.median(got):.6e})"
                )
    assert not problems, "\n".join(
        [f"{len(problems)} problem(s) under {estimator!r}:", *problems]
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_noise_and_snr_are_exactly_unchanged(estimator: str) -> None:
    """Separate from the amplitudes because it fails for different reasons.

    The noise passes through the Parseval rescale, the boost rotation and the
    interpolation onto the signal's axis. A rewrite of any of those can move
    ``bsnr`` while leaving the signal amplitudes untouched.

    ``noise_amp`` here is still the legacy value and has never been
    regenerated. ``bsnr`` has been, once — see
    :func:`test_the_only_deliberate_divergence_from_legacy_is_the_binned_noise`,
    which holds the old values and the size of the change.
    """
    expected = _reference()[estimator]
    problems: list[str] = []
    for name, pair in sorted(_run(estimator).pairs.items()):
        want = expected[name]
        rtol = RTOL_BY_ESTIMATOR.get(estimator, RTOL)
        problems += _compare(
            _summary(pair.noise.amp), want["noise_amp"], f"{name} noise", rtol
        )
        problems += _compare(_summary(pair.snr), want["bsnr"], f"{name} bsnr", rtol)
    assert not problems, "\n".join(
        [f"{len(problems)} difference(s) under {estimator!r}:", *problems]
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_only_deliberate_divergence_from_legacy_is_the_binned_noise(
    estimator: str,
) -> None:
    """One correction, bounded, with the superseded numbers kept beside it.

    The legacy code derived the binned noise and the unbinned noise by two
    different routes: it multiplied the *bins* by the boost factor, and
    separately multiplied the unbinned array by that factor interpolated up.
    Those disagree. A bin holds the geometric mean of ``log10(amp)``, so
    binning the lifted noise gives ``mean(log a) + mean(log f)`` while lifting
    the bin gives ``mean(log a) + log f(centre)`` — equal only where the factor
    is flat across the bin. Every stored pair's ``binned_noise`` was therefore
    not the binning of its own ``noise``, by up to 18.8%.

    The lift is now applied to the unbinned noise, and the binned noise
    derived from it. That changes ``bsnr`` and nothing else, so ``bsnr`` was
    regenerated and the superseded values kept as ``bsnr_legacy``.

    **What the change is, measured on the 28 fft windows.** No bin gains or
    loses a sample: the bin centres, the bin counts and the binned *signal* are
    bit-identical. Only the value representing each bin moves, because the lift
    is now averaged across the bin along with the amplitude rather than applied
    once at the centre. The result is close to symmetric — 987 bins down, 1064
    up, median ratio 1.000000 — with a slight net rise in noise (geometric mean
    1.00068) and therefore a slight net fall in signal-to-noise (0.99932).

    Median |change| is 0.08% to 0.37% per decade. The large excursions are
    rare and live at the extremes: 11.7% around 1-5 Hz, where a bin holds one
    or two samples and the interpolated factor at the sample is not the factor
    at the bin centre, and 18.8% above 60 Hz, where a bin spans enough absolute
    frequency for the factor to vary appreciably across it. Both regions sit
    outside every selected band, which is why no band and no fitted parameter
    moves.

    This test exists so that "one deliberate correction" cannot quietly become
    two. It pins the divergence rather than forgiving it.
    """
    expected = _reference()[estimator]
    ratios: list[float] = []
    for name in sorted(expected):
        want = expected[name]
        assert "bsnr_legacy" in want, (
            f"{name}: bsnr_legacy is missing. The legacy values are the record "
            "of what the published lineage produced; regenerating bsnr without "
            "keeping them discards it."
        )
        for key in ("median", "max", "sum"):
            ratios.append(abs(want["bsnr"][key] / want["bsnr_legacy"][key] - 1))
        assert want["bsnr"]["n"] == want["bsnr_legacy"]["n"], (
            f"{name}: the bin count changed ({want['bsnr_legacy']['n']} -> "
            f"{want['bsnr']['n']}). The correction changes the value in each "
            "bin, not which samples are in it — a length change is a different "
            "bug."
        )

    worst = max(ratios)
    assert worst < 0.05, (
        f"{estimator}: bsnr now differs from the legacy record by {worst:.3g}, "
        "more than the 5% the binned-noise correction accounts for. Something "
        "else has moved."
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
    for name, pair in sorted(_run(estimator).pairs.items()):
        want = expected[name]
        got = list(pair.band) if pair.band is not None else None

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
        if float(pair.resolution_floor) != pytest.approx(
            want["resolution_floor"], rel=RTOL
        ):
            problems.append(f"{name}: resolution floor moved")
    assert not problems, f"under {estimator!r}: " + "; ".join(problems)
