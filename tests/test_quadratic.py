"""Tests for the quadratic (curvature-corrected) multitaper estimator.

The numerical core is vendored from Prieto's ``multitaper`` (see
:mod:`specmod._vendor.qiinv`), so these tests do two distinct jobs: hold the
vendored code to SpecMod's own contracts, and pin the behaviour that justifies
having it at all. The second matters because vendored code has no upstream
test suite to fall back on — four latent numpy-2 bugs in one function is what
prompted vendoring in the first place.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from specmod.core import AmplitudeKind
from specmod.transforms import (
    ESTIMATORS,
    MultitaperEstimator,
    QuadraticMultitaperEstimator,
)

FS = 100.0
DT = 1.0 / FS
N = 1024
NW = 3.0
K = 5
#: Multitaper half-bandwidth in Hz — the scale everything here is measured against.
W = NW / (N * DT)


def noise(sigma: float = 1e-6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, sigma, N)


def time_domain_energy(x: np.ndarray) -> float:
    return float(np.sum((x - x.mean()) ** 2) * DT)


# ------------------------------------------------------------- the contract


def test_satisfies_the_parseval_contract() -> None:
    """The check every estimator here answers to, applied to vendored code.

    This is the reason the vendored function can be trusted at all: it was
    written against a different set of conventions, and this says the units
    survived the crossing. It holds *without* variance normalisation, so it is
    a real measurement rather than a tautology.
    """
    x = noise()
    spectrum = QuadraticMultitaperEstimator().estimate(x, DT)
    assert spectrum.energy() == pytest.approx(time_domain_energy(x), rel=0.05)


def test_reports_units_and_provenance() -> None:
    spectrum = QuadraticMultitaperEstimator().estimate(noise(), DT, motion="velocity")
    assert spectrum.kind is AmplitudeKind.FAS
    assert spectrum.unit == "m/s*s"
    assert spectrum.meta["estimator"] == "quadratic"
    assert spectrum.meta["n_tapers"] == K
    assert "curvature_rms" in spectrum.meta


def test_is_registered() -> None:
    assert ESTIMATORS["quadratic"] is QuadraticMultitaperEstimator


def test_needs_at_least_two_tapers() -> None:
    """Three basis functions are fitted to K**2 cross-spectra, so K=1 gives
    one equation for three unknowns."""
    with pytest.raises(ValueError, match="at least 2 tapers"):
        QuadraticMultitaperEstimator(time_bandwidth=1.0, n_tapers=1)


def test_rejects_too_many_tapers() -> None:
    with pytest.raises(ValueError, match=r"exceeds 2\*NW-1"):
        QuadraticMultitaperEstimator(time_bandwidth=2.0, n_tapers=6)


# --------------------------------------------------- why it exists at all


def test_recovers_peak_height_that_multitaper_smooths_away() -> None:
    """The point of the estimator, stated as a number.

    Averaging K tapers smooths the spectrum across the inner band, which pulls
    a peak down by an amount set by its curvature. A pure sine has a known
    Fourier amplitude, ``A * T / 2``, so the shortfall is measurable rather
    than relative.
    """
    t = np.arange(N) * DT
    x = np.sin(2 * np.pi * 5.0 * t) + np.random.default_rng(3).normal(0.0, 0.02, N)
    true_peak = 1.0 * (N * DT) / 2.0

    plain = MultitaperEstimator(time_bandwidth=NW, n_tapers=K).estimate(x, DT)
    quad = QuadraticMultitaperEstimator(time_bandwidth=NW, n_tapers=K).estimate(x, DT)

    assert plain.amp.max() / true_peak < 0.92, (
        "the ordinary estimate should undershoot; if it no longer does, this "
        "test has stopped measuring what it claims to"
    )
    assert quad.amp.max() / true_peak == pytest.approx(1.0, abs=0.08)
    assert quad.amp.max() > plain.amp.max()


def test_separates_lines_the_ordinary_estimate_blurs() -> None:
    """Better contrast between two lines just over the resolution limit."""
    t = np.arange(N) * DT
    rng = np.random.default_rng(3)
    f2 = 5.0 + 1.2 * 2 * W
    x = np.sin(2 * np.pi * 5.0 * t) + np.sin(2 * np.pi * f2 * t)
    x = x + rng.normal(0.0, 0.02, N)

    def contrast(spectrum: object) -> float:
        f, a = spectrum.freq, spectrum.amp  # type: ignore[attr-defined]
        trough = a[(f > 5.0) & (f < f2)]
        return float(a.max() / trough.min())

    plain = MultitaperEstimator(time_bandwidth=NW, n_tapers=K).estimate(x, DT)
    quad = QuadraticMultitaperEstimator(time_bandwidth=NW, n_tapers=K).estimate(x, DT)
    assert contrast(quad) > contrast(plain) * 1.2


def test_leaves_a_curvature_free_spectrum_alone() -> None:
    """White noise has no curvature to correct, so the correction must vanish.

    This is the control: an implementation that "sharpened" flat noise would
    be inventing structure, and the peak tests above could not tell the
    difference.
    """
    x = noise(seed=7)
    plain = MultitaperEstimator(time_bandwidth=NW, n_tapers=K).estimate(x, DT)
    quad = QuadraticMultitaperEstimator(time_bandwidth=NW, n_tapers=K).estimate(x, DT)
    band = (2.0, 40.0)
    ratio = np.median(quad.band(*band).amp / plain.band(*band).amp)
    assert ratio == pytest.approx(1.0, abs=0.10)


def brune_realisation(seed: int, fc: float = 4.0, omega: float = 1e-6) -> np.ndarray:
    """A record whose true amplitude spectrum is Brune with corner ``fc``."""
    f = np.fft.rfftfreq(N, DT)
    amp = omega / (1.0 + (f / fc) ** 2)
    amp[0] = 0.0
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, f.size)
    phase[0] = 0.0
    if N % 2 == 0:
        phase[-1] = 0.0
    return np.fft.irfft(amp * np.exp(1j * phase), n=N) * N


@pytest.mark.parametrize("adaptive", [False, True])
def test_adaptive_weights_are_renormalised_before_the_curvature_fit(
    adaptive: bool,
) -> None:
    """Regression test for a scaling bug that looked exactly like a real result.

    ``qiinv`` builds cross-spectra from ``wt * yk`` and never divides by
    ``sum(w**2)``, so its diagonal averages to ``(1/K) sum(w**2 |y|**2)`` where
    the adaptive estimate is ``sum(w**2 |y|**2) / sum(w**2)``. Feeding it raw
    Thomson weights therefore scales the result down by ``sum(w**2)/K``
    wherever the weights bite — which on a Brune spectrum is 0.80 at 10-25 Hz
    and 0.57 at 25-49 Hz.

    The symptom is a smooth droop confined to the falling tail, which reads
    convincingly as a curvature artefact. The tell is that it vanishes entirely
    under ``adaptive=False`` — a genuine property of the quadratic correction
    cannot depend on how the eigencoefficients were weighted going in.

    Hence the parametrisation: the quadratic estimate must track the ordinary
    one in a region of gentle curvature under *either* weighting. Asserting
    only the adaptive case would pass with the bug present.
    """
    plain = MultitaperEstimator(time_bandwidth=NW, n_tapers=K, adaptive=adaptive)
    quad = QuadraticMultitaperEstimator(
        time_bandwidth=NW, n_tapers=K, adaptive=adaptive
    )
    ratios = [
        float(
            np.median(
                quad.estimate(brune_realisation(s), DT).band(25.0, 49.0).amp
                / plain.estimate(brune_realisation(s), DT).band(25.0, 49.0).amp
            )
        )
        for s in range(8)
    ]
    assert float(np.median(ratios)) == pytest.approx(1.0, abs=0.15), (
        f"quadratic/multitaper is {np.median(ratios):.3f} in the tail with "
        f"adaptive={adaptive}; a discrepancy that depends on the weighting is "
        f"a normalisation bug, not a curvature effect"
    )


def test_does_not_bias_a_corner_frequency_low() -> None:
    """It is no worse than the ordinary estimate at the job SpecMod exists for.

    Paired with the weight-normalisation test above: that bug manifests as a
    suppressed high-frequency tail, and a suppressed tail drags a fitted corner
    down. This checks the consequence rather than the mechanism, so the two
    fail independently and a regression is easier to localise.

    A lightly-tapered FFT recovers ``f_c`` better than either multitaper
    variant; see docs/choosing_a_transform.md.
    """

    def fit_fc(spectrum: object) -> float:
        f, a = spectrum.freq, spectrum.amp  # type: ignore[attr-defined]
        m = (f > 0.3) & (f < 45.0)
        f, a = f[m], np.log10(a[m])
        best, best_fc = np.inf, np.nan
        for fc in np.geomspace(1.0, 16.0, 200):
            model = -np.log10(1.0 + (f / fc) ** 2)
            residual = float(np.sum((a - model - np.mean(a - model)) ** 2))
            if residual < best:
                best, best_fc = residual, fc
        return float(best_fc)

    quad = QuadraticMultitaperEstimator(time_bandwidth=NW, n_tapers=K)
    plain = MultitaperEstimator(time_bandwidth=NW, n_tapers=K)
    q = np.median([fit_fc(quad.estimate(brune_realisation(s), DT)) for s in range(12)])
    p = np.median([fit_fc(plain.estimate(brune_realisation(s), DT)) for s in range(12)])

    assert abs(q - 4.0) < 0.5, f"quadratic recovered fc = {q:.3f} against a true 4.0"
    assert abs(q - 4.0) <= abs(p - 4.0) + 0.05, (
        f"quadratic ({q:.3f}) should not be further from the true corner than "
        f"the ordinary estimate ({p:.3f})"
    )


# ---------------------------------------------------------------- robustness


@pytest.mark.parametrize(
    ("name", "record"),
    [
        ("all zeros", np.zeros(N)),
        ("constant offset", np.full(N, 5.0)),
        ("single spike", np.eye(1, N, N // 2)[0] * 1e-6),
        ("denormal amplitudes", np.random.default_rng(0).normal(0.0, 1e-300, N)),
    ],
)
def test_degenerate_records_do_not_produce_nan(name: str, record: np.ndarray) -> None:
    """The correction's damping term is 0/0 for a record with no curvature.

    ``weight = quad**2 / (quad**2 + quad_var)`` divides zero by zero when every
    cross-spectrum is zero, which is what a dead channel or a zero-filled gap
    demeans to. Upstream divides through regardless and returns NaN for the
    whole spectrum — and NaN here propagates silently into a fit.

    The ordinary estimator handles all of these, so the quadratic one has to as
    well or it cannot be a drop-in alternative.
    """
    estimator = QuadraticMultitaperEstimator(time_bandwidth=NW, n_tapers=K)
    spectrum = estimator.estimate(record, DT)
    assert np.isfinite(spectrum.amp).all(), f"{name} produced a non-finite spectrum"
    assert (spectrum.amp >= 0.0).all(), f"{name} produced negative amplitude"


def test_extreme_amplitudes_stay_correct_then_fail_loudly() -> None:
    """Absurd amplitudes must not return quietly wrong numbers.

    At 1e150 the record squares to near the top of float64 and SciPy's ``lstsq``
    overflows computing a residual norm — but ``qiinv`` discards that residual,
    so the estimate itself is unaffected and still agrees with the ordinary
    multitaper estimate. Beyond that ``scipy.optimize.nnls`` refuses a
    non-finite input and raises.

    Correct, then loudly broken, is the acceptable pair. Silently plausible is
    not, and that is what this guards.
    """
    plain = MultitaperEstimator(time_bandwidth=NW, n_tapers=K)
    quad = QuadraticMultitaperEstimator(time_bandwidth=NW, n_tapers=K)

    with warnings.catch_warnings():
        # SciPy's overflow is in an unused residual; see the docstring.
        warnings.simplefilter("ignore", RuntimeWarning)
        x = np.random.default_rng(0).normal(0.0, 1e150, N)
        ratio = float(
            np.median(quad.estimate(x, DT).band(2.0, 40.0).amp)
            / np.median(plain.estimate(x, DT).band(2.0, 40.0).amp)
        )
    assert ratio == pytest.approx(1.0, abs=0.05)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError, match="must not contain infs or NaNs"):
            quad.estimate(np.random.default_rng(0).normal(0.0, 1e160, N), DT)


def test_measurements_are_invariant_to_input_scale() -> None:
    """Every ratio SpecMod documents is dimensionless, so rescaling the record
    must not move it. This is what rules out floating-point range effects as an
    explanation for the taper and curvature biases: they survive 24 orders of
    magnitude unchanged.
    """
    t = np.arange(N) * DT
    ratios = []
    for amplitude in (1e-12, 1e-6, 1.0, 1e6, 1e12):
        x = amplitude * np.sin(2 * np.pi * 5.0 * t)
        x = x + np.random.default_rng(3).normal(0.0, amplitude * 0.02, N)
        spectrum = QuadraticMultitaperEstimator(time_bandwidth=NW, n_tapers=K).estimate(
            x, DT
        )
        ratios.append(float(spectrum.amp.max() / (amplitude * (N * DT) / 2.0)))
    assert max(ratios) - min(ratios) < 1e-9, (
        f"peak recovery varies with input scale ({min(ratios):.6f} to "
        f"{max(ratios):.6f}); a dimensionless ratio that moves with amplitude "
        f"is a numerical-range problem, not a property of the estimator"
    )


# ------------------------------------------------------------ vendored core


def test_matches_prietos_implementation() -> None:
    """Cross-validation against upstream, where upstream can be made to run.

    Prieto's ``qiinv`` raises under numpy 2 for every weighting scheme, so the
    reference is patched in-memory here with exactly the four scalar-indexing
    fixes carried in :mod:`specmod._vendor.qiinv`. Agreeing with it to 1e-9
    says the vendoring changed nothing but the bugs.
    """
    pytest.importorskip("multitaper", reason="optional extra: specmod[multitaper]")

    import pathlib
    import types

    import multitaper.utils as upstream
    from scipy.signal.windows import dpss

    from specmod._vendor.qiinv import qiinv

    src = pathlib.Path(upstream.__file__).read_text()
    for old, new in (
        ("cte2[i]  = np.real(cte_out)", "cte2[i]  = np.real(cte_out)[0]"),
        ("cte[i]   = np.real(hmodel[0])", "cte[i]   = np.real(hmodel[0])[0]"),
        ("slope[i] = -np.real(hmodel[1])", "slope[i] = -np.real(hmodel[1])[0]"),
        ("quad[i]  = np.real(hmodel[2])", "quad[i]  = np.real(hmodel[2])[0]"),
    ):
        assert old in src, f"upstream no longer contains {old!r}; re-check the patch"
        src = src.replace(old, new)
    ref = types.ModuleType("upstream_patched")
    ref.__file__ = upstream.__file__
    exec(compile(src, upstream.__file__, "exec"), ref.__dict__)

    n = 512
    tapers, lamb = dpss(n, NW, K, sym=False, return_ratios=True)
    x = np.random.default_rng(0).normal(0.0, 1.0, n)
    x = x - x.mean()
    yk = np.fft.fft(tapers * x, axis=-1).T
    wt = np.ones_like(yk, dtype=float)
    spec = (np.abs(yk) ** 2).mean(axis=1)[:, None]

    theirs = ref.qiinv(spec, yk, wt, tapers.T, lamb, NW)[0][:, 0]
    ours = qiinv(yk, wt, tapers.T, lamb, NW)[0]
    assert np.abs(ours - theirs).max() / np.abs(theirs).max() < 1e-9
