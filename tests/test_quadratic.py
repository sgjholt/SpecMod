"""Tests for the quadratic (curvature-corrected) multitaper estimator.

The numerical core is vendored from Prieto's ``multitaper`` (see
:mod:`specmod._vendor.qiinv`), so these tests do two distinct jobs: hold the
vendored code to SpecMod's own contracts, and pin the behaviour that justifies
having it at all. The second matters because vendored code has no upstream
test suite to fall back on — four latent numpy-2 bugs in one function is what
prompted vendoring in the first place.
"""

from __future__ import annotations

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


def test_droops_a_steeply_falling_tail() -> None:
    """A documented limitation, pinned so it cannot silently change.

    The correction models the spectrum as quadratic in *linear* frequency
    across the inner band. A source spectrum falling as ``f**-2`` is poorly
    described that way, and the estimate is pulled low in the far tail — about
    20% by 25-49 Hz on the synthetic Brune below. That biases a fitted corner
    frequency *low*, which is why this estimator is not the default for source
    fitting. See docs/choosing_a_transform.md.
    """
    fc, omega = 4.0, 1e-6
    f = np.fft.rfftfreq(N, DT)
    amp = omega / (1.0 + (f / fc) ** 2)
    amp[0] = 0.0

    ratios = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        phase = rng.uniform(-np.pi, np.pi, f.size)
        phase[0] = 0.0
        if N % 2 == 0:
            phase[-1] = 0.0
        x = np.fft.irfft(amp * np.exp(1j * phase), n=N) * N

        plain = MultitaperEstimator(time_bandwidth=NW, n_tapers=K).estimate(x, DT)
        quad = QuadraticMultitaperEstimator(time_bandwidth=NW, n_tapers=K).estimate(
            x, DT
        )
        tail = (25.0, 49.0)
        ratios.append(float(np.median(quad.band(*tail).amp / plain.band(*tail).amp)))

    assert np.median(ratios) < 0.95, (
        "the tail droop has gone; if this is a deliberate improvement, update "
        "docs/choosing_a_transform.md and tests/test_quadratic.py together"
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
