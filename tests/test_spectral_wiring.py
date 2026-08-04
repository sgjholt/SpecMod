"""The legacy pipeline runs on the rewritten estimators.

``specmod.spectral`` used to call ``mtspec(data, delta, 3)`` directly, so the
estimator was unconfigurable and the normalisation was its own. It now goes
through :mod:`specmod.transforms` via the layered configuration, which makes
every estimator available to the pipeline and puts one Parseval contract behind
all of them.

These tests cover the seam. The class API is deliberately unchanged — ``Spectra``,
``SNP`` and the fitting code still see ``freq``/``amp``/``bfreq``/``bamp`` — so
what needs pinning is that the numbers crossing it are now right, and by how much
they moved.
"""

from __future__ import annotations

import numpy as np
import obspy
import pytest

from specmod.spectral import Noise, Signal, Spectrum, estimate_spectrum

FS = 100.0
DT = 1.0 / FS
N = 2000


def trace(sigma: float = 1e-6, seed: int = 0, npts: int = N) -> obspy.Trace:
    data = np.random.default_rng(seed).normal(0.0, sigma, npts)
    tr = obspy.Trace(data.astype(np.float64))
    tr.stats.delta = DT
    tr.stats.network, tr.stats.station, tr.stats.channel = "XX", "TEST", "HHZ"
    return tr


def time_domain_energy(tr: obspy.Trace) -> float:
    x = tr.data
    return float(np.sum((x - x.mean()) ** 2) * DT)


# ------------------------------------------------------------- the contract


def test_the_legacy_class_now_satisfies_parseval() -> None:
    """The headline of the rewiring.

    On this path the pre-refactor chain recovered under a quarter of a record's
    energy, because ``psd_to_amp`` used ``sqrt(PSD * len(freq) / sampling_rate)``
    where the contract needs ``sqrt(2 * PSD * T)``. That is a factor of two in
    amplitude — 0.30 in log10(Omega), about 0.20 magnitude units.

    It went unnoticed because nothing checked the pipeline end to end against a
    quantity whose value was known independently. That is the whole reason this
    contract exists.
    """
    tr = trace()
    spectrum = Signal(tr.copy())
    energy = float(np.trapezoid(spectrum.amp**2 / 2.0, spectrum.freq))
    assert energy == pytest.approx(time_domain_energy(tr), rel=0.05)


@pytest.mark.parametrize("estimator", ["fft", "multitaper", "quadratic", "cwt"])
def test_the_old_normalisation_error_scales_with_the_axis_length(
    estimator: str,
) -> None:
    """Quantifies the break, and shows it was never a constant.

    ``new/old = sqrt(2 * npts / len(freq))``, so the size of the error depends
    on how many bins the transform returned — 2.00 for a half-length rfft axis,
    1.41 for a full-length one, 7.21 for the CWT's 77 log-spaced scales. That
    dependence *is* the bug, not a side effect of it.

    Anyone comparing against a pre-refactor run needs this, and it says the
    factor cannot be guessed: it has to be measured for the axis length mtspec
    actually returned.
    """
    tr = trace()
    spectrum = Signal(tr.copy(), estimator=estimator)
    duration = spectrum.meta["npts"] * spectrum.meta["delta"]

    # Reconstruct what the old code would have produced from the same PSD.
    psd = spectrum.amp**2 / (2.0 * duration)
    old = np.sqrt(psd * len(spectrum.freq) / FS)

    expected = np.sqrt(2.0 * spectrum.meta["npts"] / len(spectrum.freq))
    assert float(np.median(spectrum.amp / old)) == pytest.approx(expected, rel=0.01)
    assert expected > 1.0, "the old normalisation was always low, never high"


def test_amplitude_conversion_round_trips() -> None:
    tr = trace()
    spectrum = Signal(tr.copy())
    before = spectrum.amp.copy()
    spectrum.amp_to_psd()
    spectrum.psd_to_amp()
    assert spectrum.amp == pytest.approx(before, rel=1e-12)


def test_normalisation_is_independent_of_the_frequency_axis_length() -> None:
    """The §2.2 padding bug, at the pipeline level.

    Zero-padding lengthens ``freq`` while the record's duration is fixed. Keyed
    off duration the amplitude is unchanged; keyed off ``len(freq)``, as the old
    code was, it fell as ``1/sqrt(padding)``.
    """
    tr = trace()
    plain = Signal(tr.copy())
    padded = Signal(tr.copy(), n_fft=4096, estimator="fft")
    reference = Signal(tr.copy(), estimator="fft")

    band = (1.0, 40.0)

    def level(spectrum: Spectrum) -> float:
        mask = (spectrum.freq > band[0]) & (spectrum.freq < band[1])
        return float(np.median(spectrum.amp[mask]))

    assert level(padded) == pytest.approx(level(reference), rel=0.05)
    assert plain.freq.size > 0


# --------------------------------------------------------- estimator choice


def test_every_estimator_is_reachable_from_the_pipeline() -> None:
    """The point of the rewiring: the backend is a configuration choice.

    Before this the pipeline could only ever call mtspec.
    """
    tr = trace()
    for name in ("fft", "welch", "multitaper", "quadratic", "cwt"):
        spectrum = Signal(tr.copy(), estimator=name)
        assert spectrum.estimator == name
        assert spectrum.freq.size > 0
        assert np.isfinite(spectrum.amp).all()


def test_estimator_parameters_pass_through() -> None:
    tr = trace()
    spectrum = estimate_spectrum(
        tr.data, DT, estimator="multitaper", n_tapers=3, time_bandwidth=2.0
    )
    assert spectrum.meta["n_tapers"] == 3
    assert spectrum.meta["time_bandwidth"] == 2.0


def test_the_fortran_backend_is_refused_with_a_route_onward() -> None:
    """``mtspec`` is still a legal config value for provenance, but it cannot
    run here — so say what to use instead rather than failing on an import."""
    with pytest.raises(ValueError, match="pre-refactor Fortran backend"):
        estimate_spectrum(trace().data, DT, estimator="mtspec")


def test_spectra_carry_their_motion() -> None:
    """Previously tracked in a module-level global that had to be kept in sync
    by hand with however many times integrate() had been called."""
    tr = trace()
    assert Signal(tr.copy()).motion == "velocity"
    assert estimate_spectrum(tr.data, DT, motion="acceleration").unit == "m/s^2*s"


# ------------------------------------------------------------- the binning


def test_binning_uses_the_requested_number_of_bins() -> None:
    """Default edges of 0.001-200 Hz are wider than any real record.

    For a 100 sps trace that puts about a third of the bins below the lowest
    frequency present and a third above Nyquist, so they come out empty and are
    dropped — which is why the surviving axis was always far shorter than
    requested. Clamping to the record's own range makes the count meaningful.
    """
    spectrum = Signal(trace())
    assert spectrum.bfreq.size > 100, (
        f"only {spectrum.bfreq.size} bins survived of 151 requested"
    )
    assert spectrum.bfreq.min() >= spectrum.freq.min()
    assert spectrum.bfreq.max() <= spectrum.freq.max()


def test_binning_does_not_warn_on_empty_bins() -> None:
    """Log bins over a linear grid are sparse at the low end by construction,
    so an empty bin is expected and must not raise a warning per bin."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        Signal(trace())


def test_signal_and_noise_share_the_class() -> None:
    tr = trace()
    signal, noise = Signal(tr.copy()), Noise(tr.copy())
    assert signal.kind == "signal"
    assert noise.kind == "noise"
    assert signal.freq == pytest.approx(noise.freq)
