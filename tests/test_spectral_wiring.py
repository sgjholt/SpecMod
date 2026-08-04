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

try:  # `prieto` is behind the specmod[multitaper] extra, which CI does not install.
    import multitaper as _multitaper  # noqa: F401

    _HAS_MULTITAPER = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_MULTITAPER = False

#: Every registered estimator, with the optional one marked. Parametrise from
#: this rather than writing the list out: a hardcoded list that happens to
#: include ``prieto`` passes here and fails on a default install, which has now
#: happened twice.
ALL_ESTIMATORS = [
    "fft",
    "welch",
    "multitaper",
    pytest.param(
        "prieto",
        marks=pytest.mark.skipif(
            not _HAS_MULTITAPER, reason="optional extra: specmod[multitaper]"
        ),
    ),
    "quadratic",
    "cwt",
]


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


def test_the_pipeline_amplitude_is_the_unfolded_transform_magnitude() -> None:
    """``Omega`` is defined in this convention, so it is the one to hold fixed.

    ``|X(f)| = |rfft(x)| * dt``. The long-period plateau of the displacement
    spectrum is ``|X(f -> 0)| = |integral u dt|`` and ``M0`` is proportional to
    it, so a factor of two here is 0.2 magnitude units on every event.

    Note this is *not* the convention :class:`specmod.core.Spectrum` carries.
    That one is folded — ``2|X|`` — so its ``energy()`` integrates over
    non-negative frequencies alone. Both are self-consistent; the factor is
    removed on the way into this module.
    """
    tr = trace()
    spectrum = Signal(tr.copy(), estimator="fft", taper="boxcar")
    reference = np.abs(np.fft.rfft(tr.data - tr.data.mean())) * DT
    freq = np.fft.rfftfreq(len(tr.data), DT)

    band = (spectrum.freq > 0.5) & (spectrum.freq < 40.0)
    expected = np.interp(spectrum.freq[band], freq, reference)
    assert float(np.median(spectrum.amp[band] / expected)) == pytest.approx(
        1.0, rel=1e-6
    )


def test_energy_is_recovered_in_the_unfolded_convention() -> None:
    """Parseval for ``|X|`` is ``E = 2 * integral |X|**2 df`` over ``f >= 0``.

    The factor of two is the negative-frequency half, which an unfolded
    one-sided spectrum does not carry.
    """
    tr = trace()
    spectrum = Signal(tr.copy(), estimator="fft", taper="boxcar")
    energy = 2.0 * float(np.trapezoid(spectrum.amp**2, spectrum.freq))
    assert energy == pytest.approx(time_domain_energy(tr), rel=0.05)


def test_unpadded_amplitudes_reproduce_the_pre_refactor_run() -> None:
    """The rewiring must not move anyone's numbers on the default path.

    The old ``sqrt(PSD * len(freq) / sampling_rate)`` computes the same
    quantity as ``sqrt(PSD * T / 2)`` whenever ``len(freq) * dt == T/2``, which
    is every unpadded one-sided transform. So this is not a break: it is the
    same convention, keyed off something that does not move.
    """
    tr = trace()
    spectrum = Signal(tr.copy())
    duration = spectrum.meta["npts"] * spectrum.meta["delta"]

    # Reconstruct what the old code would have produced from the same PSD.
    psd = 2.0 * spectrum.amp**2 / duration
    old = np.sqrt(psd * len(spectrum.freq) / FS)
    assert float(np.median(spectrum.amp / old)) == pytest.approx(1.0, rel=1e-9)


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

    # And the size of what was wrong. The old formula keyed off len(freq), so
    # it drifts from the duration-keyed one by sqrt(2 * len(freq) * dt / T) —
    # here 2049 bins against a 20 s record, so about 1.43x.
    duration = padded.meta["npts"] * padded.meta["delta"]
    old_padded = np.sqrt((2.0 * padded.amp**2 / duration) * len(padded.freq) / FS)
    mask = (padded.freq > band[0]) & (padded.freq < band[1])

    expected = np.sqrt(duration / (2.0 * len(padded.freq) / FS))
    assert float(np.median(padded.amp[mask] / old_padded[mask])) == pytest.approx(
        expected, rel=1e-6
    )
    assert expected < 0.8, (
        "padding should visibly move the old formula, or this proves nothing"
    )


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_every_estimator_lands_on_the_same_amplitude_convention(
    estimator: str,
) -> None:
    """The pairing this module depends on, checked per method rather than assumed.

    ``psd_to_amp`` applies one conversion to whatever the configured estimator
    produced. That is only sound if every estimator agrees on what its
    amplitudes mean — and they reach it by very different routes: the CWT via
    ``C_delta`` and a scale integral, the multitaper family via taper
    normalisation, Welch via segment averaging.

    They do agree, because each is held to ``E = integral(FAS**2 / 2) df``. This
    asserts that directly, so a new estimator arriving on a different convention
    fails here instead of silently rescaling ``Omega``.
    """
    from specmod.spectral import estimate_spectrum as estimate

    tr = trace()
    spectrum = estimate(tr.data, DT, estimator=estimator)
    energy = float(np.trapezoid(spectrum.amp**2 / 2.0, spectrum.freq))
    assert energy == pytest.approx(time_domain_energy(tr), rel=0.10), (
        f"{estimator} is not on the folded FAS convention the pipeline assumes"
    )


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_the_pipeline_conversion_matches_the_typed_one(estimator: str) -> None:
    """``psd_to_amp`` must agree with ``to_kind("magnitude")``.

    The legacy class does the PSD-to-amplitude step arithmetically, because it
    keeps the pre-refactor call sequence. The core carries the same conversion
    as a named kind. Two implementations of one relationship is exactly how a
    factor of two survives, so they are pinned against each other here.
    """
    from specmod.spectral import estimate_spectrum as estimate

    tr = trace()
    typed = estimate(tr.data, DT, estimator=estimator).to_kind("magnitude")
    through_pipeline = Signal(tr.copy(), estimator=estimator)

    assert through_pipeline.amp == pytest.approx(np.asarray(typed.amp), rel=1e-9)


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
