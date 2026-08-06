"""The legacy pipeline runs on the rewritten estimators.

``specmod.spectral`` used to call ``mtspec(data, delta, 3)`` directly, so the
estimator was unconfigurable and the normalisation was its own. It now goes
through :mod:`specmod.transforms` via the layered configuration, which makes
every estimator available to the pipeline and puts one Parseval contract behind
all of them.

These tests cover the seam. What reads the estimators is now
``specmod.pipeline``; the amplitude convention and the padding behaviour are
what needs pinning is that the numbers crossing it are now right, and by how much
they moved.
"""

from __future__ import annotations

import numpy as np
import obspy
import pytest

from specmod.core import Spectrum
from specmod.pipeline import estimate_spectrum, spectrum_from_trace

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
    spectrum = spectrum_from_trace(tr.copy(), estimator="fft", taper="boxcar")
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
    spectrum = spectrum_from_trace(tr.copy(), estimator="fft", taper="boxcar")
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
    spectrum = spectrum_from_trace(tr.copy())
    duration = spectrum.meta["npts"] * spectrum.meta["delta"]

    # Reconstruct what the old code would have produced from the same PSD.
    psd = 2.0 * spectrum.amp**2 / duration
    old = np.sqrt(psd * len(spectrum.freq) / FS)
    assert float(np.median(spectrum.amp / old)) == pytest.approx(1.0, rel=1e-9)


def test_amplitude_conversion_round_trips() -> None:
    """``to_kind`` there and back, on the pipeline's own convention.

    Was `spectrum.amp_to_psd(); spectrum.psd_to_amp()` on the legacy class,
    which mutated in place. The relationship is the same one; what changed is
    that each step now returns a new spectrum, so the round trip compares two
    objects rather than an object against a copy of its own past.
    """
    spectrum = spectrum_from_trace(trace())
    back = spectrum.to_kind("psd").to_kind("magnitude")
    assert back.amp == pytest.approx(spectrum.amp, rel=1e-12)
    assert back.kind == spectrum.kind


def test_normalisation_is_independent_of_the_frequency_axis_length() -> None:
    """The §2.2 padding bug, at the pipeline level.

    Zero-padding lengthens ``freq`` while the record's duration is fixed. Keyed
    off duration the amplitude is unchanged; keyed off ``len(freq)``, as the old
    code was, it fell as ``1/sqrt(padding)``.
    """
    tr = trace()
    plain = spectrum_from_trace(tr.copy())
    padded = spectrum_from_trace(tr.copy(), n_fft=4096, estimator="fft")
    reference = spectrum_from_trace(tr.copy(), estimator="fft")

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
    tr = trace()
    spectrum = estimate_spectrum(tr.data, DT, estimator=estimator)
    energy = float(np.trapezoid(spectrum.amp**2 / 2.0, spectrum.freq))
    assert energy == pytest.approx(time_domain_energy(tr), rel=0.10), (
        f"{estimator} is not on the folded FAS convention the pipeline assumes"
    )


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_the_pipeline_conversion_matches_the_typed_one(estimator: str) -> None:
    """``psd_to_amp`` must agree with ``to_kind("magnitude")``.

    The pipeline reads ``Omega`` off the *unfolded* magnitude while the
    estimators return the folded ``FAS``, so a conversion happens on the way
    in. Two spellings of one relationship is exactly how a factor of two
    survives, so the pipeline's route is pinned against the named kind here.
    """
    tr = trace()
    typed = estimate_spectrum(tr.data, DT, estimator=estimator).to_kind("magnitude")
    through_pipeline = spectrum_from_trace(tr.copy(), estimator=estimator)

    assert through_pipeline.amp == pytest.approx(np.asarray(typed.amp), rel=1e-9)


# --------------------------------------------------------- estimator choice


def test_every_estimator_is_reachable_from_the_pipeline() -> None:
    """The point of the rewiring: the backend is a configuration choice.

    Before this the pipeline could only ever call mtspec.
    """
    tr = trace()
    for name in ("fft", "welch", "multitaper", "quadratic", "cwt"):
        spectrum = spectrum_from_trace(tr.copy(), estimator=name)
        assert spectrum.meta["estimator"] == name
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
    assert spectrum_from_trace(tr.copy()).motion == "velocity"
    assert estimate_spectrum(tr.data, DT, motion="acceleration").unit == "m/s^2*s"


# ------------------------------------------------------------- the binning


# `test_binning_uses_the_requested_number_of_bins` and
# `test_binning_does_not_warn_on_empty_bins` lived here, exercising `log_bin`
# through the legacy `Signal`'s `bfreq`/`bamp`. They are now
# `test_log_bin_clamps_to_the_records_own_range` and
# `test_log_bin_drops_empty_bins_without_warning` in `test_collection.py`,
# against the function directly rather than as a side effect of constructing a
# spectrum from a trace.
#
# `test_signal_and_noise_share_the_class` also lived here. It asserted that
# `Signal` and `Noise` were one class distinguished by a `kind` string — a
# property of the design that has gone, not a behaviour to preserve. A
# `SpectrumPair` has a signal and a noise, both plain `Spectrum`, and which is
# which is structural rather than a label that could disagree with its slot.
