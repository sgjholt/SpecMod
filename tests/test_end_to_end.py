"""Tier 2: recover known source parameters through the whole pipeline.

A Brune spectrum with chosen ``(Omega, fc, t*)`` is turned into a seismogram,
buried in noise, and put through cutting, transforming, bandwidth selection,
binning, fitting and the moment calculation. The assertions are on how close
the recovered parameters come back.

This is the only test that can catch an answer that has been *systematically
wrong since the snapshot was taken*. The golden references pin that numbers do
not move; they say nothing about whether the numbers are right. See §5, tier 2.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

import specmod.preprocess as pre  # noqa: E402
from specmod.magnitude import moment_magnitude, seismic_moment  # noqa: E402
from specmod.pipeline import spectrum_set_from_streams  # noqa: E402
from specmod.sources import build_model  # noqa: E402

#: The event to recover. Chosen so the corner sits well inside the resolvable
#: band at 200 Hz: low enough that the plateau has room below it, high enough
#: that the f^-2 limb is not cut off by the anti-alias edge.
TRUE_OMEGA = 3.0e-7  # m s, the long-period displacement plateau
TRUE_FC = 8.0  # Hz
TRUE_TSTAR = 0.02  # s

SAMPLING_RATE = 200.0
DT = 1.0 / SAMPLING_RATE
WINDOW_S = 20.0
NOISE_PAD_S = 40.0

#: Source-to-site distance the moment is computed at. Only the round trip is
#: asserted, so the value matters only in that both directions use it.
DISTANCE_KM = 10.0

#: Noise as a fraction of the signal's RMS. Loud enough that the bandwidth
#: selector has something to do, quiet enough to leave a wide usable band.
NOISE_FRACTION = 0.02


def _brune_displacement_fas(freq: np.ndarray) -> np.ndarray:
    """The target one-sided displacement amplitude spectrum, in m s."""
    model = build_model(source="brune", motion="displacement")
    log10_amp = model.evaluate(freq, np.log10(TRUE_OMEGA), TRUE_FC, TRUE_TSTAR)
    return np.asarray(10.0**log10_amp)


def _synthetic_velocity(seed: int = 11) -> np.ndarray:
    """A velocity record whose displacement spectrum is the Brune target.

    Built by inverting the pipeline's own convention, which is the *unfolded*
    magnitude ``|X(f)| = |rfft(x)| * dt`` — the estimators return the folded
    ``2|X|`` and ``spectrum_from_trace`` halves it, because ``Omega`` is
    defined on the unfolded spectrum. So a record whose measured spectrum is
    ``A`` has rfft magnitudes ``A / dt``, with no factor of two.

    Phases are random, which makes the record stationary noise
    with the right spectral shape rather than a physical pulse — that is what
    is wanted here, since the pipeline's window refinement should then trim
    almost nothing and the spectrum it measures is the one that was put in.

    Velocity is synthesised directly as ``2 pi f`` times the displacement
    target rather than by differentiating a displacement record, so no
    numerical derivative sits between the target and the measurement.
    """
    n = round(WINDOW_S * SAMPLING_RATE)
    freq = np.fft.rfftfreq(n, d=DT)

    amplitude = np.zeros_like(freq)
    amplitude[1:] = 2.0 * np.pi * freq[1:] * _brune_displacement_fas(freq[1:])

    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=freq.size)
    spectrum = (amplitude / DT) * np.exp(1j * phase)
    spectrum[0] = 0.0  # no DC; every estimator drops it anyway
    if n % 2 == 0:
        spectrum[-1] = np.abs(spectrum[-1])  # Nyquist must be real

    return np.asarray(np.fft.irfft(spectrum, n=n), dtype=np.float64)


def _trace(station: str, seed: int) -> Any:
    """Noise, then the synthetic, with picks placed on the join."""
    signal = _synthetic_velocity(seed=seed)
    pad = round(NOISE_PAD_S * SAMPLING_RATE)

    rng = np.random.default_rng(seed + 1000)
    noise_level = NOISE_FRACTION * float(np.sqrt(np.mean(signal**2)))
    data = np.concatenate(
        [
            rng.normal(0.0, noise_level, size=pad),
            signal + rng.normal(0.0, noise_level, size=signal.size),
        ]
    )

    trace = obspy.Trace(
        data,
        header={
            "sampling_rate": SAMPLING_RATE,
            "network": "XX",
            "station": station,
            "location": "",
            "channel": "HHN",
        },
    )
    start = trace.stats.starttime
    trace.stats["otime"] = start
    # `s_window` with rafp=0 opens the window at the P pick, so putting P on the
    # first sample of the synthetic makes the cut window the synthetic exactly.
    trace.stats["p_time"] = start + NOISE_PAD_S
    trace.stats["s_time"] = start + NOISE_PAD_S + 1.0
    trace.stats["repi"] = DISTANCE_KM
    trace.stats["rhyp"] = DISTANCE_KM
    return trace


@pytest.fixture(scope="module")
def measured() -> Any:
    """Three synthetic stations, cut and transformed."""
    stream = obspy.Stream([_trace(f"S{i:02d}", seed=11 + i) for i in range(3)])
    signal = pre.s_window(
        stream, rafp=0.0, tafs=WINDOW_S, time_after="absolute_time", refine_window=True
    )
    noise = pre.get_noise_p(stream, signal)
    return spectrum_set_from_streams(signal, noise, estimator="fft")


@pytest.fixture(scope="module")
def recovered(measured: Any) -> Any:
    """The fit table.

    Fitted in the motion the sensor recorded — velocity. The model carries a
    motion factor, so ``llpsp`` is the displacement plateau whichever domain is
    fitted, but converting first is not a neutral change of view: integrating
    to displacement implicitly low-passes, and differentiating to acceleration
    amplifies high-frequency noise.

    Velocity is also the convenient domain: it peaks at ``fc``, so
    ``initial_guess`` taking the spectral peak is exact rather than
    approximate. That holds for any omega-squared source — both registered
    models — since the stationary point of ``f * [1 + (f/fc)**(g*n)]**(-1/g)``
    sits at ``f = fc`` whenever ``n == 2``, whatever the corner sharpness
    ``g``. In displacement the spectrum is monotonic, so the peak is whichever
    band edge it was handed.
    """
    from specmod.fitting import FitSpectra  # noqa: PLC0415

    fits = FitSpectra(measured)
    fits.fit_spectra()
    return fits.table


class TestParameterRecovery:
    def test_every_station_was_fitted(self, recovered: Any) -> None:
        assert len(recovered) == 3
        assert recovered["pass_fitting"].all()

    def test_the_corner_frequency_comes_back(self, recovered: Any) -> None:
        # The shape parameter the whole exercise turns on. A mis-scaled
        # frequency axis, or a fold applied twice, moves this.
        assert recovered["fc"].to_numpy() == pytest.approx(TRUE_FC, rel=0.10)

    def test_the_plateau_comes_back(self, recovered: Any) -> None:
        # `llpsp` is log10 of the plateau, so this is the amplitude convention
        # end to end: the `dt` normalisation, the unfolding in
        # `spectrum_from_trace`, the taper correction and the model's motion
        # factor all have to be right together. Getting the fold wrong alone
        # costs 0.3 here, six times this tolerance.
        assert recovered["llpsp"].to_numpy() == pytest.approx(
            np.log10(TRUE_OMEGA), abs=0.08
        )

    def test_attenuation_comes_back(self, recovered: Any) -> None:
        assert recovered["ts"].to_numpy() == pytest.approx(TRUE_TSTAR, abs=0.005)

    def test_the_plateau_is_biased_low_and_by_how_much(self, recovered: Any) -> None:
        """The residual is a known bias, not scatter — so it is stated, not hidden.

        Window refinement trims to the 1st and 99th percentiles of cumulative
        energy, which costs ~2% of the record's duration, and amplitude scales
        with duration for a stationary record. Measured against the target
        spectrum directly the loss is ~5%; through the fit it is ~0.05 in
        log10, or **0.03 in Mw**. Small, one-directional, and worth knowing
        about before it is mistaken for a calibration error.
        """
        residual = recovered["llpsp"].to_numpy() - np.log10(TRUE_OMEGA)
        assert np.all(residual < 0.0)
        assert np.abs(residual).max() < 0.08

    def test_the_stations_agree_with_each_other(self, recovered: Any) -> None:
        """Different noise, same event: what is left is scatter, not bias."""
        assert recovered["fc"].std() < 0.5
        assert recovered["llpsp"].std() < 0.02


class TestMomentRecovery:
    """The fit's plateau, carried through to a moment and back."""

    def test_the_moment_survives_the_round_trip(self, recovered: Any) -> None:
        expected = seismic_moment([TRUE_OMEGA], [DISTANCE_KM])[0]
        got = seismic_moment(
            10.0 ** recovered["llpsp"].to_numpy(),
            np.full(len(recovered), DISTANCE_KM),
        )
        # A moment is only ever quoted to a fraction of a magnitude unit, and
        # 10% of M0 is 0.03 in Mw.
        assert got == pytest.approx(expected, rel=0.25)

    def test_the_magnitude_survives_the_round_trip(self, recovered: Any) -> None:
        expected = moment_magnitude(seismic_moment([TRUE_OMEGA], [DISTANCE_KM]))[0]
        got = moment_magnitude(
            seismic_moment(
                10.0 ** recovered["llpsp"].to_numpy(),
                np.full(len(recovered), DISTANCE_KM),
            )
        )
        assert got == pytest.approx(expected, abs=0.06)


class TestMotionConversion:
    def test_displacement_flattens_towards_the_plateau(self, measured: Any) -> None:
        """`to_motion` is exercised even though the fit does not need it."""
        pair = next(iter(measured.to_motion("displacement").pairs.values()))
        freq, amp = pair.signal.freq, pair.signal.amp
        low = (freq > 0.3) & (freq < 2.0)
        # Below the corner the displacement spectrum is flat at Omega, less the
        # same ~5% the refinement costs.
        assert np.median(amp[low]) == pytest.approx(TRUE_OMEGA, rel=0.15)


class TestTheSyntheticIsWhatItClaims:
    """Guards on the fixture itself, so a failure above is not this test's bug."""

    def test_the_record_has_the_target_spectrum(self) -> None:
        # Measured with a plain rfft rather than through specmod, so the
        # construction is checked independently of what it is used to check.
        record = _synthetic_velocity()
        freq = np.fft.rfftfreq(record.size, d=DT)
        fas = np.abs(np.fft.rfft(record)) * DT  # unfolded, as the pipeline uses

        band = (freq > 0.5) & (freq < 60.0)
        target = 2.0 * np.pi * freq[band] * _brune_displacement_fas(freq[band])
        assert fas[band] == pytest.approx(target, rel=1e-9)

    def test_the_target_is_flat_below_the_corner(self) -> None:
        # With attenuation divided out, as below: `exp(-pi f t*)` is still
        # 0.6% off unity at 0.1 Hz, so the plateau is not Omega exactly.
        freq = np.array([0.1, 0.2, 0.4])
        amp = _brune_displacement_fas(freq) * np.exp(np.pi * freq * TRUE_TSTAR)
        assert amp == pytest.approx(TRUE_OMEGA, rel=0.005)

    def test_the_target_falls_as_f_squared_above_the_corner(self) -> None:
        # Checked with attenuation divided out, which is what leaves f^-2.
        freq = np.array([40.0, 80.0])
        amp = _brune_displacement_fas(freq) * np.exp(np.pi * freq * TRUE_TSTAR)
        assert amp[0] / amp[1] == pytest.approx(4.0, rel=0.05)


class TestFittingTheWrongMotionWarns:
    """The one silent failure this exercise turned up, now audible.

    `FitSpectra(spectra.to_motion("displacement"))` is a natural thing to
    write and used to return `fc` 1.6 against a true 8.0 with nothing said.
    """

    def test_a_displacement_spectrum_warns(self, measured: Any) -> None:
        from specmod.fitting import initial_guess  # noqa: PLC0415

        with pytest.warns(UserWarning, match="the corner only in velocity"):
            initial_guess(measured.to_motion("displacement"))

    def test_the_warning_names_the_station_and_the_motion(self, measured: Any) -> None:
        from specmod.fitting import initial_guess  # noqa: PLC0415

        # One warning per station, not one for the set — otherwise a run over
        # a mixed collection names whichever spectrum happened to be first.
        # Asserting on all three also stops the two that `match=` does not
        # select being re-emitted into pytest's warning summary.
        with pytest.warns(UserWarning, match="is in displacement") as records:
            initial_guess(measured.to_motion("displacement"))

        messages = [str(record.message) for record in records]
        assert len(messages) == 3
        for station in ("XX.S00..HHN", "XX.S01..HHN", "XX.S02..HHN"):
            assert any(
                f"{station} is in displacement" in message for message in messages
            ), f"nothing warned about {station}"

    def test_velocity_is_silent(self, measured: Any) -> None:
        from specmod.fitting import initial_guess  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            guesses = initial_guess(measured)
        assert len(guesses) == 3

    def test_the_guess_it_warns_about_is_the_one_that_ruins_the_fit(
        self, measured: Any
    ) -> None:
        """Why it is worth a warning: the guess lands on the low band edge.

        The contrast is the point, not the precision. Peak-equals-corner is
        exact for the noiseless model; on a measured spectrum the argmax
        wanders, so velocity gives a guess in the right neighbourhood — a
        starting point, which is all it has to be. Displacement gives one an
        order of magnitude out, and the fit does not recover from it.
        """
        from specmod.fitting import initial_guess  # noqa: PLC0415

        with pytest.warns(UserWarning, match="the corner only in velocity"):
            displacement = initial_guess(measured.to_motion("displacement"))
        velocity = initial_guess(measured)

        for id, guess in displacement.items():
            # Measured on these three stations: displacement guesses 0.71,
            # 2.12 and 1.02 against velocity's 6.47, 7.83 and 5.72, for a true
            # 8.0. Bounding each side is the contrast; a ratio between them
            # would only be a brittle way of saying the same.
            assert guess["fc"] < TRUE_FC / 3.0
            assert velocity[id]["fc"] == pytest.approx(TRUE_FC, rel=0.35)
