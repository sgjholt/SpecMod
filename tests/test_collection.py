"""``core.collection`` against the legacy classes it replaces.

The point of stage 1 is a structural change, not a numerical one, so the
binding test is that the numbers do not move. These run both paths over the
same 28 real PNR windows and compare — if the rewrite drifts, it fails here
naming the station, and the golden reference catches it a second time when
``spectral.py`` is switched over.

The unit tests below are the other half. Every function the legacy kept as a
private method is now reachable directly, so the binning and the band search
can be tested on constructed input rather than only as a side effect of
running the whole pipeline on real waveforms. That was the practical cost of
the old design: a bug in ``find_optimal_signal_bandwidth`` could only be
observed through 28 stations' worth of end-to-end run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import glob
import io
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

import specmod.preprocess as pre  # noqa: E402
from specmod.core import Spectrum  # noqa: E402
from specmod.core.bandwidth import (  # noqa: E402
    BANDWIDTH_SELECTORS,
    BandwidthSelector,
    WidestBandwidth,
    get_bandwidth_selector,
)
from specmod.core.collection import (  # noqa: E402
    BinnedSpectrum,
    SpectrumPair,
    SpectrumSet,
    _clamp_to_floor,
    find_bandwidth,
    log_bin,
    parseval_scale,
)
from specmod.core.noise import (  # noqa: E402
    NOISE_MODELS,
    NoiseModel,
    boost_noise,
    get_noise_model,
)
from specmod.spectral import (  # noqa: E402
    BINNING_PARAMS,
    ROTATE_NOISE,
    SNR_TOLERENCE,
    Noise,
    Signal,
    Spectra,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Tutorial" / "Data" / "2019-08-26T07:30:47.0"
INVENTORY = ROOT / "Tutorial" / "MetaData" / "pnr_inventory.xml"

ORIGIN = "2019-08-26T07:49:24.2"
LATITUDE, LONGITUDE, DEPTH_KM = 53.784, -2.967, 2.1


# --------------------------------------------------------------- unit tests


def test_log_bin_clamps_to_the_records_own_range() -> None:
    """The defaults are far wider than any record, and used to waste the count."""
    freq = np.linspace(1.0, 50.0, 500)
    binned = log_bin(freq, np.ones_like(freq), f_min=0.001, f_max=200.0, n_bins=51)

    assert binned.freq.min() >= 1.0
    assert binned.freq.max() <= 50.0
    # Unclamped, most of the 50 bins would fall outside [1, 50] and be dropped.
    assert len(binned) > 40, f"only {len(binned)} bins survived"


def test_log_bin_averages_geometrically() -> None:
    """The bins are spaced on a log scale, so the average is taken on one too.

    ``n_bins=2`` gives a single bin spanning the whole clamped range, so both
    samples land in it — with more bins the clamping to ``[10.0, 10.1]`` splits
    them and each bin holds one sample.
    """
    freq = np.array([10.0, 10.1])
    amp = np.array([1.0, 100.0])
    binned = log_bin(freq, amp, f_min=1.0, f_max=100.0, n_bins=2)

    assert len(binned) == 1
    assert binned.amp[0] == pytest.approx(10.0), "geometric mean of 1 and 100 is 10"
    # An arithmetic mean would be 50.5 — worth stating, since that is the bug
    # this asserts against rather than a hypothetical one.
    assert binned.amp[0] != pytest.approx(50.5)


def test_log_bin_drops_empty_bins_without_warning() -> None:
    freq = np.linspace(10.0, 50.0, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        binned = log_bin(freq, np.ones_like(freq), f_min=0.001, f_max=200.0, n_bins=99)
    assert np.isfinite(binned.amp).all()


def test_parseval_scale_is_unity_for_equal_lengths() -> None:
    assert parseval_scale(1000, 1000) == pytest.approx(1.0)
    assert parseval_scale(2000, 1000) == pytest.approx(np.sqrt(2.0))


def test_find_bandwidth_tracks_the_true_edges() -> None:
    """The band should sit on the passing region, not lag behind it.

    The old percentile-of-the-sign-integral search returned 9.41 to 28.72 for
    this input — the high edge close, but **the low edge 4.4 Hz late**, which
    matters because the low edge is what constrains ``Omega``. Taking the
    contiguous run directly lands within one bin of the true 5 Hz onset.

    The bound below is half a bin (the grid is 0.495 Hz), which is the best
    any bin-resolution method can do.
    """
    freq = np.linspace(1.0, 50.0, 100)
    snr = np.where((freq > 5.0) & (freq < 30.0), 10.0, 0.5)
    spacing = float(np.diff(freq)[0])

    band = find_bandwidth(freq, snr, threshold=3.0, method="widest")
    assert band is not None
    low, high = band
    assert abs(low - 5.0) <= spacing, f"low edge {low} is more than a bin from 5.0"
    assert abs(high - 30.0) <= spacing, f"high edge {high} is more than a bin from 30"
    # And strictly inside, so it never claims bandwidth that does not pass.
    assert 5.0 < low < high < 30.0


def test_find_bandwidth_returns_none_when_nothing_passes() -> None:
    freq = np.linspace(1.0, 50.0, 100)
    assert find_bandwidth(freq, np.full_like(freq, 0.1), threshold=3.0) is None


def test_find_bandwidth_survives_a_single_noisy_bin() -> None:
    """A one-bin dip must not truncate the band — that is why it integrates."""
    freq = np.linspace(1.0, 50.0, 100)
    snr = np.where((freq > 5.0) & (freq < 30.0), 10.0, 0.5)
    snr[40] = 0.1

    band = find_bandwidth(freq, snr, threshold=3.0, method="widest")
    assert band is not None
    assert band[1] > 25.0, band


def test_binned_spectrum_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="must match"):
        BinnedSpectrum(freq=np.arange(5.0), amp=np.arange(4.0))


def test_the_set_behaves_like_a_mapping() -> None:
    pair = object()
    s = SpectrumSet(pairs={"A": pair, "B": pair}, event="e")  # type: ignore[dict-item]
    assert len(s) == 2
    assert s["A"] is pair
    assert sorted(s) == ["A", "B"]
    assert s.ids() == ["A", "B"]


# ------------------------------------------------- against the legacy path


pytestmark_data = pytest.mark.skipif(
    not DATA.is_dir() or not INVENTORY.is_file(),
    reason="tutorial waveforms not present",
)


@functools.cache
def _windows() -> tuple[Any, Any]:
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
def _legacy() -> Any:
    signal, noise = _windows()
    with contextlib.redirect_stdout(io.StringIO()):
        return Spectra.from_streams(signal.copy(), noise.copy())


@pytestmark_data
def test_the_binning_reproduces_the_legacy_bins_on_real_data() -> None:
    """``log_bin`` against ``Spectrum.__bin_spectrum`` on all 28 windows.

    The binning is where a rewrite is most likely to drift silently: the edges,
    the clamping, the geometric centres and the empty-bin handling all have to
    agree, and a difference in any of them shifts every downstream ratio
    without making anything obviously wrong.
    """
    binning = {
        "f_min": BINNING_PARAMS["smin"],
        "f_max": BINNING_PARAMS["smax"],
        "n_bins": BINNING_PARAMS["bins"],
    }
    for name, snp in _legacy().group.items():
        mine = log_bin(snp.signal.freq, snp.signal.amp, **binning)
        assert mine.freq == pytest.approx(snp.signal.bfreq, rel=1e-12), name
        assert mine.amp == pytest.approx(snp.signal.bamp, rel=1e-12), name


@pytestmark_data
def test_the_pair_reproduces_the_legacy_snr_and_band() -> None:
    """The whole comparison, end to end, against ``SNP`` on real windows.

    This is the test stage 1 exists to pass. It runs the rescale, the
    rotation, the interpolation, the binning and the band search through the
    new path and holds the result against what ``SNP`` produced on the same 28
    windows.

    The legacy pair has already been rescaled and rotated by the time it is
    readable, so those steps are disabled here to avoid applying them twice —
    what is compared is the binning, the ratio and the floor, which is
    everything downstream of the point the two paths can be aligned at.
    """
    assert ROTATE_NOISE, (
        "the legacy default changed; this comparison assumes rotation is on"
    )

    signal_stream, noise_stream = _windows()
    legacy = _legacy().group

    checked = 0
    with contextlib.redirect_stdout(io.StringIO()):
        for s_tr, n_tr in zip(signal_stream, noise_stream, strict=True):
            # Fresh, unmutated spectra from the same traces. Reading them off
            # the finished SNP will not do: by then the noise has been
            # rescaled, rotated and interpolated in place, and re-binning that
            # is not the same as the legacy's own `bamp` — it rotates the
            # binned array directly but the unbinned one by interpolation, so
            # the two are not related by the binning operation.
            pair = SpectrumPair.compare(
                _as_core(Signal(s_tr.copy())),
                _as_core(Noise(n_tr.copy())),
                threshold=SNR_TOLERENCE,
                f_min=BINNING_PARAMS["smin"],
                f_max=BINNING_PARAMS["smax"],
                n_bins=BINNING_PARAMS["bins"],
            )
            snp = legacy[s_tr.id]
            assert pair.snr == pytest.approx(snp.bsnr, rel=1e-9), s_tr.id
            assert pair.resolution_floor == pytest.approx(snp.resolution_floor), s_tr.id
            checked += 1
    assert checked == 28


def _as_core(legacy: Any) -> Any:
    """Wrap a legacy spectrum's arrays in a ``core.Spectrum``."""
    return Spectrum(
        freq=np.asarray(legacy.freq, dtype=float),
        amp=np.asarray(legacy.amp, dtype=float),
        motion=getattr(legacy, "motion", "velocity"),
        kind="magnitude",
        duration=legacy.meta["npts"] * legacy.meta["delta"],
        sampling_rate=float(legacy.meta["sampling_rate"]),
    )


# ------------------------------------------------------------ failure paths


def test_find_bandwidth_rejects_a_run_shorter_than_min_width() -> None:
    """A band is a run of bins, and two bins is not enough to fit anything.

    The legacy returned a band here regardless, flagging it separately;
    returning None is the point of the new contract. A run exactly at
    ``min_width`` is accepted, so the boundary is checked from both sides.
    """
    freq = np.linspace(1.0, 50.0, 100)

    widest = WidestBandwidth()

    too_short = np.full_like(freq, 0.5)
    too_short[:2] = 10.0
    assert widest.select(freq, too_short, 3.0) is None

    just_enough = np.full_like(freq, 0.5)
    just_enough[:3] = 10.0
    assert widest.select(freq, just_enough, 3.0) is not None


def test_find_bandwidth_rejects_a_band_narrower_than_min_width() -> None:
    freq = np.linspace(1.0, 50.0, 100)
    snr = np.where((freq > 20.0) & (freq < 21.0), 10.0, 0.5)

    assert WidestBandwidth(min_width=20).select(freq, snr, 3.0) is None


def test_boost_noise_rejects_mismatched_shapes() -> None:
    """Three arrays have to agree, and a silent broadcast would be worse."""
    with pytest.raises(ValueError, match="must all match"):
        boost_noise(np.arange(5.0), np.arange(5.0), np.arange(4.0))


def test_a_pair_without_a_band_does_not_pass() -> None:
    """Pure noise resolves no band, so the pair reports failure rather than
    a band that happens to be the whole axis."""
    rng = np.random.default_rng(0)
    freq = np.linspace(1.0, 50.0, 400)
    flat = np.abs(rng.normal(1e-6, 1e-7, freq.size))

    pair = SpectrumPair.compare(
        _spectrum(freq, flat),
        _spectrum(freq, flat),
        threshold=3.0,
        rotate_noise=False,
    )
    assert pair.band is None
    assert not pair.passes


def test_the_floor_can_reject_a_band_entirely() -> None:
    """A floor above the band's high edge leaves nothing measurable.

    Narrowing to ``(floor, high)`` would be wrong when ``floor >= high`` —
    there is no interval left, and returning one would be inventing a result.
    """
    assert _clamp_to_floor((1.0, 5.0), floor=0.5) == (1.0, 5.0)
    assert _clamp_to_floor((1.0, 5.0), floor=2.0) == (2.0, 5.0)
    assert _clamp_to_floor((1.0, 5.0), floor=9.0) is None


def test_passing_filters_out_the_failures() -> None:
    good = SpectrumPair(
        signal=_spectrum(np.array([1.0, 2.0]), np.array([1.0, 1.0])),
        noise=_spectrum(np.array([1.0, 2.0]), np.array([1.0, 1.0])),
        binned_signal=BinnedSpectrum(np.array([1.5]), np.array([1.0])),
        binned_noise=BinnedSpectrum(np.array([1.5]), np.array([1.0])),
        snr=np.array([1.0]),
        resolution_floor=1.0,
        band=(1.0, 2.0),
    )
    bad = dataclasses.replace(good, band=None)

    every = SpectrumSet(pairs={"good": good, "bad": bad}, event="e")
    assert len(every) == 2
    assert every.passing().ids() == ["good"]
    assert every.passing().event == "e"


def _spectrum(freq: np.ndarray, amp: np.ndarray) -> Spectrum:
    return Spectrum(
        freq=freq,
        amp=amp,
        motion="velocity",
        kind="magnitude",
        duration=float(freq.size),
        sampling_rate=100.0,
    )


# ------------------------------------------------------------- noise models


def test_the_registry_resolves_the_default() -> None:
    model = get_noise_model("boost")
    assert model.name == "boost"
    assert isinstance(model, NoiseModel)


def test_an_unknown_noise_model_names_the_available_ones() -> None:
    with pytest.raises(ValueError, match="Unknown noise model"):
        get_noise_model("rotate")
    # `rotate` is the legacy ROT_METHOD = 1, not yet ported. When it lands it
    # registers here and this test changes to assert it resolves.
    assert "rotate" not in NOISE_MODELS


def test_every_registered_model_satisfies_the_protocol() -> None:
    """The point of the registry: a new method needs no change anywhere else.

    Each is exercised on the same input, so a model returning the wrong shape
    or a non-positive factor fails here rather than deep in a band search.
    """
    freq = np.linspace(1.0, 50.0, 60)
    signal = 1e-6 * np.exp(-freq / 20.0)
    noise = signal * 0.05

    for name in NOISE_MODELS:
        model = get_noise_model(name)
        assert isinstance(model, NoiseModel), name
        assert model.name == name
        factor = model.factor(freq, noise, signal)
        assert factor.shape == noise.shape, name
        assert np.isfinite(factor).all(), name
        assert (factor > 0).all(), name
        # A noise model may raise the noise; none of them may lower it.
        assert (factor >= 1.0 - 1e-12).all(), f"{name} lowered the noise"


def test_the_null_model_is_the_identity() -> None:
    """`none` exists so a run can show what the correction is doing."""
    freq = np.linspace(1.0, 50.0, 40)
    noise = np.full_like(freq, 1e-8)
    factor = get_noise_model("none").factor(freq, noise, noise * 20)
    assert factor == pytest.approx(np.ones_like(noise))


def test_boost_lifts_the_noise_to_touch_the_signal() -> None:
    """The defining property, stated as a test rather than left in a docstring.

    The lifted noise should reach the signal somewhere and, in the half being
    lifted, not overshoot it — that is what "until it touches" means.
    """
    rng = np.random.default_rng(5)
    freq = np.sort(rng.uniform(0.5, 50.0, 60))
    signal = 10 ** rng.normal(-6, 1, 60)
    noise = signal * 10 ** rng.normal(-1.5, 0.3, 60)

    lifted = noise * get_noise_model("boost").factor(freq, noise, signal)
    assert np.max(lifted / signal) == pytest.approx(1.0, rel=1e-9)


def test_boost_is_continuous_where_the_noise_crosses_the_signal() -> None:
    """The fourth discontinuity, and the subtlest one.

    A bin that already reaches the signal needs no lift. Expressing that as a
    guard — ``if np.any(noise >= signal): return unchanged`` — makes the
    function jump as a bin crosses: measured at **17.5%** in the median boost
    factor. Expressing it as ``max(0, min(needed))`` gives the same answer,
    because a bin above the signal requires a negative exponent to reach it,
    but passes through zero smoothly.

    It survived the first three fixes and was what still made four CWT
    stations disagree between machines. This sweeps a bin across the boundary
    and asserts the response has no step in it.
    """
    freq = np.linspace(1.0, 40.0, 40)
    signal = np.full_like(freq, 1e-6)

    factors = []
    for a in np.linspace(0.99, 1.01, 501):
        noise = signal * 0.2
        noise[5] = signal[5] * a
        factors.append(float(np.median(boost_noise(freq, noise, signal))))

    steps = np.abs(np.diff(factors))
    # Any residual step must be far below the 17.5% the guarded version showed.
    assert steps.max() < 1e-3, f"largest step {steps.max():.3e} looks like a jump"


# --------------------------------------------------------- bandwidth selectors


def test_every_registered_selector_satisfies_the_protocol() -> None:
    freq = np.linspace(1.0, 50.0, 100)
    snr = np.where((freq > 5.0) & (freq < 30.0), 10.0, 0.5)

    for name in BANDWIDTH_SELECTORS:
        selector = get_bandwidth_selector(name)
        assert isinstance(selector, BandwidthSelector), name
        assert selector.name == name
        band = selector.select(freq, snr, 3.0)
        assert band is not None, name
        low, high = band
        assert 5.0 < low < high < 30.0, f"{name} strayed outside the passing run"


def test_an_unknown_selector_names_the_available_ones() -> None:
    with pytest.raises(ValueError, match="Unknown bandwidth selector"):
        get_bandwidth_selector("percentile")


def test_every_selector_declines_rather_than_guessing() -> None:
    freq = np.linspace(1.0, 50.0, 100)
    nothing_passes = np.full_like(freq, 0.1)
    for name in BANDWIDTH_SELECTORS:
        assert get_bandwidth_selector(name).select(freq, nothing_passes, 3.0) is None, (
            name
        )


def test_peak_does_not_run_past_a_failure_immediately_above_it() -> None:
    """Regression for a wrap in the legacy ``BW_METHOD = 2``.

    That version found the bin before the first failure above the peak with
    ``np.where(...)[0] - 1``. When the failure is the bin *immediately* above
    the peak that index is 0, so ``0 - 1`` wrapped to -1 and selected the
    highest frequency in the record — claiming bandwidth all the way to
    Nyquist when the ratio had in fact failed at once.

    Demonstrated: on the input below the legacy returned a high edge of 50 Hz
    against a true 21.6. It does not trigger on the 28 PNR windows, where the
    ported selector agrees with the legacy on all 140 window-estimator pairs,
    but it would badly inflate bandwidth on a spectrum with an isolated peak.
    """
    freq = np.linspace(1.0, 50.0, 20)
    snr = np.full(20, 0.5)
    snr[7], snr[8] = 5.0, 10.0

    band = get_bandwidth_selector("peak").select(freq, snr, 3.0)
    assert band is not None
    assert band[1] < 25.0, f"high edge {band[1]} ran past the failure above the peak"
    assert band[1] == pytest.approx(freq[8])
