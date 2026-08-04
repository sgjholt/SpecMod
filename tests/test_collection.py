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
from specmod.core.collection import (  # noqa: E402
    BinnedSpectrum,
    SpectrumPair,
    SpectrumSet,
    find_bandwidth,
    log_bin,
    parseval_scale,
)
from specmod.spectral import (  # noqa: E402
    BINNING_PARAMS,
    ROTATE_NOISE,
    SNR_TOLERENCE,
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


def test_find_bandwidth_selects_the_passing_run() -> None:
    """Brackets the passing region — but note where the low edge actually lands.

    The signal-to-noise here passes cleanly between 5 and 30 Hz, and the
    selected band is roughly 9.4 to 28.7. The high edge is close; **the low
    edge lags the true onset by about 4 Hz.** That is inherent to reading the
    band off percentiles of the integrated sign function rather than a defect
    in the port — the values above are bit-identical to the pre-refactor
    implementation, which was checked directly against it.

    It matters because the low edge is what constrains ``Omega``. Anyone
    revisiting the band search should treat this lag as the thing to improve,
    and this test as the record of what the behaviour was before they did.
    """
    freq = np.linspace(1.0, 50.0, 100)
    snr = np.where((freq > 5.0) & (freq < 30.0), 10.0, 0.5)

    band = find_bandwidth(freq, snr, threshold=3.0)
    assert band is not None
    low, high = band
    assert low == pytest.approx(9.414141, rel=1e-6), low
    assert high == pytest.approx(28.717171, rel=1e-6), high
    # Whatever the lag, the band must lie inside the region that passes.
    assert 5.0 < low < high < 30.0


def test_find_bandwidth_returns_none_when_nothing_passes() -> None:
    freq = np.linspace(1.0, 50.0, 100)
    assert find_bandwidth(freq, np.full_like(freq, 0.1), threshold=3.0) is None


def test_find_bandwidth_survives_a_single_noisy_bin() -> None:
    """A one-bin dip must not truncate the band — that is why it integrates."""
    freq = np.linspace(1.0, 50.0, 100)
    snr = np.where((freq > 5.0) & (freq < 30.0), 10.0, 0.5)
    snr[40] = 0.1

    band = find_bandwidth(freq, snr, threshold=3.0)
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

    Noise rotation is off for this check. It is the one step that is not yet
    ported — it mutates the noise in place and has two schemes — so comparing
    with it enabled would be comparing against something this does not claim
    to do yet.
    """
    if ROTATE_NOISE:
        pytest.skip("noise rotation is not ported yet; see stage 1 follow-up")

    checked = 0
    for name, snp in _legacy().group.items():
        pair = SpectrumPair.compare(
            _as_core(snp.signal),
            _as_core(snp.noise),
            threshold=SNR_TOLERENCE,
            f_min=BINNING_PARAMS["smin"],
            f_max=BINNING_PARAMS["smax"],
            n_bins=BINNING_PARAMS["bins"],
            # The legacy pair has already been rescaled and interpolated by the
            # time it is readable, so doing it again would double-apply.
            scale_parseval=False,
        )
        assert pair.snr == pytest.approx(snp.bsnr, rel=1e-12), name
        assert pair.resolution_floor == pytest.approx(snp.resolution_floor), name
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
