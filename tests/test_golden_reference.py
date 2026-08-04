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

Digests are over the raw float64 bytes, so a one-ULP drift still trips them.
The readable anchors alongside are what makes a failure interpretable.
"""

from __future__ import annotations

import contextlib
import functools
import glob
import hashlib
import io
import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(a, dtype=np.float64).tobytes()
    ).hexdigest()[:16]


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


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_same_windows_are_still_produced(estimator: str) -> None:
    assert sorted(_run(estimator).group) == sorted(_reference()[estimator])


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_amplitudes_are_bit_for_bit_unchanged(estimator: str) -> None:
    """The strict check. A refactor that preserves behaviour passes this.

    Reported with the readable anchors rather than the digest alone, because
    "a hash changed" is not a diagnosis — the median and max say whether
    something shifted by a factor of two or by a rounding error.
    """
    expected = _reference()[estimator]
    moved = []
    for name, snp in sorted(_run(estimator).group.items()):
        want = expected[name]
        got_amp = _digest(snp.signal.amp)
        if got_amp != want["amp_digest"]:
            moved.append(
                f"{name}: amp median {np.median(snp.signal.amp):.6e} vs "
                f"{want['amp_median']:.6e}, max {snp.signal.amp.max():.6e} vs "
                f"{want['amp_max']:.6e}"
            )
        elif _digest(snp.signal.freq) != want["freq_digest"]:
            moved.append(f"{name}: frequency axis changed, amplitudes did not")
    assert not moved, "\n".join(
        [f"{len(moved)} window(s) moved under {estimator!r}:", *moved]
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_noise_and_snr_are_unchanged(estimator: str) -> None:
    """Separate from the amplitude check because they fail for different reasons.

    The noise passes through the Parseval rescale, the rotation and the
    interpolation onto the signal's axis — three in-place steps the legacy
    classes own. A rewrite of those can move ``bsnr`` while leaving the signal
    amplitudes untouched, and that is worth seeing as its own failure.
    """
    expected = _reference()[estimator]
    moved = []
    for name, snp in sorted(_run(estimator).group.items()):
        want = expected[name]
        if _digest(snp.noise.amp) != want["noise_amp_digest"]:
            moved.append(f"{name}: noise amplitude")
        if _digest(snp.bsnr) != want["bsnr_digest"]:
            moved.append(f"{name}: binned signal-to-noise")
    assert not moved, f"under {estimator!r}: " + ", ".join(moved)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_the_selected_bandwidth_is_unchanged(estimator: str) -> None:
    """The band is what constrains Omega, so it is the number that reaches Mw."""
    expected = _reference()[estimator]
    moved = []
    for name, snp in sorted(_run(estimator).group.items()):
        want, band = expected[name], getattr(snp, "ubfreqs", None)
        has_band = band is not None and len(band) == 2
        got = [float(band[0]), float(band[1])] if has_band else None
        if got is None or want["band"] is None:
            if got is not want["band"]:
                moved.append(f"{name}: {want['band']} -> {got}")
        elif got != pytest.approx(want["band"], rel=1e-12):
            moved.append(f"{name}: {want['band']} -> {got}")
        if float(snp.resolution_floor) != pytest.approx(
            want["resolution_floor"], rel=1e-12
        ):
            moved.append(f"{name}: resolution floor moved")
    assert not moved, f"under {estimator!r}: " + "; ".join(moved)
