"""Snapshot the current pipeline's numbers so a refactor cannot move them quietly.

There is no legacy Docker image to compare against, so the reference for the
decomposition is this code, captured now. Every later stage is checked against
the file this writes: if a number moves, it moves visibly and on purpose.

**Summaries, not byte digests.** The first version of this hashed the raw
float64 bytes. That caught a 1-part-in-1e12 perturbation, which was the point
— but it also failed on every CI runner, because a different numpy/scipy build
produces last-bit differences on identical inputs. A reference that only holds
on the machine that generated it is not a reference. What is recorded instead
is a distributional summary compared with a relative tolerance: tight enough
that no real change in level or shape can slip through, loose enough to
survive a different BLAS.

Run with ``python tools/make_golden.py`` and commit the result.
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import os
import platform
import warnings
from pathlib import Path

import numpy as np
import obspy
import scipy

import specmod.preprocess as pre
from specmod.spectral import Spectra

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Tutorial" / "Data" / "2019-08-26T07:30:47.0"
INVENTORY = ROOT / "Tutorial" / "MetaData" / "pnr_inventory.xml"
OUT = ROOT / "tests" / "golden" / "pipeline_reference.json"
WINDOWS_OUT = ROOT / "tests" / "golden" / "window_reference.json"

ORIGIN = "2019-08-26T07:49:24.2"
LATITUDE, LONGITUDE, DEPTH_KM = 53.784, -2.967, 2.1


#: Fixed probabilities at which every array is sampled. Quantiles rather than
#: raw values because they summarise the whole distribution in a fixed-length
#: vector — a change in level moves all of them, a change in shape moves some.
QUANTILES = np.linspace(0.0, 1.0, 33)


def _summary(a: np.ndarray) -> dict:
    """Distributional fingerprint of an array, comparable with a tolerance."""
    a = np.asarray(a, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "max": float(a.max()),
        "sum": float(a.sum()),
        "quantiles": [float(q) for q in np.quantile(a, QUANTILES)],
    }


def _prepared_stream():
    """Metadata set, picks read, response removed — everything before the cut."""
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
        return stream


def _cut(stream, refine_window=True):
    return pre.get_signal(
        stream,
        pre.cut_s,
        rafp=0.8,
        tafs=20,
        time_after="absolute_time",
        refine_window=refine_window,
    )


def _windows():
    stream = _prepared_stream()
    signal = _cut(stream)
    return signal, pre.get_noise_p(stream, signal)


def capture(estimator: str) -> dict:
    signal, noise = _windows()
    with contextlib.redirect_stdout(io.StringIO()):
        result = Spectra.from_streams(signal.copy(), noise.copy(), estimator=estimator)

    # Read through the immutable container, not the legacy one. The values are
    # the same objects — `SNP` keeps the `SpectrumPair` its numbers came from —
    # so this reference is byte-identical to the one the legacy path produced,
    # which is what makes the swap verifiable rather than merely plausible.
    out = {}
    for name, pair in sorted(result.as_spectrum_set().pairs.items()):
        out[name] = {
            "n_freq": int(pair.signal.freq.size),
            "freq": _summary(pair.signal.freq),
            "amp": _summary(pair.signal.amp),
            "noise_amp": _summary(pair.noise.amp),
            "bsnr": _summary(pair.snr),
            "resolution_floor": float(pair.resolution_floor),
            "band": list(pair.band) if pair.band is not None else None,
        }
    return out


def _window(tr, otime) -> dict:
    """A window's geometry, expressed relative to the origin time.

    Relative seconds rather than absolute timestamps: the numbers stay legible
    in a diff, and a change of a tenth of a second shows up as a tenth of a
    second instead of as two long ISO strings that have to be subtracted by eye.
    """
    return {
        "start": float(tr.stats["wstart"] - otime),
        "end": float(tr.stats["wend"] - otime),
        "npts": int(tr.stats.npts),
        "duration": float(tr.stats.endtime - tr.stats.starttime),
    }


def capture_windows() -> dict:
    """Where every window lands, and the metadata that puts it there.

    The spectral reference starts from cut windows, so it cannot see a change
    in how those windows are chosen — such a change moves the reference rather
    than failing against it. This is the piece that closes that gap: station
    geometry, picks, the window before refinement, the window after it, and
    the noise window derived from the signal's length.
    """
    stream = _prepared_stream()
    refined = _cut(stream, refine_window=True)
    unrefined = _cut(stream, refine_window=False)
    noise = pre.get_noise_p(stream, refined)

    out = {}
    for tr, raw, sig, noi in zip(stream, unrefined, refined, noise, strict=True):
        otime = tr.stats["otime"]
        out[tr.id] = {
            "sampling_rate": float(tr.stats.sampling_rate),
            "slat": float(tr.stats["slat"]),
            "slon": float(tr.stats["slon"]),
            "selv": float(tr.stats["selv"]),
            "repi": float(tr.stats["repi"]),
            "rhyp": float(tr.stats["rhyp"]),
            "azimuth": float(tr.stats["azimuth"]),
            "back_azimuth": float(tr.stats["back_azimuth"]),
            "p_time": float(tr.stats["p_time"] - otime),
            "s_time": float(tr.stats["s_time"] - otime),
            "unrefined": _window(raw, otime),
            "signal": _window(sig, otime),
            "noise": _window(noi, otime),
            # What `signal_intensity` moved: the 1st- and 99th-percentile
            # offsets into the unrefined window, in seconds.
            "refinement": {
                "lead": float(sig.stats["wstart"] - raw.stats["wstart"]),
                "trail": float(sig.stats["wend"] - raw.stats["wstart"]),
            },
        }
    return out


def _environment() -> dict:
    """What produced these numbers.

    Recorded because parts of the pipeline are not reproducible across builds
    — see ``tests/test_golden_reference.py``. The strict noise and SNR checks
    only run where this matches.
    """
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": ".".join(platform.python_version_tuple()[:2]),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def main() -> None:
    reference = {
        "_environment": _environment(),
        **{
            est: capture(est)
            for est in ("fft", "welch", "multitaper", "quadratic", "cwt")
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reference, indent=1, sort_keys=True) + "\n")
    n = sum(len(v) for k, v in reference.items() if k != "_environment")
    est = len(reference) - 1
    print(f"wrote {OUT.relative_to(ROOT)}: {est} estimators, {n} windows")

    windows = {"_environment": _environment(), "windows": capture_windows()}
    WINDOWS_OUT.write_text(json.dumps(windows, indent=1, sort_keys=True) + "\n")
    print(f"wrote {WINDOWS_OUT.relative_to(ROOT)}: {len(windows['windows'])} traces")


if __name__ == "__main__":
    main()
