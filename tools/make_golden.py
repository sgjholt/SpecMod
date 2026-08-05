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
import warnings
from pathlib import Path

import numpy as np
import obspy

import specmod.preprocess as pre
from specmod.spectral import Spectra

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Tutorial" / "Data" / "2019-08-26T07:30:47.0"
INVENTORY = ROOT / "Tutorial" / "MetaData" / "pnr_inventory.xml"
OUT = ROOT / "tests" / "golden" / "pipeline_reference.json"

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


def _windows():
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


def capture(estimator: str) -> dict:
    signal, noise = _windows()
    with contextlib.redirect_stdout(io.StringIO()):
        result = Spectra.from_streams(signal.copy(), noise.copy(), estimator=estimator)

    out = {}
    for name, snp in sorted(result.group.items()):
        band = getattr(snp, "ubfreqs", None)
        out[name] = {
            "n_freq": int(snp.signal.freq.size),
            "freq": _summary(snp.signal.freq),
            "amp": _summary(snp.signal.amp),
            "noise_amp": _summary(snp.noise.amp),
            "bsnr": _summary(snp.bsnr),
            "resolution_floor": float(snp.resolution_floor),
            "band": [float(band[0]), float(band[1])]
            if band is not None and len(band) == 2
            else None,
        }
    return out


def main() -> None:
    reference = {
        est: capture(est) for est in ("fft", "welch", "multitaper", "quadratic", "cwt")
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(reference, indent=1, sort_keys=True) + "\n")
    n = sum(len(v) for v in reference.values())
    print(f"wrote {OUT.relative_to(ROOT)}: {len(reference)} estimators, {n} windows")


if __name__ == "__main__":
    main()
