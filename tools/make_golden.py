"""Snapshot the current pipeline's numbers so a refactor cannot move them quietly.

There is no legacy Docker image to compare against, so the reference for the
decomposition is this code, captured now. Every later stage is checked against
the file this writes: if a number moves, it moves visibly and on purpose.

Run with ``python tools/make_golden.py`` and commit the result.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
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


def _digest(a: np.ndarray) -> str:
    """Hash of the exact bytes, so a drift of one ULP is still caught."""
    raw = np.ascontiguousarray(a, dtype=np.float64).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


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
            "freq_digest": _digest(snp.signal.freq),
            "amp_digest": _digest(snp.signal.amp),
            "noise_amp_digest": _digest(snp.noise.amp),
            "bsnr_digest": _digest(snp.bsnr),
            # Human-readable anchors, so a diff is interpretable without
            # re-running: a changed digest alone says nothing about size.
            "amp_median": float(np.median(snp.signal.amp)),
            "amp_max": float(snp.signal.amp.max()),
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
