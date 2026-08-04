#!/usr/bin/env python
"""Regenerate the measured tables embedded in the documentation.

Every number in ``docs/`` that came out of a measurement is produced by a
function in this file and injected into the prose between markers:

.. code-block:: markdown

   <!-- measured: position_table -->
   | Position | ... |
   <!-- /measured -->

HTML comments are invisible in rendered Markdown and in notebook cells, so the
markers cost nothing to a reader. Usage::

    python tools/measure_docs.py write     # refresh the docs in place
    python tools/measure_docs.py check     # fail if any table is stale
    python tools/measure_docs.py show      # print without touching anything

``tests/test_docs_are_current.py`` runs ``check`` for the synthetic
measurements, so a change to an estimator that moves a published number breaks
the build rather than quietly leaving the docs wrong.

Measurements are in one of two groups, split by whether CI can be relied on to
reproduce them. **Synthetic** ones need only numpy and specmod, are fast, and
are checked on every run. **Field** ones need something not guaranteed to be
present — the PNR waveforms under ``Tutorial/Data``, or the optional
``specmod[multitaper]`` extra — so they are excluded unless ``--field`` is
passed.

That split is load-bearing, not organisational: a measurement that degrades to
a "not available" note when its dependency is missing would render that note in
CI and fail the check against a doc holding real numbers. Anything that can
fail to run belongs in ``FIELD``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np

from specmod.transforms import FFTEstimator, MultitaperEstimator

ROOT = Path(__file__).resolve().parent.parent
DOCS = [
    ROOT / "docs" / "choosing_a_transform.md",
    ROOT / "docs" / "notes" / "window_position.md",
    ROOT / "docs" / "notebooks" / "_build_notebook.py",
]

N = 2000
DT = 0.01

SYNTHETIC: dict[str, Callable[[], str]] = {}
FIELD: dict[str, Callable[[], str]] = {}


def synthetic(name: str) -> Callable[[Callable[[], str]], Callable[[], str]]:
    def register(fn: Callable[[], str]) -> Callable[[], str]:
        SYNTHETIC[name] = fn
        return fn

    return register


def field(name: str) -> Callable[[Callable[[], str]], Callable[[], str]]:
    def register(fn: Callable[[], str]) -> Callable[[], str]:
        FIELD[name] = fn
        return fn

    return register


# --------------------------------------------------------------- fixtures


def energy(x: np.ndarray) -> float:
    """Time-domain energy, the quantity every estimate is scored against."""
    return float((x**2).sum() * DT)


def transient_at(position: float, width: float = 0.10, seed: int = 1) -> np.ndarray:
    """A burst of fixed energy and width, *centred* on ``position``.

    Shared verbatim with ``tests/test_transforms.py`` so the documented numbers
    and the asserted ones cannot drift apart.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(N)
    w = int(N * width)
    start = max(0, min(N - w, int(N * position) - w // 2))
    x[start : start + w] = rng.normal(0.0, 1e-6, w)
    return x


def decaying_burst_at(start_fraction: float, width: int = 400) -> np.ndarray:
    """An exponentially-decaying burst *starting* at ``start_fraction``.

    Closer to a real arrival than white noise in a box, and the fixture the
    centring table uses.
    """
    rng = np.random.default_rng(3)
    burst = rng.normal(0.0, 1.0, width) * np.exp(-np.arange(width) / 60.0)
    x = np.zeros(N)
    start = int(N * start_fraction)
    x[start : start + width] = burst
    return x


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------- measurements


@synthetic("position_table")
def position_table() -> str:
    """Energy recovered as an identical burst moves across the window.

    The headline result: the bias is the taper envelope, it is the same under
    either weighting, and a lightly-tapered FFT barely has it.
    """
    adaptive = MultitaperEstimator(adaptive=True)
    flat = MultitaperEstimator(adaptive=False)
    fft = FFTEstimator(taper="tukey", taper_alpha=0.05)

    rows = []
    for p in (0.06, 0.10, 0.25, 0.50, 0.75, 0.90):
        x = transient_at(p)
        e = energy(x)
        rows.append(
            [
                f"{p:.0%}",
                f"{adaptive.estimate(x, DT).energy() / e:.3f}",
                f"{flat.estimate(x, DT).energy() / e:.3f}",
                f"{fft.estimate(x, DT).energy() / e:.2f}",
            ]
        )
    x = np.random.default_rng(7).normal(0.0, 1e-6, N)
    e = energy(x)
    rows.append(
        [
            "stationary noise",
            f"{adaptive.estimate(x, DT).energy() / e:.2f}",
            f"{flat.estimate(x, DT).energy() / e:.2f}",
            f"{fft.estimate(x, DT).energy() / e:.2f}",
        ]
    )
    return table(
        [
            "Position in window",
            "multitaper, adaptive",
            "multitaper, flat",
            "FFT, 5% Tukey",
        ],
        rows,
    )


@synthetic("centring_table")
def centring_table() -> str:
    """Centring removes the position dependence outright, not merely reduces it."""
    rows = []
    for start in (0.02, 0.10, 0.30, 0.50, 0.70, 0.78):
        x = decaying_burst_at(start)
        e = energy(x)
        rows.append(
            [
                f"{start:.0%}",
                f"{MultitaperEstimator(center=False).estimate(x, DT).energy() / e:.3f}",
                f"{MultitaperEstimator(center=True).estimate(x, DT).energy() / e:.3f}",
            ]
        )
    return table(["Burst start", "`center=False`", "`center=True`"], rows)


@synthetic("quadratic_table")
def quadratic_table() -> str:
    """What the curvature correction buys, and what it costs.

    Deliberately reports both directions. The peak and contrast rows are the
    case it is built for; the Brune tail row is the case it should not be used
    for, and omitting it would leave the docs recommending it for exactly the
    job it is worst at.
    """
    from specmod.transforms import QuadraticMultitaperEstimator

    n, nw, k = 1024, 3.0, 5
    t = np.arange(n) * DT
    w = nw / (n * DT)  # multitaper half-bandwidth, Hz
    plain = MultitaperEstimator(time_bandwidth=nw, n_tapers=k)
    quad = QuadraticMultitaperEstimator(time_bandwidth=nw, n_tapers=k)
    rng = np.random.default_rng(3)

    # 1. a single line, whose true Fourier amplitude is A*T/2
    line = np.sin(2 * np.pi * 5.0 * t) + rng.normal(0.0, 0.02, n)
    true_peak = 1.0 * (n * DT) / 2.0
    peak = [est.estimate(line, DT).amp.max() / true_peak for est in (plain, quad)]

    # 2. two lines just past the resolution limit
    f2 = 5.0 + 1.2 * 2 * w
    pair = line + np.sin(2 * np.pi * f2 * t)

    def contrast(est: object) -> float:
        s = est.estimate(pair, DT)  # type: ignore[attr-defined]
        trough = s.amp[(s.freq > 5.0) & (s.freq < f2)]
        return float(s.amp.max() / trough.min())

    # 3. white noise: no curvature, so nothing should change
    flat = rng.normal(0.0, 1e-6, n)
    ctrl = [
        float(np.median(est.estimate(flat, DT).band(2.0, 40.0).amp))
        for est in (plain, quad)
    ]

    # 4. a Brune corner: the job SpecMod exists for
    f = np.fft.rfftfreq(n, DT)
    amp = 1e-6 / (1.0 + (f / 4.0) ** 2)
    amp[0] = 0.0

    def brune(seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        ph = r.uniform(-np.pi, np.pi, f.size)
        ph[0] = 0.0
        ph[-1] = 0.0
        return np.fft.irfft(amp * np.exp(1j * ph), n=n) * n

    def fit_fc(spectrum: object) -> float:
        ff, a = spectrum.freq, spectrum.amp  # type: ignore[attr-defined]
        m = (ff > 0.3) & (ff < 45.0)
        ff, a = ff[m], np.log10(a[m])
        best, best_fc = np.inf, np.nan
        for fc in np.geomspace(1.0, 16.0, 300):
            model = -np.log10(1.0 + (ff / fc) ** 2)
            res = float(np.sum((a - model - np.mean(a - model)) ** 2))
            if res < best:
                best, best_fc = res, fc
        return float(best_fc)

    fcs = [
        np.median([fit_fc(est.estimate(brune(s), DT)) for s in range(12)])
        for est in (plain, quad)
    ]
    tails = [
        float(
            np.median(
                quad.estimate(brune(s), DT).band(25.0, 49.0).amp
                / plain.estimate(brune(s), DT).band(25.0, 49.0).amp
            )
        )
        for s in range(8)
    ]

    rows = [
        ["single line, peak / true", f"{peak[0]:.2f}", f"**{peak[1]:.2f}**"],
        [
            f"two lines {f2 - 5.0:.2f} Hz apart, peak/trough",
            f"{contrast(plain):.1f}",
            f"**{contrast(quad):.1f}**",
        ],
        ["white noise, level ratio to multitaper", "1.00", f"{ctrl[1] / ctrl[0]:.2f}"],
        ["Brune tail 25-49 Hz, ratio to multitaper", "1.00", f"{np.median(tails):.2f}"],
        [
            "Brune corner, fitted f_c (true 4.0 Hz)",
            f"{fcs[0]:.2f}",
            f"{fcs[1]:.2f}",
        ],
    ]
    return table(["Measurement", "multitaper", "quadratic"], rows)


@synthetic("cwt_table")
def cwt_table() -> str:
    """Energy recovered by each estimator, including a non-stationary record.

    The last row is the one the CWT exists for: multitaper assumes stationarity
    and a burst violates it, where a wavelet transform does not assume it in
    the first place.
    """
    from specmod.transforms import CWTEstimator

    n = 2048
    t = np.arange(n) * DT
    rng = np.random.default_rng(0)
    burst = np.zeros(n)
    burst[n // 2 : n // 2 + n // 8] = rng.normal(0.0, 1e-6, n // 8)

    records = {
        "white noise": rng.normal(0.0, 1e-6, n),
        "5 Hz sinusoid": 2.5 * np.sin(2 * np.pi * 5.0 * t),
        "off-centre burst": burst,
    }
    estimators = {
        "FFT": FFTEstimator(),
        "multitaper": MultitaperEstimator(),
        "CWT": CWTEstimator(),
    }
    rows = []
    for name, x in records.items():
        true = float(((x - x.mean()) ** 2).sum() * DT)
        cells = []
        for est in estimators.values():
            ratio = est.estimate(x, DT).energy() / true
            cells.append(f"{ratio:.3f}")
        rows.append([name, *cells])
    return table(["Record", *estimators], rows)


@synthetic("padding_table")
def padding_table() -> str:
    """What zero-padding does and does not do.

    Two effects get conflated. Padding does not touch spectral leakage, which
    is set by the taper; it fixes scalloping, which only matters for features
    narrower than a bin. Reporting both together is the point.
    """
    n = 2000
    t = np.arange(n) * DT
    duration = n * DT
    df = 1.0 / duration

    def sidelobe(spectrum: object, f0: float) -> float:
        f, a = spectrum.freq, spectrum.amp  # type: ignore[attr-defined]
        away = np.abs(f - f0) > 0.5
        return float(np.median(a[away]) / a.max())

    rows = []
    for pad, label in ((None, "none"), (2 * n, "2x"), (8 * n, "8x")):
        # Leakage, measured on a line deliberately off bin centre.
        off_centre = np.sin(2 * np.pi * 5.037 * t)
        floors = [
            sidelobe(
                FFTEstimator(taper=taper, n_fft=pad).estimate(off_centre, DT), 5.037
            )
            for taper in ("boxcar", "tukey")
        ]
        # Scalloping, as the worst peak loss over sub-bin placements.
        worst = min(
            FFTEstimator(taper="boxcar", n_fft=pad)
            .estimate(np.sin(2 * np.pi * (5.0 + frac * df) * t), DT)
            .amp.max()
            / duration
            for frac in np.linspace(0.0, 0.5, 11)
        )
        rows.append([label, f"{floors[0]:.0e}", f"{floors[1]:.0e}", f"{worst:.3f}"])

    return table(
        [
            "Zero-padding",
            "sidelobes, boxcar",
            "sidelobes, Tukey",
            "worst peak / true",
        ],
        rows,
    )


@synthetic("leakage_table")
def leakage_table() -> str:
    """Why adaptive weighting is the default: flat weighting does not suppress
    leakage, and the high-frequency tail is where ``t*`` and ``f_c`` are read."""
    rng = np.random.default_rng(11)
    t = np.arange(N) * DT
    weak = rng.normal(0.0, 1e-9, N)
    x = 1e-3 * np.sin(2 * np.pi * 2.0 * t) + weak

    band = (20.0, 49.0)
    truth = FFTEstimator(taper="tukey", taper_alpha=0.05).estimate(weak, DT).band(*band)
    rows = []
    for name, est in (
        ("flat", MultitaperEstimator(adaptive=False)),
        ("adaptive", MultitaperEstimator(adaptive=True)),
    ):
        s = est.estimate(x, DT).band(*band)
        ratio = float(np.median(s.amp / truth.amp))
        cell = f"**{ratio:.0f}x**" if ratio > 10 else f"{ratio:.1f}x"
        rows.append([name, cell])
    return table(["Weighting", "recovered floor / true"], rows)


@synthetic("phase_table")
def phase_table() -> str:
    """It is the *linear* component of phase — the group delay — that matters,
    not phase in general. The constant rotation is the decisive control."""
    from scipy.signal import hilbert

    base = decaying_burst_at(0.10)
    est = MultitaperEstimator(adaptive=False)

    def envelope_centroid(x: np.ndarray) -> float:
        e = x**2
        return float((np.arange(x.size) * e).sum() / e.sum() / x.size)

    def ratio(x: np.ndarray) -> float:
        return est.estimate(x, DT).energy() / energy(base)

    spec = np.fft.rfft(base)
    rng = np.random.default_rng(5)

    variants = {
        "Linear ramp (= a time shift)": np.fft.irfft(
            spec * np.exp(-2j * np.pi * np.fft.rfftfreq(N) * (N * 0.4)), n=N
        ),
        "Constant 90 deg rotation": np.imag(hilbert(base)),
        "Random phase": np.fft.irfft(
            np.abs(spec) * np.exp(1j * rng.uniform(-np.pi, np.pi, spec.size)), n=N
        ),
    }

    # DC and Nyquist are constrained to be real for a real signal, so any
    # phase manipulation necessarily disturbs them. They carry no meaningful
    # power here and are excluded rather than allowed to masquerade as a
    # magnitude change.
    interior = slice(1, -1)
    base_mag = np.abs(spec)[interior]
    base_centroid = envelope_centroid(base)

    rows = [
        ["(unchanged reference)", "--", f"{base_centroid:.1%}", f"{ratio(base):.3f}"]
    ]
    for name, x in variants.items():
        mag_change = float(
            np.abs(np.abs(np.fft.rfft(x))[interior] - base_mag).max() / base_mag.max()
        )
        # Rendered against a threshold, not as a raw value. This quantity is
        # pure floating-point noise -- its leading digit varies with platform,
        # BLAS and numpy version, which would make the published table fail on
        # some CI runners and not others. The claim is "unchanged", and a
        # bound states that more honestly than a spurious 4.7e-16.
        rows.append(
            [
                name,
                "< 1e-12" if mag_change < 1e-12 else f"{mag_change:.1e}",
                f"{envelope_centroid(x):.1%}",
                f"{ratio(x):.3f}",
            ]
        )
    return table(
        ["Change", "max change in `\\|X\\|`", "Envelope centroid", "Estimate"], rows
    )


@field("prieto_agreement")
def prieto_agreement() -> str:
    """Cross-validation against the reference implementation.

    With ``normalize_to_variance=True`` putting both on the same absolute
    scale, our estimator should be indistinguishable from Prieto's under either
    weighting. This is the check that says the adaptive fix is right rather
    than merely plausible.
    """
    try:
        from multitaper import MTSpec
    except ImportError:
        return "_Not measured: install `specmod[multitaper]` and re-run._"

    def prieto_fas(x: np.ndarray, iadapt: int) -> tuple[np.ndarray, np.ndarray]:
        p = MTSpec(x.copy(), nw=3.0, kspec=5, dt=DT, nfft=N, iadapt=iadapt)
        f, s = p.freq[:, 0], p.spec[:, 0]
        order = np.argsort(f)
        f, s = f[order], s[order]
        keep = f > 0
        return f[keep], np.sqrt(2.0 * s[keep] * 2.0 * (N * DT))

    cases = {"stationary noise": np.random.default_rng(7).normal(0.0, 1e-6, N)}
    for p in (0.10, 0.50, 0.90):
        cases[f"burst at {p:.0%}"] = transient_at(p)

    rows = []
    for name, x in cases.items():
        cells = [name]
        for adaptive, iadapt in ((True, 0), (False, 1)):
            f, ref = prieto_fas(x, iadapt)
            ours = MultitaperEstimator(
                adaptive=adaptive, normalize_to_variance=True
            ).estimate(x, DT)
            band = (f > 0.5) & (f < 45.0)
            r = np.interp(f[band], ours.freq, ours.amp) / ref[band]
            cells.append(f"{np.median(r):.4f}")
        rows.append(cells)
    return table(["Record", "adaptive", "flat"], rows)


def _field_signals():  # type: ignore[no-untyped-def]
    """The 28 PNR S-windows, cut with the published Magna workflow."""
    import glob
    import os
    import warnings

    import obspy

    import specmod.preprocess as pre

    warnings.filterwarnings("ignore")
    data = ROOT / "Tutorial" / "Data" / "2019-08-26T07:30:47.0"
    inv = obspy.read_inventory(
        str(ROOT / "Tutorial" / "MetaData" / "pnr_inventory.xml")
    )
    st = obspy.read(os.path.join(str(data), "*HH[EN]*"))
    pre.set_stream_distance(
        st,
        53.784,
        -2.967,
        2.1,
        obspy.UTCDateTime("2019-08-26T07:49:24.2"),
        inventory=inv,
        dtype="mseed",
    )
    pre.set_picks_from_pyrocko(st, glob.glob(os.path.join(str(data), "*.picks"))[0])
    st = obspy.Stream([tr for tr in st if "s_time" in tr.stats])
    st.detrend("linear")
    st.detrend("demean")
    st.taper(0.05)
    st.remove_response(inv, output="VEL")
    sig = pre.get_signal(
        st, pre.cut_s, rafp=0.8, tafs=20, time_after="absolute_time", refine_window=True
    )
    # Condition the *cut* windows, not just the parent traces. A sub-window of
    # a demeaned record carries its own offset and trend, and that energy lands
    # in the DC bin the estimators discard — which would show up as a deficit in
    # every estimator equally and bury the effect being measured.
    sig.detrend("linear")
    sig.detrend("demean")
    return sig


@field("plateau_table")
def plateau_table() -> str:
    """Variance normalisation pins total energy; it does not pin the plateau.

    ``Omega`` is read off the low-frequency plateau, so this — not the energy
    ratio — is the number that propagates into ``Mw``. A real S-arrival is
    slid through an otherwise empty window and the spread of the recovered
    1-4 Hz level is reported relative to its mid-window value.
    """
    from specmod.transforms import PrietoMultitaperEstimator

    sig = sorted(_field_signals(), key=lambda tr: -tr.stats.npts)
    tr = sig[0]
    dt = float(tr.stats.delta)
    arrival = tr.data.astype(float)
    arrival = arrival[: min(arrival.size, 400)]

    total = 2000
    # The full legal range: from hard against the start to hard against the end.
    starts = np.linspace(0.01, 1.0 - arrival.size / total - 0.01, 16)

    def placed(fraction: float) -> np.ndarray:
        x = np.zeros(total)
        i = int(total * fraction)
        x[i : i + arrival.size] = arrival
        return x

    def spread(est, band=(1.0, 4.0)) -> str:  # type: ignore[no-untyped-def]
        levels = np.array(
            [
                float(np.median(est.estimate(placed(f), dt).band(*band).amp))
                for f in starts
            ]
        )
        mid = float(np.median(est.estimate(placed(0.5), dt).band(*band).amp))
        return f"{(levels.max() - levels.min()) / mid:.0%}"

    rows = [
        ["FFT, light taper", spread(FFTEstimator(taper="tukey", taper_alpha=0.05))],
        [
            "Prieto, constant weights",
            spread(PrietoMultitaperEstimator(weighting="constant")),
        ],
        ["Prieto, adaptive", spread(PrietoMultitaperEstimator(weighting="adaptive"))],
        [
            "ours, flat, no renormalisation",
            spread(MultitaperEstimator(adaptive=False)),
        ],
        [
            "ours, adaptive, no renormalisation",
            spread(MultitaperEstimator(adaptive=True)),
        ],
        [
            "ours, adaptive, renormalised",
            spread(MultitaperEstimator(adaptive=True, normalize_to_variance=True)),
        ],
        ["ours, adaptive, `center=True`", spread(MultitaperEstimator(center=True))],
    ]
    return table(["Method", "Plateau spread"], rows)


@field("field_energy_table")
def field_energy_table() -> str:
    """Energy recovered on the real PNR windows, not synthetic bursts."""
    sig = [tr for tr in _field_signals() if tr.stats.npts > 50]
    estimators = {
        "adaptive": MultitaperEstimator(adaptive=True),
        "flat": MultitaperEstimator(adaptive=False),
        "FFT": FFTEstimator(taper="tukey", taper_alpha=0.05),
    }
    ratios: dict[str, list[float]] = {k: [] for k in estimators}
    for tr in sig:
        # Mirrors prepare_record, so the reference and the estimate see the
        # same record. A no-op on the detrended windows above; kept so the
        # comparison stays honest if that conditioning ever changes.
        x = tr.data.astype(float)
        x = x - x.mean()
        dt = float(tr.stats.delta)
        true = float((x**2).sum() * dt)
        for name, est in estimators.items():
            ratios[name].append(est.estimate(x, dt).energy() / true)

    rows = []
    for label, fn in (
        ("median", lambda v: f"{np.median(v):.3f}"),
        ("worst", lambda v: f"{min(v):.3f}"),
        ("below 0.6", lambda v: f"{np.mean(np.array(v) < 0.6):.0%} of traces"),
    ):
        rows.append([label] + [fn(ratios[k]) for k in estimators])
    return table([f"n = {len(sig)}", *estimators], rows)


# ------------------------------------------------------------------ driver

#: Body may be empty — that is how a newly-added marker looks before its first
#: ``--write``, and it has to match or the table would be silently skipped.
MARKER = re.compile(
    r"(<!-- measured: (?P<name>[a-z_]+) -->\n)(?P<body>.*?)(<!-- /measured -->)",
    re.DOTALL,
)


def render(names: dict[str, Callable[[], str]]) -> dict[str, str]:
    return {name: fn() for name, fn in names.items()}


def apply_to(path: Path, rendered: dict[str, str]) -> tuple[str, list[str]]:
    text = path.read_text()
    stale: list[str] = []

    def substitute(m: re.Match[str]) -> str:
        name = m.group("name")
        if name not in rendered:
            return m.group(0)
        fresh = rendered[name]
        if m.group("body").strip() != fresh.strip():
            stale.append(f"{path.relative_to(ROOT)}:{name}")
        return m.group(1) + fresh + "\n" + m.group(4)

    return MARKER.sub(substitute, text), stale


def _field_option(fn):  # type: ignore[no-untyped-def]
    return click.option(
        "--field",
        is_flag=True,
        help="Include measurements that read Tutorial/Data (slow).",
    )(fn)


def _selected(field_too: bool) -> dict[str, str]:
    wanted = dict(SYNTHETIC)
    if field_too:
        wanted.update(FIELD)
    return render(wanted)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Regenerate the measured tables embedded in the documentation."""


@main.command()
@_field_option
def show(field: bool) -> None:
    """Print the tables without touching any file."""
    for name, body in _selected(field).items():
        click.echo(f"\n<!-- measured: {name} -->\n{body}\n<!-- /measured -->")


@main.command()
@_field_option
def write(field: bool) -> None:
    """Refresh the documentation in place."""
    rendered = _selected(field)
    stale: list[str] = []
    for path in DOCS:
        if not path.exists():
            continue
        updated, found = apply_to(path, rendered)
        if found:
            stale += found
            path.write_text(updated)
    click.echo(f"updated {len(stale)} table(s)" if stale else "already current")


@main.command()
@_field_option
def check(field: bool) -> None:
    """Exit non-zero if any documented table no longer matches the code."""
    rendered = _selected(field)
    stale: list[str] = []
    for path in DOCS:
        if path.exists():
            _, found = apply_to(path, rendered)
            stale += found
    if stale:
        click.echo("stale measured tables:", err=True)
        for s in stale:
            click.echo(f"  {s}", err=True)
        raise click.ClickException("run: python tools/measure_docs.py write")
    click.echo("all measured tables are current")


if __name__ == "__main__":
    main()
