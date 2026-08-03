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

Measurements are in one of two groups. **Synthetic** ones are fast, need only
numpy and specmod, and are checked in CI. **Field** ones read the PNR waveforms
under ``Tutorial/Data`` and are skipped when that is unavailable — pass
``--field`` to include them.
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
        rows.append(
            [
                name,
                f"{mag_change:.1e}",
                f"{envelope_centroid(x):.1%}",
                f"{ratio(x):.3f}",
            ]
        )
    return table(
        ["Change", "max change in `\\|X\\|`", "Envelope centroid", "Estimate"], rows
    )


@synthetic("prieto_agreement")
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
