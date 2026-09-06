"""Draw the SpecMod logo from the description in ``docs/branding.md``.

    python tools/make_logo.py

Writes four SVGs into ``docs/_static/``: a horizontal lockup and a square mark,
each in a light and a dark variant. SVG rather than PNG because the sidebar
renders it at a handful of sizes and a vector has no wrong one.

**This is a draft.** The branding manual describes the mark — "a minimalist
line-art spectral window enclosing a seismic waveform that flattens into a
model curve" — and leaves the artwork as a placeholder. This is that sentence
drawn literally, so the theme can be judged whole; replace the files, or this
script, with real artwork when there is some.

The mark is the package's own subject: an observed spectrum on the left,
falling into the smooth source model that is fitted to it. Ink for the
observation and the accent for the model, which is the same division the plots
use — one accent per instance, per the manual.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
#: Keep text as text in the SVG rather than converting it to paths, so the
#: browser resolves Merriweather — which `academic.css` loads — and falls back
#: to Georgia where it cannot. A path-converted wordmark would be frozen in
#: whatever font this machine happens to have, which is neither.
matplotlib.rcParams["svg.fonttype"] = "none"
#: Deterministic output, so re-running this changes nothing unless the drawing
#: changed. matplotlib otherwise stamps each SVG with the time it ran and gives
#: clip paths a random id, which would make every regeneration a full-file diff
#: and the script something nobody runs.
matplotlib.rcParams["svg.hashsalt"] = "specmod"
#: The layout metrics still want a real font and this machine has neither of
#: the two named ones; the warning is per-glyph and says nothing useful.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STATIC = Path(__file__).resolve().parent.parent / "docs" / "_static"

#: From the palette. Ink for light backgrounds, paper for dark ones; the
#: accent is the secondary, which is what marks a fitted curve.
INK = "#1E293B"
PAPER = "#F8FAFC"
TERRACOTTA = "#C2410C"
SOFT_TERRACOTTA = "#FB923C"


def _spectrum() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A Brune-shaped spectrum, and the noisy observation of it.

    Flat below the corner, falling as ``f**-2`` above it — the shape the whole
    package exists to fit, so the mark is the science rather than decoration.
    """
    f = np.logspace(-0.7, 1.15, 400)
    model = 1.0 / (1.0 + (f / 1.1) ** 2)
    rng = np.random.default_rng(6)
    wobble = 1.0 + 0.16 * rng.standard_normal(f.size)
    wobble = np.convolve(wobble, np.ones(9) / 9, mode="same")
    return f, model, model * wobble


def _draw(ax: plt.Axes, ink: str, accent: str, *, lw: float = 2.0) -> None:
    """The mark: a spectral window, an observation, and the model in it."""
    f, model, observed = _spectrum()

    # The window: an open bracket rather than a closed box, so the curve reads
    # as continuing past it rather than being boxed in.
    ax.plot(
        [0.06, 0.06, 0.30], [0.93, 0.06, 0.06], color=ink, lw=lw, solid_capstyle="round"
    )
    ax.plot(
        [0.70, 0.94, 0.94], [0.06, 0.06, 0.93], color=ink, lw=lw, solid_capstyle="round"
    )

    x = 0.06 + 0.88 * (np.log10(f) - np.log10(f[0])) / (
        np.log10(f[-1]) - np.log10(f[0])
    )

    def y(a: np.ndarray) -> np.ndarray:
        return 0.16 + 0.70 * (a / model.max())

    # Observed first, so the model sits over it — the way a fit is read.
    ax.plot(x, y(observed), color=ink, lw=lw * 0.7, alpha=0.55, solid_capstyle="round")
    ax.plot(x, y(model), color=accent, lw=lw * 1.15, solid_capstyle="round")

    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")


def _tidy(path: Path) -> None:
    """Strip trailing whitespace, which matplotlib leaves in its SVG output.

    The `trailing-whitespace` pre-commit hook rewrites these files otherwise,
    so a regenerated logo would differ from the committed one every time and
    the script would stop being the thing that produces them.
    """
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def square(path: Path, ink: str, accent: str) -> None:
    """The mark alone, for a favicon or an avatar."""
    fig, ax = plt.subplots(figsize=(1.6, 1.6))
    _draw(ax, ink, accent, lw=2.6)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, transparent=True, format="svg", metadata={"Date": None})
    plt.close(fig)
    _tidy(path)


def horizontal(path: Path, ink: str, accent: str) -> None:
    """The mark plus the wordmark, for a page header."""
    fig = plt.figure(figsize=(5.2, 1.4))
    ax = fig.add_axes((0.0, 0.0, 0.27, 1.0))
    _draw(ax, ink, accent)
    text = fig.add_axes((0.30, 0.0, 0.70, 1.0))
    text.axis("off")
    # The heading serif, per the typography table. Georgia is the documented
    # fallback and is what renders when Merriweather is not installed, which
    # on a build machine it is not.
    text.text(
        0.0,
        0.46,
        "SpecMod",
        family=["Merriweather", "Georgia", "serif"],
        fontsize=34,
        color=ink,
        va="center",
    )
    fig.savefig(path, transparent=True, format="svg", metadata={"Date": None})
    plt.close(fig)
    _tidy(path)


def main() -> None:
    STATIC.mkdir(parents=True, exist_ok=True)
    written = []
    for name, ink, accent in (
        ("specmod-logo-academic", INK, TERRACOTTA),
        ("specmod-logo-academic-dark", PAPER, SOFT_TERRACOTTA),
    ):
        horizontal(STATIC / f"{name}.svg", ink, accent)
        square(STATIC / f"{name}-mark.svg", ink, accent)
        written += [f"{name}.svg", f"{name}-mark.svg"]
    print(f"wrote {len(written)} files into {STATIC}:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
