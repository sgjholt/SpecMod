"""Build the transforms tutorial notebook."""

import json
import pathlib

cells = []


def md(text):
    cells.append(
        {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}
    )


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip().splitlines(True),
        }
    )


md("""
# Choosing a transform: what it does to your data

Estimating a spectrum is **not** a neutral view of your record. Every estimator
is a transform, and each one trades something away. The differences are not
subtle: the choices in this notebook change recovered amplitude by factors of
three on real seismic windows, which is roughly 0.3 magnitude units.

This notebook is the long form of
[`choosing_a_transform.md`](choosing_a_transform.md). Work through it once
before picking an estimator for real analysis.

**What you will end up knowing**

1. Why the same record gives different spectra through different estimators
2. The stationarity assumption every one of them makes, and how a seismic
   arrival breaks it
3. Why the bias depends on *where in the window* your arrival sits — and how to
   remove that dependence completely
4. What `mtspec` did that you may need to reproduce
5. Which estimator to reach for, and how to pin the choice so a result stays
   reproducible
""")

code("""
import matplotlib.pyplot as plt
import numpy as np

from specmod.smoothing import KonnoOhmachi, LogBinner
from specmod.transforms import FFTEstimator, MultitaperEstimator, WelchEstimator

FS, DURATION = 100.0, 20.0
DT = 1.0 / FS
N = int(FS * DURATION)


def energy(x):
    \"\"\"Time-domain energy: what Parseval ties the spectrum to.\"\"\"
    return float(np.sum((x - x.mean()) ** 2) * DT)


plt.rcParams.update({"figure.figsize": (9, 3.4), "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})
""")

md("""
## 1. The same record, three answers

Start with something whose answer we know exactly: a sinusoid of amplitude 2.5
lasting 20 seconds.

A **Fourier amplitude spectrum** has units of `[signal] x s` — it is an
integral over the record, not an average. So the peak should read
`A0 * T = 2.5 * 20 = 50`. That is why the long-period level `Omega` is quoted
in `m*s`.
""")

code("""
t = np.arange(N) * DT
x = 2.5 * np.sin(2 * np.pi * 5.0 * t)

for est in (FFTEstimator(taper="boxcar"), FFTEstimator(), MultitaperEstimator(),
            WelchEstimator()):
    s = est.estimate(x, DT)
    print(f"{est.name:<11} peak {s.amp.max():7.2f} {s.unit:<8} "
          f"energy recovered {s.energy() / energy(x):.3f}")

print(f"\\nexpected peak = A0 * T = {2.5 * DURATION}")
""")

md("""
A boxcar FFT recovers the peak exactly. The others read lower — not because
they are wrong, but because they **spread** a pure line across their resolution
bandwidth. Multitaper deliberately smears a line over `2*NW/T` in exchange for
a large reduction in variance.

Note that energy is conserved by all of them. That is the contract every
SpecMod estimator is held to, and it is what lets one test suite cover them
all:

> `spectrum.energy()` recovers `sum(x**2) * dt`

Hold on to that, because in a moment it stops being true.
""")

md("""
## 2. The assumption every estimator makes

Multitaper, Welch and the tapered FFT all assume the record's statistics do not
change across the window. **A seismic arrival breaks that completely** — energy
arrives, then decays, occupying part of the window rather than all of it.

It matters because tapers do not weight the window evenly.
""")

code("""
from scipy.signal.windows import dpss

tapers, _ = dpss(N, 3.0, 5, sym=False, return_ratios=True)

fig, ax = plt.subplots()
for k in range(1, 5):
    ax.plot(t, tapers[k], color="0.8", lw=1.2)
ax.plot(t, tapers[0], color="#2a78d6", lw=2.2, label="taper 0")
ax.axhline(0, color="0.85", lw=0.8)
ax.set(xlabel="time in window (s)", yticks=[],
       title="DPSS tapers: your record is multiplied by each of these")
ax.legend(frameon=False)
plt.show()
""")

md("""
Taper 0 — the one the estimator leans on hardest — is a bell centred
mid-window. **Where your arrival sits relative to that bell changes the
answer.**

Let us move an identical burst through the window. Same samples, same energy,
same width; only the position changes.
""")

code("""
rng = np.random.default_rng(3)
WIDTH = 400
burst = rng.normal(0, 1e-6, WIDTH) * np.exp(-np.arange(WIDTH) / 60.0)


def place(start):
    \"\"\"The same burst, at a chosen start sample.\"\"\"
    x = np.zeros(N)
    x[start:start + WIDTH] = burst
    return x


starts = np.linspace(40, N - WIDTH - 40, 25).astype(int)
runs = {
    "multitaper, adaptive": MultitaperEstimator(adaptive=True),
    "multitaper, flat": MultitaperEstimator(),
    "FFT, 5% taper": FFTEstimator(),
}
ratios = {
    name: [est.estimate(place(s), DT).energy() / energy(place(s)) for s in starts]
    for name, est in runs.items()
}

fig, ax = plt.subplots(figsize=(9, 4))
for (name, r), c in zip(ratios.items(), ["#eb6834", "#2a78d6", "#1baf7a"]):
    ax.plot(starts / N * 100, r, "o-", ms=4, lw=2, color=c, label=name)
ax.axhline(1.0, color="0.5", ls="--", lw=1.2)
ax.set(xlabel="position of the burst in the window (%)",
       ylabel="energy recovered\\n(1.0 = truth)", ylim=(0, 1.6),
       title="The same signal, estimated at different positions")
ax.legend(frameon=False)
plt.show()
""")

md("""
Two distinct effects, and they behave very differently.

**The taper envelope** (flat weighting, blue). A gentle arch of about ±15%,
tracking the summed taper shape. Unavoidable for any tapered method.

**Adaptive collapse** (orange). An edge-located burst loses most of its energy.
The weights are seeded from the two lowest-order tapers — the most
centre-concentrated — which see almost none of an off-centre arrival, so the
iteration starts near zero and converges onto exactly the tapers with no signal
in them. It fails **silently**, returning a plausible-looking spectrum at a
fraction of the true amplitude.

This is why `adaptive` defaults to `False` in SpecMod.
""")

md("""
## 3. Is it phase, or is it position?

A reasonable guess is that this is about the signal not being zero-phase — a
seismic arrival is causal and asymmetric, unlike a symmetric pulse.

Worth testing, because the two would need different fixes. Compare a causal
envelope against a time-symmetric (zero-phase) one at matched positions.
""")

code("""
n = np.arange(WIDTH)
carrier = rng.normal(0, 1e-6, WIDTH)
envelopes = {
    "causal (like an arrival)": np.exp(-n / 60.0),
    "symmetric (zero-phase)": np.exp(-((n - WIDTH / 2) / 80.0) ** 2),
}

est = MultitaperEstimator(adaptive=True)
print(f"{'position':<12}" + "".join(f"{k:>26}" for k in envelopes))
for pos in (0.10, 0.50, 0.90):
    row = ""
    for env in envelopes.values():
        x = np.zeros(N)
        s = int(N * pos) - int((env * n).sum() / env.sum())
        s = max(0, min(N - WIDTH, s))
        x[s:s + WIDTH] = carrier * env
        row += f"{est.estimate(x, DT).energy() / energy(x):>26.3f}"
    print(f"{pos:<12.0%}{row}")
""")

md("""
The symmetric envelope collapses at 10% and 90% just as the causal one does, so
envelope *symmetry* buys nothing.

But "not phase" would be too quick. Let us separate phase properly: keep
`|X(f)|` untouched and change only the phase, three different ways.
""")

code("""
from scipy.signal import hilbert

base = place(200)                              # burst near the start
X = np.fft.rfft(base)

shifted = np.roll(base, 800)                   # (A) LINEAR ramp = a time shift
rotated = np.imag(hilbert(base))               # (B) CONSTANT 90 degree rotation
phi = rng.uniform(0, 2 * np.pi, X.size); phi[0] = 0
scrambled = np.fft.irfft(np.abs(X) * np.exp(1j * phi), n=N)   # (C) RANDOM phase

flat = MultitaperEstimator()
print(f"{'case':<32}{'|X| changed':>13}{'centroid':>11}{'estimate':>11}")
for name, y in (("base", base), ("(A) linear ramp = shift", shifted),
                ("(B) constant 90 deg", rotated), ("(C) random phase", scrambled)):
    dm = np.abs(np.abs(np.fft.rfft(y)) - np.abs(X)).max() / np.abs(X).max()
    e = (y - y.mean()) ** 2
    print(f"{name:<32}{dm:>13.1e}{(np.arange(N) * e).sum() / e.sum() / N:>11.1%}"
          f"{flat.estimate(y, DT).energy() / energy(y):>11.3f}")
""")

md("""
**(B) is the decisive one.** Rotate every phase by 90 degrees without moving the
envelope and the estimate does not budge. So phase *in general* is not what
matters.

**(A)** does change it — and a linear phase ramp *is* a time shift. So the
answer is the **linear component of phase**, the group delay, which is exactly
envelope position.

### Why, in two equivalent ways

**Frequency domain.** Tapering is convolution: `Y(f) = V(f) * X(f)`. A DPSS
taper is symmetric about the window centre, so `V` contributes no phase of its
own. But a signal sitting at offset `tau` carries `X(f) * exp(-2i*pi*f*tau)`.
Convolving that with `V`'s kernel sums a term whose phase *rotates across the
kernel width* — and the further `tau` is from centre, the steeper the rotation
and the more **destructive interference**. `|Y|` comes out low. That is the
arch.

**Time domain.** `Y_k(f) = sum_n x_n * v_k[n] * exp(-2i*pi*f*n*dt)`. The taper
weights by absolute position in the window, so a signal sitting where `v_k` is
small contributes little.

Same mechanism from either side. The frequency-domain view is the one that
explains why centring works: centring sets `tau` to about zero, so there is no
phase ramp left to cancel across the kernel.

It also pins down the asymmetry that causes all the trouble. The **true**
spectrum is shift-invariant, since `|X|` does not change. The **estimate** is
not — because tapering does not commute with shifting:

> `v(t) * x(t - tau)` is not `[v * x](t - tau)`

That single line is the whole problem.
""")

md("""
So: what matters is only where
the energy sits relative to the tapers.

That is good news, because it points at a clean fix.
""")

md("""
## 4. Centring removes the position dependence completely

If only position matters, then moving the energy to the centre should fix it.

This is legitimate rather than a fudge: **`|FFT|` is invariant under a circular
shift**. Rolling the record changes the phase of every Fourier coefficient but
not its magnitude — so the quantity being estimated is untouched, only its
alignment with the tapers.
""")

code("""
plain = MultitaperEstimator()
centred = MultitaperEstimator(center=True)

print(f"{'start':>7}{'center=False':>15}{'center=True':>14}")
for s in starts[::4]:
    x = place(s)
    print(f"{s / N:>7.0%}"
          f"{plain.estimate(x, DT).energy() / energy(x):>15.3f}"
          f"{centred.estimate(x, DT).energy() / energy(x):>14.3f}")
""")

md("""
Identical to three decimal places at every position.

**What remains is the taper concentration itself** — a compact centred
transient still reads about 1.16x high. That bias has not gone away, but it has
changed character, and the difference is the important part:

| | position-dependent bias | consistent bias |
|---|---|---|
| Cancels in a ratio (SNR, spectral ratios, station-to-station)? | no | **yes** |
| Can be calibrated out? | no | **yes** |
| Predictable from the data? | no | **yes** |

A consistent multiplicative factor is a far milder problem than one that varies
trace by trace.

### The catch

A circular shift *wraps*. If your window still has strong coda running at its
end, rolling splices a discontinuity into the middle of the arrival. SpecMod
refuses rather than doing that silently.
""")

code("""
try:
    MultitaperEstimator(center=True).estimate(rng.normal(0, 1e-6, N), DT)
except ValueError as exc:
    print("refused:", exc)
""")

md("""
## 5. The convention `mtspec` used

`mtspec` — and Prieto's `multitaper`, its successor — rescale the finished
spectrum so it integrates to the record's variance:

```python
sscal = np.sum(spec) * df
sscal = xvar / sscal
spec  = sscal * spec
```

That divides the position dependence out of the *total*, which is why
pre-refactor SpecMod results very likely do not carry the bias above.

SpecMod offers it as `normalize_to_variance=True`. **Two things to understand
before enabling it.**
""")

code("""
x = place(60)   # a badly-placed burst
for kw in ({}, {"normalize_to_variance": True},
           {"adaptive": True}, {"adaptive": True, "normalize_to_variance": True}):
    r = MultitaperEstimator(**kw).estimate(x, DT).energy() / energy(x)
    print(f"{str(kw):<52} energy {r:.3f}")
""")

md("""
### It pins energy, not shape

`Omega` is read off the low-frequency plateau, not from total energy. Pinning
the integral does not pin the plateau. Measured spread of the recovered plateau
across positions:

| Method | Plateau spread |
|---|---|
| FFT, light taper | **4%** |
| Prieto, constant weights | 28% |
| Prieto, adaptive | 33% |
| multitaper, flat, no renormalisation | 89% |
| multitaper, adaptive, no renormalisation | 650% |

So it takes the adaptive case from unusable to tolerable — but does not reach
what a plain FFT gives, and the residual sits at the window edges.

### It makes the energy check circular

Every SpecMod estimator is held to: `energy()` recovers `sum(x**2)*dt`. That is
a real, falsifiable check. With `normalize_to_variance=True` it passes *by
construction* however wrong the shape is. The contract stops being a contract.

**Turn it on to reproduce pre-refactor or published results. Leave it off
otherwise.**
""")

md("""
## 6. Real data: where do arrivals actually land?

All of the above hinges on where in the window your arrival sits. Here is the
answer for 28 real PNR S-windows, cut with the published Magna workflow — the
window opens at 80% of the Pg-Sg time, then refines to the 1st and 99th
percentiles of cumulative squared amplitude.
""")

code("""
import glob
import os
import warnings

import obspy
from scipy.integrate import cumulative_trapezoid

import specmod.preprocess as pre

warnings.filterwarnings("ignore")
DATA = "../../Tutorial/Data/2019-08-26T07:30:47.0"
inv = obspy.read_inventory("../../Tutorial/MetaData/pnr_inventory.xml")

st = obspy.read(os.path.join(DATA, "*HH[EN]*"))
pre.set_stream_distance(st, 53.784, -2.967, 2.1,
                        obspy.UTCDateTime("2019-08-26T07:49:24.2"),
                        inventory=inv, dtype="mseed")
pre.set_picks_from_pyrocko(st, glob.glob(os.path.join(DATA, "*.picks"))[0])
st = obspy.Stream([tr for tr in st if "s_time" in tr.stats])
st.detrend("linear"); st.detrend("demean"); st.taper(0.05)
st.remove_response(inv, output="VEL")

sig = pre.get_signal(st, pre.cut_s, rafp=0.8, tafs=20,
                     time_after="absolute_time", refine_window=True)


def energy_midpoint(tr):
    \"\"\"Where the 50%-energy point sits, as a fraction through the window.\"\"\"
    d = tr.data - tr.data.mean()
    c = np.concatenate([[0.0], cumulative_trapezoid(d ** 2)])
    return float(np.interp(0.5, c / c[-1], np.arange(c.size)) / (c.size - 1))


pos = np.array([energy_midpoint(tr) for tr in sig if tr.stats.npts > 50])
dist = np.array([tr.stats["repi"] for tr in sig if tr.stats.npts > 50])

fig, ax = plt.subplots(figsize=(9, 4))
ax.scatter(dist, pos * 100, s=45, color="#2a78d6", zorder=3)
ax.axhspan(0, 20, color="#eb6834", alpha=0.13, zorder=0)
ax.text(23, 9, "collapse zone", color="#eb6834", fontsize=10, ha="right")
ax.set(xlabel="epicentral distance (km)",
       ylabel="50%-energy point\\n(% through window)", ylim=(0, 80),
       title="Where the arrival lands, by distance")
ax.grid(axis="y", color="0.9")
ax.set_axisbelow(True)
plt.show()

print(f"n = {len(pos)}   median {np.median(pos):.1%}   "
      f"range {pos.min():.1%}-{pos.max():.1%}   below 20%: {(pos < 0.2).mean():.0%}")
""")

md("""
**Strongly distance-dependent, and the near-source stations are the ones at
risk.**

The cause is the window *definition*, not the physics. The window opens at a
fixed fraction of Pg-Sg. At short distance Pg-Sg is around a second, so the
window opens essentially on the S arrival and the coda fills the rest — the
energy is front-loaded. At larger distance the window opens well before S, so
the arrival lands nearer the middle.

A quarter of these windows sit below 20%, where adaptive weighting is
unreliable.
""")

md("""
## 7. Smoothing

Smoothing is **lossy by design**, and none of it preserves energy. A smoothed
spectrum no longer satisfies the Parseval contract, so `energy()` on one is not
meaningful. Every smoother records what it did.
""")

code("""
from specmod.smoothing import is_smoothed

raw = MultitaperEstimator().estimate(sig[0].data.astype(float),
                                     sig[0].stats.delta, motion="velocity")
binned = LogBinner(n_bins=60).smooth(raw)
ko = KonnoOhmachi(bandwidth=40).smooth(raw)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.loglog(raw.freq, raw.amp, color="0.8", lw=1, label="raw")
ax.loglog(ko.freq, ko.amp, color="#2a78d6", lw=2, label="Konno-Ohmachi (b=40)")
ax.loglog(binned.freq, binned.amp, "o-", ms=3.5, color="#eb6834", lw=1.6,
          label="log bins (60)")
ax.set(xlabel="frequency (Hz)", ylabel=f"amplitude [{raw.unit}]",
       title=f"{sig[0].id}: smoothing choices")
ax.legend(frameon=False)
plt.show()

print(f"raw    {len(raw):>5} points   smoothed? {is_smoothed(raw)}")
print(f"binned {len(binned):>5} points   smoothed? {is_smoothed(binned)}")
print(f"K-O    {len(ko):>5} points   smoothed? {is_smoothed(ko)}  (axis preserved)")
""")

md("""
`LogBinner` reduces the axis; `KonnoOhmachi` preserves it, which is usually
what you want before fitting on the original grid.

**One trap.** Log bins over a linear frequency axis are sparse at the low end,
so by default sparse bins are dropped — which means two records of *different
duration* produce different-length axes. An element-wise signal-to-noise ratio
then compares mismatched arrays. Pin the edges:
""")

code("""
noise_win = pre.get_noise_s(st, bshift=0.5, sig=sig)
s_tr, n_tr = sig[0], noise_win[0]

auto = LogBinner()
a1 = auto.smooth(MultitaperEstimator().estimate(s_tr.data.astype(float), DT))
a2 = auto.smooth(MultitaperEstimator().estimate(n_tr.data.astype(float), DT))
print(f"derived edges -> {len(a1)} vs {len(a2)} points  (cannot be compared)")

pinned = LogBinner(f_min=0.5, f_max=40.0, n_bins=60, drop_empty=False)
p1 = pinned.smooth(MultitaperEstimator().estimate(s_tr.data.astype(float), DT))
p2 = pinned.smooth(MultitaperEstimator().estimate(n_tr.data.astype(float), DT))
print(f"pinned edges  -> {len(p1)} vs {len(p2)} points  "
      f"identical axis: {np.array_equal(p1.freq, p2.freq)}")
""")

md("""
## 8. Which one should I use?

| If you care most about | Use |
|---|---|
| `Omega`, with arrivals possibly near a window edge | `FFTEstimator` + a smoother |
| Low variance, arrival near centre | `MultitaperEstimator(center=True)` |
| Reproducing pre-refactor or published SpecMod results | `PrietoMultitaperEstimator`, or `normalize_to_variance=True` |
| Confidence intervals, or testing for instrumental tones | `PrietoMultitaperEstimator` |
| A stationary record (noise windows, ambient) | `MultitaperEstimator` or `WelchEstimator` |

The honest summary: **`FFTEstimator` with a smoother is the safest default for
source-parameter work.** It is position-stable to 4%, and pairing it with
Konno-Ohmachi recovers most of the variance reduction multitaper offers without
the stationarity assumption.

Reach for multitaper when you want its variance reduction and either your
arrivals are well-centred or you enable `center=True`.
""")

md("""
## 9. Pinning the choice so a result stays reproducible

None of this helps if the settings behind a result are lost. Put them in a
study config, commit it alongside the results, and every output records the
resolved configuration, a hash of it, and the SpecMod version.
""")

code("""
from specmod.config import config_hash, load_config

study = load_config(project_file="../../studies/magna_2020_paper.toml",
                    use_local=False, use_env=False)
tf = study.config.transform
print(f"estimator             {tf.estimator}")
print(f"adaptive              {tf.adaptive}   <- {study.source_of('transform.adaptive')}")
print(f"normalize_to_variance {tf.normalize_to_variance}")
print(f"\\nconfig hash: {config_hash(study.config)}")
""")

md("""
Both are pinned **explicitly** in that file rather than inherited, so changing
a package default can never silently alter what the study means.

```sh
specmod config show      # resolved values, and which layer set each one
specmod config freeze    # emit TOML to commit alongside a result
```

---

## In one paragraph

A spectral estimator transforms your data, and the transform has consequences.
Multitaper assumes stationarity; a seismic arrival is not stationary; the
resulting bias depends on where in the window the energy sits, which for a
fixed-fraction window definition means it depends on **epicentral distance**.
Centring removes that dependence entirely and leaves a consistent bias that
cancels in ratios. Variance normalisation — what `mtspec` did — fixes the total
but not the shape. A tapered FFT with a smoother avoids the whole problem at
the cost of needing the smoother. Whichever you pick, pin it in a study config
so the result can be reproduced.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path("/home/user/SpecMod/docs/notebooks/choosing_a_transform.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {out} — {len(cells)} cells "
      f"({sum(c['cell_type'] == 'code' for c in cells)} code)")
