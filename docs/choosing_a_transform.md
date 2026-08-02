# Choosing a transform, and what it does to your data

Every spectral estimator is a *transform of your record*, not a neutral window
onto it. Each one trades something. This page states what each trade is, with
measured numbers, so the choice is made deliberately rather than inherited from
a default.

The short version:

| If you care most about | Use |
|---|---|
| The long-period level `Omega`, and your arrival may sit near a window edge | `FFTEstimator` + a smoother |
| A stable spectrum with low variance, arrival near window centre | `MultitaperEstimator` |
| Reproducing pre-refactor or published SpecMod results | `PrietoMultitaperEstimator` — same lineage as `mtspec` |
| Confidence intervals, or testing for instrumental tones | `PrietoMultitaperEstimator` |
| A stationary record — noise windows, ambient measurements | `MultitaperEstimator` or `WelchEstimator` |

---

## The thing that governs all of it: stationarity

Multitaper, Welch and the windowed FFT all assume the record's statistics do
not change across the window. **A seismic arrival breaks that assumption
completely.** The energy arrives, then decays; it occupies part of the window,
not all of it.

That matters because the tapers do not weight the window evenly. DPSS taper 0 —
the one the estimator leans on hardest — is a bell centred mid-window. Where
your arrival sits relative to that bell changes the answer.

### Measured

An identical burst — same samples, same energy, same width — moved through a
2000-sample window. Values are estimated energy divided by true energy, so 1.0
is correct:

| Position in window | multitaper, adaptive | multitaper, flat | FFT, 5% Tukey |
|---|---|---|---|
| 6% | 0.039 | 0.691 | 1.00 |
| 10% | 0.203 | 0.974 | 1.03 |
| 25% | 1.253 | 1.076 | 1.03 |
| 50% | 1.317 | 1.146 | 1.03 |
| 75% | 1.244 | 1.067 | 1.03 |
| 90% | 0.149 | 0.917 | 1.03 |
| stationary noise | 0.96 | 1.00 | 0.99 |

Two distinct effects:

**The taper envelope** (flat weighting). A modest arch, roughly ±15%, tracking
the summed taper shape. Unavoidable for any tapered method.

**Adaptive collapse** (adaptive weighting, *this implementation*). An
edge-located burst loses 80–96% of its energy. The weights are seeded from the
two lowest-order tapers, which are the most centre-concentrated and see almost
none of an off-centre arrival — so the iteration starts near zero and converges
onto exactly the tapers with no signal in them. It fails silently, returning a
plausible-looking spectrum at a fraction of the true amplitude.

This is why `adaptive` **defaults to `False`**.

### It is position, not phase

Worth separating, because the two would have different fixes. A *symmetric*
(zero-phase) envelope collapses identically at 10% and 90% — 0.322 at both — so
symmetry does not rescue it. What matters is only where the energy sits
relative to the tapers.

Which means centring fixes it, and fixes it completely:

| Burst start | `center=False` | `center=True` |
|---|---|---|
| 2% | 0.389 | 1.162 |
| 10% | 1.053 | 1.162 |
| 30% | 1.153 | 1.162 |
| 50% | 1.159 | 1.162 |
| 70% | 1.126 | 1.162 |
| 78% | 1.053 | 1.162 |

Identical to three decimals at every position. `MultitaperEstimator(center=True)`
circularly shifts the record so its energy centroid sits mid-window. This is
legitimate rather than a fudge: `|FFT|` is invariant under a circular shift, so
the quantity being estimated does not change — only its alignment with the
tapers.

**What remains is the taper concentration itself**: a compact centred transient
still reads about 1.16× high under flat weighting. That is a *consistent*
multiplicative bias rather than a position-dependent one, and the distinction
matters — a consistent factor cancels in any ratio (signal-to-noise, spectral
ratios, relative amplitudes between stations) and can be calibrated. A
position-dependent one cannot.

Off by default because a circular shift wraps. It refuses, rather than
silently splicing a discontinuity into the arrival, when the window edges are
not quiet — controlled by `center_edge_tolerance`. A cut S-window whose coda is
still running at the end cannot be safely rolled; taper first, widen the
window, or use `FFTEstimator`.

---

## Where arrivals actually land

Measured on the 28 PNR S-windows, cut with the published Magna workflow — the
window opens at 80% of the Pg–Sg time, then refines to the 1st and 99th
percentiles of the cumulative squared-amplitude integral.

Position of the 50%-energy point through the refined window:

| | value |
|---|---|
| median | 40.8% |
| range | 3.1% – 72.0% |
| below 20% | **25% of windows** |

**It is strongly distance-dependent:**

| Epicentral distance | Position |
|---|---|
| < 6 km | 3% – 36% |
| > 10 km | 32% – 72% |

The cause is the window *definition*, not the physics. The window opens at a
fixed fraction of Pg–Sg. At short distance Pg–Sg is ~1 s, so the window opens
essentially on the S arrival and the coda fills the remainder — the energy is
front-loaded. At larger distance the window opens well before S, so the arrival
lands nearer the middle.

**Practical consequence: near-source stations are the ones at risk.** Full
per-trace table in [`notes/window_position.md`](notes/window_position.md).

---

## Normalising to the variance

`mtspec` and Prieto's `multitaper` both rescale the finished spectrum so it
integrates to the record's variance. From `multitaper/mtspec.py`:

```python
sscal = np.sum(spec) * df
sscal = xvar / sscal
spec  = sscal * spec
```

That divides the position dependence out — which is why Prieto's package
recovers energy at exactly 1.000 at every position, under all three of its
weighting schemes.

SpecMod offers this as `MultitaperEstimator(normalize_to_variance=True)`.
**Two things to understand before turning it on.**

### It fixes energy, not shape

`Omega` is read from the low-frequency plateau, not from total energy. Pinning
the integral does not pin the plateau. Measured on a real S-arrival embedded at
different positions, spread of the recovered 1–4 Hz plateau relative to its
value at mid-window:

| Method | Plateau spread |
|---|---|
| FFT, light taper | **4%** |
| Prieto, constant weights | 28% |
| Prieto, adaptive | 33% |
| ours, flat, no renormalisation | 89% |
| ours, adaptive, no renormalisation | 650% |

So renormalisation takes the adaptive case from unusable to tolerable and the
flat case from 89% to roughly 30%. It does **not** reach the 4% an FFT gives.
The residual sits almost entirely at the extreme edges — at 10% through the
window the plateau still reads about 20–24% low.

For a near-source station that is roughly 0.1 in `log10(Omega)`, or about
**0.07 magnitude units**. Against the 0.13 m.u. scatter quoted for spectral
`Mw`, that is small but not nothing, and it is systematic rather than random.

### It makes the Parseval check circular

SpecMod's estimators are held to a contract: `Spectrum.energy()` must recover
`sum(x**2) * dt`. That is a real, falsifiable check that the normalisation is
sound, and it is what lets one test suite cover every backend.

With `normalize_to_variance=True` that check passes *by construction*, for any
estimator, however wrong the shape is. The contract stops being a contract.

Hence: off by default, on deliberately.

### When to turn it on

**Reproducing pre-refactor or published SpecMod results.** `mtspec` did this,
so pre-refactor results already carry the convention. `studies/magna_2020_paper.toml`
pins it explicitly for that reason.

---

## The estimators

### `FFTEstimator`

Taper, transform, fold, normalise. Fastest, and the most position-stable at 4%.
Highest variance — a periodogram bin is chi-square with 2 degrees of freedom, so
roughly 100% scatter bin to bin. Pair it with a smoother.

Two taper corrections, and they are not interchangeable:

- `taper_correction="energy"` (default) divides by `sqrt(mean(w**2))`,
  preserving total power. Parseval holds exactly.
- `taper_correction="amplitude"` divides by `mean(w)`, preserving the peak of a
  coherent sinusoid so it reads `A0 * T`. Right for a monochromatic line.

For a 5% Tukey taper they differ by well under a percent, but the difference is
real for heavier tapers.

### `MultitaperEstimator`

Averages `K` DPSS-tapered estimates. Large variance reduction — the reason the
published workflow uses it. Costs frequency resolution (`2*NW/T`) and carries the
position dependence above.

`n_tapers` must not exceed `2*NW - 1`; beyond that the tapers are poorly
concentrated and add leakage rather than reducing variance. Exceeding it raises
rather than silently degrading.

### `PrietoMultitaperEstimator`

Prieto's `multitaper` package, the direct successor to the Fortran library
`mtspec` wrapped. Optional: `pip install specmod[multitaper]`.

Worth having for three things the native estimator does not offer:

- **The closest available proxy for pre-refactor behaviour.** Same author, same
  lineage, so "did the old code do this?" is a pip install rather than a Docker
  build.
- **Jackknife confidence intervals** — `confidence_interval()` returns the two
  bounds as spectra. Log-symmetric about the estimate, roughly 2.5x either way
  at `nw=3, kspec=5`.
- **Thomson's F-test for periodic components** — `f_test()`. Finds instrumental
  or cultural tones that would otherwise be read as source structure.

Prieto (2022) additionally describes a **quadratic** multitaper (`qiinv`) with
better peak resolution than adaptive weighting — the estimate most relevant to
measuring a corner frequency. It raises for every weighting under current numpy
and is therefore not exposed. Verified working surface:

| Method | adaptive | constant | eigenvalue |
|---|---|---|---|
| spectrum | works | works | works |
| F-test | works | works | works |
| jackknife | works | **fails** | works |
| quadratic | **fails** | **fails** | **fails** |

Two further things to know. Its variance normalisation is **baked in and cannot
be disabled**, so `Spectrum.energy()` on its output is right by construction rather
than as a check. And `confidence_interval()` raises for
`weighting="constant"`: an upstream shape bug in `multitaper.utils.jackspec`
leaves the degrees-of-freedom array two-dimensional, so the interval broadcasts
to `(nfft, nfft)`. `adaptive` and `eigenvalue` are unaffected.

### `WelchEstimator`

Segment-averaged. Trades frequency resolution for variance, which is the right
trade for a **noise** window where a stable level matters more than structure.
Not recommended for signal windows — segmenting a transient means most segments
contain no signal.

---

## Smoothing

Smoothing is lossy, and **none of it preserves energy** — that is the point of
it. A smoothed spectrum no longer satisfies the Parseval contract, so
`Spectrum.energy()` on one is not meaningful. Every smoother records what it
did under `meta["smoothing"]`; `specmod.smoothing.is_smoothed()` checks it.

**`LogBinner`** averages into log-spaced bins. Default edges derive from the
record — `1/T` to Nyquist, the band the record can actually represent. The
`geometric` statistic (mean of `log10(amp)`) is the default and the right choice
for a quantity spanning orders of magnitude: an arithmetic mean over a decade of
amplitudes is dominated by its largest member.

Log bins over a linear frequency axis are inevitably sparse at the low end. By
default sparse bins are dropped, which means **two records of different duration
produce different-length axes.** For an element-wise signal-to-noise ratio, pin
the edges and keep the length fixed:

```python
binner = LogBinner(f_min=0.5, f_max=40.0, n_bins=60, drop_empty=False)
```

Explicit edges are honoured exactly and never clamped to each spectrum's own
range, precisely so this works.

**`KonnoOhmachi`** smooths with a constant-width window in log-frequency —
narrow at low frequency, wide at high. It preserves the frequency axis, which is
what you want before fitting on the original grid. Bandwidth `b` is inverse:
smaller smooths harder; 40 is conventional.

---

## Reproducing the published Magna configuration

```toml
[transform]
estimator = "multitaper"
time_bandwidth = 3.0
n_tapers = 5
adaptive = true                 # mtspec's default; the manuscript does not say
normalize_to_variance = true    # mtspec's convention
```

The first of those is unverified — see `docs/REFACTOR_PLAN.md` §5.2.5. The
0.1.1 re-run tests it directly.
