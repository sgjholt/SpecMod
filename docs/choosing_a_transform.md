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
| Resolving a peak, tone or resonance without smoothing it down | `QuadraticMultitaperEstimator` |
| A record that is clearly non-stationary, or seeing *where* energy sits in time | `CWTEstimator` |

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

<!-- measured: position_table -->
| Position in window | multitaper, adaptive | multitaper, flat | FFT, 5% Tukey |
|---|---|---|---|
| 6% | 0.673 | 0.690 | 1.00 |
| 10% | 0.956 | 0.973 | 1.03 |
| 25% | 1.079 | 1.075 | 1.03 |
| 50% | 1.151 | 1.145 | 1.03 |
| 75% | 1.070 | 1.066 | 1.03 |
| 90% | 0.898 | 0.916 | 1.03 |
| stationary noise | 1.01 | 1.01 | 0.99 |
<!-- /measured -->

One effect, and it is **the taper envelope**: a modest arch, roughly ±15% over
the central band and falling off hard at the extreme edges, tracking the summed
taper shape. It is unavoidable for any tapered method, it is the same under
either weighting, and — because `mtspec` used the same DPSS tapers — it is
present in pre-refactor results rather than introduced here. The FFT column is
the contrast: a 5% Tukey window is nearly flat across the record, so it barely
cares where the burst sits.

Both weightings track the envelope because the residual *is* the envelope.
With `normalize_to_variance=True` putting them on the same absolute scale, the
estimate here agrees with Prieto's `multitaper` to within **0.3%** across the
band, under either weighting, for stationary noise and for bursts at 10%, 50%
and 90%.

> **Gotcha if you implement Thomson weighting yourself.** Eq. 5.1b regularises
> each weight with `(1 − λₖ)·σ²`, and `σ²` has to be in the units of the
> spectrum being weighted. Pass a record's *time-domain* variance against
> PSD-scaled eigenspectra and you overstate it by `1/dt` — 100× at 100 sps. The
> regularisation then swamps the signal term, every weight collapses toward
> zero, and it is worst exactly where the tapers see least of the signal. The
> result looks like a plausible spectrum at a fraction of the true amplitude.
>
> Stationary noise is *insensitive* to this, so a Parseval check on white noise
> will not catch it. Test with an off-centre transient.

### Why adaptive weighting is the default

Leakage suppression is the reason to reach for multitaper at all, and flat
weighting does not provide it. A 2 Hz line 10⁶ times stronger than the
background — a mild version of what a real seismic spectrum does across its
band — with the recovered noise floor measured between 20 and 49 Hz:

<!-- measured: leakage_table -->
| Weighting | recovered floor / true |
|---|---|
| flat | **287x** |
| adaptive | 1.1x |
<!-- /measured -->

`t*` and `f_c` are both read off the high-frequency decay, so a floor nearly
three orders of magnitude too high is not a cosmetic problem — it flattens the
tail and biases both parameters.

The cost is resolution. Adaptive weighting downweights the higher-order tapers
wherever leakage would dominate, so it spends fewer effective degrees of
freedom and returns a noisier estimate in bands where the signal is strong.
Turn it off for a well-conditioned record with little dynamic range, where the
extra averaging buys more than the leakage rejection does.

### It is the *linear* component of phase — that is, position

Worth separating carefully, because the candidates have different fixes. Hold
`|X(f)|` fixed and change only the phase, three ways. The second column
confirms the magnitude spectrum really is untouched — the changes are at
machine precision — and the third tracks where the energy ends up:

<!-- measured: phase_table -->
| Change | max change in `\|X\|` | Envelope centroid | Estimate |
|---|---|---|---|
| (unchanged reference) | -- | 11.2% | 1.053 |
| Linear ramp (= a time shift) | < 1e-12 | 51.2% | 1.159 |
| Constant 90 deg rotation | < 1e-12 | 11.2% | 1.053 |
| Random phase | < 1e-12 | 52.0% | 1.007 |
<!-- /measured -->

The constant rotation is decisive: it changes every sample of the record, yet
the centroid does not move and the estimate does not budge. Phase *in general*
does not matter. What matters is the **linear** component — the group delay —
which is exactly where the envelope sits. Envelope symmetry is irrelevant
too: a symmetric, zero-phase envelope is biased identically at 10% and 90%.

**Why, in two equivalent ways.**

*Frequency domain.* Tapering is convolution, `Y(f) = V(f) * X(f)`. A DPSS taper
is symmetric about the window centre, so `V` adds no phase of its own. But a
signal at offset `τ` carries `X(f)·e^{-2πifτ}`, and convolving that with `V`'s
kernel sums a term whose phase rotates *across the kernel width*. The further
`τ` is from centre, the steeper the rotation and the more destructive
interference — so `|Y|` reads low.

*Time domain.* `Y_k(f) = Σ xₙ vₖ[n] e^{-2πifn·dt}`: the taper weights by
absolute position, so a signal where `vₖ` is small contributes little.

The same mechanism from either side. The frequency-domain view explains why
centring works — it sets `τ ≈ 0`, leaving no phase ramp to cancel.

And it pins down the asymmetry behind all of this: the **true** spectrum is
shift-invariant, but the **estimate** is not, because tapering does not commute
with shifting —

> `v(t)·x(t−τ)  ≠  [v·x](t−τ)`

Which means centring fixes it, and fixes it completely:

<!-- measured: centring_table -->
| Burst start | `center=False` | `center=True` |
|---|---|---|
| 2% | 0.374 | 1.166 |
| 10% | 1.034 | 1.166 |
| 30% | 1.155 | 1.166 |
| 50% | 1.164 | 1.166 |
| 70% | 1.129 | 1.166 |
| 78% | 1.060 | 1.166 |
<!-- /measured -->

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

`Omega` is read from the low-frequency plateau, not from total energy, and
pinning the integral does not pin the plateau. A real S-arrival slid through an
otherwise empty 2000-sample window, across the full range of positions it can
occupy; the figure is the spread of the recovered 1–4 Hz level relative to its
mid-window value:

<!-- measured: plateau_table -->
| Method | Plateau spread |
|---|---|
| FFT, light taper | 7% |
| Prieto, constant weights | 8% |
| Prieto, adaptive | 8% |
| ours, flat, no renormalisation | 15% |
| ours, adaptive, no renormalisation | 15% |
| ours, adaptive, renormalised | 10% |
| ours, adaptive, `center=True` | 0% |
<!-- /measured -->

Renormalisation helps — 15% to 10% — but does not reach what a lightly-tapered
FFT gives, and the residual sits at the extreme edges. Centring is the only
thing here that removes it outright.

For a near-source station a 10% plateau error is about 0.04 in `log10(Omega)`,
or **0.03 magnitude units**. Against the 0.13 m.u. scatter quoted for spectral
`Mw` that is small, but it is systematic rather than random, so it does not
average away across stations at similar distance.

> Earlier revisions of this page reported 89% for flat weighting and 650% for
> adaptive here. The 650% was the adaptive collapse described above and is
> gone. The remaining figures come from a differently-constructed sweep than
> the original and are not directly comparable to it; this table is the one
> `tools/measure_docs.py` reproduces.

### It costs you a diagnostic

SpecMod's estimators are held to a contract: `Spectrum.energy()` must recover
`sum(x**2) * dt`. That is a real, falsifiable check that the normalisation is
sound, and it is what lets one test suite cover every backend.

With `normalize_to_variance=True` that check passes *by construction*, for any
estimator, however wrong the shape is. That matters to you and not just to the
test suite: `energy()` on your own record stops being a measurement you can
act on. If a window is mis-cut or a response is mis-removed, an unnormalised
spectrum shows it and a normalised one does not.

Hence: off by default, on deliberately.

Note what it is *not*. It is a single scalar multiply — the ratio between the
normalised and unnormalised spectra is constant across frequency to machine
precision. So it cannot change any ratio *within* a spectrum, which is exactly
why it cannot fix the plateau above: `Omega` moves with the rest of the
spectrum or not at all.

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

**It is cross-validated against Prieto's package.** Both are put on the same
absolute scale with `normalize_to_variance=True`, then compared bin by bin over
0.5–45 Hz. Median ratio, ours over theirs:

<!-- measured: prieto_agreement -->
| Record | adaptive | flat |
|---|---|---|
| stationary noise | 1.0000 | 1.0000 |
| burst at 10% | 1.0000 | 1.0000 |
| burst at 50% | 1.0000 | 1.0000 |
| burst at 90% | 0.9999 | 0.9999 |
<!-- /measured -->

Worst-case deviation anywhere in the band is 0.3%. Two independent
implementations of Thomson's method agreeing to that level is the strongest
evidence available that the native estimator is right — and it is what
distinguishes the current adaptive weighting from the version it replaced,
which disagreed by a factor of five.

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

### `QuadraticMultitaperEstimator`

Multitaper with the curvature bias removed, after Prieto *et al.* (2007).

Averaging `K` tapers smooths the spectrum across the inner band `[-W, W]`.
Where the true spectrum is curved that smoothing does not average out — it
pulls peaks down and fills troughs in, in proportion to the second derivative.
The quadratic inverse method estimates that second derivative and subtracts the
bias it causes.

<!-- measured: quadratic_table -->
| Measurement | multitaper | quadratic |
|---|---|---|
| single line, peak / true | 0.87 | **1.02** |
| two lines 0.70 Hz apart, peak/trough | 8.9 | **12.2** |
| white noise, level ratio to multitaper | 1.00 | 1.01 |
| Brune tail 25-49 Hz, ratio to multitaper | 1.00 | 1.07 |
| Brune corner, fitted f_c (true 4.0 Hz) | 3.91 | 4.09 |
<!-- /measured -->

The first row is the clearest statement of what it does: a pure sine has a
known Fourier amplitude, `A·T/2`, and the ordinary estimate recovers 87% of it
while this recovers 102%. The white-noise row is the control — no curvature, so
nothing should change, and nothing does. Without that row the first two would
be equally consistent with an estimator that simply sharpens everything.

On a corner it is a small improvement over the ordinary estimate and no more.
Over 40 realisations of a true 4 Hz Brune corner:

| Estimator | median `f_c` | bias | IQR |
|---|---|---|---|
| FFT, light taper | 3.980 | **−0.020** | **0.055** |
| multitaper | 3.913 | −0.087 | 0.133 |
| quadratic | 4.077 | +0.077 | 0.139 |

The two multitaper variants carry opposite-signed bias of similar size and the
same scatter, so once both are counted they are indistinguishable. **A
lightly-tapered FFT recovers a corner frequency better than either**, and about
100× faster. Reach for the quadratic estimator when the feature of interest is
a peak or a line — an instrumental tone, a site resonance, a spectral hole —
not to squeeze a corner.

> **Gotcha if you call `qiinv` directly**, including through Prieto's package.
> It builds cross-spectra from `wt·yk` and never divides by `Σw²`, so its
> diagonal averages to `(1/K)·Σw²|y|²` where the adaptive estimate is
> `Σw²|y|²/Σw²`. Hand it raw Thomson weights and the result is scaled down by
> `Σw²/K` wherever the weights bite — on a Brune spectrum that is 0.80 at
> 10–25 Hz and 0.57 at 25–49 Hz.
>
> The symptom is a smooth deficit confined to the falling tail, which reads
> convincingly as a curvature artefact. The tell is that it disappears with
> flat weighting: a real property of the correction cannot depend on how the
> eigencoefficients were weighted going in. SpecMod renormalises so `Σw² = K`
> before the fit. Upstream does not, and leans on its global variance rescaling
> to mask it — which cannot work, because the deficit varies with frequency and
> that is a single scalar.

It costs a least-squares solve per frequency bin, so it is roughly two orders
of magnitude slower than the ordinary estimator. Not one for a whole catalogue.

**It is vendored, not imported.** The numerical core lives in
`specmod/_vendor/qiinv.py` under Prieto's MIT licence, because upstream's
`qiinv` raises on every weighting scheme under numpy ≥ 2 — four lines assign
shape-`(1,)` arrays into scalar slots. The vendored copy carries those fixes,
replaces a numba-jitted Goertzel recursion with an exact vectorised
equivalent (dropping numba from the dependency graph), and is cross-validated
against the patched upstream to 1e-9 in the test suite. It does **not** need
`specmod[multitaper]` installed.

---

## Zero-padding: what it fixes, and what it does not

A common expectation is that padding suppresses ringing. It does not — and the
two effects it *is* confused with pull in different directions, so they are
worth separating.

<!-- measured: padding_table -->
| Zero-padding | sidelobes, boxcar | sidelobes, Tukey | worst peak / true |
|---|---|---|---|
| none | 7e-04 | 7e-07 | 0.638 |
| 2x | 8e-04 | 7e-07 | 0.900 |
| 8x | 7e-04 | 6e-07 | 0.995 |
<!-- /measured -->

**Leakage is the taper's job, not padding's.** Read across the first two
columns: padding leaves the sidelobe floor exactly where it was, while
switching from a boxcar to a 5% Tukey moves it by three orders of magnitude.
Leakage comes from truncating the record; padding adds zeros *outside* the
truncation, so there is nothing for it to undo. It interpolates the frequency
grid — you see the same sidelobes sampled more finely.

**Scalloping is what padding fixes.** The last column places a pure line
worst-case between two bins: unpadded it reads **36% low**, because the peak
falls between samples of the spectrum. Two-fold padding recovers most of it,
eight-fold nearly all.

**But that only matters for features narrower than a bin.** On a smooth
Brune-like spectrum — which is what a source model is fitted to — padding
changes the recovered level by a factor of 1.0000 at both 2× and 8×. So for
`Omega`, `f_c` and `t*` there is nothing to gain. It matters when you are
reading the amplitude of a *line*: an instrumental tone, a site resonance, the
peaks `QuadraticMultitaperEstimator` exists to measure.

Hence `n_fft` defaults to `None`. Set it when measuring a narrow feature.

Padding is otherwise numerically free here: amplitude normalisation is keyed
off the record duration, so padding is a pure interpolation and does not move
any level. (Keyed off `len(freq)`, as the pre-refactor code was, it would have.)

### A note on record length

Cut windows are not round numbers. On the 28 PNR S-windows, lengths run
181–737 samples, **17 of them odd**, and several are prime (677, 479, 271,
181). Two consequences:

- Odd lengths have **no Nyquist bin** — the top bin sits half a bin below
  `fs/2` and is folded like any other. SpecMod handles this from the frequency
  axis rather than assuming; see `Spectrum._fold_factor`.
- A prime length makes the FFT slower, 1.77× across those windows. In absolute
  terms it is 0.2 ms per event, so padding to a fast length is not worth a
  changed default — but it is worth knowing if you ever process long records.

### `CWTEstimator`

A continuous wavelet transform on an L2-normalised Morlet, time-averaged to an
ordinary `Spectrum`. **It is the only estimator here that does not assume
stationarity**, which is the assumption a seismic arrival breaks.

<!-- measured: cwt_table -->
| Record | FFT | multitaper | CWT |
|---|---|---|---|
| white noise | 0.998 | 0.995 | 0.995 |
| 5 Hz sinusoid | 0.999 | 0.996 | 1.061 |
| off-centre burst | 1.032 | 1.118 | 0.998 |
<!-- /measured -->

The last row is the point: multitaper reads the burst 12% high because the
taper envelope does not know where the energy sits, while the wavelet transform
recovers it to 0.2%. The cost is frequency resolution — the CWT smears a line
over its scale bandwidth, which is why it reads 6% high on the sinusoid.

It gives you **two outputs from one transform**:

```python
scalogram = CWTEstimator().scalogram(trace, dt)   # full time-frequency surface
spectrum  = scalogram.time_average()              # ordinary Spectrum, fits as usual
qc        = scalogram.qc()                        # the checks below
```

`Scalogram.power` is `|W(a,b)|²` in the L2-Morlet convention, which is **not**
an amplitude spectrum — the units are `[signal]²·time`. The conversion happens
in exactly one place, `time_average()`, so there is one normalisation path and
one test rather than two things to get wrong.

**The cone of influence is a real bandwidth limit, not decoration.** A window
cannot resolve a period longer than itself, and the CWT is the only estimator
here that makes that explicit. `time_average()` drops frequencies with no
edge-free samples rather than emitting zero for them, so a masked spectrum is
shorter than an unmasked one. That is the honest answer — zero would read as
"no energy here" instead of "no measurement here", and would take a log-space
fit to `-inf`.

`scalogram.qc()` returns the §4.4.2 checks: lowest resolved frequency, median
COI coverage, temporal energy concentration (a Gini coefficient — this is what
separates a glitch from an arrival, which amplitude-only SNR cannot do), and
the first-half/second-half spectral ratio. They are computed and recorded,
never used to silently drop a trace.

> **Gotcha if you port this.** `C_δ`, the reconstruction constant, is *computed*
> against the actual scale grid rather than taken from Torrence & Compo's table.
> Substituting the tabulated 0.776 for ω₀=6 looks like a tidy-up and is not: the
> published figure is the continuum limit, the reconstruction sum is discrete,
> and using it leaves recovered energy about 7% low. The computed value drifts
> with `dj` precisely because it is absorbing that discretisation — which is why
> the recovered *energy* does not drift with `dj`.

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

```{toctree}
:hidden:

notes/window_position
```
