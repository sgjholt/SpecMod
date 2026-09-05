# How a spectrum is processed, end to end

Every step from a raw waveform to a moment magnitude, with the equation each
one applies and a pointer to the code that applies it. Together with the
normalisation and units conventions in [§3](#the-parseval-contract) and
[§4](#4-amplitude-convention), this is the reference for what SpecMod computes
and in which units.

This exists because a pipeline described only in prose is a pipeline nobody
can check. Writing the noise rotation down as an equation rather than a loop
is what revealed that it was solving for a quantity it could compute directly —
and that the search it used instead was both biased and not reproducible
([§6](#6-noise-rescaling-and-rotation)). The rest of this document is the same exercise
applied to every other stage.

**Notation.** A record is $x_n$, $n = 0 \dots N-1$, sampled at interval
$\Delta t$, so its duration is $T = N \Delta t$ and its Nyquist frequency is
$f_{\mathrm{Nyq}} = 1/(2\Delta t)$. Frequency-domain quantities are written
$X(f)$.

---

## Pipeline at a glance

| # | Stage | Code |
| --- | --- | --- |
| 1 | Instrument correction and detrending | `preprocess` |
| 2 | Window selection and refinement | `preprocess.cut_s`, `signal_intensity` |
| 3 | Spectral estimation | `transforms/` |
| 4 | Amplitude convention | `core.Spectrum.to_kind`, `spectral.Spectrum.psd_to_amp` |
| 5 | Log binning | `core.collection.log_bin` |
| 6 | Noise rescaling and rotation | `core.collection.parseval_scale`, `core.noise.NOISE_MODELS` |
| 7 | Signal-to-noise and bandwidth | `core.collection.find_bandwidth` |
| 8 | Source model fitting | `sources/`, `fitting/` |
| 9 | Moment and magnitude | `magnitude`, `spreading` |

---

## 1. Instrument correction

Applied in the caller, not inside the package, so that the choice stays
visible in the run script. The tutorial sequence is linear detrend, demean,
5% cosine taper, then response removal to velocity.

Demeaning matters more than it looks. A non-zero mean puts all of its energy
in the DC bin, which every estimator here discards — so leaving it in loses
energy that the Parseval check in [§3](#3-spectral-estimation) would report as
a failure. `transforms.prepare_record` therefore demeans unconditionally,
whatever the caller did.

## 2. Window selection

The S-window opens at a fixed fraction $r$ of the elapsed P–S time after the
P arrival:

$$t_{\text{start}} = t_P + r\,(t_S - t_P), \qquad r = 0.8 \text{ by default}$$

and runs for a fixed duration (`time_after="absolute_time"`) or a multiple of
the P–S time (`"relative_ps"`).

**Refinement.** The window is then tightened to the part of it that actually
carries energy. With the cumulative squared amplitude

$$I(t) = \int_0^t x^2(\tau)\,\mathrm{d}\tau \Big/ \int_0^{T} x^2(\tau)\,\mathrm{d}\tau$$

the refined window runs between the times at which $I$ reaches the 1st and
99th percentiles. This is why the 28 PNR windows come out at 1.8–3.7 s rather
than the nominal 20 s.

**The noise window** ends $0.2$ s before the P arrival and is *asked for* the
same length as the refined signal window, so the two would be comparable
without further correction. In practice it rarely gets it: the window is
truncated wherever the record does not start early enough, and on the PNR data
that is **all 28 windows**, which run 1.1–1.7 s against 1.8–3.7 s signals.

Each trace therefore carries two window records: `wstart`/`wend`, which are
what the trace actually holds, and `wstart_requested`/`wend_requested`, which
are what the cut asked for. They differ on every truncated noise trace, and by
up to half a sample on signal traces, where `trim` snaps to sample boundaries.
The resulting resolution difference is what
[§6](#6-noise-rescaling-and-rotation) corrects, and it is why
`SpectrumPair` carries a `resolution_floor` rather than assuming the two
spectra share a frequency axis. `tests/test_preprocess.py` pins this so that a
change which quietly started delivering full-length noise windows would move
every noise spectrum in the reference visibly rather than silently.

<a id="3-spectral-estimation"></a>

## 3. Spectral estimation

Every estimator returns a **one-sided** spectrum over $(0, f_{\mathrm{Nyq}}]$
obeying one contract, so a single test suite pins all of them.

### The FFT path

Taper, transform, fold, normalise:

$$A(f_k) = \frac{2\,\Delta t}{\sqrt{\langle w^2\rangle}}\,\bigl|\,\mathrm{DFT}[x_n w_n](k)\,\bigr|$$

with the factor of 2 applied at every bin **except** DC and — for even
transform length — Nyquist, which have no negative-frequency twin. Getting
that exception wrong is a silent error at the two ends of the band, which is
why the fold factor is computed per bin rather than applied as a scalar.

**Taper correction.** $\sqrt{\langle w^2 \rangle}$ preserves total power, which
is the right choice for a transient — and a seismic arrival is a transient.
The alternative, dividing by $\langle w \rangle$, preserves the peak of a
coherent sinusoid instead. For a Tukey taper with $\alpha = 0.05$ they differ
by well under a percent, but the choice is explicit rather than implied.

**Normalisation is keyed off $\Delta t$, never off the transform length.** The
DFT sums over the $N$ non-zero samples whatever $n_\mathrm{fft}$ is, so it
approximates the same continuous transform and merely evaluates it on a finer
grid. Zero-padding therefore changes frequency sampling and nothing else. The
pre-refactor code used $\mathrm{len}(f)$ as a stand-in for $T/2$, which is
true only for an unpadded one-sided transform — padding to $4N$ halved the
amplitude.

### The Parseval contract

Every estimator is held to

$$\sum_n x_n^2\,\Delta t \;=\; \int_0^{f_{\mathrm{Nyq}}} \frac{A^2(f)}{2}\,\mathrm{d}f$$

which is a falsifiable check rather than a convention: an estimator arriving
on a different amplitude convention fails it instead of silently rescaling
$\Omega$. It is what lets multitaper, Welch and the CWT be interchangeable
here despite reaching their amplitudes by very different routes.

### Multitaper

$K$ orthogonal DPSS tapers $v^{(k)}$ of time-bandwidth $NW$ give eigenspectra
$S_k(f)$, combined with adaptive weights (Thomson 1982, eq. 5.1b):

$$d_k(f) = \frac{\sqrt{\lambda_k}\,S(f)}{\lambda_k S(f) + b_k(f)}, \qquad b_k(f) = (1-\lambda_k)\,\sigma^2$$

iterated to convergence. The regularisation term $b_k$ has units of
$[x]^2\!\cdot\!s$ — the same as the eigenspectra — so $\sigma^2$ must be
derived from the spectra themselves and *not* from `x.var()`, which is
$[x]^2$ and too large by $1/\Delta t$. That units mismatch made the weights
collapse toward zero for off-centre transients; the recovered amplitude ratio
went from 0.203 to 0.956 once corrected.

### Continuous wavelet transform

An L2-normalised Morlet at scale $s$, with the Torrence & Compo
reconstruction factor $C_\delta$, and a cone of influence that discards the
scales a window of this length cannot resolve.

> **Open issue, `cwt` only.** Every other estimator reproduces exactly across
> machines. `cwt`'s signal amplitudes and frequency axis do too — but its
> *post-rotation noise* differs by 1-2% on 4 of 28 stations between Linux and
> macOS runners carrying identical library versions. It is the one transform
> implemented here from scratch, which makes it the natural suspect, and
> evaluating PyWavelets as an independent implementation is tracked in
> `REFACTOR_PLAN.md` §4.4.4. Note that the evidence does not yet point at the
> transform itself: what moves is downstream of it.

The COI limit is about $1.4\times$ stricter than $1/T$ in practice — measured
as the ratio of median resolution floors over the 28 PNR windows — which is why
the CWT's usable band opens higher than multitaper's on the same record.

## 4. Amplitude convention

Two conventions are in play and the factor between them is exactly 2.

| Kind | Definition | Energy relation |
| --- | --- | --- |
| `FAS` | $2\lvert X(f)\rvert$, folded | $E = \int A^2/2 \,\mathrm{d}f$ |
| `MAGNITUDE` | $\lvert X(f)\rvert = \lvert\mathrm{rfft}(x)\rvert\,\Delta t$, unfolded | $E = 2\int \lvert X\rvert^2\,\mathrm{d}f$ |

`core.Spectrum` carries `FAS`. **The pipeline converts to `MAGNITUDE`**,
because that is the convention $\Omega$ is defined in: the long-period plateau
of the displacement spectrum is $\lvert X(f\!\to\!0)\rvert = \lvert\int u\,\mathrm{d}t\rvert$
and $M_0 \propto \Omega$. Folding would put $M_0$ out by two, which is
$0.2$ magnitude units on every event.

Both are self-consistent and both recover the record's energy. Anyone reading
`core.Spectrum.amp` and calling it $\Omega$ needs to halve it first.

The conversion from a PSD is

$$A = \sqrt{\mathrm{PSD}\cdot T/2}$$

keyed off the physical duration $T$, for the reason given in [§3](#3-spectral-estimation).

## 5. Log binning

Amplitudes are averaged into $M$ bins spaced evenly in $\log_{10} f$ between
the record's own $f_{\min}$ and $f_{\max}$ — clamped to the record, so the
requested bin count is the count you get.

The average is **geometric**, matching the scale the bins are spaced on:

$$\bar{A}_i = 10^{\;\langle \log_{10} A \rangle_{\,f \in \text{bin } i}}, \qquad
\bar{f}_i = \sqrt{f_i^{\text{lo}} f_i^{\text{hi}}}$$

Empty bins are expected — log bins over a linear frequency grid are inevitably
sparse at the low end — and are dropped.

**Membership is computed, not tested:**

$$i(f) = \left\lfloor \frac{\log_{10} f - \log_{10} f_{\min}}{\Delta \log_{10} f} \right\rfloor$$

so every sample belongs to exactly one bin. The previous implementation tested
$f \ge f^{\text{lo}}_i$ **and** $f \le f^{\text{hi}}_i$ against each edge in
turn — closed at both ends, so a sample landing on an interior edge belonged
to *two* bins. On a CWT axis this reported 51 bins from 49 samples: more bins
than samples, which is only possible by double counting.

## 6. Noise rescaling and rotation

### Rescaling

The two windows are rarely the same length, and a shorter record spreads the
same power over fewer bins, so the noise is put on the signal's footing:

$$A_{\text{noise}} \leftarrow A_{\text{noise}} \sqrt{N_{\text{signal}} / N_{\text{noise}}}$$

It is then interpolated onto the signal's frequency axis. **`np.interp` does
not extrapolate** — it repeats the edge value — so below the noise window's
own lowest frequency the "noise level" is a flat continuation rather than a
measurement. [§7](#7-signal-to-noise-and-bandwidth) is what keeps the selected
band out of that region.

### Raising the noise level

A recorded noise window understates the noise *beneath* a strong signal: it is
a sample of the same process, but taken where the signal is not. How to correct
for that is a modelling choice, not a fact, so `core.noise` holds a **set** of
methods behind one signature — given frequencies, noise and signal, return a
multiplicative factor. `NOISE_MODELS` maps names to implementations, the same
way `transforms.ESTIMATORS` does for spectral estimators, so the band search
never needs to know which was used.

| Name | Status | Assumption |
|---|---|---|
| `boost` | default | Noise under the signal follows the recorded shape, scaled by a power of a frequency ramp |
| `none` | available | The recorded window is representative as measured |
| `rotate` | available | Legacy `ROT_METHOD = 1`; the discrepancy is a *tilt* in log–log space rather than a level offset |

`none` is not a placeholder. It is the honest choice when the noise window is
genuinely representative, and it is what a run needs in order to show what the
correction is doing — every other model should be compared against it before
being trusted, because **the assumption is the whole content of the method**,
and it propagates into every bandwidth and so into every $\Omega$.

The default, `boost`, lifts the low and high tails independently until each
touches the signal, then keeps the larger of the two at every frequency.

Each bin is mapped onto a scale $s(f)$ running from $\approx 0$ at one end of
the band to $\approx 1$ at the other, and the noise is raised by

$$A'(f) = A(f)\,s(f)^{-\eta}$$

Since $s < 1$, increasing $\eta$ raises the small-$s$ end fastest. The
exponent wanted is the smallest $\eta$ at which *any* bin in the half reaches
the signal. **That has a closed form.** A bin touches when

$$A\,s^{-\eta} \ge S \iff \eta \ge \frac{\ln(S/A)}{-\ln s}$$

so the first touch across the half is

$$\boxed{\;\eta^\star = \min_{i \,:\, s_i < 1} \frac{\ln(S_i/A_i)}{-\ln s_i}\;}$$

with $\eta^\star = 0$ when some bin already satisfies $A \ge S$, and no lift
at all when no bin has $s < 1$.

**Why this matters beyond elegance.** The original implementation searched for
$\eta$ by stepping it in increments of $\mathrm{inc} = 0.05$ and stopping at
the first step past the touching point — that is, it returned
$\mathrm{inc}\cdot\lceil \eta^\star/\mathrm{inc}\rceil$. Two consequences,
both measured:

- **It was irreproducible.** The stopping test is a comparison, so the result
  was a step function of its input. Two machines differing in the last bit
  landed on either side of a step and the noise moved by
  $s_{\min}^{-0.05} = 1.41$. Observed on CI: 41% and 82% — one and two steps.
- **It was biased.** Rounding was always *upward*, so the lifted noise was
  consistently overstated — a median $1.18\times$, up to $1.41\times$, across
  39 lifts on the 28 PNR windows. That made signal-to-noise pessimistic at
  exactly the band edges the ratio is read from.

Using $\eta^\star$ directly fixes both: the exponent is now a continuous
function of its input, and it is the quantity the algorithm was always trying
to compute. Measured end to end, perturbing the input by $10^{-15}$ now moves
the noise by $1.8\times10^{-11}$ and no band edge at all.

### The rotate method

`ROT_METHOD = 1`, described in the legacy source as "actual rotation, quite
aggressive". Writing $X = \log_{10} f$ and $Y = \log_{10} A$, it tilts the
noise spectrum about its low-frequency end:

$$Y'(\theta) = X\sin\theta + Y\cos\theta + Y_0\,\theta$$

The trailing $Y_0\theta$ is what makes this a rotation *about the low-frequency
end* rather than about the axis origin — without it the curve translates as
well as tilts, and the correction stops being anchored to the part of the noise
record least contaminated by signal. As with `boost`, one angle is found for
each half and the larger result kept at every frequency.

**It assumes** the recorded window has the right level somewhere and the wrong
*slope*, where `boost` assumes the right shape and the wrong *level*. Those are
genuinely different claims about the noise process, which is why both are kept:
comparing the two bands is the only way to see how much of a result is the
method. On the 28 PNR windows they disagree on 25, `rotate` always narrower —
`LV.L002..HHE` runs 1.51–38.60 Hz under `boost` and 1.51–13.87 Hz under
`rotate`.

Unlike $\eta^\star$, the touching angle has no closed form: $\theta$ appears
inside $\sin$ and $\cos$. It is bracketed on a coarse grid and then bisected
to $10^{-12}$, which is enough to make it continuous in the input — the grid
only chooses the bracket, never the answer. The legacy stepped $\theta$ by a
fixed increment and stopped at the first trial past the touch, with exactly the
irreproducibility described above.

Two departures from the legacy implementation are recorded in
[REFACTOR_PLAN §4.5.3](https://github.com/sgjholt/SpecMod/blob/main/docs/REFACTOR_PLAN.md):
the solved angle, and taking the low/high split from the signal rather than the
noise. Neither can move a published number — `ROT_METHOD = 1` was commented out
on `master` and has never produced one.

## 7. Signal-to-noise and bandwidth

The ratio is taken bin by bin on the binned spectra, which share bin edges by
construction because the noise was moved onto the signal's axis *before*
binning:

$$R_i = \bar{A}^{\,\text{signal}}_i \big/ \bar{A}^{\,\text{noise}}_i$$

The usable band is the **widest contiguous run** of bins with $R_i \ge R_{\min}$,
bridging gaps of a single failing bin so that one noisy bin does not end a
band. It returns nothing — not a band plus a flag — when no run of at least
`min_width` bins survives.

The previous method integrated $\mathrm{sign}(R - R_{\min})$ and read the
edges off the 1st and 99th percentiles, with a retry loop when they crossed.
Every step there is discontinuous and they compound: an edge could move 13
bins between machines. It also **lagged**: on a clean 5–30 Hz passing region
it returned 9.41 Hz for the low edge, because the 1st percentile of a
cumulative integral arrives late. The low edge is what constrains $\Omega$.
The contiguous-run method returns 5.45 Hz on the same input, within one bin of
the truth.

**Resolution floor.** The band is finally clamped to

$$f_{\text{low}} \ge \max\bigl(\min f^{\text{signal}},\; \min f^{\text{noise}}\bigr)$$

which refuses the region where the noise level is `np.interp`'s repeated edge
value rather than a measurement. On the 28 PNR pairs, 6 selected a band
opening below the noise window's own $1/T$ before this existed. Set
`snr.resolution_floor = false` to reproduce a run made before it.

## 8. Source model

Fitted in $\log_{10}$ space over the selected band. The generalised
Boatwright source shape is

$$\log_{10} S(f) = \log_{10}\Omega - \frac{1}{\gamma}\log_{10}\!\left[1 + \left(\frac{f}{f_c}\right)^{\gamma n}\right]$$

with $(\gamma, n) = (1,2)$ for Brune and $(2,2)$ for Boatwright. Attenuation is
a frequency-independent $t^\ast$,

$$\log_{10} D(f) = -\frac{\pi f t^\ast}{\ln 10}$$

(or $f^{1-a}$ in the frequency-dependent variant), and the motion term converts
from displacement:

$$\log_{10} G(f) = \begin{cases} 0 & \text{displacement} \\ \log_{10}(2\pi f) & \text{velocity} \\ 2\log_{10}(2\pi f) & \text{acceleration}\end{cases}$$

The fitted model is their sum:

$$\log_{10} A(f) = \log_{10} S(f) + \log_{10} D(f) + \log_{10} G(f)$$

Free parameters are $\Omega$, $f_c$ and $t^\ast$, minimised with Powell's
method by default. Turning the fitted $\Omega$ into a magnitude is
[§9](#9-moment-and-magnitude).

> **A note for anyone adding a source model.** Madariaga is omega-squared like
> Brune and sits at the *same* $(\gamma, n) = (1,2)$, so adding it as a
> spectral shape alone changes no fitted parameter. The difference is the
> constant relating $f_c$ to source radius, and since stress drop goes as
> $r^{-3}$ that is roughly an order of magnitude on identical data. The
> $f_c$-to-radius scaling has to be a named property of the model, not a
> constant buried in whatever computes stress drop. See
> `REFACTOR_PLAN.md` §4.6.5.

---

## 9. Moment and magnitude

`specmod.magnitude`. The fit gives $\Omega$ per channel; this turns it into a
seismic moment and a moment magnitude.

$$M_0 = \frac{4\pi\rho\beta^3 R_0}{\Theta F}\cdot\frac{\Omega}{G(R)}
\qquad\text{[N m]}$$

$$M_w = \tfrac{2}{3}\left(\log_{10} M_0 - 9.1\right)$$

$G(R)$ is the geometrical spreading ([`specmod.spreading`](api.md)), which
corrects the observed plateau back to the reference distance $R_0$. The
magnitude relation is Hanks and Kanamori (1979).

### The constants

Defaults are the S-wave values of Holt (2019) Ch. 1 §1.4 and Ch. 2 §2.2, all
taken **at the source**, and all overridable through `MediumConstants`:

| Symbol | Field | Default | |
|---|---|---|---|
| $\rho$ | `density` | 2700 | kg m⁻³ |
| $\beta$ | `velocity` | 3500 | m s⁻¹, cubed here |
| $\Theta$ | `radiation_pattern` | 0.55 | average SH over the focal sphere |
| $F$ | `free_surface` | 2 | vertically incident SH |
| $R_0$ | `reference_distance_m` | 1000 | metres |

**$\Theta = 0.55$ is the SH average and pairs with $F = 2$.** The textbook
0.63 is the RMS over total S — a different quantity, and substituting it while
keeping $F = 2$ is not a small correction. No partition factor is applied.

### Two distances, both correct

$R_0$ is in **metres**, because it sits beside kg m⁻³ and m s⁻¹ and that is
what makes $M_0$ come out in newton-metres. $G(R)$ takes its distance in
**kilometres**.

That looks like a units bug and is not: they are different distances. They
cancel only when the spreading exponent is exactly 1, so a model with any
other exponent genuinely needs both, in their own units.

### $\Omega$ is linear, `llpsp` is not

The fit reports `llpsp` — $\log_{10}\Omega$. `seismic_moment` takes $\Omega$
itself and raises `ValueError` on a non-positive plateau rather than returning
a moment roughly $10^{40}$ times too small, which is what passing the logarithm
straight through would otherwise produce.

And $\Omega$ must be the **displacement** plateau, in m s. That is the reason
[§4](#4-amplitude-convention) converts to `MAGNITUDE`: reading a folded `FAS`
plateau as $\Omega$ puts $M_0$ out by two, which is 0.2 magnitude units.

### Station values, then an event value

`station_moments` appends `m0` and `mw` per channel. `event_magnitude`
aggregates, following Holt (2019) §2.2:

- the **sample mean of station magnitudes**, excluding anything beyond
  `outlier_sigma` (default 2.5) standard deviations, in a single pass rather
  than iterated to convergence;
- **no event value at all below `min_stations`** (default 3). Fewer than three
  stations cannot average out the radiation pattern, so the published method
  declines to report a value rather than reporting a poor one, and this raises
  instead of returning something usable-looking.

Averaging in magnitude rather than in $M_0$ makes the result a *geometric* mean
of moments. That is the published choice, not an oversight.

:::{warning}
**The absolute calibration is unverified.** The shape of the fit is checked
against golden references, and the constants above are sourced; what has not
been confirmed is that the resulting $M_w$ reproduces the published catalogue
values on the same events. Treat station-to-station and event-to-event
*differences* as sound and the absolute level as provisional until §4.7 of
`REFACTOR_PLAN.md` is closed.
:::

---

## Reproducibility

As of the change described in [§6](#6-noise-rescaling-and-rotation) and
[§7](#7-signal-to-noise-and-bandwidth), the pipeline is continuous in its
input: perturbing a record by $10^{-15}$ moves the noise by $1.8\times10^{-11}$
and no band edge at all. Before, a last-bit difference between two machines
moved the noise by up to 82% and a band edge by 13 bins.

That is checked, not assumed. `tests/golden/pipeline_reference.json` records
the full pipeline output over 28 real windows and 5 estimators, and
`tests/test_golden_reference.py` holds every later change to it. Regenerate
with `python tools/make_golden.py`, and say in the commit message what moved
and why — regenerating without that explanation removes the only check
standing between a refactor and a silently different $\Omega$.
