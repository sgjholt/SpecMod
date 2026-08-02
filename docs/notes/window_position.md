# Where arrivals land in a refined window, and what it costs

Measured on the 28 PNR S-windows in `Tutorial/Data/`, cut with the published
Magna workflow — window opens at 80% of the Pg–Sg time, then refined to the 1st
and 99th percentiles of the cumulative squared-amplitude integral.

"Position" below is where the **50%-energy point** falls, as a fraction through
the refined window.

## The distribution

| | 50%-energy position | energy centroid |
|---|---|---|
| median | 40.8% | 44.3% |
| range | 3.1% – 72.0% | 8.5% – 66.7% |
| below 20% | **25% of windows** | — |

**It is strongly distance-dependent**, and that is the part worth internalising:

| Epicentral distance | 50%-energy position |
|---|---|
| < 6 km | 3% – 36% (mostly under 20%) |
| > 10 km | 32% – 72% |

The mechanism is the window definition, not the physics. The window opens at a
fixed *fraction* of the Pg–Sg time, so at short distance — where Pg–Sg is
around 1 s — it opens essentially on the S arrival, and the coda fills the rest.
The energy is front-loaded. At larger distance the window opens well before S,
so the arrival lands nearer the middle.

## What it costs

Energy recovered on those same windows (estimated / true):

| | adaptive | flat | FFT |
|---|---|---|---|
| median | 0.983 | 0.988 | — |
| worst | **0.286** | 0.563 | 0.790 |
| below 0.6 | **14% of traces** | 4% | 0% |

The four worst are all at position < 8%, and all are near-source: UR.AQ04
(0.9 km), UR.AQ05 (3.3 km), UR.AQ03 (6.0 km).

Worst case under adaptive weighting: 0.286 of the true energy, so `sqrt(0.286)`
= **0.53× in amplitude**, which is −0.27 in `log10(Omega)` and about **0.18
magnitude units** on `Mw` for that station.

That is not a rounding error. The paper quotes a standard deviation of 0.13 m.u.
for spectral `Mw`, and station `Mw` values are combined by median across
stations — so a handful of biased near-source stations do not necessarily move
the event `Mw` by 0.18, but they do inflate the scatter and they bias
systematically in one direction rather than randomly.

## Consequences taken

1. `MultitaperEstimator.adaptive` and `TransformConfig.adaptive` now default to
   **False**. Flat weighting is well-behaved at every position tested.
2. `studies/magna_2020_paper.toml` pins `adaptive = true` explicitly, so the
   changed default cannot silently alter what that file means. Whether the
   published run actually used it is unverified — `mtspec` defaulted to on, and
   the manuscript does not say.
3. Flat weighting is *not* unbiased either — 0.563 at worst. The FFT with a
   light taper holds 0.79 at worst and is the better choice when absolute
   amplitude fidelity matters more than variance.

## Open

Whether `mtspec`'s Fortran adaptive routine collapses the same way is untested
and needs the legacy environment. If it does, near-source stations in the
published catalogue carry this bias. If it does not, the collapse is specific to
the implementation here and should simply be fixed.

That is a question for the §5.2.6 three-way comparison, and it is now a specific
thing to look for rather than a general check: **compare near-source stations
first**, since that is where the two would diverge.
