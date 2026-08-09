# Where arrivals land in a refined window, and what it costs

Measured on the 28 PNR S-windows in `tutorial/data/events/`, cut with the published
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

<!-- measured: field_energy_table -->
| n = 28 | adaptive | flat | FFT |
|---|---|---|---|
| median | 0.988 | 0.989 | 1.006 |
| worst | 0.559 | 0.563 | 0.791 |
| below 0.6 | 4% of traces | 4% of traces | 0% of traces |
<!-- /measured -->

Regenerate with `python tools/measure_docs.py write --field`.

The worst are at position < 8%, and near-source: UR.AQ04 (0.9 km), UR.AQ05
(3.3 km), UR.AQ03 (6.0 km).

Worst case: 0.56 of the true energy, so `sqrt(0.56)` = **0.75× in amplitude**,
which is −0.12 in `log10(Omega)` and about **0.08 magnitude units** on `Mw` for
that station.

The paper quotes a standard deviation of 0.13 m.u. for spectral `Mw`, and
station `Mw` values are combined by median across stations — so this does not
move the event `Mw`, but it inflates the scatter and it biases systematically in
one direction rather than randomly.

Adaptive and flat weighting agree to within 1% on every trace here, which is
what should happen: the residual is taper shape, and both use the same tapers.
A discrepancy between them on real windows would point at a weighting bug, not
at the tapers — see the σ² units gotcha in
[`../choosing_a_transform.md`](../choosing_a_transform.md).

## Consequences taken

1. `MultitaperEstimator.adaptive` and `TransformConfig.adaptive` default to
   **True**, matching what `mtspec` did.
2. `studies/magna_2020_paper.toml` pins `adaptive = true` explicitly, so a
   changed default cannot silently alter what that file means. Whether the
   published run actually used it is unverified — `mtspec` defaulted to on, and
   the manuscript does not say.
3. Neither weighting is unbiased — 0.56 at worst, and that is taper shape, not
   a defect. Two ways out, both opt-in: `center=True` removes the position
   dependence outright, and the FFT with a light taper holds 0.79 at worst
   without needing to roll the record.

## Open

Whether `mtspec` and this implementation agree numerically on real windows is
untested and needs the legacy environment — that is the §5.2.6 three-way
comparison. Synthetic records already agree with Prieto's `multitaper`
(`mtspec`'s successor, same lineage) to within 0.3% under both weightings, so
the open question is specifically about real data.

**Still compare near-source stations first**, since that is where any remaining
divergence would show.
