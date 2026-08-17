# Audit: what `specmod.api` found in core

Answers to the five questions that had to be settled before writing
`specmod.api`, each measured rather than reasoned about. The surface itself is
small; this was the work.

It is kept because the properties it audits — path coupling, hidden state,
determinism — are the ones core's own provenance claims rest on, whether or not
anything downstream ever consumes them.

## 1. Joint per-event inversion, or per-spectrum fitting?

**Per-spectrum only.** `FitSpectra` is a loop, not a joint solver:

```python
for name, mod in self.models.items():
    mod.fit_mod(**kwargs)
```

Each station gets its own `FitSpectrum` with its own parameters. Nothing is
shared between them — no common `t*`, no common Ω₀, no event-level term. The
"event fit" is an aggregation of independent fits into one table.

A joint inversion is therefore not in core and `specmod.api` exposes the
per-spectrum primitive, `fit_spectrum`, plus everything a joint solver needs to
build its own problem: the per-bin SNR, the selected band, the model, and the
covariance of each single-station fit.

## 2. Per-bin SNR, or a scalar bandwidth?

**Per-bin, already.** `SpectrumPair.snr` is an array aligned with
`binned_signal.freq`, computed as the element-wise ratio of the two binned
spectra:

```python
snr = binned_signal.amp / binned_noise.amp
band = find_bandwidth(binned_signal.freq, snr, threshold, method=bandwidth)
```

The scalar `band` is *derived* from the curve, and both are kept. So the thing
that cannot be un-collapsed later was never collapsed: a consumer can apply its
own threshold, or admit data bin by bin instead of over a contiguous interval,
without a reprocess.

Measured on one synthetic station: 75 bins of `snr` against a `band` of
(6.47, 9.15) Hz. `compare_spectra` returns the pair whole — the curve, the
binned noise spectrum, and the resolution floor — rather than a summary.

**What core does not have** is a `valid_mask`. Bins are excluded by falling
outside the band or below the resolution floor, and there is no per-bin defect
flag. That is a consumer-side concept and building it does not need core.

## 3. Which functions take paths with no in-memory form?

Listed rather than changed. None of them is on `specmod.api`, and the
estimation and fitting paths were already path-free — `SpectralEstimator.estimate`
takes `(data, dt)`.

| Function | Takes | In-memory form today |
|---|---|---|
| `preprocess.read_picks` | path | none — wraps `picks.read` |
| `preprocess.set_picks` | path | none, though `picks.read` accepts a `Catalog` |
| `preprocess.rstfl` | paths | none |
| `picks.read` | path or `Catalog` | **yes**, `Catalog` |
| `picks.detect_reader`, every `PickReader.read` | path | none — they sniff the file |
| `tables.read_table` / `write_table` | path | none |
| `io.*` | path | none |
| `datasets.*`, `acquire.*` | paths, network | not applicable — that *is* their job |

The one worth fixing first is `set_picks`: `picks.read` already accepts an
in-memory `Catalog`, so the path-free form exists one layer down and is not
plumbed through. A caller holding a `Catalog` has to reach past `preprocess` to
use it.

## 4. Hidden global or module-level state

Four instances. Three are benign and one is not.

**A module-level config read, at import time.** `fitting/base.py` line 26:

```python
PLOT_COLUMNS = cfg.load_config().config.viz.plot_columns
```

`load_config()` with no arguments resolves against the *current working
directory* and the environment. So importing `specmod.fitting` reads whatever
`specmod.toml` happens to be next to the process, once, and freezes it for the
lifetime of the interpreter. Two jobs in one worker with different project
directories get the first one's value. This is the only genuine replay hazard
found, and it is the exact pattern §2.3 of the refactor plan set out to remove.

**Implicit config reads at call time**, in twelve places including
`fitting/event.py`, `fitting/guess.py`, `fitting/spectrum.py`, `pipeline.py`
and `sources/composite.py`. Not import-time, so not frozen, but still
working-directory-dependent. `specmod.api` closes this where it can — every
estimation and comparison argument is explicit — and documents it where it
cannot: `fit_spectrum` reads `[fitting]` for the minimiser and the initial
guess, and its docstring says so.

**The pick-reader registry.** `PICK_READERS` is a module-level dict mutated by
`register_reader` and by entry-point discovery, guarded by a `_plugins_loaded`
flag. Which readers exist is a property of what is installed, and it is not
recorded in provenance. It affects reading, never numbers, and nothing on
`specmod.api` touches it.

**No randomness anywhere.** Nothing in `src/specmod` imports `random` or
`numpy.random`, and nothing seeds. The `emcee` extra is declared but unused, so
the first sampler added is the moment an explicit `seed` argument has to be
required rather than recommended.

### Two more things the audit turned up

Not asked for, but found while looking, and both affect a consumer:

**Nine `print()` calls** in `fitting/event.py` and `utils.py`, on paths a
caller reaches — an unrecognised weight method, a station that failed to fit, a
missing id. A service capturing logs per job gets nothing, and a CLI writing to
a pipe gets its output corrupted. `specmod.api` does not currently route
through any of them, but `FitSpectra` does.

**Uncertainty depends on the minimiser, and the default provides none.**
`[fitting] method` ships as `powell`, which estimates no covariance matrix, so
every parameter's `stderr` is `None` and there is no correlation to report.
Measured on one synthetic station:

| `method` | `fc` | `fc` error | `fc`–`t*` correlation |
|---|---|---|---|
| `powell` (default) | 7.925 | — | — |
| `nelder` | 7.925 | — | — |
| `leastsq` | 7.925 | 0.129 | 0.837 |
| `least_squares` | 7.925 | 0.129 | 0.837 |

All four agree on the point estimate to three decimals against a true 8.0 Hz.
Only the least-squares family answers "how well". `SpectrumFit` reports the
absence as an absence — empty `stderr`, `covariance=None`,
`correlation()` returning `None` rather than zero — because a zero error is a
claim, and the wrong one. The 0.84 correlation is the reason neither `fc` nor
`t*` should be quoted alone.

## 5. What does one multitaper estimate cost?

Measured on the 28 real S windows of the Preston New Road event, through
`MultitaperEstimator` (DPSS, `scipy.signal.windows.dpss`).

**Re-estimation on a window change** — the full path: estimate the signal,
estimate the noise, then the Parseval rescale, interpolation onto a common
axis, log binning, the per-bin SNR and the band search.

| Operation | Median | p95 |
|---|---|---|
| `multitaper` estimate, one window | 3.05 ms | 3.31 ms |
| `fft` estimate, one window | 0.12 ms | 0.20 ms |
| **multitaper ×2 + full compare** | **7.02 ms** | 7.55 ms |
| `fft` ×2 + full compare | 2.32 ms | 2.71 ms |

Scaling is close to linear in window length, not quadratic:

| Samples | Duration | Median |
|---|---|---|
| 512 | 2.6 s | 1.92 ms |
| 1024 | 5.1 s | 3.63 ms |
| 2048 | 10.2 s | 6.31 ms |
| 4096 | 20.5 s | 11.54 ms |
| 8192 | 41.0 s | 20.85 ms |

So a 20-second window re-estimates in roughly 25 ms end to end, and a 3.7-second
one in 7 ms. Whatever budget a live window editor has, this is not what spends
it.

**Configuration.** One machine: x86_64, 4 cores, Python 3.11.15, numpy 2.4.6,
scipy 1.17.1, obspy 1.5.0, local disk, records read from the repository's own
test data. Record under test: `UR.AQ06.00.HHN`, 737 samples at 200 Hz.

**This is one configuration, and a shared cloud container at that.** Anything
that has to hold on a minimum-spec target or against remote object storage
needs measuring there; the numbers above answer "is this milliseconds or
seconds", which is the question that was blocking, and nothing more.
