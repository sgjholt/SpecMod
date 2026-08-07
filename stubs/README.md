# Type stubs for ObsPy and lmfit

Neither ObsPy 1.5 nor lmfit 1.3 ships a `py.typed` marker, and no stub package
is published for either (`obspy-stubs`, `types-obspy`, `lmfit-stubs` — all 404
on PyPI). Every object from both was therefore `Any` to mypy, which meant
`st: Any` and `result: Any` in signatures and no checking at all of what we
read off them.

These stubs cover **only the surface SpecMod uses**. That is deliberate: a
partial stub that is accurate is worth more than a complete one that is
guessed, and mypy reports an attribute missing from a stub as an error, so
anything we start using has to be added here consciously.

## What they caught

Wiring them in found four defects that `Any` had hidden. None was theoretical:

- `set_stream_distance(dtype="mseed")` passed a possibly-`None` inventory into
  `get_channel_metadata`.
- `plot_traces(plot_windows=True)` indexed `sig[i]` without checking `sig` was
  given.
- Every private reader on `FitSpectrum` went through `self.result`, which is
  `None` until `fit_mod` runs — so `quick_vis` on an unfitted model raised
  `AttributeError` from a line naming neither the station nor the missing step.
- `FitSpectrum.__param_string` computed `2 * stderr` inside a bare
  `except Exception`. `stderr` is `None` under Powell, the **shipped default**,
  so that raised `TypeError` on every fit made with the default configuration,
  swallowed it, and titled the plot `NaN`.

## What is covered

| Module | What is declared |
|---|---|
| `obspy` | `read`, `read_inventory`, and re-exports of the three core classes |
| `obspy.core.trace` | `Trace`, `Stats` |
| `obspy.core.stream` | `Stream` |
| `obspy.core.utcdatetime` | `UTCDateTime` and its arithmetic |
| `obspy.core.inventory` | `Inventory.get_channel_metadata` |
| `obspy.geodetics` | `gps2dist_azimuth` |
| `obspy.signal.konnoohmachismoothing` | `konno_ohmachi_smoothing` |
| `lmfit.model` | `Model`, `ModelResult` |
| `lmfit.parameter` | `Parameter`, `Parameters` |

`Parameter.stderr` is `float | None`, and that is the single most load-bearing
declaration here — see the fourth defect above.

## `Stats` is deliberately open

`Stats` is an `AttribDict`: it carries the SEED header fields and also whatever
else is assigned to it. SpecMod assigns fourteen of its own — `p_time`,
`s_time`, `repi`, `rhyp`, `wstart`, `wend` and the rest — and reads them back
both as attributes and as keys.

So the stub declares the standard fields with their real types and lets
everything else fall through `__getattr__`/`__getitem__` to `Any`. The effect
is that `tr.stats.delta` is checked as a `float` and `tr.stats["p_time"]`
is not checked at all, which is exactly the split that exists in reality.
Declaring SpecMod's own fields here would be worse: it would put SpecMod's
private conventions into a stub that claims to describe ObsPy.

## Keeping them honest

`tests/test_stubs.py` imports the real libraries and asserts that every name
declared here exists on the real object, with the same parameter names where
one is a function. A stub that has drifted from the library is worse than no
stub, because it is believed — and mypy cannot see the drift, since it reads
the stub *instead of* the library.

It also asserts the `stderr is None` claim directly, by fitting a line with
both minimisers.

It does **not** check return types — those were read off the running library
when the stubs were written (`UTCDateTime.matplotlib_date` is a
`numpy.float64`, `UTCDateTime - UTCDateTime` is a `float`) and are recorded in
comments where they are surprising.
