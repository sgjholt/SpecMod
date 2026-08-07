# Type stubs for ObsPy

ObsPy 1.5 ships no `py.typed` marker and no annotations, and there is no
`obspy-stubs` or `types-obspy` on PyPI (checked; both 404). Every ObsPy object
was therefore `Any` to mypy, which meant `st: Any` in signatures and no
checking at all of the attributes we read off traces — including the ones
SpecMod itself sets.

These stubs cover **only the surface SpecMod uses**. That is deliberate:
a partial stub that is accurate is worth more than a complete one that is
guessed, and mypy reports an attribute missing from a stub as an error, so
anything we start using has to be added here consciously.

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

`tests/test_obspy_stubs.py` imports the real ObsPy and asserts that every name
declared here exists on the real object, with the same parameter names where
one is a function. A stub that has drifted from the library is worse than no
stub, because it is believed.

It does **not** check return types — those were read off the running library
when the stubs were written (`UTCDateTime.matplotlib_date` is a
`numpy.float64`, `UTCDateTime - UTCDateTime` is a `float`) and are recorded in
comments where they are surprising.
