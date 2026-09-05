# Upgrading from 0.1

The version the Magna paper used is `0.1.1`, preserved on the frozen
[`master`](https://github.com/sgjholt/SpecMod/tree/master) branch. It was never
tagged or published to PyPI, so "upgrading" here means moving code written
against that branch onto a released `0.2.x`.

Nothing is aliased. Modules moved, were renamed to snake_case, and several were
deleted outright — a `0.1` script will fail at its imports rather than run and
give different numbers, which is the intended failure mode.

:::{note}
Only three names have deprecation shims, listed at the end. Everything else
below is a hard rename or a removal. That is deliberate: `0.x` ships breaking
changes in minor bumps without a deprecation cycle, because shimming an API
still being worked out costs more than it protects. See
[Releasing the software](releasing.md#what-a-version-number-means-while-this-is-0x).
:::

## Modules

| 0.1.1 | 0.2.x | |
|---|---|---|
| `specmod.PreProcess` | `specmod.preprocess` | renamed |
| `specmod.Fitting` | `specmod.fitting` | now a package |
| `specmod.config` | `specmod.config` | now a package, layered |
| `specmod.utils` | `specmod.utils` | reduced to what has callers |
| `specmod.Spectral` | `specmod.core` | **deleted** — see below |
| `specmod.Models` | `specmod.sources` | **deleted** |
| `specmod.ModelGuess` | `specmod.fitting` | **deleted** |
| `specmod.Ratios` | — | **deleted**, no replacement |

The import that breaks first is usually `import specmod.PreProcess as pre`.
It is `import specmod.preprocess as pre`.

## Containers

`Spectral.py` held `Spectrum`, `Signal`, `Noise`, `SNP` and `Spectra`. Those
are replaced by two typed containers in `specmod.core.collection`:

| 0.1.1 | 0.2.x |
|---|---|
| `SNP` | `SpectrumPair` — a signal spectrum and its noise |
| `Spectra` | `SpectrumSet` — a mapping of channel id to pair |
| `Signal`, `Noise` | `core.Spectrum`, distinguished by what holds it |

The difference that matters is not the names. **A `core.Spectrum` carries its
ground-motion domain and amplitude convention as typed attributes**, where the
old `Spectrum` carried neither and the domain lived in `Models.MOTION`, a
module global read at import time that you had to keep in sync by hand with
however many times you had called `.inte()` or `.diff()`. Getting it wrong
returned a wrong seismic moment with no error anywhere. It is now a type error.

See [§4](processing.md#4-amplitude-convention) for the conventions themselves.

## Models and guesses

`Models.py` bound a source shape and a ground-motion domain together at import
time — `MODEL = which_model(...)` and `MOTION` read from config — which is why
a Brune and a Boatwright could not be fitted in one session. Attenuation was
not import-bound; it was chosen per call site by reaching for `simple_model`
or `simple_model_fdep`. All three are now separate and composed:

| 0.1.1 | 0.2.x |
|---|---|
| `which_model` | `sources.get_source_model` |
| `scale_to_motion` | `sources.motion_scaling` (note the argument order flipped) |
| `source` | `SourceModel.log10_shape` — the source term alone |
| `simple_model`, `simple_model_fdep` | `sources.SpectralModel`, built by `build_model` or `from_config`, then `.evaluate()` |
| `t_star`, `t_star_freq` | `AttenuationModel.log10_decay` |
| `create_simple_guess`, `create_simple_guess_fdep` | `fitting.initial_guess` |

The composed model is a sum of three log-space terms — source, attenuation,
motion — written out in [§8](processing.md#8-source-model).

**Reach for `SpectralModel`, not `log10_shape`, when porting `simple_model`.**
`log10_shape` is only the source term of the three; calling it in place of the
old `simple_model` silently drops attenuation and motion, which is a wrong
number rather than an error.

## Persistence

`write_methods` and `read_methods` pickled. Nothing in the package can write a
pickle now:

| 0.1.1 | 0.2.x |
|---|---|
| `write_methods` / `read_methods` | `specmod.io.save` / `specmod.io.load` |
| `.spec` pickle | `.h5` (HDF5) for arrays, `.parquet` for tables |

**Old `.spec` files cannot be read by 0.2.x.** A pickle stores the import path
of every class it holds, and those paths no longer exist. If you need the
contents, read them with a `0.1.1` checkout of `master` and re-save, or re-run
the pipeline — which is cheaper than it sounds and gives you a file that does
not depend on the code that wrote it.

Reading and writing needs the `io` extra: `pip install "specmod[io]"`.

## Configuration

Settings were module-level constants, read at import. They are now a layered
`specmod.config` with semantic sections, resolved per access.

The practical consequence: **overriding a setting after import now works.**
Under `0.1.1` a constant read at import time had already been captured by
whatever read it, so a late override silently did nothing. Every output now
also records the configuration that produced it and a hash of it, so a run is
reproducible from its own outputs.

```sh
specmod config show      # resolved values, and which layer set each one
specmod config freeze    # emit TOML to commit alongside a result
```

## Results will not match `0.1.1` bit for bit

**The pipeline is continuous in its input**, where `0.1.1` had discontinuities
that made a last-bit difference between two machines move the noise by up to
82% and a band edge by 13 bins. Removing them changes results by design, and
they were not reproducible before. See
[Reproducibility](processing.md#reproducibility).

:::{note}
**Multitaper adaptive weighting is still on by default**, as it was in `0.1.1`
and as `mtspec` had it. The changelog for `0.2.0` carries a breaking-change
bullet saying it now defaults to `False`; that bullet is stale within its own
release. The default was turned off, then turned back on in the same release
once the collapse it was turned off for was diagnosed as a units bug in the
adaptive weighting itself — so there is nothing to restore and nothing to pass.
[Choosing a transform](choosing-a-transform.md) has the reasoning.
:::

## The three deprecation shims

These still work and warn, rather than failing:

| Name | Use instead | Removed |
|---|---|---|
| `preprocess.set_picks_from_pyrocko` | `preprocess.set_picks` | not yet dated |
| `set_stream_distance(dtype="none")` | `dtype="list"` | not yet dated |
| `fitting.PLOT_COLUMNS` | `fitting.plot_columns()` | **0.4.0** |

`set_picks` is the rename that matters: it no longer reads only Pyrocko, and
QuakeML is now the preferred format, so the old name said the opposite of what
the function does.

## The shortest path

If you are porting a script rather than a package, the
[tutorial](tutorial/SpecModTutorial.ipynb) is the same pipeline written against
`0.2.x`, executed on every documentation build so it cannot describe an API
that no longer exists. Reading it beside your `0.1` script is usually faster
than working through the tables above.

For a stable, path-free surface that will move less than the internals,
`specmod.api` is 21 names covering estimate, compare, fit and configure — see
the [API reference](api.md).
