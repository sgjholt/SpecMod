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

`Models.py` bound a source shape and an attenuation model together at import
time through `MODEL = which_model(...)`, which is why a Brune and a Boatwright
could not be fitted in one session. They are now separate and composed:

| 0.1.1 | 0.2.x |
|---|---|
| `which_model` | `sources.get_source_model`, `sources.build_model`, `sources.from_config` |
| `scale_to_motion` | `sources.motion_scaling` |
| `source`, `simple_model`, `simple_model_fdep` | `SourceModel.log10_shape` |
| `t_star`, `t_star_freq` | `AttenuationModel.log10_decay` |
| `create_simple_guess`, `create_simple_guess_fdep` | `fitting.initial_guess` |

The composed model is a sum of three log-space terms — source, attenuation,
motion — written out in [§8](processing.md#8-source-model).

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

## Defaults that changed the numbers

Two, both called out as breaking in the
[changelog](https://github.com/sgjholt/SpecMod/blob/main/CHANGELOG.md):

- **`MultitaperEstimator.adaptive` and `TransformConfig.adaptive` now default
  to `False`.** Adaptive weighting collapses for off-centre transients, and a
  seismic arrival in a refined window is one. Pass `adaptive=True` to restore
  the old behaviour, and read
  [Choosing a transform](choosing-a-transform.md) before you do.
- **The pipeline is continuous in its input**, where `0.1.1` had
  discontinuities that made a last-bit difference between two machines move
  the noise by up to 82%. Results will not match `0.1.1` bit for bit. See
  [Reproducibility](processing.md#reproducibility).

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
