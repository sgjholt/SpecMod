# API reference

Grouped by what a run does in order: get the data, cut it, transform it, fit
it, write it out.

**If you are writing a package on top of SpecMod, start with
[`specmod.api`](#the-stable-surface) instead.** It is a small, frozen subset of
what follows, and the only part that carries a compatibility promise — the rest
of this page documents internals that move between `0.x` releases. See
[`CONTRIBUTING.md`](https://github.com/sgjholt/SpecMod/blob/main/CONTRIBUTING.md)
for the exact scope of that promise.

## The stable surface

```{eval-rst}
.. automodule:: specmod.api
   :exclude-members: AmplitudeKind, Config, InternalError, InvalidInputError,
                     MissingBackendError, Motion, ResolvedConfig, SpecModError,
                     Spectrum, SpectrumPair, config_hash, load_config,
                     make_window, window_correction
.. automodule:: specmod.exceptions
```

The names excluded above are re-exports, documented at the path they are
defined — `Spectrum` and `SpectrumPair` under [Spectra](#spectra), `Config`
and `load_config` under [Configuration](#configuration), `make_window` and
`window_correction` under [Transforms](#transforms). Documenting them twice
gives every cross-reference to them two targets and makes all of them
ambiguous, which is the same trap package-level `automodule` set earlier on
this page. `specmod.api.__all__` is the authoritative list, and
`tests/test_api_surface.py` asserts it.

Packages are documented at the path you import from — `specmod.picks.PickSet`,
not `specmod.picks.base.PickSet`. Documenting both the package and its
submodules gave every re-exported name two targets and made every
cross-reference to it ambiguous.

## Getting data

```{eval-rst}
.. automodule:: specmod.datasets
.. automodule:: specmod.acquire
```

## Picks

```{eval-rst}
.. automodule:: specmod.picks
   :imported-members:
```

## Preparing waveforms

```{eval-rst}
.. automodule:: specmod.preprocess
```

## Spectra

```{eval-rst}
.. automodule:: specmod.pipeline
.. automodule:: specmod.core.spectrum
.. automodule:: specmod.core.collection
.. automodule:: specmod.core.units
.. automodule:: specmod.core.scalogram
```

## Transforms and smoothing

```{eval-rst}
.. automodule:: specmod.transforms
   :imported-members:
.. automodule:: specmod.transforms.base
.. automodule:: specmod.smoothing
   :imported-members:
```

## Source models

```{eval-rst}
.. automodule:: specmod.sources
   :imported-members:
```

## Fitting

```{eval-rst}
.. automodule:: specmod.fitting
   :imported-members:
.. automodule:: specmod.staged
```

## Magnitude

```{eval-rst}
.. automodule:: specmod.magnitude
.. automodule:: specmod.spreading
   :exclude-members: HOLT_2019_UTAH
```

`HOLT_2019_UTAH` is the piecewise model fitted in Holt (2019), pre-built:
`Piecewise(segments=((0.90, 43.0), (2.57, 76.0), (0.44, 136.0), (1.54, 400.0)))`.
It is excluded above because autodoc cannot format a signature for a callable
dataclass *instance* documented as module data — `inspect.signature` handles it
fine, autodoc's own formatter raises. Excluding one constant is cheaper than
working around that.

## Output

```{eval-rst}
.. automodule:: specmod.io
.. automodule:: specmod.tables
.. automodule:: specmod.plotting
```

## Configuration

```{eval-rst}
.. automodule:: specmod.config
.. automodule:: specmod.utils
```
