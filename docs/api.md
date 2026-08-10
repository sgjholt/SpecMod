# API reference

Grouped by what a run does in order: get the data, cut it, transform it, fit
it, write it out.

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
