# SpecMod

A Python toolbox for processing and modelling seismic spectra: cut a window,
estimate its spectrum, decide which part of it is above the noise, and fit a
source model to that band.

:::{admonition} Alpha, pre-1.0 — pin an exact version for anything you publish
:class: warning, dropdown

SpecMod is being rebuilt in the open, so treat everything here as provisional
until the API settles at 1.0:

- **Names and signatures move between `0.x` releases**, without a deprecation
  cycle. Breaking changes land in minor bumps by design — that is what `0.x`
  is for, and the alternative is a deprecation shim on an API that is still
  being worked out.
- **Some numbers still move too.** The modern layers (`specmod.config`,
  `specmod.core`, `specmod.transforms`, `specmod.picks`, `specmod.fitting`)
  are built and tested against golden references; the modules that have not
  been reached yet carry pre-refactor behaviour, and a fix there can change a
  result. Changes that move a published number are called out in the
  changelog.
- **Pin an exact version for anything you intend to publish**, and keep the
  configuration stamp that every output carries. Together they are what make a
  run reproducible while the package underneath is still moving.

What will *not* change silently: the units conventions and the Parseval
contract are pinned by tests, and the golden references fail loudly rather
than drifting. The [roadmap](roadmap.md) says what has shipped, in which
version, and what 1.0 will mean.
:::

```python
import specmod.preprocess as pre
from specmod.pipeline import spectrum_set_from_streams
from specmod.fitting import FitSpectra

stream = pre.with_picks(stream, "event.xml")
signal = pre.s_window(stream, rafp=0.8, tafs=20)
noise = pre.get_noise_p(stream, signal)

spectra = spectrum_set_from_streams(signal, noise)
fits = FitSpectra(spectra)
fits.fit_spectra()
print(fits.table[["id", "llpsp", "fc", "ts"]])
```

## Where to go

[Getting started](getting-started.md)
: Install it, run one event end to end with the tutorial, and — if you have
  code written against `0.1` — find out what moved.

[Guides](guides.md)
: What each stage does and why, and where the conventions are stated: the
  amplitude convention, the Parseval contract, and the moment and magnitude
  equations.

[API reference](api.md)
: Every public object. `specmod.api` is the narrower, more stable subset to
  import from if you are building on this.

[Working on SpecMod](contributing.md)
: The roadmap, the developer guide, how this site is built, and how a merged
  commit becomes a release.

## Two things worth knowing early

**Units are typed.** A spectrum carries its ground-motion domain and amplitude
convention as attributes. Converting between them is a method that returns a
new spectrum, so a moment computed from the wrong domain is a type error rather
than a wrong number.

**Fit in the units the sensor recorded.** Integrating to displacement
implicitly low-passes and differentiating to acceleration amplifies
high-frequency noise, so the record to fit is the one that was measured. The
model carries a motion factor, so the plateau it reports is the displacement
one either way.

```{toctree}
:maxdepth: 3
:hidden:

getting-started
guides
api
contributing
```
