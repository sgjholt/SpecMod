# SpecMod

A Python toolbox for processing and modelling seismic spectra: cut a window,
estimate its spectrum, decide which part of it is above the noise, and fit a
source model to that band.

```python
import specmod.preprocess as pre
from specmod.pipeline import spectrum_set_from_streams
from specmod.fitting import FitSpectra

pre.set_picks(stream, "event.xml")
signal = pre.get_signal(stream, pre.cut_s, rafp=0.8, tafs=20)
noise = pre.get_noise_p(stream, signal)

spectra = spectrum_set_from_streams(signal, noise)
fits = FitSpectra(spectra)
fits.fit_spectra()
print(fits.table[["id", "llpsp", "fc", "ts"]])
```

## Where to start

[Processing](processing.md)
: Every step of the pipeline with the equation it implements — what a window
  is, how the noise is compared against it, and what the bandwidth selector
  does.

[Choosing a transform](choosing_a_transform.md)
: What each estimator does to your data, measured. The choices here change
  recovered amplitude by factors of three on real windows, which is about 0.3
  magnitude units.

[Reading picks](pick-formats.md)
: What arrival formats are read out of the box, how to add one, and how a pick
  is matched to a trace.

[Publishing a dataset](releasing-data.md)
: Taking an event from an FDSN archive to a hash-pinned entry in the registry.

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
:maxdepth: 2
:hidden:

processing
choosing_a_transform
pick-formats
releasing-data
api
```
