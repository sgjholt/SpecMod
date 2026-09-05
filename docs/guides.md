# Guides

What each stage does, why it is done that way, and what the choices cost.

## The pipeline, and the conventions

[How a spectrum is processed](processing.md)
: Every step from a raw waveform to a moment magnitude, with the equation it
  applies and a pointer to the code that applies it.

This is also where the conventions are stated, deliberately in the sections
that apply them rather than on a page of their own:

[Amplitude convention](processing.md#4-amplitude-convention)
: `FAS` is the folded $2\lvert X\rvert$; `MAGNITUDE` is the unfolded
  $\lvert X\rvert$, **and that is the one $\Omega$ is defined in**. Reading a
  folded plateau as $\Omega$ puts $M_0$ out by two — 0.2 magnitude units.

[The Parseval contract](processing.md#the-parseval-contract)
: The one energy check every estimator is held to, which is what lets
  multitaper, Welch, FFT and the CWT be interchangeable rather than merely
  similar.

[Moment and magnitude](processing.md#9-moment-and-magnitude)
: The $M_0$ and $M_w$ equations, the medium constants and where they come
  from, and the two distances that are in different units on purpose.

## Choices that change the answer

[Choosing a transform](choosing-a-transform.md)
: What each estimator does to your data, measured. On real windows the plateau
  moves by 7–15% between estimators — about 0.03 magnitude units — and window
  position alone reaches 0.08. Small against the 0.13 m.u. scatter quoted for
  spectral $M_w$, but systematic rather than random, so it does not average
  away across stations at similar distance.

[Reading picks](pick-formats.md)
: What arrival formats are read out of the box, how to add one, and how a pick
  is matched to a trace.

## Working with data

[Publishing a dataset](releasing-data.md)
: Taking an event from an FDSN archive to a hash-pinned entry in the registry,
  so a result can be reproduced from the same bytes.

```{toctree}
:maxdepth: 2
:hidden:

processing
choosing-a-transform
pick-formats
releasing-data
```
