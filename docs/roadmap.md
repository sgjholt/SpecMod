# Roadmap

SpecMod is being rebuilt in stages, each one independently usable. This page
says what is done, what is being worked on, and what 1.0 will mean.

**No dates.** The stages are ordered by dependency, not by calendar, and the
order after the current one can still change. A stage is listed as done when
it is merged and tested, not when it is designed —
[`REFACTOR_PLAN.md`](https://github.com/sgjholt/SpecMod/blob/main/docs/REFACTOR_PLAN.md)
is the working document behind this one and carries the reasoning, the
measurements and the open questions.

## Where it is now

Alpha, pre-1.0. Everything below marked done is merged, tested against golden
references, and usable — but names and signatures still move between `0.x`
releases without a deprecation cycle. See
[Releasing the software](releasing.md) for what a version number means here.

## The stages

### 1. An installable package ✅

A real `pyproject.toml`, `src/` layout, snake_case modules, linting, type
checking and a test suite on CI. Before this, the package could not be
installed or imported without editing paths by hand.

### 2. Configuration without globals ✅

Settings live in a `config/` package with semantic sections and layered
overrides, instead of module-level constants read at import time. Every output
records the configuration that produced it, so a locally-overridden run is
reproducible from its own outputs.

### 3. Publishing: docs, PyPI, DOI ✅

This site, built from the repository and deployed on merge; automated
changelog and version derivation from tags; PyPI upload through Trusted
Publishing; a Zenodo DOI per release. Deliberately built early, while the
package is small enough that debugging the pipeline is cheap.

The repository side is complete. Publishing also needs a handful of one-time
repository settings, which are listed in
[Releasing the software](releasing.md).

### 4. The transform layer ✅

One `SpectralEstimator` protocol with interchangeable implementations —
`FFTEstimator`, `WelchEstimator`, `MultitaperEstimator` and Prieto's — plus
Konno–Ohmachi smoothing and log-binning as separate, composable steps. The
`mtspec` Fortran dependency, which no longer builds on current toolchains, is
demoted to an optional legacy backend rather than being the only path.

What each estimator does to real data is measured in
[Choosing a transform](choosing-a-transform.md); the differences are large
enough to matter to a magnitude.

### 5. Wavelets ✅

A continuous wavelet transform alongside the Fourier estimators:
`CWTEstimator`, a `Scalogram` that tracks its cone of influence, quality
checks over it, and time-averaging back to a spectrum with ground-motion units
preserved.

### 6. Decomposition and typed I/O 🚧

Breaking the remaining large modules into packages with narrow
responsibilities, and replacing pickle-based persistence with typed, portable
formats.

Done so far: the spectral core, the fitting layer and the pick readers are
packages; the type-checking backlog is empty. Still to come: the I/O and
plotting layers, and making the remaining operations return new objects
instead of mutating in place.

### 7. 1.0 — the API stops moving

The theory page that states the normalisation and units conventions
explicitly, the tutorial rebuilt as an executed notebook so it cannot rot
silently, and a "what changed since 0.1" guide for anyone upgrading. Then the
1.0 release, which is the promise that names and signatures stop moving
without a deprecation cycle.

## After 1.0

Designed but deliberately not on the path to 1.0, because each is blocked on
an input rather than on effort — mostly a real file from a real tool, which a
guess written from documentation cannot substitute for:

- **Presets for picker output** (PhaseNet, EQTransformer, SeisBench), so those
  users write no column mapping. The configurable reader already exists; a
  preset is a mapping and a test against one real file.
- **Confirming two more pick formats** (NonLinLoc `.hyp`, IMS/GSE bulletins)
  that cannot be round-tripped through ObsPy and so are listed as unconfirmed
  rather than claimed.
- **SeisComP picks**, which ObsPy's SCML reader currently discards. A test
  fails when that starts working upstream.
- **Recording pick-resolution policy in the configuration**, alongside the
  provenance stamp.
- **A fuller `acquire --verify`** that re-fetches and diffs against the
  manifest, rather than only detecting local tampering.

## How this page will change

Stages are the honest unit while nothing has been released: there is no
version to point at, so "done" can only mean merged. Once releases exist each
completed stage gets the version it shipped in, and this page becomes a list
of milestones against those versions rather than a list of stages — the same
content, anchored to something a reader can install and check.
