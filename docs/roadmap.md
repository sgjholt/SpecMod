# Roadmap

SpecMod is being rebuilt in stages, each one independently usable. This page
says what has shipped, in which version, what is being worked on, and what 1.0
will mean.

**No dates.** The order is by dependency, not by calendar, and what comes after
the current milestone can still change. Shipped work names the version that
carries it, so it can be installed and checked rather than taken on trust;
unreleased work says so plainly.
[`REFACTOR_PLAN.md`](https://github.com/sgjholt/SpecMod/blob/main/docs/REFACTOR_PLAN.md)
is the working document behind this one and carries the reasoning, the
measurements and the open questions.

## Where it is now

**v0.2.0**, the first release of the rebuilt package —
[on PyPI](https://pypi.org/p/specmod), with a Zenodo DOI and its own
[documentation version](https://specmod.readthedocs.io/en/stable/). Install it
with `pip install specmod`.

Still alpha, still pre-1.0: names and signatures move between `0.x` releases
without a deprecation cycle. Everything below marked shipped is released,
tested against golden references, and usable — but pin an exact version for
anything you intend to publish. See
[Releasing the software](releasing.md) for what a version number means here.

## Shipped in v0.2.0

The version before this one, `0.1.1`, is the pre-refactor code that produced
the published Magna results. It is preserved unchanged on the
[`master`](https://github.com/sgjholt/SpecMod/tree/master) branch, which is
protected and frozen — it was never tagged or published to PyPI, so that
branch is the only reference to it. Everything below is the delta between it
and v0.2.0, which is why the first changelog is enormous and correctly so.

### An installable package

A real `pyproject.toml`, `src/` layout, snake_case modules, linting, type
checking and a test suite on CI. Before this, the package could not be
installed or imported without editing paths by hand.

### Configuration without globals

Settings live in a `config/` package with semantic sections and layered
overrides, instead of module-level constants read at import time. Every output
records the configuration that produced it, so a locally-overridden run is
reproducible from its own outputs.

### Publishing: docs, PyPI, DOI

This site, built from the repository on every merge; automated changelog and
version derivation from tags; PyPI upload through Trusted Publishing; a Zenodo
DOI per release. Deliberately built early, while the package was small enough
that debugging the pipeline was cheap — and v0.2.0 is the release that proved
every leg of it end to end.

A fresh clone still needs the one-time repository settings listed in
[Releasing the software](releasing.md); they are account and repository state,
not something a commit can carry.

### The transform layer

One `SpectralEstimator` protocol with interchangeable implementations —
`FFTEstimator`, `WelchEstimator`, `MultitaperEstimator` and Prieto's — plus
Konno–Ohmachi smoothing and log-binning as separate, composable steps. The
`mtspec` Fortran dependency, which no longer builds on current toolchains, is
demoted to an optional legacy backend rather than being the only path.

What each estimator does to real data is measured in
[Choosing a transform](choosing-a-transform.md); the differences are large
enough to matter to a magnitude.

### Wavelets

A continuous wavelet transform alongside the Fourier estimators:
`CWTEstimator`, a `Scalogram` that tracks its cone of influence, quality
checks over it, and time-averaging back to a spectrum with ground-motion units
preserved.

### Typed, portable persistence

HDF5 for arrays and Parquet for tables, replacing pickle everywhere — nothing
in the package can write a pickle any more, so a saved result is readable
without the code that wrote it and cannot execute anything on load.

### A stable import surface

`specmod.api` — a path-free, deterministic, non-mutating import surface with
typed errors, for downstream packages that need something narrower than the
whole package and less volatile than its internals.

## In progress — not yet released

### Finishing the decomposition

Breaking the remaining large modules into packages with narrow
responsibilities. The spectral core, the fitting layer and the pick readers
are already packages, and the type-checking backlog is empty.

What is left: `io.py` and `plotting.py` are still single modules, and the
operations that mutate in place need to return new objects instead. Neither is
large — the split is about responsibilities, not line count.

## Planned

### 1.0 — the API stops moving

The theory page that states the normalisation and units conventions
explicitly, the tutorial rebuilt as an executed notebook so it cannot rot
silently, and a "what changed since 0.1" guide for anyone upgrading. Then the
1.0 release, which is the promise that names and signatures stop moving
without a deprecation cycle.

That promise is the whole content of the number, which is why it has to be a
decision rather than something a breaking commit does on its way past. Until
it is made, breaking changes bump the minor.

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

## How this page works

Each shipped entry names the version that carries it, so a reader can install
that version and check the claim. Work that is merged but unreleased sits under
*In progress* and gets its version when a release goes out — merged is not
shipped, and this page does not blur the two.

Before v0.2.0 this was a list of stages, because there was no released version
to point at and "done" could only mean merged. The full changelog for every
release is in
[`CHANGELOG.md`](https://github.com/sgjholt/SpecMod/blob/main/CHANGELOG.md).
