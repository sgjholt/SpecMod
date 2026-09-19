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

SpecMod is [on PyPI](https://pypi.org/p/specmod), with a Zenodo DOI per release
and its own [documentation version](https://specmod.readthedocs.io/en/stable/)
for each. Install it with `pip install specmod`; `v0.2.0` was the first release
of the rebuilt package.

*This page was built from version* **{{ release }}**. On `stable` that is the
release you are reading about; on `latest` it is a development build ahead of
it, which is the honest answer to which code the milestones below describe.

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
branch is the only reference to it. Its tip,
[`453c77c`](https://github.com/sgjholt/SpecMod/commit/453c77c), is the commit
the paper cites as v0.1.1; cite that rather than the branch name, which is a
moving target in principle even though this one will not move. Everything below is the delta between it
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
dropped rather than being the only path: multitaper is implemented natively on
SciPy's DPSS tapers, and Prieto's `multitaper` package is available through the
optional `[multitaper]` extra as the same-lineage substitute. Asking the
pipeline for `mtspec` by name raises, and says which two to use instead.

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

## Shipped in v0.2.2

### The tutorial, executed

The tutorial is published as part of this site and **re-executed on every
build**, so every figure and number on the page came from running the code
being documented. A notebook that stops working fails the build rather than
going quietly stale, which is why it was a 1.0 requirement.

## Shipped in v0.3.0

### The decomposition, finished

This was the last item 1.0 was waiting on.

`io.py` and `plotting.py` are now packages, alongside the spectral core, the
fitting layer and the pick readers; the type-checking backlog is empty. The
split is by responsibility rather than line count — for `io/`, what a file
contains against how one is written against where the optional dependency
lives.

**`specmod.preprocess` no longer modifies the stream it is given.** Every
function returns a new one, which is the package-wide rule the containers have
always followed and the only place it did not hold. Ten functions were renamed
to say so — `cut_s` is `s_window`, `set_picks` is `with_picks` — because a
function that starts returning instead of mutating while keeping its name
leaves every existing call running and silently doing nothing.
[Upgrading](upgrading.md#from-02-to-03) is the table.

### The documentation 1.0 asked for

The conventions are stated explicitly and can be found: the amplitude
convention, the Parseval contract, and now the moment and magnitude equations
with their constants, in
[How a spectrum is processed](processing.md) — in the sections that apply them,
signposted from [Guides](guides.md) rather than split onto a page of their own.
[Upgrading](upgrading.md) covers moving code off the pre-refactor
`master`.

## Shipped in v0.4.0

### Smoothing became a choice, and the choice is applied

`[smoothing] method` selected nothing. The registry mapped the names, the config
validated against them, and no code read the result: every spectrum was
log-binned whatever the setting said. Measured on the 28 PNR windows,
`log_bins`, `konno_ohmachi` and `none` produced bit-identical output — `none`
included, which reads as "leave my spectra unsmoothed".

It is wired now, with three methods beside the two:

- **`log_window`** — constant relative bandwidth over a window you name:
  boxcar, Bartlett, Hann, Hamming, Blackman or Gaussian, a fraction of an
  octave wide everywhere. Konno–Ohmachi is this family with its own window.
  `statistic="median"` discards a spike rather than averaging it in, and
  discards a genuine narrow peak with it.
- **`savitzky_golay`** — a local polynomial fit in log–log, which keeps peak
  amplitude and width where a running mean flattens both.
- **`none`** — no smoothing, which now means it.

Log-binning replaces the frequency axis; the others preserve it. That is the
distinction the setting had no way to express, and the reason wiring it up was
a design change rather than a patch.

One detail that is easy to get wrong and is worth knowing about: a window
symmetric in log frequency is not a symmetric average over samples, because a
Fourier grid is uniform in Hz. Weighting samples equally tilts the result by
1.5e-3 dex on an `f**-2` power law; weighting each by its share of the log
axis brings that to 3.2e-8 dex. The latter is the default. Konno–Ohmachi
applies the former, by definition, and keeps it.

## In progress — not yet released

Merged on `main`, and it will name its version when a release goes out.

### The station selection means what the config says

Three ways a fetched dataset did not match the config that asked for it, all
found by a live fetch rather than by the suite:

- `max_radius_km` reached the station query and never the waveform request,
  because FDSN dataselect has no geographic parameters. A 50 km fetch returned
  channels from 225 km out, with no metadata beside them. Waveforms are now
  asked for by name, from the inventory the station query returned.
- The radius was converted at 111.195 km per degree. A degree of arc is
  110.574 km at the equator and 111.691 km at the poles, so the boundary moved
  with latitude — ±0.3 km at 50 km, ±2 km at 400 km. The query is now sent
  deliberately wide and the cut made against the true WGS84 distance.
- A `channel` pattern naming several instruments returned all of them at every
  station, so a broadband and the accelerometer beside it entered the pipeline
  as two independent records of one ground motion. `channel_priorities` and
  `location_priorities` rank them, first match wins per station.

## Planned

### 1.0 — the API stops moving

**Everything this was waiting on shipped in v0.3.0** — the documentation, and
the decomposition above with the signature changes it implied. What remains is
the release itself, which says that names and signatures stop moving without a
deprecation cycle.

That promise is the whole content of the number, which is why it has to be a
decision rather than something a breaking commit does on its way past. Until it
is made, breaking changes bump the minor. The honest test before making it is
whether a release goes by without one — the API is not settled because the
backlog is empty, it is settled when it stops moving.

**By that test, 1.0 is not ready.** v0.3.0 is the release that emptied the
backlog and it carried two breaking changes itself: the ten renamed
`preprocess` functions, and the tutorial's page moving. So what 1.0 now waits
on is a release going out without one — which is a thing that has to be
observed rather than declared, and cannot be brought forward by finishing
anything.

**v0.4.0 is the first observation, and it is not clean.** No name or signature
moved: the smoothing work was additive, and the `SpectrumPair.compare`
parameter it added is optional. But `[smoothing] method` changed what it does,
so a config that set `konno_ohmachi` or `none` gets different numbers out of
the same code path. Nothing breaks at import or at the call; results move. That
is a weaker promise than the one 1.0 makes and a real one to have broken, so
the count starts again rather than standing at one.

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
- **A fuller `specmod fetch --verify`** that re-fetches and diffs against the
  manifest, rather than only detecting local tampering.

## How this page works

Each shipped entry names the version that carries it, so a reader can install
that version and check the claim. Work that is merged but unreleased sits under
*In progress* and gets its version when a release goes out — merged is not
shipped, and this page does not blur the two. Between releases there is no such
section at all, which is a state the page should be in rather than a section
that went missing.

The move at release time is the step this page got wrong once. The v0.3.0 work
sat under *In progress — not yet released* for a week after v0.3.0 was
released, because `release-please` writes the changelog and the manifest and
knows nothing about this file. `tests/test_release_config.py` now fails when
the released minor version is ahead of every version this page names, which is
the check that was missing rather than a rule that was.

Before v0.2.0 this was a list of stages, because there was no released version
to point at and "done" could only mean merged. The full changelog for every
release is in
[`CHANGELOG.md`](https://github.com/sgjholt/SpecMod/blob/main/CHANGELOG.md).
