# SpecMod

[![PyPI](https://img.shields.io/pypi/v/specmod.svg)](https://pypi.org/project/specmod/)
[![Documentation](https://readthedocs.org/projects/specmod/badge/?version=stable)](https://specmod.readthedocs.io/en/stable/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22071454.svg)](https://doi.org/10.5281/zenodo.22071454)

A Python toolbox for processing and modelling seismic spectra, following the
method of Edwards *et al.* (2010).

SpecMod estimates source parameters — long-period spectral level Ω, corner
frequency `f_c`, and the attenuation operator `t*` — by fitting a Brune-type
source model to direct-phase spectra.

> **Status: alpha, and under active reconstruction.**
> The package is pre-1.0 and mid-refactor. The modern layers
> (`specmod.config`, `specmod.core`, `specmod.transforms`, `specmod.picks`,
> `specmod.fitting`) are built and tested; the older pipeline modules still
> carry pre-refactor behaviour and are being replaced stage by stage. Expect
> breaking changes at every `0.x` release until the API settles at 1.0 — they
> land in minor bumps by design, with no deprecation cycle. Pin an exact
> version for anything you intend to publish.
> [`docs/roadmap.md`](docs/roadmap.md) says what has shipped, in which version,
> and what 1.0 will mean; [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md) is
> the working document behind it.

## Installation

Requires Python 3.11 or newer.

```sh
pip install specmod
```

While this is `0.x`, pin the exact version in anything you intend to publish:
`pip install specmod==<version>`, taking the current one from the badge above.
The reason is in the status note — `0.x` releases move names and numbers, and
the pin plus the configuration stamp each output carries are together what
make a run reproducible.

To work on SpecMod rather than with it, clone the repository and install it
editable with the test and lint tooling:

```sh
pip install -e ".[dev]"
```

Optional extras, installable as `pip install "specmod[multitaper]"` or in
combination — `pip install "specmod[multitaper,wavelet]"`:

| Extra | Adds |
|---|---|
| `io` | `h5py` and `pyarrow` — needed to save or load spectra, as HDF5 for arrays and Parquet for tables |
| `multitaper` | Prieto's `multitaper` package — jackknife confidence intervals, F-test for spectral lines |
| `wavelet` | PyWavelets, for wavelet families beyond the built-in Morlet |
| `mcmc` | `emcee`, for Markov-chain Monte Carlo parameter search |

`io` is the one most people want: without it SpecMod computes and plots
normally, but `specmod.io` raises on the first save telling you to install it.

No Fortran compiler is needed. Multitaper estimation is implemented natively on
SciPy's DPSS tapers, so the historical `mtspec` dependency — Fortran source with
no wheels and no release since 2016 — is no longer required.

## Estimating a spectrum

Every estimator returns a `Spectrum` that knows its own units:

```python
import numpy as np
from specmod.transforms import FFTEstimator, MultitaperEstimator

dt = 0.01
trace = np.random.default_rng(0).normal(0, 1e-6, 2000)

spectrum = MultitaperEstimator(time_bandwidth=3.0, n_tapers=5).estimate(
    trace, dt, motion="velocity"
)

spectrum.unit           # 'm/s*s' — a Fourier amplitude spectrum
spectrum.duration       # 20.0 s, the physical record length
spectrum.energy()       # recovers sum(x**2) * dt
```

Conversions return new objects and are unit-aware:

```python
spectrum.to_motion("displacement")   # divides by 2*pi*f
spectrum.to_kind("psd")              # A**2 / (2T)
spectrum.band(0.5, 25.0)
```

The wavelet estimator additionally exposes the full time-frequency surface,
which is what to look at when a fit comes out wrong:

```python
from specmod.transforms import CWTEstimator

scalogram = CWTEstimator().scalogram(trace, dt)
scalogram.time_average()     # an ordinary Spectrum, fits like any other
scalogram.coi_coverage()     # how much of the window each frequency resolves
scalogram.qc()               # concentration, coda balance, resolved bandwidth
```

Which estimator to use, and what each one does to your data — including
measured position-dependence and the variance-normalisation convention `mtspec`
used — is set out in
[`docs/choosing-a-transform.md`](docs/choosing-a-transform.md), with a
worked walkthrough in
[`docs/notebooks/choosing-a-transform.ipynb`](docs/notebooks/choosing-a-transform.ipynb).

### Why the units are typed

A spectrum carries its ground-motion domain and amplitude convention as
attributes rather than in module-level globals. The alternative is keeping those
globals in sync by hand with however many times `.inte()` or `.diff()` has been
called, where getting it wrong returns a wrong seismic moment with no error
anywhere. Here it is a type error.

Amplitude normalisation is keyed off the physical record duration, never off the
length of the frequency axis, so zero-padding refines the frequency grid and
changes nothing else.

## Reading picks

`with_picks` returns a stream with arrivals attached, detecting the format
from the file rather than from its name:

```python
import specmod.preprocess as pre

stream = pre.with_picks(stream, "event.xml")
```

Everything `obspy.read_events` parses is read through one delegate — QuakeML,
SEISAN Nordic, HypoDD, NonLinLoc, IMS/GSE bulletins — plus Snuffler marker
files, plus delimited tables whose columns you name. A format registered with
ObsPy's own plugin system is read here with no SpecMod-side registration at
all.

Most formats supply less than a full sensor identity — a bare station code is
common — so a pick matches a trace on the fields it *states*. A pick that fits
several sensors is an error rather than a broadcast: a station's surface and
borehole instruments differ only by location code and do not see the same
arrival.

Adding a format, the column mapping for a picker's CSV, and the policies for
duplicate picks and multi-event files are in
[`docs/pick-formats.md`](docs/pick-formats.md).

## Configuration

Settings are grouped by pipeline stage and resolved through five layers —
package defaults, a committed `specmod.toml`, a gitignored
`specmod.local.toml`, `SPECMOD_*` environment variables, then explicit
arguments:

```sh
specmod config show      # resolved values, and which layer set each one
specmod config freeze    # emit TOML to commit alongside a result
```

Package defaults reproduce the behaviour that shipped before the refactor, so
upgrading does not silently move anyone's numbers. A study pins its own values
in a committed file — see [`studies/magna_2020_paper.toml`](studies/magna_2020_paper.toml),
which transcribes the workflow of the Magna paper below.

Every output records the configuration that produced it, a hash of it, and the
SpecMod version, so a locally-overridden run is still reproducible from its
outputs.

The published Magna results were produced with **0.1.1**, which predates this
refactor. That code is preserved unchanged on the
[`master`](https://github.com/sgjholt/SpecMod/tree/master) branch, which is
protected and frozen; `main` is the trunk now. `0.1.1` was never tagged or
published to PyPI, so the branch is the reference — there is no release to
install. Read it there when you need to see exactly what the paper ran.

## Documentation

The full documentation — the pipeline with its equations, the estimator
comparison, pick formats, and an API reference — builds with Sphinx:

```bash
uv pip install -e '.[docs,io,tutorial]'
sphinx-build -b html docs docs/_build/html
```

`docs/REFACTOR_PLAN.md` is excluded from the built site on purpose: it is a
working document that records decisions and the measurements behind them, not
documentation for using the package.

## Development

```sh
uv venv && uv pip install -e ".[dev]"
pre-commit install                 # both hook types; not optional
pytest                             # test suite
pytest --without-optional-extras   # as a default install and CI see it
ruff check src/ tests/ tools/      # lint
ruff format src/ tests/ tools/
mypy                               # strict on the rewritten modules
```

[`docs/development.md`](docs/development.md) is the full guide — the repository
mapped, every tool and CI check, the branch and commit conventions, and where
development stops and releasing begins. [`AGENTS.md`](AGENTS.md) is the short
version that binds AI coding sessions.

Run `--without-optional-extras` before pushing. A development environment
with `specmod[multitaper]` installed will pass tests that a default install
fails, and CI installs only `[dev]`.

The `mypy` override list in `pyproject.toml` is the migration backlog: modules
leave it as they are rewritten, and the target is an empty list.

Command-line entry points are built on [`click`](https://click.palletsprojects.com),
including internal tooling — one convention across the whole surface.

### Measured numbers in the docs

The documentation quotes a lot of measurements, and stale numbers are worse
than no numbers. Every table that came out of a measurement is generated:

```sh
python tools/measure_docs.py show     # print the tables
python tools/measure_docs.py write    # refresh the docs in place
python tools/measure_docs.py check    # fail if any table is stale
```

`tests/test_docs_are_current.py` runs `check`, so a change that moves a
published number fails the suite rather than quietly leaving the prose wrong.
Measurements that read `tutorial/data/events/` are slower and opt-in via `--field`;
refresh those by hand after changing an estimator.

### Releasing

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org),
which is what makes the changelog and the version automatic: `release-please`
opens a standing release pull request, and merging it creates the tag, the
GitHub Release, the PyPI upload and the Zenodo DOI. Nothing is released until
that merge. See [`docs/releasing.md`](docs/releasing.md), which also lists the
repository settings that have to be turned on once.

## References

Edwards, B., Allmann, B., Fäh, D., Clinton, J. (2010). Automatic computation of
moment magnitudes for small earthquakes and the scaling of local to moment
magnitude. *Geophysical Journal International* 183(1), 407–420.
<https://doi.org/10.1111/j.1365-246X.2010.04743.x>

Holt, J., Whidden, K.M., Koper, K.D., Pankow, K.L., Mayeda, K., Pechmann, J.C.,
Edwards, B., Gök, R., Walter, W.R. Towards robust and routine determination of
Mw for small earthquakes: application to the 2020 Mw 5.7 Magna, Utah, seismic
sequence. *Seismological Research Letters*.

Thomson, D.J. (1982). Spectrum estimation and harmonic analysis.
*Proceedings of the IEEE* 70(9), 1055–1096.

## Contributing

Issues and pull requests are welcome. Commits follow
[Conventional Commits](https://www.conventionalcommits.org), which drives the
changelog and version bumps.

## License

MIT — see [LICENSE](LICENSE).
