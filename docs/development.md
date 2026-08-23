# Developer guide

Everything needed to work on SpecMod: the tools, the loop, the conventions,
and — the part that is easiest to get wrong — where development stops and
releasing begins.

This page is the hub. The three companions are
[Documentation workflow](documentation.md) (previewing, publishing, versions),
[Releasing the software](releasing.md) (tags, PyPI, DOI) and
[Publishing a dataset](releasing-data.md) (`data-v*` artefacts).

**Where to look for what**

| I want to… | Go to |
|---|---|
| Get a working checkout | [Quick start](#quick-start) |
| Know what lives where | [The repository, mapped](#the-repository-mapped) |
| Make a change | [The daily loop](#the-daily-loop) |
| Understand a tool or a check | [Tooling reference](#tooling-reference) |
| Write or fix a test | [Testing](#testing) |
| Read a CI failure | [What CI runs](#what-ci-runs) |
| Change a workflow file | [The `ci/` mirror](#the-ci-mirror) |
| Understand versions and releases | [Development versus release](#development-versus-release) |
| Preview or publish docs | [Documentation workflow](documentation.md) |
| Use Claude or Codex on this repo | [Working with agents](#working-with-agents) |
| Build a package on top of SpecMod | [The stable surface](#the-stable-surface) |

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/sgjholt/SpecMod.git
cd SpecMod
uv venv && uv pip install -e ".[dev]"
pre-commit install                       # both hook types; see below
pytest -m "not dataset and not notebook" # ~3 minutes, should be all green
```

Four things about that, in order of how often they bite:

1. **`pre-commit install` is a required step, not a nicety.** It installs the
   `pre-commit` *and* `commit-msg` hooks — the config names both stages under
   `default_install_hook_types`, because a plain `pre-commit install` used to
   wire only the first and left the commit-msg check inert.
2. **Install with `[dev]`, not bare.** The I/O suite needs `h5py` and
   `pyarrow`, which `[dev]` pulls in; CI does the same.
3. **`-m "not dataset and not notebook"`** is what CI runs. `dataset` tests
   need a network download and `notebook` executes the tutorial with a Jupyter
   kernel (~40 s).
4. **The editable install must come from a git checkout with history.** The
   version is derived by `hatch-vcs` from `git describe`; a tarball without
   `.git` reports the fallback `0.0.0`.

Optional extras, if you are working on those code paths:

| Extra | Adds | Needed for |
|---|---|---|
| `multitaper` | [`multitaper`](https://github.com/gaprieto/multitaper) | Prieto's estimator, jackknife CIs |
| `wavelet` | [PyWavelets](https://pywavelets.readthedocs.io) | wavelet families beyond the built-in Morlet |
| `io` | h5py, pyarrow | HDF5 and Parquet persistence (in `[dev]` already) |
| `docs` | Sphinx and friends | building this site |
| `tutorial` | ipykernel, nbclient | executing the tutorial notebook |
| `mcmc` | [emcee](https://emcee.readthedocs.io) | sampling-based fits |

## The repository, mapped

```
src/specmod/          the package
  config/             layered settings, provenance stamping
  core/               Spectrum, collections, noise, bandwidth, scalogram, units
  transforms/         FFT, Welch, multitaper, Prieto, quadratic, CWT
  smoothing/          Konno–Ohmachi, log binning
  sources/            source models, attenuation, motion factors
  fitting/            the fitter: base, guess, spectrum, event
  picks/              pick readers, sensor resolution, the registry
  acquire.py          the only module that touches the network
  datasets.py         hash-pinned published datasets, via pooch
  cli.py              the `specmod` command
tests/                the suite, plus tests/golden/ reference numbers
tools/                repository scripts, each with a CI job or test behind it
ci/workflows/         staged copies of .github/workflows (see below)
docs/                 this site
  REFACTOR_PLAN.md    the working document — not part of the built site
datasets/             dataset definitions for `specmod fetch`
tutorial/             the tutorial notebook and its data
stubs/                hand-written ObsPy type stubs
```

Two files that are not what they look like:

- **`docs/REFACTOR_PLAN.md`** is a working document, deliberately excluded from
  the built site. It records decisions, the measurements behind them, and an
  audit (§6.6) of claims in it that turned out to be false. When something here
  says "why", that is usually where the long answer is.
- **`ci/workflows/`** holds complete copies of the live GitHub Actions
  workflows. See [The `ci/` mirror](#the-ci-mirror).

## The daily loop

```sh
git switch -c my-change                      # branch off main
# ... edit ...
pytest -m "not dataset and not notebook"     # the suite CI runs
pytest --without-optional-extras             # what a default install sees
mypy                                         # strict, on the whole package
git commit                                   # Conventional Commits; hooks run
git push -u origin my-change                 # open a PR against main
```

`ruff` runs automatically on commit via pre-commit; run it by hand with
`ruff check src/ tests/ tools/` and `ruff format src/ tests/ tools/`.

**Run `--without-optional-extras` before pushing.** A development environment
with `specmod[multitaper]` installed passes tests that CI, which installs only
`[dev]`, fails. It has happened twice.

### Branches

- **`main`** is the trunk. Everything lands here, and it is the default branch.
- **`master`** is frozen: the permanent record of the pre-refactor code, doing
  the job a `v0.1.0` tag would have done. Never commit to it.
- Feature branches are short-lived and merge into `main` via pull request.

Every pull request must target **`sgjholt/SpecMod`**. See §6.7 of the plan for
why that is worth checking rather than assuming.

### Commit messages

[Conventional Commits](https://www.conventionalcommits.org), because
`release-please` parses them to decide the version and write the changelog —
see [Development versus release](#development-versus-release). The type
controls where the commit lands:

| Type | Effect on the release |
|---|---|
| `feat:` | minor bump; **Features** |
| `fix:` | patch bump; **Bug Fixes** |
| `perf:` | patch bump; **Performance** |
| `refactor:`, `docs:`, `build:` | no bump; shown in the changelog |
| `test:`, `ci:`, `style:`, `chore:` | no bump; hidden |
| `feat!:`, or a `BREAKING CHANGE:` footer | **minor** bump while below 1.0 |

There is no `commitlint` hook — the convention is followed by hand, and the
plan's §6.6 records that as a claim it once made falsely.

**No session URLs from AI coding tools** in commit messages or anywhere else
published. The repository is public and those links are private state; a
`commit-msg` hook rejects them, which is the reason `pre-commit install`
appears in the quick start rather than further down.

## Tooling reference

| Tool | Runs | Configured in |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | environments, installs, builds | `pyproject.toml` |
| [ruff](https://docs.astral.sh/ruff/) | lint + format, pre-commit and CI | `[tool.ruff]` |
| [mypy](https://mypy.readthedocs.io) | strict types over `src/specmod` | `[tool.mypy]` |
| [pytest](https://docs.pytest.org) | the suite | `[tool.pytest.ini_options]` |
| [hypothesis](https://hypothesis.readthedocs.io) | property tests | in-test |
| [pre-commit](https://pre-commit.com) | hooks on commit | `.pre-commit-config.yaml` |
| [hatch-vcs](https://github.com/ofek/hatch-vcs) | version from git tags | `[tool.hatch.version]` |
| [Sphinx](https://www.sphinx-doc.org) + [MyST](https://myst-parser.readthedocs.io) | this site | `docs/conf.py` |
| [release-please](https://github.com/googleapis/release-please) | changelog and version decision | `release-please-config.json` |

### The `tools/` scripts

Each one exists because something was claimed and not enforced. All are
standard-library-only unless noted.

| Script | Does | Enforced by |
|---|---|---|
| `check_ci_mirror.py` | staged workflows match the live ones | `lint` job |
| `check_floors.py` | the installed versions really are the declared minimums | `floors` job |
| `check_built_version.py` | the built wheel's version is the tag | `publish` job |
| `make_golden.py` | regenerates `tests/golden/*.json` | run by hand, deliberately |
| `measure_docs.py` | regenerates the measured tables in `docs/` | `tests/test_docs_are_current.py` |

`measure_docs.py` is worth knowing about before editing a table by hand:

```sh
python tools/measure_docs.py show     # print the tables
python tools/measure_docs.py write    # refresh the docs in place
python tools/measure_docs.py check    # fail if any table is stale
```

Numbers that came from a measurement live between markers and are generated.
`tests/test_docs_are_current.py` runs `check`, so a change that moves a
published number fails the suite instead of leaving the prose quietly wrong.
The `--field` measurements read `tutorial/data/events/` and are slower;
refresh those by hand after changing an estimator.

### Configuration and provenance

Settings live in `src/specmod/config/` as semantic sections with layered
overrides, not module-level constants read at import time. Two commands:

```sh
specmod config show      # the resolved configuration, with its layers
specmod config freeze    # write it out, pinned
```

Every output records the configuration that produced it, a hash of it, and the
SpecMod version. That is what makes a locally-overridden run reproducible from
its own outputs, and it is the mechanism that lets the package stay alpha
without published results becoming unrepeatable.

## Testing

```sh
pytest -m "not dataset and not notebook"   # what CI runs
pytest --without-optional-extras           # as a default install sees it
pytest -m notebook                         # executes the tutorial (~40 s)
pytest -m dataset                          # needs a network download
pytest tests/test_transforms.py -q         # one module
```

Markers are declared in `pyproject.toml` and `--strict-markers` is on, so a
typo in a marker name is an error rather than a silently-skipped filter.

**The tiers**, as the plan lays them out:

1. **Property tests** (`hypothesis`) encoding the physics — Parseval, scaling,
   units — which hold for any input rather than one recorded case.
2. **Synthetic end-to-end**: generate a spectrum with known source parameters,
   run the whole pipeline, recover them.
3. **Golden/regression**: run the current code on the tutorial event and on
   Magna, and compare against committed summaries in `tests/golden/`.
4. **Unit tests** for specific bugs, each written as a failing test first.

### Golden references, and the one rule about them

`tests/golden/*.json` records what this code produced at a known-good point.
It is compared as a **distributional summary** — median, quantile profile,
length — with a relative tolerance, not as a byte digest: an earlier version
hashed the raw float64 bytes and failed on every CI runner, because a
different numpy or BLAS build produces last-bit differences on identical
input. A reference that only holds on the machine that generated it is not a
reference.

**Do not regenerate it to make a test pass.** If a change moves a number, that
is the finding: say which number, by how much, and why. Then regenerate
deliberately:

```sh
python tools/make_golden.py    # and commit the result, with the reason
```

Tolerances carry comments explaining what was measured to choose them. The
`cwt` entry is the worked example — it records a per-runner residual that is
bounded rather than explained, and says so in as many words.

## What CI runs

Five jobs in `test.yml`, plus two more workflows. All of them run on every pull
request.

| Job | Workflow | Does |
|---|---|---|
| `lint` | `test.yml` | `ruff check`, `ruff format --check`, and the `ci/` mirror check |
| `typecheck` | `test.yml` | `mypy`, strict, over the whole package |
| `test` | `test.yml` | pytest on 3.11/3.12/3.13 × ubuntu/macOS, coverage to Codecov from one cell |
| `floors` | `test.yml` | installs the *declared minimum* versions and runs the suite |
| `notebook` | `test.yml` | executes the tutorial notebook |
| `ci` | `test.yml` | succeeds only if every job above did. **This is what branch protection requires** — see below |
| `build` | `build.yml` | sdist + wheel, `twine check`, install-from-wheel smoke test |
| `docs` | `docs.yml` | builds the site as a check; publishing is Read the Docs' job |
| `release` | `release.yml` | the release PR, and publishing — see below |

Two of those are worth understanding before you read a failure from them:

**`floors`** installs `--resolution lowest-direct`, exercising the oldest
dependency set the project claims to support. It caught two floors that could
never have worked: `lmfit>=1.2` with `numpy>=2.0` (lmfit below 1.3 calls
`np.asfarray`, removed in NumPy 2), and `scipy>=1.13` silently breaking the
quadratic multitaper. If you raise or add a dependency, this is the job that
tells you whether the floor you wrote is real.

**`ci` exists so branch protection has one name to require.** Required status
checks match *job* names, and a matrix job reports one check per cell —
`test (ubuntu-latest, 3.11)` and five more — so requiring "the test workflow"
means listing ten names that change whenever the matrix does. `ci` depends on
all of them and is required in their place, alongside `docs` and `build`.

Two details in it are not decoration. It runs `if: always()`, because a job
whose dependency failed is *skipped*, and **GitHub treats a skipped required
check as satisfied** — so without that line branch protection would go green
over a red build. And it counts `skipped` as a failure, because nothing in this
workflow is conditionally skipped, so a skip means something upstream broke.

Job names are also why `docs.yml`'s job is called `docs` rather than `build`:
it collided with `build.yml`'s job, leaving two unrelated checks sharing one
name and nothing named `docs` at all.

**`docs`** deliberately does **not** use `-W`. Intersphinx resolves seven
inventories over the network and warns whenever one is briefly unreachable;
turning a third party's downtime into a red build is flakiness, not a check.

## The `ci/` mirror

`ci/workflows/*.yml` holds complete, ready-to-paste copies of
`.github/workflows/*.yml`. The reason is narrow: the GitHub App token used by
AI coding sessions has no `workflows` permission, so a push touching
`.github/workflows/` is rejected. Rather than describe an edit in a comment and
hope it is applied correctly, the intended file is committed in full.

To apply one, copy the whole file over its counterpart — no merging, no partial
application. `tools/check_ci_mirror.py` runs in the `lint` job and fails while
a pair differs, so a staged change that has not been copied across shows up as
a red build rather than being forgotten. **That failure is the reminder.** It
clears the moment the files match.

The mirror is the *intended* state, which is not always the current one — in
either direction. If you fix a workflow through the GitHub web editor, copy it
back into `ci/` so the next person staging a change starts from the working
version.

Full detail in [`ci/README.md`](https://github.com/sgjholt/SpecMod/blob/main/ci/README.md).

## Development versus release

The thing to hold onto: **merging to `main` releases nothing.** `main` is
continuously integrated and continuously *documented*, but it is not
continuously published. Three separate clocks:

| | What moves it | What it produces | Who decides |
|---|---|---|---|
| **Development** | any merge to `main` | updated `main`, updated docs site | whoever merges the PR |
| **Software release** | merging the release PR | a `v*` tag, a GitHub Release, a PyPI upload, a DOI | a human, deliberately |
| **Data release** | creating a `data-v*` tag by hand | a dataset artefact pinned by hash | a human, deliberately |

### How a version comes to exist

No version string is committed anywhere. `hatch-vcs` derives it from
`git describe`, so **the tag is the version**:

- On `main` between releases you get `<last-tag>.postN.devN` — a version that
  claims nothing.
- On a `v*` tag you get exactly that tag without its `v`.

`pyproject.toml` constrains which tags count, with both a `--match v[0-9]*` on
the describe command and a `tag_regex` on the parse. Both are needed: a
`data-v1` tag was measured to take the package version from
`0.1.0.post1.dev173` to `1`, because setuptools-scm's default pattern strips
the `data-` prefix and reads what is left.

That is why the two release channels use different tag prefixes, and why
`tests/test_versioning.py` pins the parse.

### What a release actually does

`release-please` watches `main` and keeps a standing pull request titled
`chore(main): release <version>`, carrying the generated `CHANGELOG.md`.
Merging it creates the tag and the GitHub Release; a gated job then builds from
the tag, checks the built version against it, and uploads to PyPI via Trusted
Publishing; Zenodo mints a DOI from the release webhook.

Nothing publishes until that merge. The gate exists because **a DOI cannot be
retracted** — fully automatic tagging plus Zenodo means a typo fix can mint a
citable version of the software.

The step-by-step, including the six repository settings that have to be turned
on once and cannot be expressed in a commit, is in
[Releasing the software](releasing.md).

### What this means day to day

- **Land work whenever it is ready.** The changelog accrues; you are not
  choosing a version when you merge.
- **Write the commit message for the changelog**, because that is where it ends
  up verbatim.
- **Breaking changes are allowed** and land in minor bumps while the project is
  `0.x`. There is no deprecation cycle, by design — see the
  [roadmap](roadmap.md) for what 1.0 will change about that.
- **Datasets are versioned by registry name**, not by the package version:
  `magna_2020_v1` and `magna_2020_v2` are separate entries, so a published
  result pinned to v1 keeps fetching v1 forever. Nothing revises an entry in
  place.

## The stable surface

`specmod.api` is a small re-export module, and the only part of SpecMod that
carries a compatibility promise: one minor release of `DeprecationWarning`
before anything on it is removed or changes signature, even while the package
is `0.x`. Everything else may move in any release.

It exists for downstream packages. SpecMod's internals are still being
refactored; a package that imports `specmod.core` or `specmod.fitting` directly
takes on that churn, and one that imports `specmod.api` does not.

Five properties hold across the surface, and they are enforced by
`tests/test_api_surface.py` rather than promised in prose: **path-free**
(nothing on it opens a file — a consumer that stores its data on S3 has to be
able to hand in arrays), **deterministic**, **non-mutating**, **quiet** (no
`print`), and **typed errors** rooted at `SpecModError`, distinguishing a
caller's bad input from a missing optional backend from an internal bug.

Adding an export is a compatibility obligation, and the procedure is in
[`CONTRIBUTING.md`](https://github.com/sgjholt/SpecMod/blob/main/CONTRIBUTING.md).

The audit that established what could go on it — path coupling, hidden state,
determinism, and what one multitaper estimate actually costs — is in
[Audit: what `specmod.api` found in core](notes/api_audit.md). Two of its
findings were defects in core rather than in the surface, both since fixed and
both now guarded package-wide by `tests/test_ambient_state.py`: **no module
reads configuration at import time**, and **no module prints**. Those two
properties are worth knowing before adding code — a config read at module
level freezes the working directory the process started in, and a `print` is
invisible to a caller capturing logs.

```{toctree}
:hidden:

notes/api_audit
```

## Working with agents

Claude Code, Codex and similar tools are used on this repository. The durable
rules live in [`AGENTS.md`](https://github.com/sgjholt/SpecMod/blob/main/AGENTS.md)
at the repository root, with `CLAUDE.md` pointing at it so there is one copy;
agents read those automatically. What follows is the context for a human
supervising one.

**The failure modes are environmental, not intellectual.** Every one of these
has happened here:

- **A fresh container has no git hooks.** `pre-commit install` has to be run in
  the session, or the `commit-msg` check that rejects session links is simply
  not there. Three commits went out with session trailers before this was
  noticed; the config now installs both hook types from one command, and the
  first thing `AGENTS.md` says is to run it.
- **An agent's token cannot push `.github/workflows/`.** This is what the
  `ci/` mirror exists for. An agent that does not know about it will either
  fail the push or, worse, quietly drop the change.
- **A development container often lacks the optional extras**, so an agent's
  green run can be greener than CI's. `--without-optional-extras` is the check.
- **A harness may append its own commit trailers.** The repository's rules take
  precedence over a tool's defaults, and this one is a publishing rule rather
  than a style preference.

**What to ask for in review.** The habit this repository is built around is
saying what was checked and what was not. An agent that reports "fixed" should
be able to show the command and its output; one that widens a tolerance or
regenerates a golden file to reach green has moved the goalposts rather than
found the problem. §6.6 of the plan is an audit of exactly that failure — three
claims stated as fact with no mechanism behind them — and it is worth reading
once before delegating anything that touches a check.

**What agents are good at here.** The mechanical, checkable work: splitting
modules while keeping public names, writing the test that pins a behaviour
before changing it, running the same verification five ways, and the tedious
correctness of the docs — which is the same skill as the tests, since numbers
in prose go stale silently.
