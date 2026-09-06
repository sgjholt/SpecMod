# Documentation workflow

How this site is written, previewed, checked and published — and what the
difference is between a build you made to look at, the build CI makes on a
pull request, and the versions people actually read.

Companion to the [Developer guide](development.md).

## Where each build goes

| Build | Made by | Lives | Official? |
|---|---|---|---|
| **Local** | you, `sphinx-build` | `docs/_build/html/`, gitignored | no — nobody else can see it |
| **Pull request check** | the `docs` job in GitHub Actions | a downloadable artefact on the run | no — it checks, it does not publish |
| **Pull request preview** | Read the Docs | its own temporary URL, linked from the PR | no — it disappears when the PR closes |
| **`latest`** | Read the Docs, from `main` | `/en/latest/` | yes, but it is the trunk |
| **`stable`** | Read the Docs, from the newest `v*` tag | `/en/stable/`, and the default | yes — this is the site |
| **`v0.2.0`, `v0.3.0`, …** | Read the Docs, from each tag | `/en/v0.2.0/` | yes, and frozen |

**Read the Docs publishes; GitHub Actions only checks.** That split is
deliberate, and the reason is versions. This package is alpha and the docs tell
people to pin an exact version, so the documentation for a release has to stay
readable after the trunk moves on. One site that always shows `main` — which is
what deploying to GitHub Pages gives you — means someone pinned to `0.2.0`
reads about code they do not have.

Merging to `main` updates `latest` within a few minutes and changes `stable`
not at all. `stable` moves when a release is tagged. See
[Development versus release](development.md#development-versus-release) for why
those are separate clocks.

## Building it locally

```sh
uv pip install -e '.[docs,io,tutorial]'
sphinx-build -b html docs docs/_build/html
python -m http.server -d docs/_build/html 8000   # then open localhost:8000
```

Rebuild after editing; Sphinx is incremental, so only changed pages are redone.
To force a clean build — worth doing before trusting a warning count —
`rm -rf docs/_build` first.

**Expect warnings about intersphinx.** The build resolves seven inventories
(python, numpy, scipy, pandas, matplotlib, obspy, lmfit) over the network, and
each unreachable one is a warning. Offline you will see seven of them and a
site that builds correctly, with cross-references to other projects left as
plain text. That is also why neither the CI job nor `.readthedocs.yaml` fails
on warnings: turning a third party's downtime into a red build is flakiness
rather than a check.

Anything *other* than those seven is a real warning. A clean build today looks
like:

```
build succeeded, 7 warnings.     # all of them intersphinx, offline
```

## The two automatic builds on a pull request

**Read the Docs builds a preview** at its own URL and posts the link as a
status check on the pull request. That is the one to click when you want to
*look* at a change. It is torn down when the pull request closes.

**The `docs` job in GitHub Actions builds the same site and keeps the HTML**
as an artefact. It is kept alongside the Read the Docs preview for two reasons:
it does not depend on a third-party service being up, and because autodoc
imports every module it documents, it doubles as an import check. A module that
no longer imports fails here.

To open what Actions built: the pull request → **Checks** → **docs** → the run
summary → download the **docs-html** artefact, then

```sh
unzip docs-html.zip -d preview
python -m http.server -d preview 8000
```

## Read the Docs setup

One-time. None of it can be done from a commit.

**Install the GitHub App first.** This is the step to get right, because the
alternative looks like it works and does not. Read the Docs has moved from an
OAuth connection to a GitHub App, and the App is what subscribes to the events
— *"No need to create webhooks on your repositories. The GitHub App subscribes
to all required events when you install it."*

1. **Install the Read the Docs GitHub App** — from Read the Docs under
   *Settings → Connected Services*, or directly at
   <https://github.com/apps/readthedocs>. Grant it `sgjholt/SpecMod`; *only
   select repositories* is enough and is the point of the App. It asks for read
   access to code, and **read and write to checks, commit statuses and pull
   requests** — that write access is what puts the preview link on a pull
   request.
2. **Create the project**, or connect an existing one. *Add project* and search
   for the repository. If the project already exists — imported by URL, say —
   installing the App does not link it retroactively: go to *Settings* and pick
   the repository from the **Connected repository** dropdown.
3. **Check the first build found the config.** `.readthedocs.yaml` is in the
   repository root and is read automatically; the build log names the
   configuration file it used. If it found none, Read the Docs falls back to
   defaults that will not install this package.
4. **Turn on pull request builds.** *Admin → Settings → Advanced settings →
   Build pull requests for this project*.
5. **Add an automation rule for tags.** *Admin → Automation Rules → Add rule*,
   version type **Tag**, action **Activate version**, matching `^v.*`. New
   releases then publish themselves; without a rule each tag is built and then
   left unpublished until someone activates it by hand.
6. **Set the default version to `stable`** once a tag exists. *Admin →
   Settings → Default version*. Until then it is `latest`, which is correct
   while there are no releases.

**Do not add a webhook by hand.** It is possible, it delivers `200`s, and it
will still leave the feature half-built: a manual webhook can trigger a build,
but without the App's write access to commit statuses there is nothing able to
post the preview back onto the pull request. Read the Docs' own error in that
state — *"Unable to attach webhook to this project"* — points at permissions
rather than at the App, which is what makes it worth writing down. If a manual
webhook already exists, delete it once the App is installed, or every push
fires two triggers.

**How to know it actually worked.** The error banner disappearing proves
little; it is dismissible and is only raised when Read the Docs tries to attach
a webhook, which it no longer needs to do. In increasing order of conclusiveness:
the *Connected repository* field survives a page reload; a build starts on its
own after a push to `main`; and a `docs/readthedocs.org` check appears on a
pull request with a preview link. The last one is the real proof, because it is
the part only the App can do.

The build configuration itself is
[`.readthedocs.yaml`](https://github.com/sgjholt/SpecMod/blob/main/.readthedocs.yaml).
Two things in it are load-bearing and easy to lose:

- **`post_checkout` unshallows the clone and fetches tags.** Read the Docs
  clones shallow to save time, and `hatch-vcs` derives the version from
  `git describe`. Without those two lines every build — including a tagged one
  — reports the fallback `0.0.0` in the sidebar.
- **`extra_requirements: [docs, io, tutorial]`.** `io` because autodoc imports
  `specmod.io`, which imports h5py and pyarrow — and because the tutorial saves
  an HDF5 file while executing. `tutorial` for the Jupyter kernel that executes
  it. The CI `docs` job installs the same three for the same reasons; keep them
  in step or one of the two builds fails on a missing kernel.

## Versions, and what to link to

- **Link to `/en/stable/`** in papers, READMEs and anywhere durable. It follows
  releases without becoming stale.
- **Link to `/en/v0.2.0/`** when the reference is to a specific version's
  behaviour — for a published result, this is the honest link.
- **`/en/latest/` is the trunk**, ahead of any release during alpha. It is what
  contributors should read and what nobody should cite.

A version's build is frozen at what it said when that tag was cut, which is the
point: rebuilding 0.2's documentation later would not make it more true.

## Writing for the site

### What belongs here, and what does not

The site is for someone *using or developing* SpecMod. `REFACTOR_PLAN.md` is
excluded from it on purpose: it is a working document, written for whoever is
doing the refactor, recording decisions and the evidence behind them. Link to
it on GitHub rather than adding it to the build.

`docs/notes/` **is** included, which was a correction: `choosing-a-transform.md`
links to `notes/window-position.md` for a per-trace table, and a page another
page depends on is documentation whatever its folder is called.

### Adding a page

1. Write `docs/<name>.md` in MyST Markdown.
2. **Add it to the `toctree` of the section it belongs to** — not to
   `index.md`. The site has four section pages, each owning its own toctree:
   `getting-started.md`, `guides.md`, `api.md` and `contributing.md`. Add a
   description to the list on that page too, since the sidebar shows titles
   only.
3. Build. A page in no toctree builds but warns, and is reachable only by a
   direct link.

`index.md`'s toctree holds only those four. That is what gives the sidebar its
nesting: this theme puts top-level toctree entries in the header and gives the
sidebar the current section's children, so a page added at the top level lands
in the header and flattens the navigation for everything else.

If a page is a supporting note rather than a top-level one, put it in a hidden
toctree on the page that cites it — that is how `notes/window-position.md` is
attached to `choosing-a-transform.md`:

````markdown
```{toctree}
:hidden:

notes/window-position
```
````

### Markdown, math and directives

[MyST](https://myst-parser.readthedocs.io) with `colon_fence`, `deflist`,
`dollarmath` and `substitution` enabled, and heading anchors down to `h3` so
pages can link to each other's sections.

- Inline math is `$...$`, display math `$$...$$`. `dollarmath` is what makes
  those work; without it MyST does not read `$` at all.
- `amsmath` is **not** enabled — nothing here uses a bare `\begin{align}`
  outside `$` delimiters. A `\begin{cases}` inside `$$` renders without it.
- Markdown and notebooks. `source_suffix` maps `.md` and `.ipynb`, both to
  `myst-nb` — there is no reStructuredText in `docs/`, so there is one prose
  syntax rather than two. Both suffixes name `myst-nb` because that is the only
  parser `myst_nb` registers; pointing `.md` at `myst_parser`'s `markdown`
  fails the build with "Source parser for markdown not registered".
- For a Sphinx directive that has no MyST spelling, drop into rST:

````markdown
```{eval-rst}
.. automodule:: specmod.picks
   :imported-members:
```
````

### The API reference

`docs/api.md` is hand-ordered — grouped by what a run does in order, rather
than alphabetically — and each entry is an `automodule`. Two rules learned from
the first build:

- **Document a package at the path you import from.** `specmod.picks.PickSet`,
  not `specmod.picks.base.PickSet`. Documenting both the package and its
  submodules gives every re-exported name two targets and makes every
  cross-reference to it ambiguous.
- **Some objects break autodoc's signature formatter.** `HOLT_2019_UTAH` is a
  callable dataclass instance documented as module data; `inspect.signature`
  handles it, autodoc's own formatter raises. It is excluded with
  `:exclude-members:` and written out in prose instead.

Type hints come from the annotations via `autodoc_typehints = "description"`.
`sphinx-autodoc-typehints` is deliberately **not** used: measured, it produced
the same number of documented objects while calling an API Sphinx 10 removes.

### The tutorial notebook

`tutorial/SpecModTutorial.ipynb` is written by
[a builder](#the-notebook-builders), published as part of the site, and
**executed on every build** (`nb_execution_mode = "force"`). Every figure and
number on the page came from running that code against the code being
documented, and a notebook that raises fails the build
(`nb_execution_raise_on_error = True`). That is the whole point: a tutorial
nobody runs is the first thing to rot, and this one broke three times before
anything executed it.

Three things about the arrangement are worth knowing before changing it.

**It is copied into `docs/tutorial/` by `conf.py`, not moved.** Sphinx builds
only what is under `docs/`, and the notebook reads its 1 MB of waveforms
through paths relative to itself — so the data has to travel with it. The copy
is gitignored. `tutorial/` stays canonical because eight other files name
`tutorial/data/events/`, and a renamed data directory has broken this before.

**The copy is also what the notebook writes into.** It saves an HDF5 file and
two flatfiles as part of the lesson. Executed where it lives, a docs build
would leave those in your working tree; `tests/test_tutorial.py` copies to
`tmp_path` for exactly the same reason.

**Outputs are stripped in git**, by the `nbstripout` pre-commit hook, and
regenerated at build time. Do not commit them back: `force` ignores them, so a
committed output is never what a reader sees — only a stale diff.

The kernel comes from the `tutorial` extra. Both `.readthedocs.yaml` and
`.github/workflows/docs.yml` install `[docs,io,tutorial]`; keep them in step or
one of the two builds fails on a missing kernel. This does mean the notebook
executes twice per pull request — once here, and once in the `notebook` CI job,
which runs the single `-m notebook` test that executes it in a `tmp_path` copy.
Roughly 40 seconds, paid twice, to catch a broken notebook either as a failed
build or as a failed test. The cheaper checks around it — that every name the
notebook imports resolves, and that deleted modules stay unmentioned even in
prose — are unmarked, so they run in the `test` matrix job instead.

### The notebook builders

**Every notebook in the repository is written by a script**, one per notebook,
named after the notebook it writes with underscores where the notebook has
hyphens. `docs/_builders/` is the one place to look for all of them:

```text
docs/_builders/
    _notebook.py             shared: md(), code(), the envelope
    choosing_a_transform.py  writes docs/notebooks/choosing-a-transform.ipynb
    SpecModTutorial.py       writes tutorial/SpecModTutorial.ipynb
```

```sh
uv run python docs/_builders/SpecModTutorial.py
```

The correspondence is derived, not declared: `builder_for(__file__)` takes the
output name from the calling script's filename, so the two cannot be given
different names without renaming the file. A stem with no underscore is left
alone, which is how `SpecModTutorial.py` writes `SpecModTutorial.ipynb`
without the convention needing an exception.

The two notebooks are unlike each other and both live where they do for a
reason. The tutorial is published and executed; it stays in `tutorial/`
because it reads `tutorial/data/events/` through paths relative to itself, and
`conf.py` copies it into `docs/tutorial/` at build time. The transform
comparison is neither published nor executed — `choosing-a-transform.md` is
the page that carries that material. So a builder says where its notebook
goes with `into=`, and only the name is derived.

**The builder is the source of truth, and `tests/test_docs_are_current.py`
holds it to that** — it runs every builder and fails if a notebook changes,
and separately checks that every builder has a notebook and every notebook a
builder. Edit the builder and re-run it; do not edit the `.ipynb`.

That test exists because the invariant had already been lost once. The
transform notebook had been through `ruff format` and had cell ids added, its
builder had not, and the builder had been edited five times against the
notebook's two — rebuilding would have reverted the formatting of every code
cell. The two still agreed cell for cell, which is the only reason it was
recoverable.

Consequences for writing one. Cell sources are emitted verbatim, so **write
them already formatted** — the builders deliberately do not shell out to
`ruff`, which is not importable from the docs environment and is pinned to
three different versions across a local checkout, the pre-commit hook and CI.
Sources are also stripped, so leading and trailing blank lines inside a cell
do not survive a rebuild. Cell ids are sequential rather than the random hex
`nbformat` assigns, so a rebuild diffs only where content changed.

`docs/_builders/` is outside the ruff and mypy scopes on purpose — builders
carry long prose lines and mathematical unicode that the source rules reject —
so keep new builders there rather than under `tools/`.

### Numbers in prose

Any table that came from a measurement is generated, not typed. Edit
`tools/measure_docs.py` and run `python tools/measure_docs.py write`;
`tests/test_docs_are_current.py` fails if a table drifts from what the code
does. See the [Developer guide](development.md#the-tools-scripts).
