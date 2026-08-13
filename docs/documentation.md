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
uv pip install -e '.[docs]'
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

One-time, at the point the first version is published. None of it can be done
from a commit; all of it is on <https://readthedocs.org>.

1. **Import the project.** Log in with GitHub, *Import a Project*, pick
   `sgjholt/SpecMod`. The slug chosen here becomes the URL —
   `https://<slug>.readthedocs.io` — so it is worth getting right first time.
2. **Check it found the config.** `.readthedocs.yaml` is in the repository
   root and is read automatically; the first build's log says which
   configuration file it used. If it did not find one, Read the Docs falls back
   to defaults that will not install this package.
3. **Turn on pull request builds.** *Admin → Settings → Advanced settings →
   Build pull requests for this project*. Without it there are no previews.
4. **Add an automation rule for tags.** *Admin → Automation Rules → Add rule*,
   version type **Tag**, action **Activate version**, matching `^v.*`. New
   releases then publish themselves; without a rule each tag has to be
   activated by hand before anyone can read it.
5. **Set the default version to `stable`** once a tag exists. *Admin →
   Settings → Default version*. Until then it is `latest`, which is correct
   while there are no releases.

The build configuration itself is
[`.readthedocs.yaml`](https://github.com/sgjholt/SpecMod/blob/main/.readthedocs.yaml).
Two things in it are load-bearing and easy to lose:

- **`post_checkout` unshallows the clone and fetches tags.** Read the Docs
  clones shallow to save time, and `hatch-vcs` derives the version from
  `git describe`. Without those two lines every build — including a tagged one
  — reports the fallback `0.0.0` in the sidebar.
- **`extra_requirements: [docs, io]`.** The `io` extra is there because autodoc
  imports `specmod.io`, which imports h5py and pyarrow. The CI job installs the
  same pair for the same reason.

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

`docs/notes/` **is** included, which was a correction: `choosing_a_transform.md`
links to `notes/window_position.md` for a per-trace table, and a page another
page depends on is documentation whatever its folder is called.

### Adding a page

1. Write `docs/<name>.md` in MyST Markdown.
2. Add it to the `toctree` at the bottom of `docs/index.md`, and usually to the
   "Where to start" list above it.
3. Build. A page in no toctree builds but warns, and is reachable only by a
   direct link.

If a page is a supporting note rather than a top-level one, put it in a hidden
toctree on the page that cites it — that is how `notes/window_position.md` is
attached to `choosing_a_transform.md`:

````markdown
```{toctree}
:hidden:

notes/window_position
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
- Markdown only. `source_suffix` maps `.md`; there is no reStructuredText in
  `docs/`, so there is one syntax rather than two.
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
the same 367 documented objects while calling an API Sphinx 10 removes.

### Numbers in prose

Any table that came from a measurement is generated, not typed. Edit
`tools/measure_docs.py` and run `python tools/measure_docs.py write`;
`tests/test_docs_are_current.py` fails if a table drifts from what the code
does. See the [Developer guide](development.md#the-tools-scripts).

## Changing the docs workflow

`docs.yml` cannot be pushed by an AI coding session — no `workflows` token
permission — so the intended file lives at `ci/workflows/docs.yml` and is
copied across by hand. The `lint` job fails while the two differ, in **either**
direction: if you fix the live workflow in the web editor, copy it back into
`ci/`. See [The `ci/` mirror](development.md#the-ci-mirror).
