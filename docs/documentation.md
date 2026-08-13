# Documentation workflow

How this site is written, previewed, checked and published — and what the
difference is between a build you made to look at, the build CI makes on a
pull request, and the official site.

Companion to the [Developer guide](development.md).

## The three builds

| Build | Made by | Lives | Is it official? |
|---|---|---|---|
| **Local** | you, `sphinx-build` | `docs/_build/html/`, gitignored | no — nobody else can see it |
| **Pull request** | the `docs` job, on every PR | an artefact on the workflow run | no — it checks, it does not publish |
| **Official** | the `docs` job, on `main` only | GitHub Pages | yes — this is the site |

There is exactly one published site and it tracks `main`. Nothing you do on a
branch changes it, and merging a branch changes it within a few minutes,
without any release being involved. See
[Development versus release](development.md#development-versus-release) for
why documentation and releases move on different clocks.

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
plain text. That is also why the CI job does not use `-W`: turning a third
party's downtime into a red build is flakiness rather than a check.

Anything *other* than those seven is a real warning. A clean build today looks
like:

```
build succeeded, 7 warnings.     # all of them intersphinx, offline
```

## What the pull request build does

`docs.yml` runs on every pull request. It installs the package — not just the
docs extra, because autodoc imports every module it documents, which makes a
docs build an import check too — and builds the site. It does **not** deploy.

A broken cross-reference, a page missing from a toctree, or a module that no
longer imports fails here, before it can reach the published site.

### Previewing a pull request's build

There is no preview URL. What there is: the job uploads the built site as a
workflow artefact, so you can look at the exact HTML CI produced.

1. Open the pull request → **Checks** → the **docs** workflow → the **build**
   job's run page.
2. Download the **github-pages** artefact from the run summary.
3. It is a zip containing a tar. Extract both, then serve the result:

```sh
unzip artifact.zip && tar xf artifact.tar -C preview/
python -m http.server -d preview 8000
```

Clunky, and it is the honest answer today. If per-PR preview URLs are wanted,
that is a real change — see [What is not built](#what-is-not-built).

## The official site

Deployed by the `deploy` job in `docs.yml`, which is gated on
`github.ref == 'refs/heads/main'`, so a pull request builds and stops. Once
Pages is enabled the site is at:

<https://sgjholt.github.io/SpecMod/>

Enabling it is a one-time repository setting — **Settings → Pages → Source →
GitHub Actions** — and until it is done the `deploy` step fails with a
permissions error *even though the build succeeded*, which is a confusing
pairing to meet cold. `ci/README.md` lists it alongside the other one-time
settings.

The site is rebuilt and redeployed on every merge to `main`. It always
describes `main`, which during alpha is ahead of the newest release.

## What is not built

Two things people reasonably expect from a documentation site, neither of which
exists here yet. Both are listed so the absence is a decision rather than a
surprise.

### Historic versions

**Today there is one site and it tracks `main`.** To read the documentation for
a released version you have to build it from its tag:

```sh
git worktree add /tmp/docs-v0.2.0 v0.2.0
uv pip install -e '/tmp/docs-v0.2.0[docs]'
sphinx-build -b html /tmp/docs-v0.2.0/docs /tmp/docs-v0.2.0/_build
```

This matters more than usual here, because the package is alpha and users are
told to pin an exact version — so the documentation they need is not the
documentation they get. Two credible ways to fix it:

**Deploy into versioned subdirectories.** On a release, build the tag and
deploy to `/<version>/` while `main` keeps deploying to the root; add a version
dropdown. The theme supports one directly: `html_theme_options["switcher"]`
takes a `json_url` pointing at a `switcher.json` listing the versions and a
`version_match` naming the current one (verified against pydata-sphinx-theme
0.20.0's own validation code). No new service and no new dependency; the cost
is workflow logic and a `switcher.json` that has to be kept correct.

**[Read the Docs](https://docs.readthedocs.io).** Versioning per tag and
branch, a version switcher, server-side search and per-pull-request preview
builds, all native. The cost is a second service, a `.readthedocs.yaml`, and
the canonical URL moving off `github.io`.

There is also [sphinx-multiversion](https://holzhaus.github.io/sphinx-multiversion/),
which builds every tag into subdirectories from one checkout. It works and it
keeps everything on GitHub Pages, but weigh the maintenance: 0.2.4, released
October 2024, four years after the previous release, one maintainer.

**Recommendation:** do it at the first release, not before — there is nothing
to keep historic until a `v*` tag exists. If per-PR previews are wanted too,
Read the Docs gets both for one setup; if the priority is keeping everything in
this repository and on GitHub Pages, versioned subdirectories plus the theme's
switcher is a contained change to `docs.yml`.

### Per-pull-request preview URLs

GitHub Pages serves one site per repository, so previews need somewhere else to
put them: Read the Docs (above), or a host with native PR previews such as
Netlify or Cloudflare Pages. Both mean granting a third party access to the
repository. Until then, the artefact download above is the preview.

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
