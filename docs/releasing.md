# Releasing the software

How a merged commit becomes a tag, a PyPI release and a DOI. The companion to
[Publishing a dataset](releasing-data.md), which covers the `data-v*` tags
instead — the two are deliberately separate, and `pyproject.toml` enforces the
separation.

Nothing here is triggered by hand. What a human does is **merge the release
pull request**, and everything after that is automatic. That gate exists
because of Zenodo: every GitHub Release mints a DOI, and a DOI cannot be
retracted. A typo fix should not be able to mint a citable version of the
software without someone deciding it should.

## What the pieces are

| Piece | Decides |
|---|---|
| Conventional Commit messages | which section of the changelog a commit lands in, and whether it bumps |
| `release-please-config.json` | the bump rules, the changelog sections, the tag format |
| `.release-please-manifest.json` | the version we are on now |
| `.github/workflows/release.yml` | opens and maintains the release PR; publishes when it merges |
| `pyproject.toml` (`tag_regex`) | which tags `hatch-vcs` will read as a version |

No version string is committed anywhere. `hatch-vcs` derives it from
`git describe`, so the tag *is* the version and there is nothing to forget to
bump.

## One-time setup

Six things, none of which can be done from a commit. Until they are done the
release workflow is inert rather than wrong — it opens a release PR and stops.

1. **Copy the workflow into place.** `ci/workflows/release.yml` →
   `.github/workflows/release.yml`. See [`ci/README.md`](https://github.com/sgjholt/SpecMod/blob/main/ci/README.md)
   for why the file is staged rather than pushed.

2. **Let Actions open pull requests.** Settings → Actions → General →
   Workflow permissions → tick *Allow GitHub Actions to create and approve pull
   requests*. Without it release-please fails with `GitHub Actions is not
   permitted to create or approve pull requests`.

3. **Create the `pypi` environment.** Settings → Environments → New
   environment → `pypi`. Add required reviewers here if you want a second
   human gate on the upload itself.

4. **Register the trusted publisher on PyPI.** On the project page (or, for
   the first ever upload, under *Publishing* → *Add a new pending publisher*):

   | Field | Value |
   |---|---|
   | Owner | `sgjholt` |
   | Repository | `SpecMod` |
   | Workflow | `release.yml` |
   | Environment | `pypi` |

   The workflow name is matched against the OIDC token, so it must be the file
   that actually runs. If the workflow is ever renamed, this has to be changed
   in the same sitting or the upload fails authentication.

5. **Turn on the Zenodo webhook.** Log into Zenodo with GitHub, find `SpecMod`
   in the repository list, and flip the switch. It takes effect for releases
   made *after* it is on, so do it before the first release rather than after.
   `CITATION.cff` supplies the metadata.

6. **Require the checks on `main`.** Branch protection → require the status
   checks from `test.yml`, `docs.yml` and `build.yml`. The release pull request
   is an ordinary pull request and goes through them like any other.

## What a version number means while this is 0.x

SemVer, with the pre-1.0 rule taken literally: **a breaking change bumps the
minor**, and there is no deprecation cycle. The package is alpha, the API is
still being worked out, and shipping shims for names that are about to move
again would cost more than it protects.

So `0.2.0 → 0.3.0` may rename things, and `0.2.0 → 0.2.1` will not. Anyone
depending on this for published work should pin an exact version; the
configuration stamp on every output covers the rest.

`1.0.0` is the release that says the API has stopped moving, which is why
`bump-minor-pre-major` exists — it must be a decision, not something a `feat!:`
does on its way past.

## The cycle

1. Merge work into `main` with Conventional Commit messages. Nothing is
   released.
2. `release.yml` opens (or updates) a pull request titled
   `chore(main): release <version>`, carrying the generated `CHANGELOG.md` and
   the new version in `.release-please-manifest.json`. It accrues every commit
   since the last release.
3. Read the changelog. This is the review, and the only one — after the merge
   nothing else is asked.
4. Merge it. release-please creates the tag and the GitHub Release; the
   `publish` job builds the sdist and wheel from **the tag**, checks their
   version against it, and uploads to PyPI; Zenodo mints the DOI from the
   release webhook.

To skip a release, do not merge the PR. It stays open and keeps accruing.

## Things that are the way they are for a reason

**The publish job is in `release.yml`, not a separate `publish.yml`.** The
obvious design — a workflow keyed on `release: published` — never fires.
release-please creates the release with the default `GITHUB_TOKEN`, and
"events triggered by the `GITHUB_TOKEN` will not create a new workflow run".
The workaround is a personal access token in secrets, which is the one thing
Trusted Publishing exists to avoid, so the publish job gates on
release-please's own `release_created` output instead. Zenodo is unaffected:
it listens to the release *webhook*, which is not subject to that restriction.

**`include-component-in-tag` is `false`.** Left on, the tag is
`specmod-v0.2.0`, which `pyproject.toml`'s `--match v[0-9]*` does not describe
and its `tag_regex` does not parse — so the build would fall back to
`0.1.1.postN.devN` and upload under that name, permanently.
`tests/test_release_config.py` asserts the two formats agree, and
`tools/check_built_version.py` checks the actual artefact between the build
and the upload.

**`bump-minor-pre-major` is `true`.** The history already contains two
breaking commits (`feat!: make the package installable and importable`,
`fix!: default multitaper adaptive weighting off`). Without this setting
either one reads as a `1.0.0` — a number that says the API has stopped
moving, minted with a DOI that cannot be withdrawn. Below 1.0 a breaking
change bumps the minor instead, which is what §3.1 of the refactor plan
assumes throughout.

**The changelog sections are listed explicitly.** The default preset hides
`refactor`, `docs`, `build`, `test`, `ci`, `style` and `chore`. Measured over
this repository's 146 conventional commits, that default would print 73 of
them and drop the other 73 — a release that is mostly a refactor would ship an
almost empty changelog. `refactor`, `docs` and `build` are shown; `test`,
`ci`, `style` and `chore` stay hidden.

**The first release will be enormous, and that is correct.** There are no tags
in this repository, so the manifest declares `0.1.1` — the version the Magna
paper cites, which is the pre-refactor code. Everything since then genuinely
is the delta, so the first changelog covers the whole refactor. With the two
breaking commits and `bump-minor-pre-major`, the version it proposes is
`0.2.0`, which is what §7 of the plan expects at the end of Phase 2.

## Checking a release went out

- **PyPI**: the version appears at <https://pypi.org/p/specmod>, and
  `pip install specmod==<version>` in a clean environment works.
- **The version is real**: `python -c "import specmod; print(specmod.__version__)"`
  from that install prints the tag without its `v`, not a `.postN.devN`.
- **Zenodo**: a new DOI under the concept DOI, with `CITATION.cff`'s metadata.
- **Docs**: `docs.yml` deploys from `main`, so the site updates on the release
  PR merge rather than on the tag.

If the `publish` job fails after the release exists — a transient PyPI error,
say — re-run that job from the Actions UI. Do not re-run `release-please`
expecting a second attempt: `release_created` is only true on the run that
created the release.
