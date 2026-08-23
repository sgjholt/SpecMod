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

1. **Mint `RELEASE_PLEASE_TOKEN`.** Settings → Developer settings →
   Personal access tokens → Fine-grained tokens → *Generate new token*.

   | Field | Value |
   |---|---|
   | Repository access | Only select repositories → `SpecMod` |
   | Contents | Read and write |
   | Pull requests | Read and write |
   | Expiration | up to 366 days — see the renewal note below |

   Then Settings → Secrets and variables → Actions → *New repository secret*,
   named exactly `RELEASE_PLEASE_TOKEN`.

   This is what makes the release PR run CI. Without it the PR is opened by
   `GITHUB_TOKEN`, which starts no workflow runs at all, so the three required
   checks never report and the PR is blocked with nothing to fix. The token
   never touches PyPI — the upload authenticates by OIDC — so this does not
   reintroduce a publishing secret.

   **It expires.** When it does, the symptom is the old one returning: a
   release PR with no checks on it. Renew the token and update the secret;
   nothing in the repository changes. A GitHub App token via
   `actions/create-github-app-token` avoids the expiry entirely and is the
   better answer if this becomes annoying.

2. **Let Actions open pull requests.** Settings → Actions → General →
   Workflow permissions → tick *Allow GitHub Actions to create and approve pull
   requests*. Only needed as a fallback — with `RELEASE_PLEASE_TOKEN` set the
   PR is opened by a user, not by Actions — but leave it on so the workflow
   still works if the token lapses.

3. **Create the `pypi` environment.** Settings → Environments → New
   environment → `pypi`.

   Required reviewers here are optional, and worth thinking about before
   adding: they hold the `publish` job at `waiting` until a human approves the
   deployment, and **that approval cannot be given from the GitHub mobile
   app** — the button exists only on the web UI and the REST API. Merging the
   release PR is already a deliberate human act, and
   `tools/check_built_version.py` is what actually prevents a wrong version
   reaching PyPI, so the reviewer gate buys less than it looks like it does.

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

6. **Require the checks on `main`.** Branch protection → *Require status
   checks to pass* → add exactly three: **`ci`**, **`docs`**, **`build`**.

   Those are job names, not workflow names, and only these three are stable:
   `ci` is an aggregator that fails unless every job in `test.yml` succeeded,
   so the six `test (os, version)` matrix checks do not need naming and the
   list survives a matrix change. Requiring `test` or the workflow names
   instead matches nothing, and a required check that never reports blocks
   every merge.

   The release pull request goes through them like any other — but only
   because of step 1. This page previously said GitHub *holds* the runs on a
   `github-actions[bot]` pull request until a maintainer approves them, and
   that is not what happens: no runs are created, there is nothing to approve,
   and the PR sits blocked with an empty check list. `RELEASE_PLEASE_TOKEN` is
   what makes this paragraph true.

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
obvious design — a workflow keyed on `release: published` — could not fire at
all when release-please ran on the default `GITHUB_TOKEN`, because "events
triggered by the `GITHUB_TOKEN` will not create a new workflow run". So the
publish job gates on release-please's own `release_created` output instead.

`RELEASE_PLEASE_TOKEN` removes that constraint — a release created by a PAT
*does* start workflow runs — so a separate `publish.yml` would now work. It is
still not worth splitting: one workflow means one `concurrency` group, one
place to read, and no dependency on the ordering of two runs that would race
whenever two merges land together. The `release_created` gate is also exact,
where an event key would fire on any release including one made by hand.

Zenodo was never affected either way: it listens to the release *webhook*,
which is not subject to the token restriction.

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
- **Docs**: Read the Docs builds the new tag as its own version and moves
  `stable` onto it. Two things to check the first few times: that the tag was
  activated (an automation rule does it, see
  [Documentation workflow](documentation.md#read-the-docs-setup)), and that the
  new version's sidebar shows the release number rather than `0.0.0` — the
  latter means the shallow-clone fix in `.readthedocs.yaml` did not take.
  `latest` moved earlier, when the release PR merged.

If the `publish` job fails after the release exists — a transient PyPI error,
say — re-run that job from the Actions UI. Do not re-run `release-please`
expecting a second attempt: `release_created` is only true on the run that
created the release.

## When the release PR has no checks on it

Not a red check — *no checks at all*, an empty list, and the PR blocked
because the three required ones never reported. This means release-please fell
back to `GITHUB_TOKEN`: either `RELEASE_PLEASE_TOKEN` is unset, or it expired,
or its repository access no longer covers `SpecMod`. Renew it (one-time setup,
step 1); nothing in the repository needs changing.

To unblock the release PR that is already open, without waiting for a new
commit: **close it and immediately reopen it.** That re-fires
`pull_request.reopened` from your account rather than from the token, and the
checks run. It is the same trick that got `v0.2.0` out, and it is safe — the
`autorelease: pending` label survives, so release-please still recognises the
merge as a release, and no history is touched.
