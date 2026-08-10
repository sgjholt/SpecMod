# `ci/` — workflow files staged for copying

Complete, ready-to-paste copies of the files under `.github/workflows/`.

## Why this exists

The GitHub App token used by Claude Code sessions has no `workflows`
permission, so a push touching `.github/workflows/` is rejected:

```
refusing to allow a GitHub App to create or update workflow
`.github/workflows/test.yml` without `workflows` permission
```

Everything else in a change can be pushed; the workflow cannot. Rather than
describe the edit in a PR comment and leave it to be applied by hand — awkward
on a phone, and easy to apply partially — the intended file is committed here
in full.

## Using it

Copy the whole file over its counterpart. No merging, no partial application:

| staged copy | destination |
|---|---|
| `ci/workflows/test.yml` | `.github/workflows/test.yml` |
| `ci/workflows/docs.yml` | `.github/workflows/docs.yml` |
| `ci/workflows/release.yml` | `.github/workflows/release.yml` |

Two of them need a repository setting turning on as well, once each, and
neither can be done from a commit:

- `docs.yml` — **Settings → Pages → Source → GitHub Actions**. Without it the
  deploy step fails with a permissions error even though the build succeeded.
- `release.yml` — **Settings → Actions → General → Allow GitHub Actions to
  create and approve pull requests**. Without it release-please fails with
  `GitHub Actions is not permitted to create or approve pull requests`.

`release.yml` needs four more one-time steps before it can publish anything —
a `pypi` environment, a trusted publisher registered on PyPI, the Zenodo
webhook, and branch protection. They are listed in
[`docs/releasing.md`](../docs/releasing.md). Until they are done the workflow
opens a release pull request and stops there, which is inert rather than
wrong.

**The PyPI trusted publisher names this file.** It is registered against the
workflow filename `release.yml`, matched from the OIDC token, so renaming the
workflow breaks authentication at upload time. Rename both in the same
sitting or not at all.

A file here is the **intended** state, which is not necessarily the current
one. `tools/check_ci_mirror.py` reports which of the two each pair is in, and
prints a diff for any that have diverged:

```
python tools/check_ci_mirror.py
```

It exits non-zero when they differ, and the `lint` job runs it, so a staged
change that has not been copied across fails CI rather than waiting to be
noticed. Trailing blank lines are ignored — GitHub's web editor leaves them
behind, and a check that fails on invisible whitespace is one people stop
reading.

The script uses only the standard library, so the step needs no install.

There is one ordering quirk worth knowing: a change to a workflow lands in
`ci/` first, so the check fails until the copy is made. That failure is the
reminder, and it clears the moment the files match.
