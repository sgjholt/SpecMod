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

`docs.yml` also needs Pages turning on once: **Settings → Pages → Source →
GitHub Actions**. Without that the deploy step fails with a permissions error
even though the build succeeded.

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
