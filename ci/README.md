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

A file here is the **intended** state, which is not necessarily the current
one. `tools/check_ci_mirror.py` reports which of the two each pair is in, and
prints a diff for any that have diverged:

```
python tools/check_ci_mirror.py
```

It exits non-zero when they differ, so it can be wired into CI once the pair is
in sync. It is deliberately not a test yet: the mirror is ahead of the live
workflow by design until the copy is made, and a test that fails for the
expected reason teaches people to ignore it.
