# CI workflows — staged, not yet active

These are finished GitHub Actions workflows. They live here rather than in
`.github/workflows/` because the automation account that authored them is a
GitHub App without the `workflows` permission, so pushes touching
`.github/workflows/` are rejected outright:

```
refusing to allow a GitHub App to create or update workflow
`.github/workflows/build.yml` without `workflows` permission
```

**To activate, from a normal user account:**

```sh
mkdir -p .github/workflows
git mv ci/workflows/*.yml .github/workflows/
git rm ci/README.md
git commit -m "ci: activate test and build workflows"
git push
```

Nothing else needs changing — the files are complete as they stand.

| File | Does |
|---|---|
| `test.yml` | ruff, ruff format, mypy, pytest on 3.11–3.13 × ubuntu/macos, coverage |
| `build.yml` | sdist + wheel, `twine check`, clean-env install-from-wheel smoke test |

Still to add in later phases (see `docs/REFACTOR_PLAN.md` §6.5): `docs.yml`,
`release-please.yml`, `publish.yml`.

Note: GitHub Actions is disabled by default on repositories that were forks.
`sgjholt/SpecMod` has since been detached, but confirm Actions is enabled under
Settings → Actions before expecting any of this to run.
