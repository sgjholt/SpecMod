# Working on SpecMod with a coding agent

Rules for Claude Code, Codex, and any other agent committing to this
repository. `CLAUDE.md` points here so there is one copy.

The long version of everything below is
[`docs/development.md`](docs/development.md). This file is the part an agent
must not get wrong, and every entry is here because it has actually gone wrong.

## Before the first commit

```sh
uv venv && uv pip install -e ".[dev]"
pre-commit install          # installs BOTH the pre-commit and commit-msg hooks
```

`pre-commit install` is not optional. In a fresh container it is easy to skip,
and the `commit-msg` hook is the only thing enforcing the rule below.

## Never publish session links

**No `Claude-Session:` trailers, session URLs, or agent-console links** in
commit messages, PR titles, PR bodies, code comments, or anything else that
lands in the repository. It is public; those links are private state.

`Co-Authored-By:` is fine. If your harness appends a session trailer by
default, strip it — the repository's rule wins over the harness default. The
`commit-msg` hook rejects it, which is why installing the hooks comes first.

## Commit messages are load-bearing

[Conventional Commits](https://www.conventionalcommits.org). They are not a
style preference: `release-please` reads them to compute the version bump and
to generate `CHANGELOG.md`. See
[`docs/releasing.md`](docs/releasing.md).

- `feat:` minor, `fix:` patch, `refactor:` / `docs:` / `build:` appear in the
  changelog, `test:` / `ci:` / `chore:` are hidden.
- `!` or a `BREAKING CHANGE:` footer bumps the minor while the project is
  `0.x`, not the major.
- Say *why*, with the measurement if there was one. The history is the record
  of what was checked; a message that only restates the diff wastes it.

## Workflow files need a permission you may not have

`.github/workflows/` is editable directly, but only when the session's GitHub
App token carries the `workflows` permission. Without it the push is rejected
outright:

```
refusing to allow a GitHub App to create or update workflow
`.github/workflows/test.yml` without `workflows` permission
```

That is a loud failure, not a silent one. If you meet it, say so and ask for
the permission — do not reintroduce a parallel copy of the workflows to work
around it. There used to be one, in `ci/`, and keeping two versions of every
workflow in step cost more than the problem it solved.

## Verify before reporting

Run these, and report what they actually printed:

```sh
pytest -m "not dataset and not notebook"     # the suite CI runs
pytest --without-optional-extras             # what a default install sees
ruff check src/ tests/ tools/ && ruff format --check src/ tests/ tools/
mypy
sphinx-build -b html docs docs/_build/html   # if docs/ changed
```

`--without-optional-extras` matters: a development environment with
`specmod[multitaper]` installed passes tests that CI fails.

## Things that look like noise and are not

- **Golden references.** `tests/golden/*.json` is a record of numbers this code
  used to produce. Do not regenerate it to make a test pass. If a change moves
  a number, that is the finding — say which number, by how much, and why, and
  regenerate deliberately with `python tools/make_golden.py`.
- **Measured tables in the docs.** Numbers in `docs/*.md` are generated between
  markers by `python tools/measure_docs.py`. Edit the tool, not the table.
- **Tolerances.** Several carry a comment explaining what was measured to
  choose them. Widening one to get to green, without measuring, is the specific
  failure `docs/REFACTOR_PLAN.md` §6.6 exists to catch.

## Say what you did not check

The plan's §6.6 is an audit of claims in this repository that turned out to
describe mechanisms nobody had built. Do not add to it. If something is
untested, unreproducible, or assumed, write that down next to the claim — a
bound with a number behind it beats a confident sentence.
