# Contributing to SpecMod

The working guide is [`docs/development.md`](docs/development.md) — setup, the
daily loop, the tooling, the CI jobs, and how development relates to releases.
If you are an AI coding agent, read [`AGENTS.md`](AGENTS.md) first.

This file covers the one thing that is a promise rather than a practice.

## The stability promise, and its exact scope

SpecMod is `0.x` and says so loudly: names and signatures move between minor
releases without a deprecation cycle, because the API is still being worked
out and shipping shims for names that are about to move again costs more than
it protects.

**`specmod.api` is the exception.** It exists so that downstream packages have
something that does not move while the internals do, and it is the only part of
SpecMod that carries a compatibility guarantee.

| | Everything else | `specmod.api` |
|---|---|---|
| Rename or remove | any minor release, no notice | one minor release of `DeprecationWarning` first |
| Change a signature | any minor release, no notice | one minor release of `DeprecationWarning` first |
| Add | freely | freely, but see below |
| Behaviour change that moves a number | called out in the changelog | called out in the changelog **and** in the warning |

"One minor release" means: if a removal is decided during `0.4.x`, the warning
ships in `0.5.0` and the removal is `0.6.0` at the earliest. A deprecation that
has not been through a release has not been announced.

### Deprecating something in `specmod.api`

```python
warnings.warn(
    "specmod.api.old_name is deprecated and will be removed in 0.6.0; "
    "use specmod.api.new_name, which takes the same arguments.",
    DeprecationWarning,
    stacklevel=2,
)
```

Three things that make the difference between a warning people act on and one
they filter out:

- **Name the replacement**, or say plainly that there is none.
- **Name the release it goes in**, not "a future version".
- **`stacklevel=2`**, so the warning points at the caller's line rather than at
  SpecMod's.

Keep the old name working for the whole cycle. A `DeprecationWarning` on
something that already raises is not a deprecation, it is a breakage with a
note attached.

### Adding to `specmod.api`

Every export is a compatibility obligation, so the surface is deliberately
small and does not grow opportunistically. To add one:

1. Have a caller that needs it. "Studio might want this" is not one.
2. Make it satisfy the five properties in the module docstring — path-free,
   deterministic, non-mutating, quiet, typed errors. If the underlying
   function does not, the wrapper is where that gets fixed, not the caller.
3. Add it to `EXPECTED_EXPORTS` in `tests/test_api_surface.py`. The list is
   duplicated there on purpose, so an addition shows up as a diff in review
   rather than as a passing test.
4. Give it a docstring with `Parameters`, `Returns` and `Raises`, and a type
   annotation on everything. Both are tested.

**Do not reach around the surface.** If a downstream package needs something
`specmod.api` does not export, extend `specmod.api` in its own pull request
with the reason stated. An import of `specmod.core` or `specmod.fitting` from
a downstream package is a bug in the boundary, not a shortcut.

### What is *not* promised

- **Internals.** `specmod.core`, `specmod.fitting`, `specmod.transforms`,
  `specmod.picks`, `specmod.config` and everything else may change in any
  release. They are documented because SpecMod's own users read them; that is
  not a stability claim.
- **Numerical output.** The promise is about names and signatures. A bug fix
  that moves a number is still a bug fix, and it will move it — that is what
  the golden references and the changelog are for. If you depend on exact
  values, pin an exact version and keep the config hash.
- **The objects the surface returns**, beyond the attributes its docstrings
  name. `SpectrumPair` gaining a field is not a breaking change.

## Everything else

Conventional Commits, `pre-commit install` before your first commit, and the
rest of the mechanics are in [`docs/development.md`](docs/development.md).
