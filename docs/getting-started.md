# Getting started

Install it, run one event end to end, and find out what moved if you have code
written against `0.1`.

```sh
pip install specmod
```

Python 3.11 or newer. Saving and loading spectra needs the `io` extra —
`pip install "specmod[io]"` — without which SpecMod computes and plots normally
but cannot write a result. The other extras are listed in the
[README](https://github.com/sgjholt/SpecMod#installation).

While this is `0.x`, pin an exact version in anything you intend to publish.
The [roadmap](roadmap.md) says what has shipped and what 1.0 will mean.

[Tutorial](tutorial/specmod-tutorial.ipynb)
: One event from waveforms and picks to source parameters. Every figure and
  number on the page is produced by executing the notebook when this site is
  built, so it cannot describe an API that no longer exists. **Start here.**

[Upgrading](upgrading.md)
: For code written against the pre-refactor `0.1.1` on `master`. Modules moved,
  were renamed to snake_case, and several were deleted, so old code fails at
  its imports rather than running and quietly giving different numbers.

## Then

[How a spectrum is processed](processing.md) is the reference for what each
stage does and the conventions it uses — the amplitude convention, the Parseval
contract, and the moment and magnitude equations. It is the page to read once
the tutorial has run.

```{toctree}
:maxdepth: 2
:hidden:

tutorial/specmod-tutorial
upgrading
```
