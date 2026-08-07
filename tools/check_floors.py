"""Assert that the installed versions really are the declared minimums.

The ``floors`` CI job exists to exercise the oldest dependency set the project
claims to support. It did not. The job ran::

    uv pip install --resolution lowest-direct -e ".[dev]"
    uv run pytest -m "not dataset" -q

and ``uv run`` re-syncs the project environment before running — there is no
committed lock file, so it resolved fresh to the newest of everything and
uninstalled what the previous step had just put in place. The job was a
duplicate of the ubuntu/3.11 test matrix entry, and green for that reason.

The comment above it read *"Exercise the declared minimums. CI otherwise
installs the newest of everything, so the floors are only ever tested by a
user."* — describing precisely what continued to happen.

What it would have caught, once fixed:

* ``lmfit>=1.2`` and ``numpy>=2.0`` could never both be satisfied. lmfit below
  1.3.0 calls ``np.asfarray``, removed in NumPy 2.0, so every fit raised
  ``AttributeError``. 15 tests fail on the declared floor.
* ``scipy>=1.13`` silently broke the quadratic multitaper. 1.15 reimplemented
  ``scipy.optimize.nnls``; before that the vendored ``qiinv`` inversion does
  not converge for every input scale, and peak recovery moves between 0.53 and
  1.02 for the same signal at different amplitudes.

Neither was theoretical and neither was visible from a green CI.

So this script is the guard: a job that stops installing floors now *fails*
rather than passing quietly, because that failure mode is invisible by
construction — a floors job testing the newest versions looks exactly like a
floors job that works.

Run with no arguments; exits non-zero and prints every mismatch.

**``--no-sync`` in the workflow is the other half, and it is load-bearing.**
Both ``uv run`` lines in the ``floors`` job carry it::

    - run: uv pip install --resolution lowest-direct -e ".[dev]"
    - run: uv run --no-sync python tools/check_floors.py
    - run: uv run --no-sync pytest -m "not dataset" -q

Dropping it from either line puts the job straight back to testing the newest
of everything. The difference is that this script now turns that into a
failure rather than a silent pass — which is the whole reason it exists, since
the two states are otherwise indistinguishable from a green run.
"""

from __future__ import annotations

import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: ``name>=X.Y`` — the only form used, and the only one this understands. A
#: requirement written any other way is reported rather than skipped, so a new
#: spelling cannot quietly drop out of the check.
FLOOR = re.compile(r"^([A-Za-z0-9_.\-]+)\s*>=\s*([0-9][0-9A-Za-z.\-]*)$")


def _requirements() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    required: list[str] = list(project["dependencies"])
    # `dev` pulls in the test tooling and repeats the io extra; its versions
    # are installed by the same command, so they are checked the same way.
    for name, extra in project.get("optional-dependencies", {}).items():
        if name == "dev":
            required += extra
    return required


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _same_version(declared: str, installed: str) -> bool:
    """``2.0`` and ``2.0.0`` are the same version; ``2.0`` and ``2.4.6`` are not.

    Floors are written at whatever precision reads well — ``numpy>=2.0``,
    ``pandas>=2.2.2`` — while the resolver installs a full release number. So
    both are padded to equal length before comparison rather than truncated to
    the shorter, which would accept 2.0.5 for a floor of 2.0.
    """
    a = [int(p) for p in re.findall(r"\d+", declared)]
    b = [int(p) for p in re.findall(r"\d+", installed)]
    width = max(len(a), len(b))
    return a + [0] * (width - len(a)) == b + [0] * (width - len(b))


def main() -> int:
    problems: list[str] = []
    checked = 0

    for requirement in _requirements():
        # Strip any environment marker; nothing here uses one, but a marker
        # would otherwise be parsed as part of the version.
        text = requirement.split(";")[0].strip()
        match = FLOOR.match(text)
        if match is None:
            problems.append(
                f"{text!r} is not of the form 'name>=X.Y', so its floor is not "
                "being checked. Extend tools/check_floors.py rather than "
                "leaving it unchecked."
            )
            continue

        name, declared = match.groups()
        try:
            installed = version(_normalise(name))
        except PackageNotFoundError:
            problems.append(f"{name} is declared but not installed")
            continue

        checked += 1
        if not _same_version(declared, installed):
            problems.append(
                f"{name}: declared floor {declared}, but {installed} is "
                "installed. This job is meant to run the floors — if the "
                "install step resolved something newer, it is not testing "
                "what it claims to."
            )

    if problems:
        print("floors are not what is installed:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"all {checked} declared floors are the installed versions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
