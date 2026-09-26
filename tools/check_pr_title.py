#!/usr/bin/env python
"""Check that a pull request title is a Conventional Commit.

Run as::

    PR_TITLE='feat(acquire): rank co-sited instruments' python tools/check_pr_title.py

The title matters because of how a squash merge composes the commit that lands
on ``main``. GitHub replaces the branch's subjects with the pull request title,
server-side, after every local hook has run. `release-please` then parses that
subject to compute the version bump and the changelog section — and a subject
it cannot parse contributes neither.

That is not hypothetical. #45 was squash-merged as "Make the station selection
mean what the config says: radius, and one instrument per station (#45)", and
the release job recorded:

.. code-block:: text

   commit could not be parsed: 2af57e8 Make the station selection mean what ...
   error message: Error: unexpected token ' ' at 1:5, valid tokens [(, !, :]
   Considering: 0 commits
   No commits for path: ., skipping

A radius fix, a WGS84 cut and two new configuration fields went out with no
entry anywhere, and the changelog had to be repaired by hand afterwards. A
local ``commit-msg`` hook cannot prevent this: the squash subject does not
exist until GitHub builds it.

The title is untrusted input — anyone who can open a pull request chooses it —
so it arrives through the environment rather than as a shell argument, and
nothing here interpolates it into a command.

Standard library only: this runs before any environment is set up.
"""

from __future__ import annotations

import os
import re
import sys

#: The types `release-please-config.json` declares, and nothing else. Keeping
#: the two lists identical is the point: a type this accepts but the release
#: configuration does not know lands in no changelog section at all.
TYPES = (
    "feat",
    "fix",
    "perf",
    "refactor",
    "docs",
    "build",
    "revert",
    "test",
    "ci",
    "style",
    "chore",
)

#: ``type(scope)!: description``. The scope is optional and, when given, may
#: not be empty. ``!`` marks a breaking change, which bumps the minor while
#: this is ``0.x`` — see `bump-minor-pre-major` in the release configuration.
PATTERN = re.compile(
    r"^(?P<type>" + "|".join(TYPES) + r")"
    r"(?:\((?P<scope>[^()\s][^()]*)\))?"
    r"(?P<breaking>!)?"
    r": (?P<description>\S.*)$"
)

#: What GitHub appends to a squash subject. Present in the commit but not in
#: the title, so it is stripped before matching rather than rejected.
TRAILING_NUMBER = re.compile(r"\s*\(#\d+\)$")


def problem_with(title: str) -> str | None:
    """Why ``title`` is not a Conventional Commit, or ``None`` if it is."""
    subject = TRAILING_NUMBER.sub("", title.strip())
    if not subject:
        return "the title is empty"
    if PATTERN.match(subject):
        return None
    return _diagnosis(subject)


def _diagnosis(subject: str) -> str:
    """Which part of a subject that did not match to point the author at.

    One message, naming one thing to change. "Does not match
    ``^(feat|fix|...)(\\(...\\))?!?: .+$``" is accurate and useless.
    """
    if ":" not in subject:
        return (
            f"there is no `type: ` prefix. The title begins {subject[:24]!r}, "
            f"which release-please reads as a description with no type"
        )
    prefix = re.sub(r"[(!].*$", "", subject.split(":", 1)[0]).strip()
    if prefix not in TYPES:
        if prefix.lower() in TYPES:
            return f"the type {prefix!r} must be lower case"
        return f"{prefix!r} is not one of the known types"
    if re.match(r"^[^:]*\(\s*\)", subject):
        return "the scope is empty; write `type: ` or `type(scope): `"
    return "there is no space after the colon, or no description after it"


def main() -> int:
    title = os.environ.get("PR_TITLE")
    if title is None:
        print("PR_TITLE is not set", file=sys.stderr)
        return 2

    problem = problem_with(title)
    if problem is None:
        print(f"ok: {title}")
        return 0

    # `::error::` renders the message on the pull request itself, which is
    # where the person who has to fix it is looking.
    print(
        f"::error::The pull request title is not a Conventional Commit: "
        f"{problem}. A squash merge makes this title the commit subject on "
        f"main, and release-please computes the version and the changelog "
        f"from it, so an unparseable title ships the work with no changelog "
        f"entry. Known types: {', '.join(TYPES)}. Example: "
        f"'fix(acquire): ask for waveforms by name'."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
