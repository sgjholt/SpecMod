"""The pull request title check, which stands in for a hook that cannot exist.

`release-please` computes the version bump and the changelog from commit
subjects on ``main``. A squash merge composes that subject from the pull
request title, server-side, after every local hook has run — so the
``commit-msg`` hook cannot see it, and the one time it mattered the work
shipped with no changelog entry at all.

These are the cases that actually occurred or are one keystroke away, not an
exhaustive grammar: the title as #45 carried it, a capitalised type, a missing
space, an empty scope.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from check_pr_title import TYPES, problem_with  # noqa: E402

VALID = [
    "feat(acquire): rank co-sited instruments",
    "fix: ask for waveforms by name",
    "docs(roadmap): file v0.4.0 as shipped",
    "feat!: make the package installable and importable",
    "fix(multitaper)!: default adaptive weighting off",
    "chore(main): release 0.5.0",
    # What a squash merge appends; the title itself never carries it, but a
    # person copying a merged subject back into a title will.
    "feat(acquire): rank co-sited instruments (#45)",
]

INVALID = [
    # The one that happened.
    "Make the station selection mean what the config says: radius, and one "
    "instrument per station (#45)",
    "Roadmap: file v0.4.0 as shipped, and the acquire work as merged",
    "Feat(acquire): rank co-sited instruments",
    "feat(acquire):no space after the colon",
    "feat(): an empty scope",
    "feat(acquire): ",
    "update the acquire module",
    "",
    "   ",
]


@pytest.mark.parametrize("title", VALID)
def test_a_conventional_title_passes(title: str) -> None:
    assert problem_with(title) is None, title


@pytest.mark.parametrize("title", INVALID)
def test_an_unparseable_title_is_refused(title: str) -> None:
    assert problem_with(title) is not None, title


@pytest.mark.parametrize("type_", TYPES)
def test_every_known_type_is_accepted(type_: str) -> None:
    assert problem_with(f"{type_}: a description") is None


def test_the_types_are_the_ones_release_please_knows() -> None:
    """Two lists of types that must not drift apart.

    A type this check accepts but `release-please-config.json` does not
    declare passes review and then lands in no changelog section — the same
    silent outcome the check exists to prevent, arrived at from the other side.
    """
    config = json.loads((ROOT / "release-please-config.json").read_text())
    declared = {
        section["type"] for section in config["packages"]["."]["changelog-sections"]
    }
    assert set(TYPES) == declared


def test_the_failure_says_what_to_do() -> None:
    """The message is the whole interface: it is read on a red check."""
    problem = problem_with("Make the station selection mean what the config says")
    assert problem is not None
    assert "type" in problem
