"""No module may be exempted from type checking.

``[tool.mypy] strict = true`` covers the package, and an override used to set
``ignore_errors = true`` for ``fitting``, ``preprocess`` and ``utils`` — 144
suppressed errors between them. The list is empty now, so these tests guard a
property rather than tracking a countdown.

They still matter, and more than they did. mypy reports ``Success: no issues
found in 39 source files`` whether or not a module is exempt: it parses the
exempt ones and discards what it finds. So a green mypy run is not evidence
that the list is empty, and re-adding a module to it is invisible in CI. That
is what these tests see and nothing else does.

The comment above the override once said "CI asserts it never grows". Nothing
did — that sentence was the entire mechanism. Same defect class as
``config.model.source`` before the ``sources`` package existed: a value in a
configuration file, a claim about it somewhere else, and nothing tying the two
together.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Empty, and meant to stay that way. Adding a module to the mypy override is
#: a deliberate act that has to change this line too, which is the whole point.
EXPECTED_EXEMPT: frozenset[str] = frozenset()


def _exempt_modules() -> frozenset[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    exempt: set[str] = set()
    for override in config["tool"]["mypy"].get("overrides", []):
        if override.get("ignore_errors"):
            exempt.update(override["module"])
    return frozenset(exempt)


def test_nothing_is_exempt_from_type_checking() -> None:
    """The assertion the comment in ``pyproject.toml`` claimed to have."""
    exempt = _exempt_modules()
    added = exempt - EXPECTED_EXEMPT
    assert not added, (
        f"{', '.join(sorted(added))} was added to the mypy exemption list, "
        "which is empty and should stay empty. Annotate the module instead — "
        "and note that mypy reports success either way, so nothing but this "
        "test will tell you."
    )


def test_the_backlog_is_recorded_accurately() -> None:
    """A module annotated and removed from the override must leave here too.

    Otherwise this file becomes a record of what the backlog used to be, and
    the test above stops being able to tell growth from drift.
    """
    removed = EXPECTED_EXEMPT - _exempt_modules()
    assert not removed, (
        f"{', '.join(sorted(removed))} is no longer exempt in pyproject.toml "
        "— delete it from EXPECTED_EXEMPT here as well."
    )


def test_every_exempt_module_exists() -> None:
    """An exemption for a module that was deleted or renamed silences nothing
    and hides that the backlog is shorter than it looks."""
    for module in EXPECTED_EXEMPT:
        path = ROOT / "src" / Path(*module.split(".")).with_suffix(".py")
        assert path.exists(), f"{module} is exempt but {path} does not exist"


#: Skips cleanly rather than silently generating zero cases when the list is
#: empty, so the suite still reports that this check exists.
_NOTHING_EXEMPT = [pytest.param("", marks=pytest.mark.skip(reason="nothing is exempt"))]


@pytest.mark.slow
@pytest.mark.parametrize("module", sorted(EXPECTED_EXEMPT) or _NOTHING_EXEMPT)
def test_an_exempt_module_still_needs_its_exemption(
    module: str, tmp_path: Path
) -> None:
    """Or the exemption is stale, and the module can be checked for real.

    Without this the list only ever shrinks by someone thinking to retry it,
    which over three phases of decomposition is exactly what does not happen.
    Marked slow because it runs mypy once per module.

    **The empty config file is the whole trick.** mypy reads ``pyproject.toml``
    from its working directory whatever is on the command line, so naming the
    file and passing ``--strict`` still picks up the very override being tested
    and reports ``Success``. The first version of this test did exactly that
    and passed against all three modules for that reason.

    Skipped while the list is empty; it is kept because it is what turned the
    countdown into a measurement — each module came off the list when this
    test said it could, not when someone guessed.
    """
    config = tmp_path / "mypy.ini"
    config.write_text("[mypy]\n")

    path = ROOT / "src" / Path(*module.split(".")).with_suffix(".py")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            f"--config-file={config}",
            "--strict",
            "--ignore-missing-imports",
            "--no-incremental",
            str(path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert result.returncode != 0, (
        f"{module} now type-checks cleanly under --strict. Remove it from the "
        f"mypy override in pyproject.toml and from EXPECTED_EXEMPT here.\n"
        f"{result.stdout}"
    )
