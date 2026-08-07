"""The list of modules exempted from type checking may shrink, never grow.

``[tool.mypy] strict = true`` covers the package, and then an override sets
``ignore_errors = true`` for three legacy modules. mypy still parses them and
still reports ``Success: no issues found in 39 source files`` — which reads
like the whole package is clean and is not what it means.

The comment above that override said "CI asserts it never grows". Nothing
did. CI ran ``mypy`` with the override in place, so adding a fourth module
would have been green, and the sentence claiming otherwise was the only thing
standing between the backlog and quiet expansion.

That is the defect class this suite exists for: a value in a configuration
file, a claim about it somewhere else, and nothing tying the two together.
Same shape as ``config.model.source`` before the ``sources`` package, and as
``snr.bandwidth_method`` naming a strategy the registry would not accept.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Exempt today. Delete entries as the modules are annotated; adding one is a
#: deliberate act that has to change this line, which is the whole point.
EXPECTED_EXEMPT = frozenset(
    {
        "specmod.fitting",
    }
)


def _exempt_modules() -> frozenset[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    exempt: set[str] = set()
    for override in config["tool"]["mypy"].get("overrides", []):
        if override.get("ignore_errors"):
            exempt.update(override["module"])
    return frozenset(exempt)


def test_the_backlog_has_not_grown() -> None:
    """The assertion the comment in ``pyproject.toml`` claimed to have."""
    exempt = _exempt_modules()
    added = exempt - EXPECTED_EXEMPT
    assert not added, (
        f"{', '.join(sorted(added))} was added to the mypy exemption list. "
        "The backlog shrinks; it does not grow. Annotate the module instead."
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


@pytest.mark.slow
@pytest.mark.parametrize("module", sorted(EXPECTED_EXEMPT))
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
