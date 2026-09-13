"""Every ``specmod`` name a script in ``tools/`` uses must still exist.

The 0.3.0 renames — ``cut_s`` to ``s_window``, ``set_picks`` to ``with_picks``,
``get_signal`` removed — landed in the package, in the tests and in the docs,
and missed ``tools/``. Nothing noticed. ``make_golden.py`` and
``measure_docs.py`` both import cleanly, because the dead names sit inside
function bodies, and neither function is executed by the suite:
``make_golden.py`` is run by hand to regenerate the golden references, and
``measure_docs.py``'s field measurements are run by hand with ``--write
--field``. So ``python tools/make_golden.py`` — the documented way to
regenerate the one artefact the whole rewrite leans on, named as such in
``AGENTS.md`` — raised ``AttributeError`` on its fourth line of real work, and
would have gone on doing so until someone needed it, which is the moment at
which finding out is most expensive.

Executing the scripts here is not the fix: both want the PNR waveforms, and
between them several minutes of CPU. This is the cheap half — resolve the
names against the installed package without running anything — and it is
enough, because what rots is names. A renamed function is an ``AttributeError``
at the first call, and that is exactly what this sees.

Only module attributes are checked. ``PNR_2019.directory`` is an attribute of
an object, not of a module, and following it would mean evaluating the script.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLS = sorted((ROOT / "tools").glob("*.py"))

# Every tool that touches the package imports obspy transitively.
pytest.importorskip("obspy")


def _specmod_aliases(tree: ast.AST) -> dict[str, str]:
    """Map each local name bound to a ``specmod`` module to that module.

    ``ast.walk`` rather than a scope-aware pass on purpose: ``measure_docs``
    imports ``specmod.preprocess`` inside the one function that needs it, to
    keep ``--help`` fast, and an import that only runs sometimes is exactly the
    one nothing else checks.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name.startswith("specmod") and name.asname:
                    aliases[name.asname] = name.name
    return aliases


def _from_imports(tree: ast.AST) -> list[tuple[str, str]]:
    """``(module, name)`` for every ``from specmod.x import name``."""
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "specmod"
        ):
            assert node.module is not None
            out += [(node.module, alias.name) for alias in node.names]
    return out


def _attribute_uses(
    tree: ast.AST, aliases: dict[str, str]
) -> list[tuple[str, str, int]]:
    """``(module, attribute, line)`` for every ``alias.attr`` in the file."""
    return [
        (aliases[node.value.id], node.attr, node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    ]


@pytest.mark.parametrize("path", TOOLS, ids=lambda p: p.name)
def test_every_specmod_name_the_script_uses_still_exists(path: Path) -> None:
    tree = ast.parse(path.read_text())
    aliases = _specmod_aliases(tree)

    missing: list[str] = []
    for module_name, attribute, line in _attribute_uses(tree, aliases):
        module = importlib.import_module(module_name)
        if not hasattr(module, attribute):
            missing.append(
                f"  {path.name}:{line}: {module_name}.{attribute} does not exist"
            )
    for module_name, name in _from_imports(tree):
        module = importlib.import_module(module_name)
        if not hasattr(module, name):
            missing.append(f"  {path.name}: cannot import {name} from {module_name}")

    assert not missing, "\n".join(
        [
            f"{len(missing)} name(s) in {path.name} no longer exist in specmod. "
            "The script is broken at the line named; see docs/upgrading.md for "
            "what each one became:",
            *missing,
        ]
    )


def test_the_scan_finds_something_to_check() -> None:
    """A scan that silently matches nothing would pass forever.

    Same failure class as the one above: a mechanism that is believed to run
    and does not. If the tools stop importing specmod under an alias — or the
    parsing above stops recognising it — this says so instead of going green.
    """
    found: dict[str, int] = {}
    for path in TOOLS:
        tree = ast.parse(path.read_text())
        found[path.name] = len(_attribute_uses(tree, _specmod_aliases(tree)))

    assert sum(found.values()) > 0, found
    # `make_golden.py` alone calls four `pre.*` functions, and every one of
    # them was renamed in 0.3.0.
    assert found.get("make_golden.py", 0) >= 4, found
