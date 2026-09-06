"""Shared machinery for the notebook builders in this directory.

Every notebook in this repository is written by a script here, named after the
notebook it writes. One copy of ``md``, ``code`` and the notebook envelope,
rather than one per builder: the first builder carried all three inline, which
was fine while there was one, and a second would have started by copying them.
The two copies would then have drifted on cell metadata, ``nbformat_minor``, or
the trailing newline — none of it visible in a rendered notebook, all of it
noise in the diff of a regenerated one.

The two notebooks do not live in the same place, and should not:

``docs/notebooks/choosing-a-transform.ipynb``
    Not published. ``choosing-a-transform.md`` is the page that carries this
    material, and ``conf.py`` excludes ``notebooks/**``.

``tutorial/SpecModTutorial.ipynb``
    Published, and executed on every docs build. It stays outside ``docs/``
    because it reads ``tutorial/data/events/`` through paths relative to
    itself, and because eight other files name that directory.

So a builder says where its notebook goes, and only the name is derived. This
directory is the one place to look for all of them.

Builders are **not** part of the published site — ``conf.py`` excludes
``_builders/**`` — and are deliberately outside the ruff and mypy scopes,
which ``.pre-commit-config.yaml`` pins to ``src/``, ``tests/`` and ``tools/``.
They carry long prose lines and mathematical unicode that the source rules
reject, so keep new builders here rather than under ``tools/``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

#: The repository root: this file is ``<root>/docs/_builders/_notebook.py``.
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

#: Where a builder writes unless it says otherwise.
NOTEBOOKS = ROOT / "docs" / "notebooks"

#: The envelope a notebook gets when its builder does not supply one.
DEFAULT_METADATA: dict[str, Any] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}


class NotebookBuilder:
    """Cells in, one ``.ipynb`` out.

    Prefer :func:`builder_for`, which takes the output name from the calling
    script's own filename so the two cannot drift apart.
    """

    def __init__(
        self,
        name: str,
        *,
        into: pathlib.Path = NOTEBOOKS,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.into = into
        self.metadata = DEFAULT_METADATA if metadata is None else metadata
        self.cells: list[dict[str, Any]] = []

    def md(self, text: str) -> None:
        """A markdown cell."""
        self.cells.append(
            {
                "cell_type": "markdown",
                "id": None,
                "metadata": {},
                "source": text.strip().splitlines(True),
            }
        )

    def code(self, text: str) -> None:
        """A code cell, with no outputs — the docs build executes it."""
        self.cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "id": None,
                "metadata": {},
                "outputs": [],
                "source": text.strip().splitlines(True),
            }
        )

    def write(self) -> pathlib.Path:
        """Write the notebook and report where it went.

        The output is what is committed, byte for byte —
        ``tests/test_docs_are_current.py`` asserts it. That is why the cell
        sources in a builder are written already formatted rather than being
        passed through ``ruff format`` here: ruff is not importable from this
        environment, and the pre-commit hook, CI and a local checkout pin
        three different versions, so a formatting step at build time would
        make "rebuilding changes nothing" depend on which ruff you had.
        """
        for position, cell in enumerate(self.cells):
            # Sequential rather than the random hex `nbformat` assigns, so a
            # regenerated notebook diffs against the committed one only where
            # its content changed. Random ids would rewrite every cell.
            #
            # Filled in here, but the key is created in `md`/`code` so it lands
            # in the position `nbformat` writes it. Key order is not meaningful
            # to a notebook and is entirely meaningful to `git diff`.
            cell["id"] = str(position)
        notebook = {
            "cells": self.cells,
            "metadata": self.metadata,
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        # Relative to this file, not to whoever happened to run it: the path
        # was once hardcoded to one machine's checkout, so the script only
        # worked there.
        out = self.into / f"{self.name}.ipynb"
        out.parent.mkdir(parents=True, exist_ok=True)
        # `ensure_ascii=False`: the prose is full of em dashes and mathematical
        # symbols, and `\\u2014` in a source file is unreadable and unsearchable.
        out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        code_cells = sum(c["cell_type"] == "code" for c in self.cells)
        print(f"wrote {out} — {len(self.cells)} cells ({code_cells} code)")
        return out


def builder_for(
    script: str,
    *,
    into: pathlib.Path = NOTEBOOKS,
    metadata: dict[str, Any] | None = None,
) -> NotebookBuilder:
    """The builder writing the notebook this script is named after.

    Pass ``__file__``. ``choosing_a_transform.py`` writes
    ``choosing-a-transform.ipynb``: underscores become hyphens, because Python
    module names cannot carry a hyphen and notebook filenames in this project
    do. A stem with no underscore is left alone, which is how
    ``SpecModTutorial.py`` writes ``SpecModTutorial.ipynb`` without the
    convention needing an exception for it.

    Deriving the name rather than repeating it is what makes the convention
    hold. A builder and its notebook cannot be given different names without
    renaming the file, so there is no way to end up with
    ``_build_notebook.py`` again — a name that said nothing about which of
    several notebooks it built.

    ``into`` is the directory the notebook goes in, for the one that cannot
    live beside the others.
    """
    return NotebookBuilder(
        pathlib.Path(script).stem.replace("_", "-"), into=into, metadata=metadata
    )
