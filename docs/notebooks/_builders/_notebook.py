"""Shared machinery for the notebook builders in this directory.

One copy of ``md``, ``code`` and the notebook envelope, rather than one per
builder. The first builder carried all three inline, which was fine while
there was one; a second would have started by copying them, and the two copies
would then have drifted on cell metadata, ``nbformat_minor``, or the trailing
newline — none of which is visible in a rendered notebook and all of which
show up as noise in the diff of a regenerated one.

Builders in this directory are **not** part of the published site:
``docs/conf.py`` excludes ``notebooks/*`` wholesale, and the pre-commit ruff
hooks are scoped to ``src/``, ``tests/`` and ``tools/``. That exclusion is
deliberate — these files carry long prose lines and mathematical unicode that
the source rules would reject — so keep new builders here rather than under
``tools/``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

#: Where the built notebooks go: the parent of this directory.
NOTEBOOKS = pathlib.Path(__file__).resolve().parent.parent


class NotebookBuilder:
    """Cells in, one ``.ipynb`` out.

    Prefer :func:`builder_for`, which takes the output name from the calling
    script's own filename so the two cannot drift apart.
    """

    def __init__(self, name: str) -> None:
        self.name = name
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
        ``tests/test_docs.py`` asserts it. That is why the cell sources in a
        builder are written already formatted rather than being passed
        through ``ruff format`` here: ruff is not importable from this
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
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        # Relative to this file, not to whoever happened to run it: the path
        # was once hardcoded to one machine's checkout, so the script only
        # worked there.
        out = NOTEBOOKS / f"{self.name}.ipynb"
        out.parent.mkdir(parents=True, exist_ok=True)
        # `ensure_ascii=False`: the prose is full of em dashes and mathematical
        # symbols, and `\\u2014` in a source file is unreadable and unsearchable.
        out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        code_cells = sum(c["cell_type"] == "code" for c in self.cells)
        print(f"wrote {out} — {len(self.cells)} cells ({code_cells} code)")
        return out


def builder_for(script: str) -> NotebookBuilder:
    """The builder writing the notebook this script is named after.

    Pass ``__file__``. ``choosing_a_transform.py`` writes
    ``choosing-a-transform.ipynb``: underscores become hyphens, because Python
    module names cannot carry a hyphen and notebook filenames in this project
    do.

    Deriving the name rather than repeating it is what makes the convention
    hold. A builder and its notebook cannot be given different names without
    renaming the file, so there is no way to end up with
    ``_build_notebook.py`` again — a name that said nothing about which of
    several notebooks it built.
    """
    return NotebookBuilder(pathlib.Path(script).stem.replace("_", "-"))
