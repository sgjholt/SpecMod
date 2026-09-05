"""Sphinx configuration for the SpecMod documentation.

Build with::

    uv pip install -e '.[docs,io,tutorial]'
    sphinx-build -b html docs docs/_build/html

``-W`` is deliberately **not** used. Intersphinx resolves seven inventories
over the network, and a warning is emitted whenever one of them is briefly
unreachable — turning a third party's downtime into a red build. The docs job
fails on a non-zero exit instead, which is what a genuinely broken build
gives.
"""

from __future__ import annotations

import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

project = "SpecMod"
author = "James Holt"
#: Sphinx substitutes `%Y` with the build year, so this does not need editing
#: — and it honours `SOURCE_DATE_EPOCH`, so a reproducible build stamps the
#: source date rather than the day it happened to run. 2020 is the year in
#: `LICENSE` and the year the history starts.
copyright = "2020-%Y, James Holt"

try:
    release = version("specmod")
except PackageNotFoundError:  # pragma: no cover - docs built without an install
    release = "0.0.0"
#: The short X.Y form, which is what the sidebar shows.
version = ".".join(release.split(".")[:2])

extensions = [
    #: `myst_nb` supersedes `myst_parser` — it is a superset, and loading both
    #: makes Sphinx complain that the `.md` parser is registered twice.
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    #: Makes `:class: dropdown` collapse an admonition. The alpha warning on the
    #: landing page is the reason: it has to be read once and then stops earning
    #: the screen it occupies, and a caveat nobody scrolls past is worse than
    #: one folded behind its own headline.
    "sphinx_togglebutton",
]
#: `sphinx_autodoc_typehints` is deliberately not used. `autodoc_typehints`
#: below is built into `sphinx.ext.autodoc` and does the same job here, and the
#: extension calls an API Sphinx 10 removes — it emits a deprecation warning
#: per module on Sphinx 9. One less dependency for no loss.

#: `REFACTOR_PLAN.md` is a working document, not documentation — it is written
#: for whoever is doing the refactor and records decisions and their evidence.
#: `notebooks/` holds the source for the transform comparison, which
#: `tools/measure_docs.py` renders into `choosing-a-transform.md`; the page is
#: what is published, so building the notebook too would duplicate it.
#: `notes/` *is* included: `choosing-a-transform.md` links to it for a
#: per-trace table, so excluding it broke that link.
exclude_patterns = [
    "_build",
    "REFACTOR_PLAN.md",
    "notebooks/*",
    "Thumbs.db",
    ".DS_Store",
]

# ------------------------------------------------------- the tutorial notebook

#: The tutorial lives in `tutorial/`, outside this source directory, next to the
#: 1 MB of PNR waveforms it reads through paths relative to itself. Sphinx only
#: builds what is under `docs/`, so it is copied in here.
#:
#: A copy rather than a move, and rather than executing it where it sits,
#: because **the notebook writes**: cell 53 saves an HDF5 file under
#: `data/events/<event>/spectra/` and cell 54 writes flatfiles beside it. Run in
#: place, a docs build would leave those artefacts in the working tree — which
#: is why `tests/test_tutorial.py` executes it in a `tmp_path` copy too. This is
#: the same trick, and the copy is what the artefacts land in.
#:
#: Copying the data with it is what keeps the notebook's `Path("data/events")`
#: working, so the notebook is identical whether opened from `tutorial/`, run by
#: pytest, or built here. `tutorial/` stays canonical: eight other files name
#: `tutorial/data/events/`, and the CI job that executes it records that a
#: renamed data directory has already broken this once.
_HERE = Path(__file__).parent
_TUTORIAL_SRC = _HERE.parent / "tutorial"
_TUTORIAL_DST = _HERE / "tutorial"

if _TUTORIAL_SRC.is_dir():
    shutil.rmtree(_TUTORIAL_DST, ignore_errors=True)
    shutil.copytree(_TUTORIAL_SRC, _TUTORIAL_DST)

#: `force`, not `auto`. The `nbstripout` hook means the notebook arrives with no
#: outputs, so `auto` would execute it today and quietly stop the moment anyone
#: committed a notebook with outputs saved — publishing whatever was last run by
#: hand. Forcing it means the page can only ever show output the code produced
#: against the code being documented, whatever is in the file.
nb_execution_mode = "force"
#: Matches `tests/test_tutorial.py`. The whole notebook runs in ~40s; the
#: default 30s is per cell, and the two-stage fit is the one that would trip it.
nb_execution_timeout = 600
#: A notebook that raises fails the build. That is the point: an executed
#: tutorial is only a guarantee if a broken one is loud.
nb_execution_raise_on_error = True
#: `False` keeps the kernel's working directory at the notebook's own, which is
#: what `Path("data/events")` resolves against.
nb_execution_in_temp = False

#: Markdown for the prose pages — every one in `docs/` is written that way, and
#: allowing reStructuredText too would mean two syntaxes for the same job — plus
#: `.ipynb` for the tutorial.
#:
#: Both map to `myst-nb`, which is the only parser name `myst_nb` 1.4 registers:
#: it does not re-register `myst_parser`'s `markdown`, so leaving `.md` pointing
#: at that fails the build outright with "Source parser for markdown not
#: registered". The `myst-nb` parser is a superset and reads the prose pages
#: identically.
source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb"}

#: `linkify` is deliberately absent: it needs `linkify-it-py` and every link in
#: these pages is already explicit.
myst_enable_extensions = [
    "colon_fence",  # ::: fences, so a directive can hold a code block
    "deflist",
    "dollarmath",
    "substitution",
]
#: Heading anchors down to h3, so `docs/*.md` can link to each other's sections.
myst_heading_anchors = 3

#: `{{ release }}` in a page resolves to the version this site was built from.
#: The roadmap used to state it in prose, which meant it was wrong from the
#: moment of the next release — it said v0.2.0 through two that followed. A
#: version is derivable, so deriving it removes the only part of that page that
#: went stale on a timetable rather than on a decision.
myst_substitutions = {"release": release}

# ------------------------------------------------------------------ autodoc

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "obspy": ("https://docs.obspy.org", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py", None),
}
#: Resolving these needs the network. A build without it still produces a
#: site, with the cross-references left as plain text.
intersphinx_disabled_reftypes = ["*"]

# --------------------------------------------------------------------- html

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {version}"
html_theme_options = {
    "github_url": "https://github.com/sgjholt/SpecMod",
    "show_prev_next": True,
    "navigation_with_keys": False,
    #: This theme puts *top-level* toctree entries in the header and gives the
    #: sidebar the current section's children. A flat toctree therefore
    #: produces a header of ten items and an empty sidebar, with `:caption:`
    #: nowhere to render — which is what this site had. The four section pages
    #: are what give the sidebar something to nest.
    #:
    #: `show_nav_level: 1` expands each section's own entries rather than
    #: leaving them behind a disclosure triangle, so a reader can see a
    #: section's contents without a click.
    "show_nav_level": 1,
    #: Deep enough for a section, its pages, and their headings — which is what
    #: makes a long page like `processing` navigable from the sidebar rather
    #: than by scrolling.
    "navigation_depth": 3,
}
html_static_path: list[str] = []
