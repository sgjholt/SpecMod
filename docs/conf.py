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

#: `REFACTOR_PLAN.md` and `branding.md` are working documents, not
#: documentation — the first records refactor decisions and their evidence, the
#: second is the design source `_static/academic.css` and `tools/make_logo.py`
#: are written against. Both are cited from source comments rather than linked
#: from any built page, so neither has a URL to keep.
#: `notebooks/` holds the long-form transform comparison;
#: `choosing-a-transform.md` is the published page and carries the same
#: measurements, so building the notebook too would duplicate it.
#: `_builders/` holds the scripts that write both notebooks — the tutorial's
#: included — and is source, not documentation.
#: `notes/` *is* included: `choosing-a-transform.md` links to it for a
#: per-trace table, so excluding it broke that link.
exclude_patterns = [
    "_build",
    "REFACTOR_PLAN.md",
    "branding.md",
    "notebooks/**",
    "_builders/**",
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

#: `furo`, per `branding.md`. It replaced `pydata_sphinx_theme`, and the
#: navigation model is the substantive difference rather than the colours:
#: pydata put top-level toctree entries in a *header* and gave the sidebar the
#: current section's children, so the four section pages existed to give that
#: sidebar something to nest. furo has one sidebar holding the whole tree, so
#: those four pages now nest inside it instead — the structure still earns its
#: keep, and `index.md`'s toctree is still what orders the site.
html_theme = "furo"
html_title = f"{project} {version}"

#: Every colour the site uses, in one place, as the manual requires:
#: `academic.css` reads them back through `var()` and hard-codes none.
html_theme_options = {
    "light_css_variables": {
        "color-background-primary": "#FDFBF7",  # warm manuscript off-white
        "color-background-secondary": "#F8FAFC",
        "color-background-border": "#E2E8F0",
        "color-foreground-primary": "#1E293B",  # ink slate
        "color-brand-primary": "#1E40AF",  # journal navy
        "color-brand-content": "#C2410C",  # terracotta
        #: Declared under `light` only. furo emits these on `body` as the base
        #: declaration, so they carry into dark mode without a second copy.
        "font-stack": "'Open Sans', sans-serif",
        "font-stack--monospace": (
            "'Fira Code', 'Computer Modern Typewriter', monospace"
        ),
    },
    "dark_css_variables": {
        "color-background-primary": "#0F172A",  # deep slate
        "color-background-secondary": "#1E293B",
        "color-background-border": "#334155",
        "color-foreground-primary": "#F8FAFC",  # paper white
        "color-brand-primary": "#60A5FA",  # soft navy
        "color-brand-content": "#FB923C",  # soft terracotta
    },
    "light_logo": "specmod-logo-academic.svg",
    "dark_logo": "specmod-logo-academic-dark.svg",
    #: The logo carries the wordmark, so repeating the project name under it
    #: says the same thing twice.
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/sgjholt/SpecMod",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_static_path = ["_static"]
#: `fonts.css` first: it declares the `@font-face` rules `academic.css` and the
#: theme variables then refer to by name. Generated by `tools/vendor_fonts.py`.
html_css_files = ["fonts.css", "academic.css"]
#: The square mark, which is the browser-tab half of the logo system.
html_favicon = "_static/specmod-logo-academic-mark.svg"
