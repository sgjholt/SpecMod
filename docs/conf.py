"""Sphinx configuration for the SpecMod documentation.

Build with::

    uv pip install -e '.[docs]'
    sphinx-build -b html docs docs/_build/html

``-W`` is deliberately **not** used. Intersphinx resolves seven inventories
over the network, and a warning is emitted whenever one of them is briefly
unreachable — turning a third party's downtime into a red build. The docs job
fails on a non-zero exit instead, which is what a genuinely broken build
gives.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

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
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
#: `sphinx_autodoc_typehints` is deliberately not used. `autodoc_typehints`
#: below is built into `sphinx.ext.autodoc` and does the same job here, and the
#: extension calls an API Sphinx 10 removes — it emits a deprecation warning
#: per module on Sphinx 9. One less dependency for no loss.

#: `REFACTOR_PLAN.md` is a working document, not documentation — it is written
#: for whoever is doing the refactor and records decisions and their evidence.
#: `notebooks/` is built by Phase 6 with myst-nb; until then the `.ipynb` files
#: would be copied in without being executed, which is worse than leaving them
#: out. `notes/` *is* included: `choosing_a_transform.md` links to it for a
#: per-trace table, so excluding it broke that link.
exclude_patterns = [
    "_build",
    "REFACTOR_PLAN.md",
    "notebooks/*",
    "Thumbs.db",
    ".DS_Store",
]

#: Markdown only. Every page in `docs/` is already written that way, and
#: allowing both means two syntaxes for the same job.
source_suffix = {".md": "markdown"}

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
}
html_static_path: list[str] = []
