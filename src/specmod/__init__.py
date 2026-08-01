"""SpecMod — a toolbox for processing and modelling seismic spectra.

The public API is re-exported here. Submodules may be imported directly for
anything not listed in ``__all__``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("specmod")
except PackageNotFoundError:  # pragma: no cover - only when running from source
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
