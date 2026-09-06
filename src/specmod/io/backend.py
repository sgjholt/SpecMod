"""The optional-dependency boundary.

The only module that knows the library is called ``h5py`` and the extra is
called ``io``. Keeping that in one place is what lets the rest of the package
be written as though HDF5 were always available: every path that needs it
comes through here first and gets one message if it is missing.
"""

from __future__ import annotations

from typing import Any


def require_h5py() -> Any:
    """Import ``h5py``, or say what to install.

    Deferred rather than imported at module scope so that the message names the
    extra. A bare ``ModuleNotFoundError: h5py`` in the middle of a save tells a
    user nothing about how to fix it.
    """
    try:
        import h5py  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading and writing spectra needs h5py. Install it with "
            "`pip install specmod[io]`."
        ) from exc
    return h5py
