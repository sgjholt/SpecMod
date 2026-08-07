"""Fit results as a table, on disk.

The other half of ``REFACTOR_PLAN`` §4.6. Spectra are arrays asked about one
event at a time and live in HDF5 (:mod:`specmod.io`); fit results are a
columnar scan over every event ever fitted — *"give me f_c and Omega for all
635 events and regress them"* — and live here. The published Magna run produced
**11,226 rows** of exactly that, and a multi-event catalogue is larger.

**Parquet as the primary, CSV as an export.** This is not fashion. CSV loses
dtypes, so a column of floats comes back as floats only if pandas guesses
right and as strings if one cell says ``None``; it round-trips every float
through decimal text; and it has to be read in full to read any of it. Parquet
is typed, compressed, and queryable with DuckDB or polars *without loading the
file*. CSV stays because journal supplements want it, and because a human with
a text editor is a legitimate reader.

The format follows from the suffix, so the choice is visible at the call site
rather than buried in a keyword.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

__all__ = ["read_table", "write_table"]

#: Suffix to format. ``.parquet`` is the default because it is what the
#: analysis actually wants; the others exist because other tools do.
_FORMATS = {".parquet": "parquet", ".pq": "parquet", ".csv": "csv"}


def _resolve(path: str | Path) -> tuple[Path, str]:
    path = Path(path)
    try:
        return path, _FORMATS[path.suffix.lower()]
    except KeyError:
        raise ValueError(
            f"cannot tell what format {path.name!r} should be from its suffix. "
            f"Use one of {', '.join(sorted(_FORMATS))}."
        ) from None


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401, PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "writing Parquet needs pyarrow. Install it with "
            "`pip install specmod[io]`, or use a .csv path."
        ) from exc


def write_table(
    path: str | Path,
    table: pd.DataFrame,
    *,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write a fit table, choosing the format from ``path``'s suffix.

    ``meta`` is stored in the file's own metadata for Parquet — where it
    survives as key-value pairs a reader can get at without parsing the data —
    and dropped for CSV, which has nowhere to put it. That asymmetry is stated
    rather than hidden: a CSV export is lossy about provenance, and pretending
    otherwise is how a run stops being reproducible.

    The parent directory is created if absent.
    """
    path, format = _resolve(path)
    parent = path.parent
    if parent != Path():
        parent.mkdir(parents=True, exist_ok=True)

    if format == "csv":
        table.to_csv(path, index=False)
        return path

    _require_pyarrow()
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    arrow = pa.Table.from_pandas(table, preserve_index=False)
    if meta:
        # Parquet key-value metadata is bytes-to-bytes, and the existing schema
        # metadata carries pandas' own type information — dropping it would
        # lose the dtypes this format exists to preserve.
        existing = arrow.schema.metadata or {}
        arrow = arrow.replace_schema_metadata(
            {
                **existing,
                **{str(k).encode(): str(v).encode() for k, v in meta.items()},
            }
        )
    pq.write_table(arrow, path, compression="zstd")
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a fit table back, choosing the format from ``path``'s suffix."""
    path, format = _resolve(path)
    if format == "csv":
        return pd.read_csv(path)
    _require_pyarrow()
    return pd.read_parquet(path)
