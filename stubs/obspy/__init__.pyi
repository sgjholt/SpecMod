from os import PathLike
from typing import Any

from . import geodetics as geodetics
from .core.inventory import Inventory as Inventory
from .core.stream import Stream as Stream
from .core.trace import Stats as Stats, Trace as Trace
from .core.utcdatetime import UTCDateTime as UTCDateTime

def read(
    pathname_or_url: str | PathLike[str] | None = ...,
    format: str | None = ...,
    headonly: bool = ...,
    starttime: UTCDateTime | None = ...,
    endtime: UTCDateTime | None = ...,
    nearest_sample: bool = ...,
    dtype: Any = ...,
    apply_calib: bool = ...,
    check_compression: bool = ...,
    **kwargs: Any,
) -> Stream: ...
def read_inventory(
    path_or_file_object: str | PathLike[str] | None = ...,
    format: str | None = ...,
    level: str = ...,
    *args: Any,
    **kwargs: Any,
) -> Inventory: ...
