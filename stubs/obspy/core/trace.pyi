from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .utcdatetime import UTCDateTime

class Stats:
    """SEED header fields, plus whatever else is assigned to it.

    `Stats` is an `AttribDict`, so it is both a mapping and an object with
    attributes, and callers add keys freely — SpecMod adds fourteen of its own
    (`p_time`, `s_time`, `repi`, `rhyp`, `wstart`, `wend`, ...). Only the
    standard fields are declared; the rest fall through to `Any`, which is
    exactly the split that exists at runtime.
    """

    network: str
    station: str
    location: str
    channel: str
    npts: int
    sampling_rate: float
    calib: float
    starttime: UTCDateTime
    endtime: UTCDateTime
    delta: float

    def __init__(self, header: dict[str, Any] | None = ...) -> None: ...

    # The open half. Declared explicitly so reading an undeclared field is
    # `Any` rather than an error: the alternative is listing SpecMod's private
    # conventions in a stub that claims to describe ObsPy.
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...
    def get(self, key: str, default: Any = ...) -> Any: ...
    def keys(self) -> Any: ...
    def items(self) -> Any: ...
    def values(self) -> Any: ...
    def update(self, adict: Any) -> None: ...
    def copy(self) -> Stats: ...

class Trace:
    stats: Stats
    data: NDArray[Any]

    def __init__(
        self, data: NDArray[Any] | None = ..., header: dict[str, Any] | None = ...
    ) -> None: ...
    @property
    def id(self) -> str: ...
    def copy(self) -> Trace: ...
    def trim(
        self,
        starttime: UTCDateTime | None = ...,
        endtime: UTCDateTime | None = ...,
        pad: bool = ...,
        nearest_sample: bool = ...,
        fill_value: float | None = ...,
    ) -> Trace: ...
    def detrend(self, type: str = ..., **options: Any) -> Trace: ...
    def taper(
        self,
        max_percentage: float | None,
        type: str = ...,
        max_length: float | None = ...,
        side: str = ...,
        **kwargs: Any,
    ) -> Trace: ...
    def normalize(self, norm: float | None = ...) -> Trace: ...
    def times(
        self, type: str = ..., reftime: UTCDateTime | None = ...
    ) -> NDArray[np.float64]: ...
