# Minimal: the parts of the QuakeML object model specmod touches. The full
# model is large, and stubbing it wholesale would be a maintenance burden for
# attributes nothing here reads.
from collections.abc import Iterator
from typing import Any

from .utcdatetime import UTCDateTime

class WaveformStreamID:
    network_code: str | None
    station_code: str | None
    location_code: str | None
    channel_code: str | None
    def __init__(
        self,
        network_code: str | None = ...,
        station_code: str | None = ...,
        location_code: str | None = ...,
        channel_code: str | None = ...,
        **kwargs: Any,
    ) -> None: ...

class Pick:
    time: UTCDateTime
    phase_hint: str | None
    evaluation_status: str | None
    waveform_id: WaveformStreamID
    def __init__(self, **kwargs: Any) -> None: ...

class Origin:
    time: UTCDateTime
    latitude: float
    longitude: float
    depth: float
    def __init__(self, **kwargs: Any) -> None: ...

class Magnitude:
    mag: float
    magnitude_type: str | None
    def __init__(self, **kwargs: Any) -> None: ...

class Event:
    picks: list[Pick]
    origins: list[Origin]
    magnitudes: list[Magnitude]
    def __init__(self, **kwargs: Any) -> None: ...
    def preferred_origin(self) -> Origin | None: ...
    def preferred_magnitude(self) -> Magnitude | None: ...

class Catalog:
    events: list[Event]
    def __init__(self, events: list[Event] | None = ..., **kwargs: Any) -> None: ...
    def __iter__(self) -> Iterator[Event]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Event: ...
    def write(self, filename: str, format: str = ..., **kwargs: Any) -> None: ...
