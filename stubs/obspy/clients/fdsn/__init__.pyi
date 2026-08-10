from typing import Any

from ...core.inventory import Inventory
from ...core.stream import Stream
from ...core.utcdatetime import UTCDateTime

class Client:
    base_url: str
    def __init__(self, base_url: str = ..., **kwargs: Any) -> None: ...
    # Returns an `obspy.core.event.Catalog`, which is not stubbed: only its
    # sequence behaviour is used here, and stubbing the event model is a much
    # larger job than this needs.
    def get_events(self, **kwargs: Any) -> Any: ...
    def get_stations(
        self,
        starttime: UTCDateTime | None = ...,
        endtime: UTCDateTime | None = ...,
        **kwargs: Any,
    ) -> Inventory: ...
    def get_waveforms(
        self,
        network: str,
        station: str,
        location: str,
        channel: str,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        **kwargs: Any,
    ) -> Stream: ...
