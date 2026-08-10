# The plugin-registry surface `specmod.picks.events` uses to ask ObsPy which
# of its formats recognises a file. Private to ObsPy, and stubbed rather than
# ignored so a signature change is a type error instead of a runtime one.
from collections.abc import Callable
from importlib.metadata import EntryPoint
from typing import Any

ENTRY_POINTS: dict[str, dict[str, EntryPoint]]

def buffered_load_entry_point(
    dist: str, group: str, name: str
) -> Callable[..., Any]: ...
