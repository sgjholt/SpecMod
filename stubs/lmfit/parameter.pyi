from collections.abc import ItemsView, Iterator, ValuesView
from typing import Any

class Parameter:
    name: str
    value: float
    vary: bool
    min: float
    max: float
    expr: str | None
    #: `None` whenever the minimiser produced no covariance matrix. Every read
    #: of this in SpecMod has to cope with that; see `__determine_pass_or_fail`.
    stderr: float | None
    correl: dict[str, float] | None
    def __init__(
        self,
        name: str,
        value: float | None = ...,
        vary: bool = ...,
        min: float = ...,
        max: float = ...,
        expr: str | None = ...,
        brute_step: float | None = ...,
        user_data: Any = ...,
    ) -> None: ...
    def set(
        self,
        value: float | None = ...,
        vary: bool | None = ...,
        min: float | None = ...,
        max: float | None = ...,
        expr: str | None = ...,
        brute_step: float | None = ...,
    ) -> None: ...

class Parameters:
    def __init__(self, usersyms: dict[str, Any] | None = ...) -> None: ...
    def __getitem__(self, key: str) -> Parameter: ...
    def __setitem__(self, key: str, value: Parameter) -> None: ...
    def __contains__(self, key: str) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def keys(self) -> Any: ...
    def values(self) -> ValuesView[Parameter]: ...
    def items(self) -> ItemsView[str, Parameter]: ...
    def add(self, name: str, **kwargs: Any) -> None: ...
    def copy(self) -> Parameters: ...
    def valuesdict(self) -> dict[str, float]: ...
