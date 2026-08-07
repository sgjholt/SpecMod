from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .parameter import Parameters

class ModelResult:
    params: Parameters
    #: The model evaluated at the best-fit parameters, on the data's own grid.
    best_fit: NDArray[np.float64]
    residual: NDArray[np.float64]
    init_fit: NDArray[np.float64]
    aic: float
    bic: float
    chisqr: float
    redchi: float
    success: bool
    #: False when the minimiser estimated no covariance matrix — which Powell,
    #: the shipped default, never does. `stderr` on every parameter is then
    #: `None`, and that is a property of the method rather than a failed fit.
    errorbars: bool
    def fit_report(self, **kwargs: Any) -> str: ...
    def eval(self, params: Parameters | None = ..., **kwargs: Any) -> Any: ...

class Model:
    def __init__(
        self,
        func: Callable[..., Any],
        independent_vars: list[str] | None = ...,
        param_names: list[str] | None = ...,
        nan_policy: str = ...,
        prefix: str = ...,
        name: str | None = ...,
        **kws: Any,
    ) -> None: ...
    @property
    def param_names(self) -> list[str]: ...
    def make_params(self, verbose: bool = ..., **kwargs: Any) -> Parameters: ...
    def fit(
        self,
        data: NDArray[Any],
        params: Parameters | None = ...,
        weights: NDArray[Any] | None = ...,
        method: str = ...,
        iter_cb: Callable[..., Any] | None = ...,
        scale_covar: bool = ...,
        verbose: bool = ...,
        fit_kws: dict[str, Any] | None = ...,
        nan_policy: str | None = ...,
        calc_covar: bool = ...,
        max_nfev: int | None = ...,
        # `coerce_farray` exists on lmfit 1.3 and NOT on 1.2.0, which is this
        # project's declared floor. Left out rather than declared: a stub must
        # be true across the supported range, and nothing here passes it.
        **kwargs: Any,
    ) -> ModelResult: ...
