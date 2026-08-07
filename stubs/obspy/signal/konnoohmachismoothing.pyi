from typing import Any

import numpy as np
from numpy.typing import NDArray

def konno_ohmachi_smoothing(
    spectra: NDArray[Any],
    frequencies: NDArray[Any],
    bandwidth: float = ...,
    count: int = ...,
    enforce_no_matrix: bool = ...,
    max_memory_usage: int = ...,
    normalize: bool = ...,
) -> NDArray[np.float64]: ...
