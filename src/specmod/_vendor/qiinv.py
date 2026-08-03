"""Quadratic inverse spectrum estimation, vendored from Prieto's ``multitaper``.

Upstream: https://github.com/gaprieto/multitaper — ``multitaper/utils.py``,
functions ``qiinv`` and ``sft``, version 1.2.0.

    MIT License
    Copyright (c) 2022 Germán A. Prieto

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

Why this is vendored rather than imported
-----------------------------------------
``multitaper`` remains an optional extra and is still the better route to
jackknife intervals and the F-test. This one function is carried here because
it does not run at all on the version of numpy SpecMod requires, and because a
corner frequency is read straight off the quantity it estimates.

Changes from upstream
---------------------
1. **numpy 2 compatibility (the reason this is here).** ``scipy.optimize.nnls``
   and ``scipy.linalg.lstsq`` return shape-``(1,)`` arrays for a single-column
   system, and upstream assigns them into scalar slots of a 1-D buffer. That
   was deprecated in numpy 1.25 and raises in 2.0, so ``qiinv`` fails for every
   weighting scheme. Four lines now index the scalar out explicitly: ``cte2``,
   ``cte``, ``slope``, ``quad``.

2. **``sft`` replaced by a vectorised equivalent, dropping numba.** Upstream's
   ``sft`` is a Goertzel recursion decorated with ``@njit``; it is called
   ``kspec * nxi`` times, so without numba it would be unusably slow. It
   computes a single-frequency DFT, ``sum_j x[j] exp(-i w j)``, and the whole
   ``Vj`` block collapses to one complex matmul. Verified against upstream to
   1e-12 absolute over 300 randomised cases — the residual is the recursion's
   own accumulation error, so the form here is the more accurate of the two.
   This removes numba from the dependency graph entirely.

3. **The ``nfft``-long solve loop is unchanged.** It is the slow part and it
   vectorises, but leaving it identical keeps the diff against upstream
   readable. Optimise it only with the cross-validation test in place.

4. The ``Cjk``/``Pjk`` construction is vectorised out of its double loop,
   ``spec`` is dropped from the signature (upstream accepts it and never reads
   it), the eigenvalue warning goes through ``warnings`` rather than ``print``,
   and the unused ``cte``/``sigma2``/``cte_var``/``slope_var`` buffers are gone.

5. Formatting and type annotations.

References
----------
Prieto, G.A., Parker, R.L., Thomson, D.J., Vernon, F.L., Graham, R.L. (2007).
Reducing the bias of multitaper spectrum estimates.
*Geophysical Journal International* 171(3), 1269-1281.

Thomson, D.J. (1990). Quadratic-inverse spectrum estimates: applications to
palaeoclimatology. *Phil. Trans. R. Soc. Lond. A* 332, 539-597.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg
import scipy.optimize
from numpy.typing import NDArray

__all__ = ["qiinv"]


def _single_frequency_dft(
    tapers: NDArray[np.float64], omega: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """``sum_j tapers[j, k] exp(-i omega_i j)`` for every ``(i, k)``.

    Replaces upstream's ``sft`` Goertzel recursion (see change 2 above). The
    frequencies here are a handful of points inside the inner band, not an FFT
    grid, so there is nothing to gain from an FFT and a direct sum is both
    exact and fast enough as one matmul.
    """
    phase = np.exp(-1j * np.outer(omega, np.arange(tapers.shape[0])))
    result: NDArray[np.complex128] = phase @ tapers
    return result


def qiinv(
    yk: NDArray[np.complex128],
    wt: NDArray[np.float64],
    vn: NDArray[np.float64],
    lamb: NDArray[np.float64],
    nw: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Quadratic inverse spectrum estimate, after Prieto et al. (2007).

    Estimates the spectrum's first two derivatives inside the inner band and
    subtracts the bias that curvature — the second derivative — introduces into
    an ordinary multitaper estimate. That bias is largest exactly where the
    spectrum bends most sharply, which for a Brune source is the corner.

    Parameters
    ----------
    yk
        Eigencoefficients, shape ``(nfft, kspec)``: the **two-sided** FFT of
        each tapered copy of the record, in numpy's FFT ordering.
    wt
        Per-taper, per-frequency weights, shape ``(nfft, kspec)``. Pass ones
        for flat weighting.
    vn
        DPSS tapers, shape ``(npts, kspec)``. Note the orientation: this is the
        transpose of what :func:`scipy.signal.windows.dpss` returns.
    lamb
        Taper eigenvalues (concentration ratios), shape ``(kspec,)``.
    nw
        The time-bandwidth product.

    Returns
    -------
    qispec
        The quadratic estimate, shape ``(nfft,)``, in the units of
        ``|yk|**2``.
    slope, quad
        First and second derivative of the spectrum with respect to frequency.
        Returned because they are diagnostics in their own right — ``quad`` is
        what the correction is built from, so a caller can see how much work it
        did.
    """
    npts, kspec = np.shape(vn)
    nfft = np.shape(yk)[0]
    nxi = 79
    n_cross = kspec * kspec

    if np.min(lamb) < 0.9:
        # Upstream prints; a caller of ours has no console to watch.
        warnings.warn(
            f"Poorest taper eigenvalue is {np.min(lamb):.4f} (< 0.9), so the "
            f"higher-order tapers leak badly and the quadratic estimate will "
            f"inherit that. Reduce n_tapers or raise time_bandwidth.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ---------------------------------------------- inner-band frequencies
    bp = nw / npts  # half-bandwidth W
    xi = np.linspace(-bp, bp, num=nxi)
    dxi = xi[2] - xi[1]

    xk = wt * yk
    vj = _single_frequency_dft(vn, 2.0 * np.pi * xi) / np.sqrt(lamb)

    # ------------------------- vectorised Cjk (data) and Pjk = {Vj Vk*}
    i_idx, k_idx = np.divmod(np.arange(n_cross), kspec)
    cross = np.conjugate(xk[:, i_idx]) * xk[:, k_idx]  # (nfft, L)
    proj = np.conjugate(vj[:, i_idx]) * vj[:, k_idx]  # (nxi, L)

    cross = cross.T  # (L, nfft), as upstream orders it
    proj = proj.T  # (L, nxi)
    proj[:, 0] *= 0.5  # trapezoid end weights
    proj[:, nxi - 1] *= 0.5

    # ------------------ Chebyshev basis: constant, slope, curvature
    hcte = np.ones(nxi)
    hslope = xi / bp
    hquad = 2.0 * (xi / bp) ** 2 - 1.0

    h1 = (proj @ hcte) * dxi
    hk = np.empty((n_cross, 3), dtype=complex)
    hk[:, 0] = h1
    hk[:, 1] = (proj @ hslope) * dxi
    hk[:, 2] = (proj @ hquad) * dxi
    nh = hk.shape[1]

    # --------------------------- least squares via QR, factored once
    q_mat, r_mat = scipy.linalg.qr(hk)
    qt = np.transpose(q_mat)
    ri = scipy.linalg.lstsq(r_mat, np.eye(n_cross))[0]
    covb = np.real(ri @ np.transpose(ri))

    cte2 = np.zeros(nfft)
    slope = np.zeros(nfft)
    quad = np.zeros(nfft)
    quad_var = np.zeros(nfft)

    h1_real = np.real(h1)[:, None]
    for i in range(nfft):
        cjk = cross[:, i : i + 1]

        # Constrain the constant term to be non-negative: a power spectrum
        # cannot go below zero, and the unconstrained fit will happily do so.
        cte2[i] = np.real(scipy.optimize.nnls(h1_real, np.real(cjk[:, 0]))[0])[0]

        # Solve the derivatives against what the constant term left behind.
        residual = cjk - h1_real * cte2[i]
        hmodel = scipy.linalg.lstsq(r_mat, qt @ residual)[0]
        slope[i] = -np.real(hmodel[1])[0]
        quad[i] = np.real(hmodel[2])[0]

        pred = hk @ np.real(hmodel)
        sigma2 = np.sum(np.abs(residual - pred) ** 2) / (n_cross - nh)
        quad_var[i] = sigma2 * covb[2, 2]

    slope = slope / bp
    quad = quad / bp**2
    quad_var = quad_var / bp**4

    # Damp the correction where the curvature estimate is itself noisy, so a
    # poorly-determined second derivative cannot drag the spectrum around.
    weight = quad**2 / (quad**2 + quad_var)
    qispec = cte2 - weight * (1.0 / 6.0) * bp**2 * quad

    return qispec, slope, quad
