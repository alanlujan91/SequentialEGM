"""ENGINE: ENdogenous Grid INterpolation and Extrapolation.

A high-performance 2D interpolation library for curvilinear grids produced
by the Endogenous Grid Method (EGM) in computational economics.

This module implements the algorithm described in Section 4 of the EGM^n
paper ("Multivariate Interpolation on Curvilinear Grids"). The code is
tagged with paper cross-references; see ``content/paper/4_multinterp.md``
for the definitions and theorems.

Paper alignment
---------------
- The grid regularity conditions ``def-fold-free`` and ``def-ipm``
  (Index-Preserved Monotonicity) are the formal assumptions ENGINE relies
  on. Section 4's bracket-displacement constant alpha (the largest bracket
  displacement between adjacent rows) governs only the search cost.
  Callers should verify the conditions on the mesh they supply.
- The two-pass algorithm in ``fig:engine_visual`` (Pass 1: find row
  intersections of the vertical line at x*; Pass 2: 1D interp in y across
  intermediate points) is implemented by the ``log_engine2d_*`` family
  for scattered queries and by the ``engine2d_regrid*`` family for
  meshgrid (sorted) queries.
- Section 4's complexity bound: O(log J * log K) worst case per query. For
  sorted P x Q targets, the walking-pointer regrid costs
  O(P log J + P K + P V + P Q), where V is total bracket variation across rows.
- Section 4's accuracy statement: on a rectangular grid the two passes reduce
  to bilinear interpolation, second order in the mesh width.

Key Classes
-----------
Engine2DInterp : Drop-in replacement for HARK's Curvilinear2DInterp.
    See ``def-ipm`` and ``def-fold-free`` for required input conditions.

Kernel Variants
---------------
- ``log_engine2d_scalar_multi`` / ``log_engine2d_batch_multi``: one scalar
  query, or a parallel-over-queries batch, via binary search on rows + a
  32-slot probe cache; both run the shared per-query body ``multi_query_into``.
  The batch kernel takes tuples of value and output arrays, so each function
  count (the common EGM case is three) compiles its own specialization.
- ``engine2d_regrid``: parallel-over-P meshgrid regrid; walks rows
  forward as y_q advances within each p.
- ``engine2d_regrid_serial``: same algorithm, no ``parallel=True``;
  for hot-inner-loop usage where ``prange`` dispatch dominates.
- ``engine2d_regrid_ipo``: full walking-pointer (no per-p reset);
  realizes the sorted-target total-variation bound from Section 4.

Performance
-----------
- JIT-compiled with Numba for near-C performance.
- For inner-loop EGM regrids (many small per-slice calls), prefer
  ``regrid(mode='serial')``. The parallel variant pays ~0.5 ms per call
  in ``prange`` thread-pool setup that dominates small slices.
- For HARK-style ``__call__`` patterns with scattered queries (e.g.,
  ``solution_next(R*a + shock*H, (1-shock)*H)``), ``log_engine2d_batch_multi``
  runs the probe-caching kernel in parallel over the queries.
- Regrid time versus mesh size: ``code/engine/bench_regrid_scaling.py``
  measures it and ``validate_regrid_linear_scaling`` in
  ``code/validate_claims.py`` checks it.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from engine.core import KERNEL_JIT

ENGINE_JIT = {**KERNEL_JIT, "boundscheck": False}

__all__ = [
    "Engine2DInterp",
    "warmup_jit",
]

# Maximum number of probes cached during binary search on rows.
# log2(4 billion) ~ 32, so 32 entries suffice for any practical grid size
# (K >= 2**32 would silently fall through to fallback search).
_MAX_PROBE_CACHE = 32

# =============================================================================
# Logarithmic ENGINE Helper Functions
# =============================================================================


@njit(**ENGINE_JIT, inline="always")
def binary_search_in_range(xq: float, xp, lo: int, hi: int) -> int:
    """Binary-search ``xp[lo..hi]`` for the bracketing index of ``xq``.

    Shared inner loop of ``binary_search_bracket`` (``lo, hi = 0, n-1``) and
    ``bounded_binary_search`` (``lo, hi = j_lo, j_hi+1``). Requires ``hi > lo``.
    """
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if xp[mid] <= xq:
            lo = mid
        else:
            hi = mid
    return lo


@njit(**ENGINE_JIT, inline="always")
def binary_search_bracket(xq: float, xp) -> int:
    """Binary search to find bracketing index (no hint).

    Returns lo such that xp[lo] <= xq < xp[lo+1], or boundary cases.
    Used by regrid kernel where no prior hint is available.

    Degenerate inputs: for ``n < 2`` returns 0 (caller is responsible
    for not indexing ``xp[lo+1]`` in that case; ``interp_at_bracket``
    handles ``n == 1`` correctly).
    """
    n = len(xp)
    if n < 2:
        return 0

    if xq <= xp[0]:
        return 0
    if xq >= xp[n - 1]:
        return n - 2

    return binary_search_in_range(xq, xp, 0, n - 1)


@njit(**ENGINE_JIT, inline="always")
def bounded_binary_search(xq: float, xp, j_lo: int, j_hi: int) -> int:
    """Binary search within a bounded range [j_lo, j_hi].

    Returns the bracketing index within the specified range.
    Used by regrid kernel for efficient row walking with curvature bounds.

    Window-miss guard: a query outside [j_lo, j_hi] but inside the grid has
    its true bracket outside the window, so it falls back to the full binary
    search rather than silently returning a wrong bracket; an under-set
    search margin then costs one extra O(log J) search. At the grid's own
    edges the clamped return still handles extrapolation.
    """
    n = len(xp)

    # Clamp bounds to valid range
    j_lo = max(0, j_lo)
    j_hi = min(n - 2, j_hi)

    if j_lo > j_hi:
        j_lo, j_hi = j_hi, j_lo

    # Window-miss guard; see the docstring above.
    if xq <= xp[j_lo]:
        if j_lo == 0:
            return 0
        return binary_search_bracket(xq, xp)
    if xq >= xp[j_hi + 1]:
        if j_hi == n - 2:
            return n - 2
        return binary_search_bracket(xq, xp)

    return binary_search_in_range(xq, xp, j_lo, j_hi + 1)


@njit(**ENGINE_JIT, inline="always")
def advance_j_walk(xq: float, xp, j_prev: int) -> int:
    """Walk j from ``j_prev`` to a valid bracket for ``xq``.

    Forward-walks while ``xp[j+1] <= xq``, then backward-walks while
    ``xp[j] > xq``. Under bounded bracket displacement (``|j_star(k+1) - j_star(k)| <=
    curvature_bound``) this returns in O(curvature_bound) steps, faster
    than ``bounded_binary_search`` because it skips the logarithmic
    midpoint arithmetic and benefits from straight-line code that the
    branch predictor handles well.

    Boundary semantics match ``binary_search_bracket``: returns
    ``max(0, min(n-2, j))`` so callers can pass the result directly to
    ``interp_at_bracket`` for linear extrapolation outside the mesh.
    """
    n = len(xp)
    if n < 2:
        return 0
    j = j_prev
    if j < 0:
        j = 0
    elif j > n - 2:
        j = n - 2
    # Forward walk
    while j < n - 2 and xp[j + 1] <= xq:
        j += 1
    # Backward walk
    while j > 0 and xp[j] > xq:
        j -= 1
    return j


@njit(**ENGINE_JIT, inline="always")
def interp_at_bracket(xq: float, xp, fp, j: int) -> float:
    """Interpolate given a known bracketing index j.

    Clamps boundary indices and interpolates linearly; a zero-width segment,
    which ``def-ipm`` rules out on any ``Engine2DInterp`` mesh, takes the
    lower node's value.

    For ``n < 2`` returns ``fp[0]`` (or 0.0 if n == 0).
    """
    n = len(xp)
    if n < 2:
        return fp[0] if n == 1 else 0.0

    # Handle boundary cases
    if j < 0:
        j = 0
    elif j >= n - 1:
        j = n - 2

    xl = xp[j]
    dx = xp[j + 1] - xl
    if dx == 0.0:
        return fp[j]

    t = (xq - xl) / dx
    return fp[j] + t * (fp[j + 1] - fp[j])


@njit(**ENGINE_JIT, inline="always")
def probe_row(x_q: float, k: int, fx, fy, j_lo: int, j_hi: int):
    """Probe row k: find j* and compute y_k(x*).

    Uses bounded binary search if j_lo != j_hi, else full search.

    Called from the LOG ENGINE Phase-1 binary-search-on-rows loop. The
    search window ``[j_lo, j_hi]`` can grow to ``cb * best_dist + 1``
    where ``best_dist`` is the row distance to the nearest cached probe
    (Section 4's bracket-displacement constant). When ``best_dist`` is
    moderate (early in the
    Phase-1 binary search), the window can be wide enough that
    ``O(log window)`` bounded binary search is faster than the
    ``O(window)`` :func:`advance_j_walk`. We keep bounded
    binary search here; the regrid kernels use the linear walk because
    their per-row advance always has ``window = cb + 1`` (small).

    Returns: (y_intermediate, j_star)
    """
    # Get row data
    fx_row = fx[:, k]
    fy_row = fy[:, k]

    # Find j* in bounded range
    if j_hi > j_lo:
        j_star = bounded_binary_search(x_q, fx_row, j_lo, j_hi)
    else:
        j_star = binary_search_bracket(x_q, fx_row)

    # Interpolate y-coordinate
    y_interp = interp_at_bracket(x_q, fx_row, fy_row, j_star)

    return y_interp, j_star


# =============================================================================
# Probe-Caching Logarithmic ENGINE Kernels
# =============================================================================


@njit(**ENGINE_JIT, inline="always")
def find_log_engine_bracket(x_q, y_q, fx, fy, curvature_bound):
    """Probe-caching row bracket search shared across log-engine kernels.

    Walks a binary search on rows ``k`` to bracket the query ``y_q``,
    caching up to ``_MAX_PROBE_CACHE`` (``k, j*, y(x_q;k)``) tuples so
    later kernels can interpolate without re-probing. Returns
    ``(k_lo, k_hi, j_lo_star, j_hi_star, y_lo, y_hi)`` with
    ``y_lo <= y_q < y_hi`` (or the closest equivalents at boundaries).

    Inlined by ``inline="always"`` into ``multi_query_into``, the per-query
    body of both scattered-query kernels. The j_lo fallback margin is
    ``curvature_bound * best_dist + 1``, with best_dist the distance to the
    nearest probed row. The window is a performance hint:
    ``bounded_binary_search`` verifies containment and
    escalates a window miss to a full row search, so an under-declared
    ``curvature_bound`` costs one extra O(log J) search per missed probe.

    Empirical guidance
    ------------------
    The health-model warp sweep (code/figures/health_warp_sweep.py) measures
    empirical bracket-displacement constants between adjacent rows from 10
    through 16, so it runs ENGINE with ``curvature_bound=16``, which avoids the
    fallback searches on those meshes; the default value 2 triggers them.
    Sampled displacement is a speed diagnostic, not a bound over every
    admissible query.
    """
    J, K = fx.shape
    k_lo, k_hi = 0, K - 1
    probed_k = np.empty(_MAX_PROBE_CACHE, dtype=np.int32)
    probed_j = np.empty(_MAX_PROBE_CACHE, dtype=np.int32)
    probed_y = np.empty(_MAX_PROBE_CACHE, dtype=fx.dtype)
    n_probed = 0

    while k_hi - k_lo > 1:
        k_mid = (k_lo + k_hi) >> 1
        j_lo_search, j_hi_search = bracket_search_window(
            k_mid,
            probed_k,
            probed_j,
            n_probed,
            J,
            K,
            curvature_bound,
        )
        y_mid, j_mid = probe_row(x_q, k_mid, fx, fy, j_lo_search, j_hi_search)
        n_probed = record_probe(
            probed_k,
            probed_j,
            probed_y,
            n_probed,
            k_mid,
            j_mid,
            y_mid,
        )
        if y_q < y_mid:
            k_hi = k_mid
        else:
            k_lo = k_mid

    j_lo_star, y_lo, j_hi_star, y_hi = lookup_probed_pair(
        probed_k,
        probed_j,
        probed_y,
        n_probed,
        k_lo,
        k_hi,
    )

    if j_lo_star < 0:
        j_lo_search, j_hi_search = bracket_search_window(
            k_lo,
            probed_k,
            probed_j,
            n_probed,
            J,
            K,
            curvature_bound,
        )
        y_lo, j_lo_star = probe_row(x_q, k_lo, fx, fy, j_lo_search, j_hi_search)

    if j_hi_star < 0:
        margin = curvature_bound + 1
        y_hi, j_hi_star = probe_row(
            x_q,
            k_hi,
            fx,
            fy,
            max(0, j_lo_star - margin),
            min(J - 2, j_lo_star + margin),
        )

    return k_lo, k_hi, j_lo_star, j_hi_star, y_lo, y_hi


@njit(**ENGINE_JIT, inline="always")
def bracket_search_window(
    k_target,
    probed_k,
    probed_j,
    n_probed,
    J,
    K,
    curvature_bound,
):
    """Return (j_lo_search, j_hi_search) centered on the nearest probed row."""
    best_dist = K + 1
    j_center = J >> 1
    for p in range(n_probed):
        dist = abs(k_target - probed_k[p])
        if dist < best_dist:
            best_dist = dist
            j_center = probed_j[p]
    margin = curvature_bound * best_dist + 1
    return max(0, j_center - margin), min(J - 2, j_center + margin)


@njit(**ENGINE_JIT, inline="always")
def record_probe(probed_k, probed_j, probed_y, n_probed, k, j, y):
    """Append (k, j, y) to the probe cache if not yet at capacity."""
    if n_probed < _MAX_PROBE_CACHE:
        probed_k[n_probed] = k
        probed_j[n_probed] = j
        probed_y[n_probed] = y
        return n_probed + 1
    return n_probed


@njit(**ENGINE_JIT, inline="always")
def lookup_probed_pair(probed_k, probed_j, probed_y, n_probed, k_lo, k_hi):
    """Look up cached (j*, y) for k_lo and k_hi; -1 indicates a miss."""
    j_lo_star, y_lo = -1, 0.0
    j_hi_star, y_hi = -1, 0.0
    for p in range(n_probed):
        if probed_k[p] == k_lo:
            j_lo_star, y_lo = probed_j[p], probed_y[p]
        if probed_k[p] == k_hi:
            j_hi_star, y_hi = probed_j[p], probed_y[p]
    return j_lo_star, y_lo, j_hi_star, y_hi


@njit(**ENGINE_JIT, inline="always")
def band_weight(y_q, y_lo, y_hi):
    """Linear weight of ``y_q`` in the band ``[y_lo, y_hi]``. A zero-width band
    takes the lower value (weight 0). Shared by every ENGINE kernel.
    """
    dy = y_hi - y_lo
    if dy == 0.0:
        return 0.0
    return (y_q - y_lo) / dy


@njit(**ENGINE_JIT, inline="always")
def multi_query_into(x_q, y_q, fx, fy, fv_list, results, i, curvature_bound) -> None:
    """One scattered query for every function in ``fv_list``: brackets the rows
    once (probe cache), then writes ``results[f][i]`` for each function ``f``.
    """
    k_lo, k_hi, j_lo_star, j_hi_star, y_lo, y_hi = find_log_engine_bracket(
        x_q,
        y_q,
        fx,
        fy,
        curvature_bound,
    )
    t = band_weight(y_q, y_lo, y_hi)
    for f_idx in range(len(fv_list)):
        fv = fv_list[f_idx]
        v_lo = interp_at_bracket(x_q, fx[:, k_lo], fv[:, k_lo], j_lo_star)
        v_hi = interp_at_bracket(x_q, fx[:, k_hi], fv[:, k_hi], j_hi_star)
        results[f_idx][i] = v_lo + t * (v_hi - v_lo)


@njit(**ENGINE_JIT)
def log_engine2d_scalar_multi(
    x_q: float,
    y_q: float,
    fx,
    fy,
    fv_list,
    curvature_bound: int = 2,
):
    """Multi-function Logarithmic ENGINE scalar interpolation with probe caching
    (O(log J * log K) worst case) of each array in the tuple ``fv_list``. Returns
    one value per function.
    """
    results = np.empty((len(fv_list), 1), dtype=fx.dtype)
    multi_query_into(x_q, y_q, fx, fy, fv_list, results, 0, curvature_bound)
    return results[:, 0]


# =============================================================================
# Optimized Regridding Kernel
# =============================================================================


@njit(**ENGINE_JIT, inline="always")
def init_row_state(x_q, fx, fy):
    """Initial (k=0, k=1) row brackets for one x-query: shared by the reset
    and ``"ipo"`` regrid variants (the ``"ipo"`` variant only calls this once,
    at p=0).
    """
    k_current = 0
    j_star_k = binary_search_bracket(x_q, fx[:, 0])
    y_k = interp_at_bracket(x_q, fx[:, 0], fy[:, 0], j_star_k)
    j_star_k1 = binary_search_bracket(x_q, fx[:, 1])
    y_k1 = interp_at_bracket(x_q, fx[:, 1], fy[:, 1], j_star_k1)
    return k_current, j_star_k, y_k, j_star_k1, y_k1


@njit(**ENGINE_JIT, inline="always")
def regrid_per_p_reset(x_q, y_q_1d, fx, fy, fv, result_2d_p, K: int, Q: int) -> None:
    """Reset-per-p regrid body: initial brackets at k=0,1 then q-loop.

    Used by the parallel and serial regrid kernels, which both reset
    ``k_current = 0`` at every p. The ``"ipo"`` kernel (``engine2d_regrid_ipo``)
    does NOT use this helper: it maintains state across the outer p loop and
    performs a forward j-walk initialization for ``p > 0`` before calling
    ``regrid_q_loop``. Inlined.
    """
    k_current, j_star_k, y_k, j_star_k1, y_k1 = init_row_state(x_q, fx, fy)
    # State is not carried across p in the reset variant; discard the return.
    _ = regrid_q_loop(
        x_q,
        y_q_1d,
        fx,
        fy,
        fv,
        result_2d_p,
        K,
        Q,
        k_current,
        j_star_k,
        j_star_k1,
        y_k,
        y_k1,
    )


@njit(**ENGINE_JIT, inline="always")
def regrid_q_loop(
    x_q,
    y_q_1d,
    fx,
    fy,
    fv,
    result_2d_p,
    K: int,
    Q: int,
    k_current: int,
    j_star_k: int,
    j_star_k1: int,
    y_k: float,
    y_k1: float,
):
    """Inner q-loop shared by ``engine2d_regrid``, ``engine2d_regrid_serial``,
    and ``engine2d_regrid_ipo``.

    Walks rows forward as ``y_q`` advances; interpolates within each
    band; writes results into ``result_2d_p`` (a view of one row of the
    output). Returns the updated
    ``(k_current, j_star_k, j_star_k1, y_k, y_k1)`` tuple so the ``"ipo"``
    kernel can carry state across the outer p loop. Inlined by
    ``inline="always"`` so each caller compiles to the same machine
    code as before deduplication.
    """
    for q in range(Q):
        y_q = y_q_1d[q]

        while k_current < K - 2 and y_q >= y_k1:
            k_current += 1
            j_star_k = j_star_k1
            y_k = y_k1
            k_next = k_current + 1
            j_star_k1 = advance_j_walk(x_q, fx[:, k_next], j_star_k)
            y_k1 = interp_at_bracket(x_q, fx[:, k_next], fy[:, k_next], j_star_k1)

        t = band_weight(y_q, y_k, y_k1)
        v_k = interp_at_bracket(x_q, fx[:, k_current], fv[:, k_current], j_star_k)
        v_k1 = interp_at_bracket(
            x_q,
            fx[:, k_current + 1],
            fv[:, k_current + 1],
            j_star_k1,
        )
        result_2d_p[q] = v_k + t * (v_k1 - v_k)

    return k_current, j_star_k, j_star_k1, y_k, y_k1


@njit(**ENGINE_JIT, inline="always")
def regrid_query_dims(x_q_1d, y_q_1d, fx):
    """Query counts (P, Q) and mesh shape (J, K) shared by the regrid kernels,
    which read ``fx[:, 1]`` unconditionally and interpolate along rows of at
    least two nodes: a mesh below 2 x 2 raises.
    """
    P = len(x_q_1d)
    Q = len(y_q_1d)
    J, K = fx.shape
    if J < 2 or K < 2:
        msg = "ENGINE regrid needs a mesh of at least 2 x 2"
        raise ValueError(msg)
    return P, Q, J, K


@njit(**ENGINE_JIT, parallel=True)
def engine2d_regrid(
    x_q_1d,
    y_q_1d,
    fx,
    fy,
    fv,
    result_2d,
) -> None:
    """Parallel regridding from curvilinear to rectangular grid.

    For each x-query the kernel resets ``k_current = 0`` and walks rows
    forward as the y-query advances, maintaining four state variables
    ``(j_star_k, j_star_k1, y_k, y_k1)``. Unlike the scattered-query
    ``log_engine2d_*`` kernels this kernel does NOT use a probe cache:
    the structured-output query order makes the forward walk sufficient,
    so the 32-slot cache would be dead weight. The per-p reset is forced
    by ``parallel=True`` (each ``prange`` iteration must be independent
    of the others); for inner-loop call patterns where the parallel
    dispatch overhead dominates, use :func:`engine2d_regrid_serial`.

    For each p, initialization costs ``O(log J)``; the forward row walk costs
    ``O(K)``; bracket updates cost ``O(V_p)``, where ``V_p`` is total bracket
    variation across rows; and interpolation costs ``O(Q)``. The kernel's
    upper bound is ``O(P log J + P K + P V + P Q)``, where
    ``V = max_p V_p``.

    Parameters
    ----------
    x_q_1d : 1D array of length P
        New x-coordinates. Must be sorted ascending.
    y_q_1d : 1D array of length Q
        New y-coordinates. Must be sorted ascending.
    fx, fy : 2D arrays (J, K)
        Curvilinear grid coordinates. Both must be C-contiguous; ``fx``
        is monotone in the J axis at each k, and ``fy`` is monotone in the K
        axis at each j (the IPM property).
    fv : 2D array (J, K)
        Function values on the curvilinear mesh.
    result_2d : 2D array (P, Q)
        Output array, written in place.

    Raises
    ------
    ValueError
        If the mesh is smaller than 2 x 2.

    """
    P, Q, _J, K = regrid_query_dims(x_q_1d, y_q_1d, fx)

    # Parallel over x-queries, each running the inlined reset-per-p body.
    for p in prange(P):
        regrid_per_p_reset(x_q_1d[p], y_q_1d, fx, fy, fv, result_2d[p], K, Q)


@njit(**ENGINE_JIT)
def engine2d_regrid_serial(
    x_q_1d,
    y_q_1d,
    fx,
    fy,
    fv,
    result_2d,
) -> None:
    """Serial variant of :func:`engine2d_regrid`.

    Identical algorithm, no ``parallel=True`` decoration. Recommended
    for the EGM inner-loop call pattern: many small per-(e)-slice
    regrids where the fixed ``prange`` thread-pool dispatch cost
    (~0.5 ms per call) of the parallel variant exceeds the per-slice
    work many times over. The parallel variant remains the right choice for large
    one-shot regrids (post-processing, visualization).

    The same ``O(P log J + P K + P V + P Q)`` upper bound applies as for the
    parallel variant, just without ``prange``.
    """
    P, Q, _J, K = regrid_query_dims(x_q_1d, y_q_1d, fx)
    for p in range(P):
        regrid_per_p_reset(x_q_1d[p], y_q_1d, fx, fy, fv, result_2d[p], K, Q)


@njit(**ENGINE_JIT)
def engine2d_regrid_ipo(
    x_q_1d,
    y_q_1d,
    fx,
    fy,
    fv,
    result_2d,
) -> None:
    """``"ipo"`` regrid: state across BOTH p and q, with no per-p reset.

    Differs from :func:`engine2d_regrid_serial` in that the kernel state
    ``(j_star_k, j_star_k1, y_k, y_k1, k_current)`` is maintained across
    the outer ``p`` loop. At the start of each new p:

    1. Forward-walk ``j_star_k`` and ``j_star_k1`` for the new ``x_q``
       via :func:`advance_j_walk` without a per-p binary-search reset.
    2. If the new ``y_q[0]`` falls below the current band's lower edge,
       rewind ``k_current`` backward step-by-step until the band
       contains ``y_q[0]``. The rewind cost is bounded by how far the
       (j*, k*) bracket shifted between p iterations; in practice the
       rewind is short for smooth EGM meshes.

    Both inner walks use :func:`advance_j_walk`. State reuse can reduce
    initialization work and constants on favorable monotone paths, but the
    stated general upper bound remains ``O(P log J + P K + P V + P Q)``.
    Requires sorted ascending ``x_q_1d`` and ``y_q_1d``; for unsorted
    queries, use a variant that resets state per p.
    """
    P, Q, _J, K = regrid_query_dims(x_q_1d, y_q_1d, fx)
    if P == 0:
        return

    # Initialize state from p = 0 (one-time binary search, then walks)
    x_q = x_q_1d[0]
    k_current, j_star_k, y_k, j_star_k1, y_k1 = init_row_state(x_q, fx, fy)

    for p in range(P):
        x_q = x_q_1d[p]

        if p > 0:
            # Forward-walk j brackets for new x_q (no binary search).
            # Under monotone x_q_1d, each walk is amortized O(1).
            j_star_k = advance_j_walk(x_q, fx[:, k_current], j_star_k)
            y_k = interp_at_bracket(x_q, fx[:, k_current], fy[:, k_current], j_star_k)
            k_next = k_current + 1
            j_star_k1 = advance_j_walk(x_q, fx[:, k_next], j_star_k1)
            y_k1 = interp_at_bracket(x_q, fx[:, k_next], fy[:, k_next], j_star_k1)

            # Rewind k_current if y_q resets below current band.
            # Bounded by how far the (j*, k*) shifted between p iterations.
            y_q0 = y_q_1d[0]
            while k_current > 0 and y_q0 < y_k:
                # Demote (k_current, k_current+1) -> (k_current-1, k_current)
                k_current -= 1
                j_star_k1 = j_star_k
                y_k1 = y_k
                j_star_k = advance_j_walk(x_q, fx[:, k_current], j_star_k)
                y_k = interp_at_bracket(
                    x_q,
                    fx[:, k_current],
                    fy[:, k_current],
                    j_star_k,
                )

        # Inner q-loop: carry final state across p (no per-p reset).
        k_current, j_star_k, j_star_k1, y_k, y_k1 = regrid_q_loop(
            x_q,
            y_q_1d,
            fx,
            fy,
            fv,
            result_2d[p],
            K,
            Q,
            k_current,
            j_star_k,
            j_star_k1,
            y_k,
            y_k1,
        )


@njit(**ENGINE_JIT, parallel=True)
def log_engine2d_batch_multi(
    x_q_flat,
    y_q_flat,
    fx,
    fy,
    fv_list,
    results,
    curvature_bound: int = 2,
) -> None:
    """Probe-caching kernel for N functions (O(log J * log K) worst case):
    ``multi_query_into`` in parallel over the queries. ``fv_list`` and
    ``results`` are same-length tuples of arrays (the common EGM case is three:
    vNvrs, c, n).
    """
    for i in prange(x_q_flat.shape[0]):
        multi_query_into(
            x_q_flat[i],
            y_q_flat[i],
            fx,
            fy,
            fv_list,
            results,
            i,
            curvature_bound,
        )


# =============================================================================
# Main Interpolator Class
# =============================================================================


def validate_constructor_scalars(curvature_bound, axes_order) -> None:
    """Validate scalar args to Engine2DInterp.__init__ (raises ValueError)."""
    if not isinstance(curvature_bound, int) or curvature_bound < 1:
        msg = f"curvature_bound must be a positive integer, got {curvature_bound!r}"
        raise ValueError(
            msg,
        )
    if (
        not isinstance(axes_order, tuple)
        or len(axes_order) != 2
        or set(axes_order) != {0, 1}
    ):
        msg = f"axes_order must be (0, 1) or (1, 0), got {axes_order!r}"
        raise ValueError(
            msg,
        )


def validate_grid_shapes(fx_in, fy_in, fv_in) -> None:
    """Validate that all grid coordinate and value arrays share a shape."""
    named = [("y_values", fy_in)] + [
        (f"f_values[{i}]", fv) for i, fv in enumerate(fv_in)
    ]
    for name, arr in named:
        if arr.shape != fx_in.shape:
            msg = f"{name} shape {arr.shape} != x_values shape {fx_in.shape}"
            raise ValueError(msg)


def validate_grid_finite(fx_in, fy_in, fv_in) -> None:
    """Reject NaN/Inf in input arrays before they propagate through kernels."""
    for label, arr in [("x_values", fx_in), ("y_values", fy_in)]:
        if not np.all(np.isfinite(arr)):
            msg = f"{label} contains NaN or Inf"
            raise ValueError(msg)
    for idx, fv in enumerate(fv_in):
        if not np.all(np.isfinite(fv)):
            msg = f"f_values[{idx}] contains NaN or Inf"
            raise ValueError(msg)


def ipm_violations(x, y):
    """Counts ``(n_x, n_y)`` of steps breaking ``def-ipm``: ``x`` not strictly
    increasing along axis 0, ``y`` not strictly increasing along axis 1. A NaN
    step counts as a violation.
    """
    n_x = int(np.sum(~(np.diff(x, axis=0) > 0)))
    n_y = int(np.sum(~(np.diff(y, axis=1) > 0)))
    return n_x, n_y


def apply_axes_order(fx_in, fy_in, fv_in, axes_order):
    """Materialize C-contiguous (fx, fy, fv_arrays) honouring axes_order.

    With ``axes_order=(1, 0)`` the user's x and y are swapped, and arrays
    are transposed once at construction so all downstream kernels can
    assume ``x sweeps J, y sweeps K`` without branching.
    """
    if axes_order == (1, 0):
        fx = np.ascontiguousarray(fy_in.T)
        fy = np.ascontiguousarray(fx_in.T)
        fv_arrays = tuple(np.ascontiguousarray(fv.T) for fv in fv_in)
    else:
        fx = np.ascontiguousarray(fx_in)
        fy = np.ascontiguousarray(fy_in)
        fv_arrays = tuple(np.ascontiguousarray(fv) for fv in fv_in)
    return fx, fy, fv_arrays


class Engine2DInterp:
    """High-performance 2D interpolator for curvilinear EGM grids.

    Drop-in replacement for HARK's Curvilinear2DInterp with identical API.
    Implements the ENGINE algorithm from Section 4 of the EGM^n paper:
    sequential 1D interpolation along curvilinear grid lines (Pass 1 +
    Pass 2 in ``fig:engine_visual``), with logarithmic complexity per
    query (Section 4's complexity bound).

    Required input conditions: the grid ``(x_values, y_values)`` must
    satisfy ``def-fold-free`` and ``def-ipm`` (Index-Preserved Monotonicity).
    Section 4's bracket-displacement constant ``alpha = curvature_bound``
    sets the search window. Callers remain responsible for checking the
    supplied finite mesh.

    Complexity: O(log J * log K) worst case per scalar query. For sorted
    P x Q targets, ``regrid()`` costs O(P log J + P K + P V + P Q), where V is
    total bracket variation across rows.

    Parameters
    ----------
    f_values : list of np.ndarray
        List of (J, K) arrays of function values to interpolate.
    x_values : np.ndarray, shape (J, K)
        First coordinate of the endogenous grid (``def-ipm`` axis 0).
    y_values : np.ndarray, shape (J, K)
        Second coordinate of the endogenous grid (``def-ipm`` axis 1).
    axes_order : tuple, default ``(0, 1)``
        Memory-layout flag controlling which axis the inner kernel walks
        outer-first. ``(0, 1)`` (default) keeps the input arrays in
        their original orientation: the kernel's outer ``p`` loop sweeps
        the J (axis 0) coordinate. ``(1, 0)`` transposes the stored
        ``fx``/``fy``/``f_values`` at construction AND swaps the
        ``__call__`` query coordinates so the user-visible math is
        unchanged; the kernel ends up with the original K axis as its
        outer loop, which can improve cache locality when the EGM mesh
        is K-major. This is NOT a pure performance switch: only on an
        index-aligned mesh (one straight coordinate family) do the two
        orders return identical values modulo floating-point order of
        operations. On a warped mesh the two pass orders
        define DIFFERENT second-order interpolants (sequential 1D
        interpolation does not commute across warped coordinate lines),
        so expect O(h^2)-sized value differences in addition to the
        performance asymmetry documented in the paper's
        interpolation-order subsection. Use ``(1, 0)`` only when
        profiling shows a win AND the value difference is acceptable
        (or the mesh is index-aligned, where there is none).
    curvature_bound : int, default 2
        Section 4's bracket-displacement constant alpha:
        ``|j*(k1; x) - j*(k2; x)| <= alpha * |k1 - k2|`` for all queries x
        and row indices k1, k2. When probing row k after row k', the
        j-bracket search range is bounded by
        ``+/-(curvature_bound * |k - k'| + 1)`` indices around the
        previous bracket. This parameter affects scattered ``__call__``
        queries; regrid modes use complete linear walks and ignore it. The
        bounded search verifies containment and falls back to a full row
        search on a miss, so this parameter tunes speed, never correctness;
        set it near the measured bracket displacement to avoid the fallback
        searches.

    Raises
    ------
    ValueError
        If ``curvature_bound < 1``, the grid is smaller than 2 x 2, or
        any of ``x_values``, ``y_values``, ``f_values`` contains NaN or
        Inf (a one-pass O(JK) check, so a NaN from upstream fails at
        construction).

    Notes
    -----
    Extrapolation: queries outside the grid domain are handled by linear
    extrapolation using the two nearest boundary grid points. This can
    produce unbounded values for queries far from the grid. Callers
    should clip query points to the grid domain if bounded results are
    required.

    Examples
    --------
    >>> interp = Engine2DInterp([values, policy], x_endog, y_endog)
    >>> v_result, p_result = interp(m_query, h_query)

    """

    def __init__(
        self,
        f_values: list,
        x_values,
        y_values,
        axes_order: tuple = (0, 1),
        curvature_bound: int = 2,
    ) -> None:
        """Initialize the ENGINE interpolator."""
        validate_constructor_scalars(curvature_bound, axes_order)

        if not isinstance(f_values, list):
            f_values = [f_values]

        self.n_funcs = len(f_values)
        self.curvature_bound = curvature_bound
        self.axes_order = axes_order

        fx_in = np.asarray(x_values)
        fy_in = np.asarray(y_values)
        fv_in = [np.asarray(fv) for fv in f_values]

        validate_grid_shapes(fx_in, fy_in, fv_in)
        validate_grid_finite(fx_in, fy_in, fv_in)

        # Kernels always assume x sweeps J and y sweeps K; we transpose once
        # here instead of branching inside every kernel.
        self.fx, self.fy, self._fv_arrays = apply_axes_order(
            fx_in,
            fy_in,
            fv_in,
            axes_order,
        )
        self.J, self.K = self.fx.shape

        if self.J < 2 or self.K < 2:
            msg = (
                f"Grid must be at least 2x2 for interpolation, got "
                f"({self.J}, {self.K}) after applying axes_order={axes_order}"
            )
            raise ValueError(
                msg,
            )
        n_x, n_y = ipm_violations(self.fx, self.fy)
        if n_x or n_y:
            msg = (
                f"mesh breaks def-ipm in {n_x} x-steps and {n_y} y-steps after "
                f"applying axes_order={axes_order}"
            )
            raise ValueError(msg)

    def __call__(self, x_query, y_query):
        """Interpolate at query points using the logarithmic ENGINE algorithm.

        Worst case O(log J * log K). Handles scalar, 1D, and 2D/3D+ query
        arrays automatically, using parallelized kernels for large sets.

        Parameters
        ----------
        x_query, y_query : array-like
            Query coordinates in the user's original axis order. When
            ``axes_order=(1, 0)``, the stored arrays are transposed at
            construction; this method swaps the query coordinates so the
            kernels still see ``(x_q sweeps J, y_q sweeps K)``.

        Returns
        -------
        tuple
            Tuple of interpolated values, one per input function.

        """
        x_q = np.asarray(x_query)
        y_q = np.asarray(y_query)
        if self.axes_order == (1, 0):
            x_q, y_q = y_q, x_q

        # Scalar query: probe-caching algorithm (O(log J * log K) worst case)
        # Uses log_engine2d_scalar_multi to share row probing work across all functions
        if x_q.ndim == 0 and y_q.ndim == 0:
            result_arr = log_engine2d_scalar_multi(
                float(x_q),
                float(y_q),
                self.fx,
                self.fy,
                self._fv_arrays,
                self.curvature_bound,
            )
            return tuple(float(result_arr[i]) for i in range(self.n_funcs))

        # Array query: flatten and dispatch to logarithmic ENGINE kernel
        original_shape = x_q.shape
        n_points = x_q.size
        x_flat = np.ascontiguousarray(x_q.ravel())
        y_flat = np.ascontiguousarray(y_q.ravel())

        # A tuple of output arrays, like the values: the kernel specializes per count.
        result_arrays = tuple(
            np.empty(n_points, dtype=self.fx.dtype) for _ in range(self.n_funcs)
        )
        log_engine2d_batch_multi(
            x_flat,
            y_flat,
            self.fx,
            self.fy,
            self._fv_arrays,
            result_arrays,
            self.curvature_bound,
        )

        # Reshape and return
        return tuple(arr.reshape(original_shape) for arr in result_arrays)

    def regrid(self, x_new, y_new, mode: str = "parallel"):
        """Efficient regridding to a new rectangular grid.

        Three kernel variants are available:

        ``"parallel"``
            ``engine2d_regrid`` with ``prange`` over the outer ``p``
            loop. Per-call thread-pool dispatch (~0.5 ms fixed) is
            amortized for large one-shot regrids.
        ``"serial"``
            ``engine2d_regrid_serial``. Same algorithm without
            ``parallel=True``. Use for many small per-slice regrids
            inside a hot inner loop (e.g. EGM backward step).
        ``"ipo"``
            ``engine2d_regrid_ipo``. Like ``serial`` but also maintains
            ``(j*, k*)`` state across the outer ``p`` loop: a new ``p``
            walks the brackets on from the previous ``p`` with
            ``advance_j_walk``, the same linear walk all three modes use
            along ``q``, where the other two restart with a binary search.
            Fastest for sorted-query inner-loop regrids on smooth meshes with
            small bracket displacement.

        All three modes obey the ``O(P log J + P K + P V + P Q)`` upper bound.
        State reuse in ``"ipo"`` may reduce work on favorable monotone paths.
        ``curvature_bound`` does not affect these regrid modes.

        Parameters
        ----------
        x_new : 1D array of length P
            New x-coordinates in the user's original axis convention
            (swapped internally when ``axes_order=(1, 0)``). Must be
            sorted ascending for ``mode="ipo"``.
        y_new : 1D array of length Q
            New y-coordinates in the user's original axis convention.
            Must be sorted ascending for ``mode="ipo"``.
        mode : str, default ``"parallel"``
            Kernel selection: ``"parallel"``, ``"serial"``, or ``"ipo"``.

        Returns
        -------
        tuple of (P, Q) arrays
            Regridded function values, one per input function. The
            output shape follows the user's original (x, y) axis order;
            it is transposed back to ``(P, Q)`` after kernel execution
            when ``axes_order=(1, 0)``.

        Examples
        --------
        With ``interp`` built as in the class docstring:

        >>> v_grid, p_grid = interp.regrid(x_regular, y_regular, mode="ipo")

        """
        x_new_arr = np.ascontiguousarray(x_new)
        y_new_arr = np.ascontiguousarray(y_new)

        # Swap axes for the kernel; remember to swap output back after.
        swap = self.axes_order == (1, 0)
        if swap:
            x_kernel, y_kernel = y_new_arr, x_new_arr
        else:
            x_kernel, y_kernel = x_new_arr, y_new_arr
        P, Q = len(x_kernel), len(y_kernel)

        if mode == "parallel":
            kernel = engine2d_regrid
        elif mode == "serial":
            kernel = engine2d_regrid_serial
        elif mode == "ipo":
            kernel = engine2d_regrid_ipo
        else:
            msg = f"mode must be 'parallel', 'serial', or 'ipo', got {mode!r}"
            raise ValueError(msg)

        results = []
        for fv in self._fv_arrays:
            result = np.empty((P, Q), dtype=self.fx.dtype)
            kernel(x_kernel, y_kernel, self.fx, self.fy, fv, result)
            results.append(result.T if swap else result)

        return tuple(results)


# =============================================================================
# JIT Warmup
# =============================================================================


def warmup_jit() -> None:
    """Compile every ENGINE kernel for float64, the dtype of every caller in this
    repository, on a small (5, 6) curvilinear mesh, so a timed first call does
    not pay the 5-15 second compile. Another dtype compiles on its first call.
    The ``inline="always"`` helpers compile into the kernels that call them.
    """
    J, K = 5, 6
    x_base = np.linspace(0.1, 1.0, J)
    y_base = np.linspace(0.1, 1.0, K)
    fx = x_base[:, None] * (1.0 + 0.1 * y_base[None, :])
    fy = y_base[None, :] * (1.0 + 0.1 * x_base[:, None])
    fv = np.ones((J, K))

    x_new = np.array([0.3, 0.5, 0.7])
    regrid_result = np.empty((3, 3))
    for kernel in (engine2d_regrid, engine2d_regrid_serial, engine2d_regrid_ipo):
        kernel(x_new, x_new, fx, fy, fv, regrid_result)

    log_engine2d_scalar_multi(0.5, 0.5, fx, fy, (fv,), 2)

    fvs = (fv, 2.0 * fv, 3.0 * fv)
    outs = tuple(np.empty(3) for _ in fvs)
    log_engine2d_batch_multi(x_new, x_new, fx, fy, fvs, outs, 2)
