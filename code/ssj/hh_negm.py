"""NEGM two-asset household ``@het`` block: the nested-EGM baseline EGM$^n$ replaces.

Drop-in for the same two-asset household as ``ssj.hh_egm.hh``, differing in ONE
component: where EGM$^n$ inverts the illiquid FOC in closed form (Prop 2), NEGM
solves it by a per-node safeguarded root-find on ``a'`` over the regular ``(b, a)``
grid (Illinois on the ratio FOC ``v1ap/v1z - (1+Psi_1)``), with interpolated Stage
1 envelopes inside each iteration. This is the "one inversion plus one search"
baseline; the root-find is intentional here, it is the comparator.

Stage 1, the constrained segment, and the ``a'=0`` corner are shared with EGM$^n$.
NEGM solves directly on ``(b, a)``, so it skips the regrid and the Euler correction.
"""

from __future__ import annotations

import numpy as np
from engine.core import SAFE_FASTMATH
from numba import njit, prange

from ssj.hh_egm import (
    advance_bracket,
    bilin,
    clamp01,
    kernel_tuples,
    policy_arrays,
    psi_and_derivs,
    split_outputs,
    stage1_extension,
    store_cell,
    two_asset_het,
)

# Relative tolerance of the Illinois bracket in the published comparator.
NEGM_INNER_REL_TOL = 1e-11


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def foc(ap, ag, m, v1z_e, v1ap_e, l_grid, a_grid, Ra, chi0, chi1, chi2):
    """Illiquid FOC residual in ratio form, ``F(a') = v1ap/v1z - (1 + Psi_1)``,
    evaluated at ``(ell(a'), a')`` with ``ell = m - a' - Psi(a', a)``. F is
    strictly decreasing in ``a'``. The ratio normalization keeps F order-one where
    v1z explodes (``c -> 0``); where liquid is exhausted, ``F -> -(1+Psi_1)``,
    carrying the correct sign on both the deposit and deep-withdrawal sides.
    Returns ``(F, ell)``.
    """
    psi, psi1, _psi2 = psi_and_derivs(ap, ag, Ra, chi0, chi1, chi2)
    s = 1.0 + psi1
    ell = m - ap - psi
    if ell <= l_grid[0]:
        return -s, ell
    # Near ``l_grid[0]`` the interpolant takes in the ``v1z = +inf`` node (c = 0),
    # where the ratio's limit is ``F = -s``. There ``0 * inf`` reads as NaN.
    vz = bilin(l_grid, a_grid, v1z_e, ell, ap)
    if not (vz > 0.0) or vz == np.inf:
        return -s, ell
    vap = bilin(l_grid, a_grid, v1ap_e, ell, ap)
    f = vap / vz - s
    if not np.isfinite(f):
        return -s, ell
    return f, ell


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def feasible_upper(a0, ag, m, aN, Ra, chi0, chi1, chi2):
    """Largest ``a'`` in ``[a0, aN]`` with positive liquid resources ``ell``.
    Anchored at the no-adjustment point ``Ra*a`` (liquid there is ``z+Rb*b>0``, and
    ``ell`` is decreasing beyond it). When ``Ra*a`` lies above the grid top ``aN``
    the anchor is unusable: if ``aN`` is feasible the guard below returns it, else
    the bracket would collapse onto an unchecked ``aN``, so we search down from
    ``a0`` instead (the caller's FOC guard then handles a fully infeasible top).
    """
    if m - aN - psi_and_derivs(aN, ag, Ra, chi0, chi1, chi2)[0] > 0.0:
        return aN
    lo = min(Ra * ag, aN)
    if lo >= aN:
        lo = a0
    hi = aN
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if m - mid - psi_and_derivs(mid, ag, Ra, chi0, chi1, chi2)[0] > 0.0:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def illinois(
    x0,
    f0,
    x1,
    f1,
    ag,
    m,
    v1z_e,
    v1ap_e,
    l_grid,
    a_grid,
    Ra,
    chi0,
    chi1,
    chi2,
    tol,
):
    """Illinois (modified regula-falsi) on the FOC root bracketed by ``(x0, x1)``
    with opposite-sign ``(f0, f1)``. Halving the stale endpoint's value prevents
    the one-sided stalls of plain regula-falsi.
    """
    for _ in range(100):
        if abs(x1 - x0) < tol:
            break
        denom = f1 - f0
        if denom == 0.0:
            break
        x2 = x1 - f1 * (x1 - x0) / denom
        lo = min(x1, x0)
        hi = x0 + x1 - lo
        if x2 <= lo or x2 >= hi:
            x2 = 0.5 * (lo + hi)
        f2, _ = foc(x2, ag, m, v1z_e, v1ap_e, l_grid, a_grid, Ra, chi0, chi1, chi2)
        if f2 == 0.0:
            return x2
        if (f2 > 0.0) == (f1 > 0.0):
            x1, f1 = x2, f2
            f0 *= 0.5
        else:
            x0, f0 = x1, f1
            x1, f1 = x2, f2
    return 0.5 * (x0 + x1)


@njit(parallel=True, cache=True, fastmath=SAFE_FASTMATH)
def negm_stage1(Va_p, Vb_p, b_grid, l_grid, beta, eis):
    """Stage 1 consumption EGM, the step EGM$^n$ runs before its illiquid
    inversion. Returns consumption, its marginal utility and the discounted
    illiquid marginal, each on ``(e, l, a')``.
    """
    nE, _nBp, nAp = Va_p.shape
    nL = l_grid.shape[0]
    inv_eis = 1.0 / eis
    c_s1 = np.empty((nE, nL, nAp))
    v1z = np.empty((nE, nL, nAp))
    v1ap = np.empty((nE, nL, nAp))
    for p in prange(nE * nAp):
        e = p // nAp
        j = p % nAp
        lext, cext, _bext, waext = stage1_extension(Va_p, Vb_p, e, j, b_grid, beta, eis)
        ip = 0
        for il in range(nL):
            ip, pi = advance_bracket(lext, l_grid[il], ip)
            cval = pi * cext[ip] + (1.0 - pi) * cext[ip + 1]
            c_s1[e, il, j] = cval
            # u'(c); +inf at the l = 0 constraint node, where c = 0.
            v1z[e, il, j] = cval ** (-inv_eis)
            pcl = clamp01(pi)
            v1ap[e, il, j] = pcl * waext[ip] + (1.0 - pcl) * waext[ip + 1]
    return c_s1, v1z, v1ap


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def negm_choice(
    ag,
    mres,
    v1z_e,
    v1ap_e,
    l_grid,
    a_grid,
    Ra,
    chi0,
    chi1,
    chi2,
    inner_rel_tol,
):
    """Safeguarded root-find on ``a'`` at one ``(b, a)`` node with cash on hand
    ``mres`` (the NEGM step). F is decreasing: ``F(a_grid[0]) <= 0`` is the
    ``a' = 0`` KKT corner, ``F(a_hi) >= 0`` the upper edge.
    """
    a_lo = a_grid[0]
    f_lo, _ = foc(a_lo, ag, mres, v1z_e, v1ap_e, l_grid, a_grid, Ra, chi0, chi1, chi2)
    if f_lo <= 0.0:
        return a_lo
    a_hi = feasible_upper(a_lo, ag, mres, a_grid[-1], Ra, chi0, chi1, chi2)
    f_hi, _ = foc(a_hi, ag, mres, v1z_e, v1ap_e, l_grid, a_grid, Ra, chi0, chi1, chi2)
    if f_hi >= 0.0:
        return a_hi
    tol = inner_rel_tol * (1.0 + a_hi)
    return illinois(
        a_lo,
        f_lo,
        a_hi,
        f_hi,
        ag,
        mres,
        v1z_e,
        v1ap_e,
        l_grid,
        a_grid,
        Ra,
        chi0,
        chi1,
        chi2,
        tol,
    )


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def negm_cell(a_e, ag, mres, c_s1_e, grids, params):
    """NEGM delivery at the illiquid choice ``a_e``: consumption read off Stage 1's
    policy at the liquid resources left after ``a_e``. Returns ``(c, b', cb,
    psi2)`` with ``cb`` the budget-delivered consumption and ``psi2`` the
    ``a``-derivative of the adjustment cost.
    """
    a_grid, b_grid, l_grid, _z_grid = grids
    _beta, _eis, Ra, _Rb, chi0, chi1, chi2 = params
    psi, _psi1, psi2 = psi_and_derivs(a_e, ag, Ra, chi0, chi1, chi2)
    ell = mres - a_e - psi
    c_e = max(bilin(l_grid, a_grid, c_s1_e, ell, a_e), 0.0)
    b_e = max(ell - c_e, b_grid[0])
    return c_e, b_e, mres - psi - a_e - b_e, psi2


@njit(parallel=True, cache=True, fastmath=SAFE_FASTMATH)
def negm_deliver(c_s1, v1z, v1ap, grids, e_grid, params, inner_rel_tol):
    """Stage 0 per ``(e, b)``: root-find ``a'`` at each node, then read consumption
    and the envelopes off Stage 1. Returns ``(Va, Vb, a, b, c, uce)``.
    """
    a_grid, b_grid, l_grid, z_grid = grids
    _beta, _eis, Ra, Rb, chi0, chi1, chi2 = params
    nE, nB, nA = c_s1.shape[0], b_grid.shape[0], a_grid.shape[0]
    outs = policy_arrays(nE, nB, nA)
    for p in prange(nE * nB):
        e, ib = p // nB, p % nB
        for ia in range(nA):
            ag = a_grid[ia]
            mres = z_grid[e] + Rb * b_grid[ib] + Ra * ag
            a_e = negm_choice(
                ag,
                mres,
                v1z[e],
                v1ap[e],
                l_grid,
                a_grid,
                Ra,
                chi0,
                chi1,
                chi2,
                inner_rel_tol,
            )
            c_e, b_e, cb, psi2 = negm_cell(a_e, ag, mres, c_s1[e], grids, params)
            store_cell(outs, e, ib, ia, a_e, b_e, cb, c_e, psi2, e_grid[e], params)
    return split_outputs(outs)


def backward_negm(
    Va_p,
    Vb_p,
    a_grid,
    b_grid,
    l_grid,
    z_grid,
    e_grid,
    beta,
    eis,
    rb,
    ra,
    chi0,
    chi1,
    chi2,
    inner_rel_tol,
):
    """NEGM backward step: Stage 1 EGM, then the per-node illiquid root-find.

    ``inner_rel_tol`` is the Illinois bracket's relative tolerance; the published
    comparator passes ``NEGM_INNER_REL_TOL``, and the robustness module
    ``ssj.bench_negm_innertol`` re-times the solve at ``1e-9``.
    """
    grids, params = kernel_tuples(
        a_grid,
        b_grid,
        l_grid,
        z_grid,
        beta,
        eis,
        rb,
        ra,
        chi0,
        chi1,
        chi2,
    )
    c_s1, v1z, v1ap = negm_stage1(Va_p, Vb_p, b_grid, l_grid, beta, eis)
    return negm_deliver(c_s1, v1z, v1ap, grids, e_grid, params, inner_rel_tol)


hh_negm = two_asset_het(
    backward_negm,
    NEGM_INNER_REL_TOL,
    "hh_negm",
    """NEGM two-asset backward step: Stage 1 EGM shared with EGM$^n$, illiquid step
    by a safeguarded root-find on a' (the search EGM$^n$ removes).
    """,
)
