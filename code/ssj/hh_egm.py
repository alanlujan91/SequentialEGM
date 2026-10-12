"""EGM$^n$ two-asset household as a sequence-jacobian ``@het`` block.

A drop-in for upstream ``hetblocks.hh_twoasset.hh`` with a closed-form illiquid
stage. The backward step runs three
closed-form stages as ``@njit(parallel=True)`` kernels, with no root-finding:

- **Stage 1 (liquid)**: invert the Euler equation ``c = (W^b)^{-eis}`` with the
  kinked ``(l=0, c=0, b'=0)`` prepend, realigned to ``l_grid``.
- **Stage 0 (illiquid)**: invert ``Phi_1`` in closed form (Prop 2, replacing SSJ's
  ``lhs_equals_rhs_interpolate`` search), then ENGINE-regrid the curvilinear
  ``(b_endo, a_endo)`` mesh of ``a'`` to the regular ``(b, a)`` grid. Only ``a'``
  crosses the curvilinear mesh; consumption and ``b'`` never do.
- **Regular-grid inversion**: seed consumption from the Stage 1 policy on its own
  regular ``(l, a')`` mesh, then re-apply the same closed-form inversion
  ``c = (beta E[Vb])^{-eis}`` once at the regridded ``a'`` (``K`` passes, one by
  default). ``K`` is fixed with no convergence test, and ``a'`` comes from the
  regrid. The seed only locates the evaluation point; the delivered ``c`` is
  the inversion's output.

The kernel parallelizes over ``(e, a')`` and ``(e, b)``. The ``@het`` contract
matches upstream: ``policy=["b", "a"]``, ``backward=["Vb", "Va"]``, returns
``Va, Vb, a, b, c, uce``; ``build_l_grid`` is the one added hetinput.
"""

from __future__ import annotations

import numpy as np
from engine.core import SAFE_FASTMATH
from engine.engine2d import binary_search_bracket, engine2d_regrid_serial
from numba import njit, prange
from sequence_jacobian.blocks.het_block import het
from sequence_jacobian.hetblocks.hh_twoasset import adjustment_costs, hh_init

# Euler inversion passes (module docstring). One reaches the grid's off-grid accuracy
# (euler_passes_abrs.csv); a second only polishes the on-grid residual 1e-5 to 1e-6.
# K=0 reads c off the Stage 1 policy and fails to converge at coarse grids.
EULER_INVERSION_PASSES = 1

# Stage 0 column rule (clamp_c, clamp_wa, floor_wa): past the last endogenous node
# c and b' extrapolate linearly while the carried Va term is clamped.
PRODUCTION_RULE = (False, True, False)


def build_l_grid(b_grid, a_grid, z_grid, rb, ra, chi0, chi1, chi2):
    """Hetinput: Stage 1 extended liquid-resources grid.

    ``b_grid`` plus geometrically spaced withdrawal headroom up to the
    FOC-implied liquidity bound, so the high-illiquid rows of the Stage 1
    realign keep mesh coverage (reusing ``b_grid`` truncates that band into
    budget-infeasible policies). The node count is fixed (scaled to ``nB``, not
    the differentiated ``rb``/``ra``/``chi``/``z``), so the hetinput stays
    shape-stable under SSJ Jacobian differencing; only the node positions adapt
    to ``l_max``. ``z_grid`` is labor income, a block input in partial
    equilibrium and the ``income`` hetinput's output in general equilibrium.
    """
    if chi2 <= 1.0 or chi0 <= 0.0 or chi1 <= 0.0:
        msg = (
            f"require chi2>1, chi0>0, chi1>0; got chi2={chi2}, chi0={chi0}, chi1={chi1}"
        )
        raise ValueError(msg)
    if b_grid.shape[0] < 2:
        msg = f"b_grid needs at least 2 nodes, got {b_grid.shape[0]}"
        raise ValueError(msg)
    l_max = liquidity_bound(b_grid, a_grid, z_grid, rb, ra, chi0, chi1, chi2)
    extra = geometric_tail(b_grid, l_max, max(8, b_grid.shape[0] // 4))
    # @het names the hetinput's output from this line's text (RET504 is off here).
    l_grid = np.concatenate([b_grid, extra])
    return l_grid


def liquidity_bound(b_grid, a_grid, z_grid, rb, ra, chi0, chi1, chi2):
    """FOC-implied upper bound on liquid resources (``build_l_grid``).

    ``l_max - b_max = rb b_max + z_max + ((chi2 - 1)/chi2) u_cap ((1 + ra) a_max
    + chi0)``, whose last term is positive under ``build_l_grid``'s ``chi``
    restrictions, so ``l_max > b_max`` whenever ``rb >= 0``. Where it fails there
    is no withdrawal headroom for the fixed-length tail to cover, so it raises.
    """
    u_cap = chi1 ** (-1.0 / (chi2 - 1.0))
    l_max = (
        (1.0 + rb) * b_grid[-1]
        + float(np.max(z_grid))
        + ((chi2 - 1.0) / chi2) * u_cap * ((1.0 + ra) * a_grid[-1] + chi0)
    )
    if l_max <= b_grid[-1]:
        msg = f"liquidity bound {l_max} does not exceed b_grid[-1] = {b_grid[-1]}"
        raise ValueError(msg)
    return l_max


def geometric_tail(b_grid, l_max, n_extra):
    """``n_extra`` nodes past ``b_grid[-1]`` up to ``l_max``, geometric in
    ``l + 0.25`` (the pivot of ``sequence_jacobian.grids.agrid``).
    """
    pivot = 0.25
    ratio = ((l_max + pivot) / (b_grid[-1] + pivot)) ** (1.0 / n_extra)
    k = np.arange(1, n_extra + 1)
    return (b_grid[-1] + pivot) * ratio**k - pivot


@njit(cache=True, fastmath=SAFE_FASTMATH)
def interp1(xq, xp, fp):
    """Scalar linear interpolation with endpoint clamping (matches np.interp)."""
    n = xp.shape[0]
    if xq <= xp[0]:
        return fp[0]
    if xq >= xp[n - 1]:
        return fp[n - 1]
    lo = binary_search_bracket(xq, xp)
    w = (xq - xp[lo]) / (xp[lo + 1] - xp[lo])
    return fp[lo] + w * (fp[lo + 1] - fp[lo])


@njit(cache=True, fastmath=SAFE_FASTMATH)
def bilin(xg, yg, v, xq, yq):
    """Scalar bilinear interpolation of ``v`` on the regular ``xg x yg`` grid at
    ``(xq, yq)``, with linear extrapolation past the edges (index clamped,
    weights unclamped) to match ``RegularGridInterpolator(fill_value=None)``.
    """
    ix = binary_search_bracket(xq, xg)
    iy = binary_search_bracket(yq, yg)
    wx = (xq - xg[ix]) / (xg[ix + 1] - xg[ix])
    wy = (yq - yg[iy]) / (yg[iy + 1] - yg[iy])
    v0 = (1.0 - wy) * v[ix, iy] + wy * v[ix, iy + 1]
    v1 = (1.0 - wy) * v[ix + 1, iy] + wy * v[ix + 1, iy + 1]
    return (1.0 - wx) * v0 + wx * v1


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def stage1_extension(Va_p, Vb_p, e, j, b_grid, beta, eis):
    """Stage 1 kinked liquid EGM on one ``(e, a')`` column: the endogenous liquid
    resources ``l``, consumption, ``b'`` and discounted illiquid marginal at each
    post-decision ``b'`` node, with the ``(l=0, c=0, b'=0)`` constraint point
    prepended. Returns ``(lext, cext, bext, waext)``, each of length ``nB' + 1``.
    """
    nBp = Vb_p.shape[1]
    m = nBp + 1
    lext = np.empty(m)
    cext = np.empty(m)
    bext = np.empty(m)
    waext = np.empty(m)
    lext[0] = 0.0
    cext[0] = 0.0
    bext[0] = 0.0
    waext[0] = beta * Va_p[e, 0, j]
    for k in range(nBp):
        c = (beta * Vb_p[e, k, j]) ** (-eis)
        cext[k + 1] = c
        lext[k + 1] = b_grid[k] + c
        bext[k + 1] = b_grid[k]
        waext[k + 1] = beta * Va_p[e, k, j]
    return lext, cext, bext, waext


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def advance_bracket(lext, lq, ip):
    """Advance the sorted-target bracket ``ip`` on ``lext`` to cover ``lq`` (it
    never moves backward) and return ``(ip, pi)``, where ``pi`` is the weight on
    node ``ip``: unclamped, so it extrapolates past the last node. ``lext = b' +
    c(b')`` rises strictly when ``Vb`` falls in ``b'`` (concavity). In the
    ``Vb = +inf`` limit at ``b' = 0`` the first node repeats the prepended
    constraint point, every weight on that flat step is exact, and the weight
    is 1. On a falling step the Stage 1 inversion does not exist, so the weight
    is NaN and the solve's non-finite stop reports non-convergence.
    """
    m = lext.shape[0]
    while ip < m - 2 and lext[ip + 1] < lq:
        ip += 1
    dl = lext[ip + 1] - lext[ip]
    if dl == 0.0:
        return ip, 1.0
    if dl < 0.0:
        return ip, np.nan
    return ip, (lext[ip + 1] - lq) / dl


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def clamp01(pi):
    """Clamp an interpolation weight to ``[0, 1]`` (no extrapolation)."""
    return 0.0 if pi < 0.0 else (min(pi, 1.0))


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def psi_and_derivs(a_e, ag, Ra, chi0, chi1, chi2):
    """Adjustment cost ``Psi(a_e, a)`` of moving the illiquid balance from ``Ra a``
    to ``a_e``, and its derivatives in ``a_e`` and in ``a``. Returns ``(psi,
    psi1, psi2)``.
    """
    cexp = chi2 - 1.0
    aret = Ra * ag
    ach = a_e - aret
    aa = abs(ach)
    sgn = 1.0 if ach > 0.0 else (-1.0 if ach < 0.0 else 0.0)
    den = aret + chi0
    core = (aa / den) ** cexp
    psi = chi1 / chi2 * aa * core
    psi1 = chi1 * sgn * core
    return psi, psi1, -Ra * (psi1 + cexp * psi / den)


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def stage0_node(v1ap, cval, apj, lq, z_e, inv_eis, Ra, Rb, chi0, chi1, chi2):
    """Closed-form ``Phi_1`` inversion at one mesh node: the incoming balances
    ``(a_endo, b_endo)`` at which the choice ``a' = apj`` is optimal with liquid
    resources ``lq``, given Stage 1 consumption ``cval`` and the discounted
    illiquid marginal ``v1ap`` there.

    The FOC sets the adjustment rate ``u = (a' - Ra a)/(Ra a + chi0)``. If
    ``u <= -1``, then ``a' = Ra a (1 + u) + u chi0 < 0`` for every ``a >= 0``, so
    no balance makes ``apj`` optimal: the node's preimage is the ``u -> -1``
    limit, ``a = +inf``. Along it ``Psi - Ra a`` grows like ``(chi1/chi2 - 1) Ra
    a``, so ``b`` diverges with the sign of ``chi1 - chi2``; at ``chi1 = chi2``
    the first-order terms leave ``Psi - Ra a -> chi0 - chi2 (a' + chi0)``. With
    ``v1ap >= 0`` the rate stays above ``-chi1^{-1/(chi2-1)}``, so this needs
    ``chi1 <= 1`` or a negative carried envelope. For ``u > -1``, ``Ra a +
    chi0 = (a' + chi0)/(1 + u) > 0``, so ``Psi`` is evaluated at the exact ``a``,
    negative or not.
    """
    cexp = chi2 - 1.0
    target = v1ap * cval**inv_eis - 1.0
    if target > 0.0:
        u = (target / chi1) ** (1.0 / cexp)
    elif target < 0.0:
        u = -((-target / chi1) ** (1.0 / cexp))
    else:
        u = 0.0
    if u <= -1.0:
        if chi1 == chi2:
            return np.inf, (lq - z_e + (1.0 - chi2) * (apj + chi0)) / Rb
        return np.inf, np.inf if chi1 > chi2 else -np.inf
    ae = (apj - u * chi0) / (Ra * (1.0 + u))
    psi, _psi1, _psi2 = psi_and_derivs(apj, ae, Ra, chi0, chi1, chi2)
    return ae, (lq + apj + psi - Ra * ae - z_e) / Rb


@njit(cache=True, fastmath=SAFE_FASTMATH)
def mesh_column(Va_p, Vb_p, e, j, grids, params, rule, out) -> None:
    """One ``(e, a')`` column of the Stage 0 endogenous mesh: Stage 1 kinked liquid
    EGM, then closed-form ``Phi_1`` inversion. Writes rows ``(c, b', a_endo,
    b_endo)`` on ``l_grid`` into ``out[:, e, :, j]``. ``rule = (clamp_c, clamp_wa,
    floor_wa)`` sets how the Stage 1 weights extend past the last endogenous node
    (see ``PRODUCTION_RULE``).
    """
    a_grid, b_grid, l_grid, z_grid = grids
    beta, eis, Ra, Rb, chi0, chi1, chi2 = params
    clamp_c, clamp_wa, floor_wa = rule
    inv_eis = 1.0 / eis
    lext, cext, bext, waext = stage1_extension(Va_p, Vb_p, e, j, b_grid, beta, eis)
    ip = 0
    for il in range(l_grid.shape[0]):
        lq = l_grid[il]
        ip, pi = advance_bracket(lext, lq, ip)
        pc = clamp01(pi) if clamp_c else pi
        pw = clamp01(pi) if clamp_wa else pi
        cval = pc * cext[ip] + (1.0 - pc) * cext[ip + 1]
        v1ap = pw * waext[ip] + (1.0 - pw) * waext[ip + 1]
        if floor_wa:
            # The illiquid marginal value is nonnegative; extrapolation can cross 0.
            v1ap = max(v1ap, 0.0)
        out[0, e, il, j] = cval
        out[1, e, il, j] = pc * bext[ip] + (1.0 - pc) * bext[ip + 1]
        out[2, e, il, j], out[3, e, il, j] = stage0_node(
            v1ap,
            cval,
            a_grid[j],
            lq,
            z_grid[e],
            inv_eis,
            Ra,
            Rb,
            chi0,
            chi1,
            chi2,
        )


@njit(parallel=True, cache=True, fastmath=SAFE_FASTMATH)
def stage0_columns(Va_p, Vb_p, grids, params, rule):
    """Phase 1: ``mesh_column`` for every ``(e, a')``, in parallel. Returns the
    Stage 1 policy ``(c_s1, bp_s1)`` and the Stage 0 mesh ``(a_endo, b_endo)``,
    each indexed ``(e, l, a')``.
    """
    nE, _nBp, nAp = Va_p.shape
    out = np.empty((4, nE, grids[2].shape[0], nAp))
    for p in prange(nE * nAp):
        e = p // nAp
        j = p % nAp
        mesh_column(Va_p, Vb_p, e, j, grids, params, rule, out)
    return out[0], out[1], out[2], out[3]


@njit(parallel=True, cache=True, fastmath=SAFE_FASTMATH)
def regrid_illiquid(a_grid, b_grid, a_endo, b_endo):
    """Phase 2: ENGINE regrid of ``a'`` from the curvilinear ``(b_endo, a_endo)``
    mesh to the regular ``(b, a)`` grid, per ``e``. Only ``a'`` crosses the mesh;
    ``c`` and ``b'`` come from the Stage 1 policy on its regular ``(l, a')`` mesh.
    """
    nE, nL, nAp = a_endo.shape
    nB, nA = b_grid.shape[0], a_grid.shape[0]
    ap_choice = np.empty((nL, nAp))
    for il in range(nL):
        for j in range(nAp):
            ap_choice[il, j] = a_grid[j]
    a_pol = np.empty((nE, nB, nA))
    for e in prange(nE):
        be = np.ascontiguousarray(b_endo[e])
        ae = np.ascontiguousarray(a_endo[e])
        engine2d_regrid_serial(b_grid, a_grid, be, ae, ap_choice, a_pol[e])
    return a_pol


def stage0_policy(Va_p, Vb_p, grids, params, rule=PRODUCTION_RULE):
    """Phases 1 and 2: the Stage 1 policy ``(c_s1, bp_s1)`` on ``(e, l, a')`` and
    the regridded illiquid choice ``a_pol`` on ``(e, b, a)``.
    """
    c_s1, bp_s1, a_endo, b_endo = stage0_columns(Va_p, Vb_p, grids, params, rule)
    return c_s1, bp_s1, regrid_illiquid(grids[0], grids[1], a_endo, b_endo)


def kernel_tuples(a_grid, b_grid, l_grid, z_grid, beta, eis, rb, ra, chi0, chi1, chi2):
    """The ``(grids, params)`` tuples the kernels take, from loose arguments:
    ``grids = (a, b, l, z)`` and ``params = (beta, eis, Ra, Rb, chi0, chi1, chi2)``.
    """
    grids = (a_grid, b_grid, l_grid, z_grid)
    return grids, (beta, eis, 1.0 + ra, 1.0 + rb, chi0, chi1, chi2)


@njit(cache=True, fastmath=SAFE_FASTMATH)
def corner_costs(a_grid, Ra, chi0, chi1, chi2):
    """Corner cost ``Psi(0, a)`` at each ``a``: it depends only on ``a``, so it is
    computed once per ``a`` rather than once per ``(e, b, a)`` cell.
    """
    psi_a0 = np.empty(a_grid.shape[0])
    for ia in range(a_grid.shape[0]):
        psi_a0[ia], _psi1, _psi2 = psi_and_derivs(0.0, a_grid[ia], Ra, chi0, chi1, chi2)
    return psi_a0


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def cell_choice(e, ib, ia, a_pol, psi_a0, grids, params):
    """Illiquid choice and adjustment cost at cell ``(e, b, a)``.

    The FOC ``v1ap = v1z (1 + Psi_1)`` with ``v1ap >= 0`` admits no ``a'`` below
    ``a_star = Ra a - u_cap (Ra a + chi0)``, ``u_cap = chi1^{-1/(chi2-1)}``,
    where ``1 + Psi_1 = 0`` and the liquid resources ``l(a') = cash - a' - Psi``
    peak: below it a larger ``a'`` also leaves a larger ``l``. A regridded choice
    below the KKT floor ``max(0, a_star)`` is delivered at the floor. When the
    floor is zero it is the ``a' = 0`` corner; ``l`` then falls in ``a'`` on
    ``[0, Ra a]`` and the corner resources ``zc = l(0)`` exceed ``l(Ra a) = z + Rb b > 0``. Returns ``(corner, a', l,
    cash, psi, psi2)`` where ``l = b' + c`` is ``zc`` at the corner and ``l(a')``
    elsewhere.
    """
    a_grid, b_grid, _l_grid, z_grid = grids
    _beta, _eis, Ra, Rb, chi0, chi1, chi2 = params
    ag = a_grid[ia]
    cash = z_grid[e] + Rb * b_grid[ib] + Ra * ag
    a_floor = max(0.0, Ra * ag - chi1 ** (-1.0 / (chi2 - 1.0)) * (Ra * ag + chi0))
    apv = a_pol[e, ib, ia]
    below = apv < a_floor
    corner = below and a_floor == 0.0
    a_e = a_floor if below else apv
    psi, _psi1, psi2 = psi_and_derivs(a_e, ag, Ra, chi0, chi1, chi2)
    ell = cash - psi_a0[ia] if corner else cash - a_e - psi
    return corner, a_e, ell, cash, psi, psi2


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def stage1_seed(corner, l_avail, a_e, c_s1_e, bp_s1_e, grids):
    """Stage 1 policy read at liquid resources ``l_avail``: on the ``a' = 0``
    column at the corner, else at ``(l_avail, a')`` clamped to the feasible
    interval with the consumption floor LAST (when ``cmax < 0``, a
    resource-infeasible cell, ``c >= 0`` binds and ``b'`` absorbs the violation).
    Returns ``(c, b', cmax)``.
    """
    a_grid, b_grid, l_grid, _z_grid = grids
    if corner:
        c0 = interp1(l_avail, l_grid, c_s1_e[:, 0])
        return c0, interp1(l_avail, l_grid, bp_s1_e[:, 0]), 0.0
    cmax = l_avail - b_grid[0]
    c0 = max(min(bilin(l_grid, a_grid, c_s1_e, l_avail, a_e), cmax), 0.0)
    return c0, l_avail - c0, cmax


@njit(cache=True, fastmath=SAFE_FASTMATH, inline="always")
def euler_passes(Vb_p_e, a_grid, b_grid, l_avail, cmax, a_e, b_e, c, beta, eis, K):
    """``K`` closed-form regular-grid Euler inversions ``c = (beta E[Vb])^{-eis}``
    at fixed ``b' + c = l_avail`` and illiquid choice ``a_e``, each clamped like
    the seed. Linear extrapolation of ``Vb_p`` can reach ``wb <= 0``; the
    inversion's limit there is ``c -> +inf`` as ``wb -> 0+``, so ``cmax`` binds.
    A NaN ``wb`` (from a NaN seed) stays NaN. Returns ``(c, b')``.
    """
    for _ in range(K):
        wb = beta * bilin(b_grid, a_grid, Vb_p_e, b_e, a_e)
        c = np.inf if wb <= 0.0 else wb ** (-eis)
        c = max(min(c, cmax), 0.0)
        b_e = l_avail - c
    return c, b_e


@njit(cache=True)
def policy_arrays(nE, nB, nA):
    """One empty ``(6, e, b, a)`` block for a backward step's six outputs
    ``(Va, Vb, a, b, c, uce)``: a single array, so the per-cell ``store_cell``
    passes one reference instead of six.
    """
    return np.empty((6, nE, nB, nA))


@njit(cache=True)
def split_outputs(out):
    """The ``(Va, Vb, a, b, c, uce)`` views of a ``policy_arrays`` block."""
    return out[0], out[1], out[2], out[3], out[4], out[5]


@njit(cache=True, fastmath=SAFE_FASTMATH)
def envelope_marginals(c_env, psi2, cb, e_val, params):
    """Envelope marginals ``(Va, Vb)`` from the inversion's consumption ``c_env``,
    and marginal utility ``uce`` at the budget-delivered consumption ``cb``;
    at ``c = 0`` marginal utility is ``+inf`` (Inada).
    """
    _beta, eis, Ra, Rb, _chi0, _chi1, _chi2 = params
    inv_eis = 1.0 / eis
    v1z = c_env ** (-inv_eis)
    uce = e_val * cb ** (-inv_eis)
    return (Ra - psi2) * v1z, Rb * v1z, uce


@njit(cache=True, fastmath=SAFE_FASTMATH)
def store_cell(out, e, ib, ia, a_e, b_e, cb, c_env, psi2, e_val, params) -> None:
    """Write cell ``(e, b, a)`` of the ``policy_arrays`` block ``out``: the
    policies ``(a', b', cb)`` and the envelopes from ``c_env``.
    """
    out[0, e, ib, ia], out[1, e, ib, ia], out[5, e, ib, ia] = envelope_marginals(
        c_env,
        psi2,
        cb,
        e_val,
        params,
    )
    out[2, e, ib, ia] = a_e
    out[3, e, ib, ia] = b_e
    out[4, e, ib, ia] = cb


@njit(parallel=True, cache=True, fastmath=SAFE_FASTMATH)
def deliver(a_pol, c_s1, bp_s1, Vb_p, grids, e_grid, params, K):
    """Phase 3: corner, seeded regular-grid Euler inversion and envelopes,
    independent per ``(e, b)``. Returns ``(Va, Vb, a, b, c, uce)``.
    """
    nE, nB, nA = a_pol.shape
    a_grid, b_grid = grids[0], grids[1]
    beta, eis, Ra, _Rb, chi0, chi1, chi2 = params
    psi_a0 = corner_costs(a_grid, Ra, chi0, chi1, chi2)
    outs = policy_arrays(nE, nB, nA)
    for p in prange(nE * nB):
        e, ib = p // nB, p % nB
        for ia in range(nA):
            corner, a_e, l_avail, cash, psi, psi2 = cell_choice(
                e,
                ib,
                ia,
                a_pol,
                psi_a0,
                grids,
                params,
            )
            c_env, b_e, cmax = stage1_seed(
                corner,
                l_avail,
                a_e,
                c_s1[e],
                bp_s1[e],
                grids,
            )
            if not corner:
                c_env, b_e = euler_passes(
                    Vb_p[e],
                    a_grid,
                    b_grid,
                    l_avail,
                    cmax,
                    a_e,
                    b_e,
                    c_env,
                    beta,
                    eis,
                    K,
                )
            cb = cash - psi - a_e - b_e
            store_cell(outs, e, ib, ia, a_e, b_e, cb, c_env, psi2, e_grid[e], params)
    return split_outputs(outs)


def backward_step(
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
    K,
    rule=PRODUCTION_RULE,
):
    """EGM$^n$ backward step: Stage 1 + Stage 0 mesh (Phase 1), ENGINE regrid of
    ``a'`` (Phase 2), then the corner and seeded Euler inversion (Phase 3), each a
    parallel kernel. ``rule`` is the Stage 0 column rule (``PRODUCTION_RULE``).
    Returns ``(Va, Vb, a, b, c, uce)``.
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
    c_s1, bp_s1, a_pol = stage0_policy(Va_p, Vb_p, grids, params, rule)
    return deliver(a_pol, c_s1, bp_s1, Vb_p, grids, e_grid, params, K)


def two_asset_het(backward_fun, last_arg, name, doc):
    """``backward_fun(..., last_arg)`` as a block under upstream's two-asset
    ``@het`` contract plus the ``build_l_grid`` hetinput.

    ``@het`` reads the block's inputs from its signature and its outputs from the
    text of its return statement, so the wrapper spells out every argument and
    returns named locals; ``name`` and ``doc`` label the block.
    """

    def block(
        Va_p,
        Vb_p,
        a_grid,
        b_grid,
        z_grid,
        e_grid,
        l_grid,
        beta,
        eis,
        rb,
        ra,
        chi0,
        chi1,
        chi2,
    ):
        Va, Vb, a, b, c, uce = backward_fun(
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
            last_arg,
        )
        return Va, Vb, a, b, c, uce

    block.__name__ = block.__qualname__ = name
    block.__doc__ = doc
    return het(
        exogenous="Pi",
        policy=["b", "a"],
        backward=["Vb", "Va"],
        hetinputs=[build_l_grid],
        hetoutputs=[adjustment_costs],
        backward_init=hh_init,
    )(block)


hh = two_asset_het(
    backward_step,
    EULER_INVERSION_PASSES,
    "hh",
    """EGM$^n$ two-asset backward step: closed-form inversions + ENGINE regrid of
    the illiquid choice + one seeded regular-grid Euler inversion, all in one
    parallel kernel. No root-finding. See the module docstring for the method.
    """,
)
