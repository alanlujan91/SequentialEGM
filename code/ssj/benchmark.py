"""Benchmark harness: sequence-jacobian ``hh_twoasset``, EGM$^n$ ``hh`` and NEGM.

All three households are driven through their raw ``@het`` backward functions (no
hand-copied bodies), iterated to a stationary household steady state at fixed
prices on matched grids and matched linear interpolation. The module supplies the
calibration, the solvers, the policy-distance and robust Euler instruments, and
the Stage 0 mesh checks that the ``ssj.bench_*`` scripts report.
"""

from __future__ import annotations

import numpy as np
from engine.utilities import corner_jacobians, ipm_violations
from numerics import median_p95_max, rel_diff
from scipy.interpolate import RegularGridInterpolator
from sequence_jacobian import grids
from sequence_jacobian.hetblocks import hh_twoasset as _ssj

from ssj import hh_egm, hh_negm

# Convergence tolerance and iteration cap of every published solve: the defaults
# of every solver below.
SOLVE_TOL = 1e-9
MAXIT = 8000

# Fixed-price calibration (sequence_jacobian/examples/two_asset baseline).
_CAL = {
    "eis": 0.5,
    "beta": 0.976,
    "rb": 0.0075,
    "ra": 0.0125,
    "chi0": 0.25,
    "chi1": 6.5,
    "chi2": 2.0,
    "rho_z": 0.966,
    "sigma_z": 0.92,
    "bmax": 50.0,
    "amax": 4000.0,
    "nK": 4,
}

_PRICES = ("beta", "eis", "rb", "ra", "chi0", "chi1", "chi2")

_SHIPPED = _ssj.hh.backward_fun.f
_COMPACT = hh_egm.hh.backward_fun.f
_NEGM = hh_negm.hh_negm.backward_fun.f


def calibration(nB, nA, nZ=3):
    """Matched grids + fixed prices for a partial-equilibrium household solve."""
    e, _, Pi = grids.markov_rouwenhorst(rho=_CAL["rho_z"], sigma=_CAL["sigma_z"], N=nZ)
    b = grids.agrid(amax=_CAL["bmax"], n=nB)
    a = grids.agrid(amax=_CAL["amax"], n=nA)
    return {"e": e, "Pi": Pi, "b": b, "a": a, "z": e, **_CAL}


def l_grid_for(cal):
    """``build_l_grid`` at the calibration's grids and prices."""
    return hh_egm.build_l_grid(
        cal["b"],
        cal["a"],
        cal["z"],
        cal["rb"],
        cal["ra"],
        cal["chi0"],
        cal["chi1"],
        cal["chi2"],
    )


def backward_args(cal, l_grid=None):
    """The arguments after ``(Va_p, Vb_p)`` of ``hh_egm.backward`` and every kernel
    sharing its signature: ``(a, b, l, z, e, beta, eis, rb, ra, chi0, chi1, chi2)``,
    on ``l_grid`` when given, else on ``l_grid_for(cal)``.
    """
    return (
        cal["a"],
        cal["b"],
        l_grid_for(cal) if l_grid is None else l_grid,
        cal["z"],
        cal["e"],
        *(cal[k] for k in _PRICES),
    )


def kernel_inputs(cal, l_grid=None):
    """The kernels' ``(grids, params)`` tuples at a calibration (``hh_egm.
    kernel_tuples``), on ``l_grid`` when given, else on ``l_grid_for(cal)``.
    """
    l_grid = l_grid_for(cal) if l_grid is None else l_grid
    return hh_egm.kernel_tuples(
        cal["a"],
        cal["b"],
        l_grid,
        cal["z"],
        *(cal[k] for k in _PRICES),
    )


def stage_inversions(Va_p, Vb_p, cal, l_grid):
    """Phase 1 of the EGM$^n$ kernel at a calibration: the Stage 1 policy and the
    Stage 0 mesh, ``(c_s1, bp_s1, a_endo, b_endo)`` on ``(e, l, a')``.
    """
    grids, params = kernel_inputs(cal, l_grid)
    return hh_egm.stage0_columns(Va_p, Vb_p, grids, params, hh_egm.PRODUCTION_RULE)


def regular_grid_inversion(a_pol, c_s1, bp_s1, Vb_p, cal, l_grid):
    """Phase 3 of the EGM$^n$ kernel at a calibration and an illiquid choice
    ``a_pol``: the ``a' = 0`` corner, the seeded regular-grid Euler inversion and
    the envelopes. Returns ``(Va, Vb, a, b, c, uce)``.
    """
    grids, params = kernel_inputs(cal, l_grid)
    return hh_egm.deliver(
        a_pol,
        c_s1,
        bp_s1,
        Vb_p,
        grids,
        cal["e"],
        params,
        hh_egm.EULER_INVERSION_PASSES,
    )


def expect(Pi, V):
    """Markov expectation ``Pi @ V`` over the leading (productivity) axis."""
    return (Pi @ V.reshape(Pi.shape[0], -1)).reshape(V.shape)


def marginal_change(Va_n, Va, Vb_n, Vb):
    """Larger of the two marginals' sup-norm changes, each relative to ``1 +``
    the previous sup norm; every solver here converges on this metric.
    """
    return max(
        np.max(np.abs(Va_n - Va)) / (1.0 + np.max(np.abs(Va))),
        np.max(np.abs(Vb_n - Vb)) / (1.0 + np.max(np.abs(Vb))),
    )


def initial_marginals(cal):
    """Upstream ``hh_init`` marginals ``(Va, Vb)``: the start of every solve."""
    return _ssj.hh_init(cal["b"], cal["a"], cal["z"], cal["eis"])


def iterate(step, cal, tol=SOLVE_TOL, maxit=MAXIT):
    """Iterate ``step(Va_p, Vb_p) -> (Va, Vb, a, b, c)`` from upstream's
    ``hh_init`` until the relative change in both marginals falls below ``tol``;
    stops early on a non-finite update.
    """
    Va, Vb = initial_marginals(cal)
    a = b = c = None
    it = 0
    converged = False
    for it in range(1, maxit + 1):
        Va_n, Vb_n, a, b, c = step(expect(cal["Pi"], Va), expect(cal["Pi"], Vb))
        d = marginal_change(Va_n, Va, Vb_n, Vb)
        Va, Vb = Va_n, Vb_n
        if d < tol:
            converged = True
            break
        if not np.isfinite(d):
            # A non-finite distance never recovers under further iteration.
            break
    return {
        "a": a,
        "b": b,
        "c": c,
        "Va": Va,
        "Vb": Vb,
        "iter": it,
        "converged": converged,
    }


def require_converged(ok, msg) -> None:
    """Raise ``RuntimeError(msg)`` unless ``ok``: no unconverged solve may reach a CSV."""
    if not ok:
        raise RuntimeError(msg)


def solve_backward(
    cal,
    backward,
    *args,
    tol=SOLVE_TOL,
    maxit=MAXIT,
    require=True,
    **kwargs,
):
    """``iterate`` a raw backward function ``backward(Va_p, Vb_p, *args,
    **kwargs)`` whose first five outputs are ``(Va, Vb, a, b, c)``. Raises unless
    the solve converges when ``require``; a caller that records the failure
    itself passes ``require=False``.
    """

    def step(Va_p, Vb_p):
        return backward(Va_p, Vb_p, *args, **kwargs)[:5]

    out = iterate(step, cal, tol=tol, maxit=maxit)
    if require:
        require_converged(
            out["converged"],
            f"backward iteration failed to converge at grid "
            f"({len(cal['b'])}, {len(cal['a'])}) after {out['iter']} iterations",
        )
    return out


def solve_shipped(cal, tol=SOLVE_TOL):
    """Upstream ``hh_twoasset`` via its raw backward function (no hand-copy)."""
    return solve_block(_SHIPPED, cal, tol=tol, **shipped_extras(cal))


def shipped_extras(cal):
    """Upstream ``hh_twoasset``'s extra backward inputs: its ``k_grid`` of
    adjustment-cost shares and the marginal-cost grid ``Psi1``.
    """
    return {
        "k_grid": grids.agrid(amax=1.0, n=cal["nK"])[::-1].copy(),
        "Psi1": _ssj.marginal_cost_grid(
            cal["a"],
            cal["ra"],
            cal["chi0"],
            cal["chi1"],
            cal["chi2"],
        ),
    }


def solve_block(backward_fun, cal, tol=SOLVE_TOL, **extra):
    """Iterate a two-asset ``@het`` block's raw backward function to its fixed
    point. The blocks share the upstream keyword signature; ``extra`` carries the
    block-specific grids (``k_grid``/``Psi1`` upstream), and blocks that take an
    ``l_grid`` hetinput (EGM$^n$ ``hh``, NEGM ``hh_negm``) get it by default.
    """
    kwargs = {
        "a_grid": cal["a"],
        "b_grid": cal["b"],
        "z_grid": cal["z"],
        "e_grid": cal["e"],
        **{k: cal[k] for k in _PRICES},
        **(extra or {"l_grid": l_grid_for(cal)}),
    }

    return solve_backward(cal, backward_fun, tol=tol, **kwargs)


def solve_compact(cal, tol=SOLVE_TOL):
    """Compact EGM$^n$ ``hh`` via its raw backward function."""
    return solve_block(_COMPACT, cal, tol=tol)


def solve_negm(cal, tol=SOLVE_TOL):
    """NEGM baseline ``hh_negm`` via its raw backward function (root-find on a')."""
    return solve_block(_NEGM, cal, tol=tol)


def node_mesh(cal):
    """The solver's regular ``(b, a)`` nodes as ``ij``-indexed meshes ``(B, A)``."""
    return np.meshgrid(cal["b"], cal["a"], indexing="ij")


def cash_on_hand(cal, B, A, ie):
    """Cash on hand ``(1 + r_b) b + (1 + r_a) a + z`` at balances ``(B, A)`` in
    income state ``ie``.
    """
    return (1.0 + cal["rb"]) * B + (1.0 + cal["ra"]) * A + cal["z"][ie]


def policy_rel_diffs(out, cal, ref_out, ref_cal, pts, key="c"):
    """``rel_diff`` of ``out[key]`` against ``ref_out[key]``, each linearly
    interpolated on its own grid at the ``(b, a)`` rows of ``pts``, pooled over
    every income state.
    """
    return np.concatenate(
        [
            rel_diff(
                grid_interp(out[key][ie], cal)(pts),
                grid_interp(ref_out[key][ie], ref_cal)(pts),
            )
            for ie in range(cal["Pi"].shape[0])
        ],
    )


def policy_distance(out, cal, ref_out, ref_cal, key):
    """Median relative diff of ``out[key]`` vs ``ref_out[key]`` at cal's grid
    nodes, the reference interpolated there unless it shares cal's grid.
    """
    if ref_cal is cal:
        return float(np.median(rel_diff(out[key], ref_out[key])))
    B, A = node_mesh(cal)
    pts = np.column_stack([B.ravel(), A.ravel()])
    return float(np.median(policy_rel_diffs(out, cal, ref_out, ref_cal, pts, key)))


def robust_euler_stats(out, cal):
    """Median, p95, max of the robust (constraint-valid) consumption-unit Euler
    error at the policy, evaluated at the solver's own regular grid nodes
    (on-grid; the off-grid family lives in ``ssj.bench_interp``).
    """
    return median_p95_max(np.abs(liquid_euler_gap(out, cal, out["Vb"])).ravel())


def grid_interp(values, cal):
    """Linear interpolant of ``values`` on ``cal``'s regular ``(b, a)`` grid,
    extrapolating linearly past its edges.
    """
    return RegularGridInterpolator(
        (cal["b"], cal["a"]),
        values,
        bounds_error=False,
        fill_value=None,
    )


def continuation_at_policy(out, cal, marginal, query=None):
    """Yield, per productivity state, the discounted expected marginal
    ``beta E[marginal](b', a')`` at the policy (``marginal`` is ``Va`` or ``Vb`` on
    the grid, before the Markov mix), with the state's ``a'`` and ``c`` policies.
    The policies are those at the solver's nodes, or with ``query = (bq, aq)``
    the policies linearly interpolated at those points.
    """
    V_e = np.einsum("ef,fij->eij", cal["Pi"], marginal)
    for ie, V in enumerate(V_e):
        if query is None:
            ap, bp, c = out["a"][ie], out["b"][ie], out["c"][ie]
        else:
            pts = np.column_stack(query)
            ap, bp, c = (grid_interp(out[k][ie], cal)(pts) for k in "abc")
        at = np.column_stack([bp.ravel(), ap.ravel()])
        yield cal["beta"] * grid_interp(V, cal)(at).reshape(bp.shape), ap, c


def euler_consumption(W, cal):
    """Consumption ``W^(-eis)`` the Euler equation implies from a marginal ``W``."""
    return W ** (-cal["eis"])


def constrained_euler_consumption(Wb, m, ap, psi, cal):
    """Consumption the liquid Euler equation implies from the continuation ``Wb``,
    capped by the borrowing-constraint branch ``m - a' - Psi - b_min``: the
    min-at-constraint geometry every liquid Euler instrument scores against.
    """
    return np.minimum(euler_consumption(Wb, cal), m - ap - psi - cal["b"][0])


def liquid_euler_gap(out, cal, marginal, query=None):
    """Signed relative liquid-FOC error ``(c - c_implied)/|c|`` per income state
    at each grid node, or at the points ``query = (bq, aq)`` from the linearly
    interpolated policies, with ``c_implied`` from the continuation ``marginal``
    at the policy.
    """
    B, A = node_mesh(cal) if query is None else query
    r = []
    states = continuation_at_policy(out, cal, marginal, query)
    for ie, (W, ap, c) in enumerate(states):
        m = cash_on_hand(cal, B, A, ie)
        psi = adjustment_cost(ap, A, cal)[0]
        r.append((c - constrained_euler_consumption(W, m, ap, psi, cal)) / np.abs(c))
    return np.stack(r)


def adjustment_cost(ap, A, cal):
    """Upstream's ``(Psi, Psi_1)`` at choice ``ap`` from illiquid balance ``A``."""
    return _ssj.get_Psi_and_deriv(
        ap,
        A,
        cal["ra"],
        cal["chi0"],
        cal["chi1"],
        cal["chi2"],
    )


def budget_violation(out, cal):
    """Absolute budget-identity residual ``|c + b' + a' + Psi - m|`` at every node."""
    B, A = node_mesh(cal)
    res = np.empty_like(out["c"])
    for ie, (ap, bp, c) in enumerate(zip(out["a"], out["b"], out["c"], strict=True)):
        psi = adjustment_cost(ap, A, cal)[0]
        res[ie] = np.abs(c + bp + ap + psi - cash_on_hand(cal, B, A, ie))
    return res


def stage0_mesh(out, cal):
    """Stage 0 endogenous ``(b, a)`` mesh + consumption at the converged marginals,
    on the ``(e, l, a')`` grid: the grid-transformation figure's data. Re-derives the
    mesh from the compact block's Phase 1, the exact arithmetic ``solve_compact`` runs.
    """
    l_grid = l_grid_for(cal)
    c, _bp, a_endo, b_endo = stage_inversions(
        expect(cal["Pi"], out["Va"]),
        expect(cal["Pi"], out["Vb"]),
        cal,
        l_grid,
    )
    return {
        "l_grid": l_grid,
        "a_grid": cal["a"],
        "b_endo": b_endo,
        "a_endo": a_endo,
        "c": c,
    }


def assert_mesh_fold_free(b_endo, a_endo) -> None:
    """Raise unless every Stage 0 mesh corner has a positive Jacobian.

    The grid-transformation figure's caption claims mesh lines of constant ``l``
    and constant ``a'`` never cross; a folded cell would falsify it. Checks all
    four corner Jacobians of every cell in every productivity slice and raises
    ``RuntimeError`` (not ``assert``, which ``-O`` strips) on a fold, degenerate
    cell, or reversed orientation.
    """
    for ie in range(b_endo.shape[0]):
        det_all = corner_jacobians(b_endo[ie], a_endo[ie])
        if not np.all(det_all > 0):
            n_bad = int(np.sum(det_all <= 0))
            msg = (
                f"Stage 0 mesh has invalid orientation at productivity slice {ie}: "
                f"{n_bad} corner Jacobians are non-positive."
            )
            raise RuntimeError(
                msg,
            )


def assert_mesh_ipm(b_endo, a_endo) -> None:
    """Raise unless the Stage 0 mesh is index-monotone (IPM), slice by slice.

    Index-Preserved Monotonicity is ENGINE's second requirement beside
    fold-freeness: each endogenous coordinate must increase strictly along its
    own exogenous index, ``b`` along the ``l`` axis and ``a`` along the ``a'``
    axis. Checks every step of every line in every productivity slice and
    raises ``RuntimeError`` (not ``assert``, which ``-O`` strips) on any
    step that does not strictly increase, a NaN step included.
    """
    for ie in range(b_endo.shape[0]):
        n_b, n_a = ipm_violations(b_endo[ie], a_endo[ie])
        if n_b or n_a:
            msg = (
                f"Stage 0 mesh violates IPM at productivity slice {ie}: "
                f"{n_b} not strictly increasing steps in the liquid coordinate along its "
                f"own axis, {n_a} in the illiquid coordinate along its own axis."
            )
            raise RuntimeError(msg)
