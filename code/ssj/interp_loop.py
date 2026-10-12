"""In-loop interpolator comparison on the compact EGM$^n$ two-asset block.

Drives the SAME EGM$^n$ backward iteration as ``ssj.hh_egm`` (Stage 1 kinked EGM,
the closed-form ``Phi_1`` inversion, and the corner-plus-Euler-correction finish),
swapping ONLY the Stage-0 realign primitive:

- ``"engine"``  : ENGINE (``engine.engine2d``), the production realign;
- ``"curvilinear"``: the White (2015) sector-walking interpolant (HARK
  ``Curvilinear2DInterp``);
- ``"delaunay"`` : the Ludwig-Schoenle (2018) triangulation
  (``engine.utilities.Delaunay2DInterp``, the health comparator's class), linear
  off its hull from the nearest boundary triangle.

The stage inversions and the regular-grid inversion call the production kernel's
own phases (``benchmark.stage_inversions`` and ``benchmark.regular_grid_inversion``), and
``test_interp_loop`` pins the ENGINE configuration to
``ssj.benchmark.solve_compact``. Only ``a'`` crosses the curvilinear mesh
(matching the kernel), so the realign under comparison is the single field the
method actually asks of the interpolator. Used by ``ssj.bench_interp``.
"""

from __future__ import annotations

import numpy as np
from engine.engine2d import engine2d_regrid_serial
from engine.utilities import Delaunay2DInterp
from HARK.interpolation import Curvilinear2DInterp

from ssj.benchmark import (
    MAXIT,
    SOLVE_TOL,
    expect,
    initial_marginals,
    l_grid_for,
    marginal_change,
    regular_grid_inversion,
    stage_inversions,
)


def a_choice_mesh(cal, l_grid):
    """The Stage 0 mesh's ``a'`` values, ``a_grid`` repeated over ``l_grid``."""
    return np.ascontiguousarray(
        np.broadcast_to(cal["a"][None, :], (l_grid.shape[0], cal["a"].shape[0])),
    )


def initial_state(cal):
    """Start of a hand-rolled backward iteration: the initial marginals
    ``(Va, Vb)``, the liquid grid and the Stage 0 choice mesh.
    """
    l_grid = l_grid_for(cal)
    return *initial_marginals(cal), l_grid, a_choice_mesh(cal, l_grid)


def regrid_with(interp, b_endo, a_endo, ap_choice, b_grid, a_grid):
    """Realign ``a'`` from the curvilinear ``(b_endo, a_endo)`` mesh to the regular
    ``(b, a)`` grid with the chosen interpolator. Returns ``a_pol``, the one field
    the backward step asks of the interpolator (c and b' come from the Stage 1
    policy on its regular mesh).
    """
    nE = b_endo.shape[0]
    nB, nA = b_grid.shape[0], a_grid.shape[0]
    a_pol = np.empty((nE, nB, nA))
    if interp == "engine":
        for e in range(nE):
            be = np.ascontiguousarray(b_endo[e])
            ae = np.ascontiguousarray(a_endo[e])
            engine2d_regrid_serial(b_grid, a_grid, be, ae, ap_choice, a_pol[e])
        return a_pol
    if interp == "engine_reversed":
        # Pass 1 walks the warped illiquid axis first (every array transposed); on a
        # warped mesh this is not a no-op, the two pass orders define different
        # interpolants. Section 4's order subsection benchmarks the cost and consequence.
        ta = np.empty((nA, nB))
        for e in range(nE):
            be_t = np.ascontiguousarray(b_endo[e].T)
            ae_t = np.ascontiguousarray(a_endo[e].T)
            engine2d_regrid_serial(
                a_grid,
                b_grid,
                ae_t,
                be_t,
                np.ascontiguousarray(ap_choice.T),
                ta,
            )
            a_pol[e] = ta.T
        return a_pol
    mesh_interp = {"curvilinear": Curvilinear2DInterp, "delaunay": Delaunay2DInterp}
    if interp not in mesh_interp:
        msg = f"unknown interpolator {interp!r}"
        raise ValueError(msg)
    bq, aq = np.meshgrid(b_grid, a_grid, indexing="ij")
    bqf, aqf = bq.ravel(), aq.ravel()
    for e in range(nE):
        out = mesh_interp[interp]([ap_choice], b_endo[e], a_endo[e])(bqf, aqf)
        a_pol[e] = out[0].reshape(nB, nA)
    return a_pol


def solve_interp(cal, interp, tol=SOLVE_TOL, maxit=MAXIT):
    """Iterate the compact EGM$^n$ backward step to convergence with the chosen
    Stage-0 interpolator. Returns the converged policies, the ``converged``
    flag and the ``reason`` a solve stopped short: ``no_convergence`` at
    ``maxit`` iterations, or ``nan_policy`` / ``nan_value`` when a regridded
    policy or marginal value turns non-finite.
    """
    Va, Vb, l_grid, ap_choice = initial_state(cal)
    a = b = c = None
    it = 0
    reason = "no_convergence"
    for it in range(1, maxit + 1):
        Va_p, Vb_p = expect(cal["Pi"], Va), expect(cal["Pi"], Vb)
        c_s1, bp_s1, a_endo, b_endo = stage_inversions(Va_p, Vb_p, cal, l_grid)
        a_pol = regrid_with(
            interp,
            b_endo,
            a_endo,
            ap_choice,
            cal["b"],
            cal["a"],
        )
        if not np.isfinite(a_pol).all():
            reason = "nan_policy"
            break
        Va_n, Vb_n, a, b, c, _uce = regular_grid_inversion(
            a_pol,
            c_s1,
            bp_s1,
            Vb_p,
            cal,
            l_grid,
        )
        if not (np.isfinite(Va_n).all() and np.isfinite(Vb_n).all()):
            reason = "nan_value"
            break
        d = marginal_change(Va_n, Va, Vb_n, Vb)
        Va, Vb = Va_n, Vb_n
        if d < tol:
            reason = ""
            break
    return {
        "a": a,
        "b": b,
        "c": c,
        "Va": Va,
        "Vb": Vb,
        "iter": it,
        "converged": reason == "",
        "reason": reason,
    }
