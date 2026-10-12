"""Numerical helpers shared across the package: the Delaunay interpolator with
the curvilinear interpolators' API, the HARK interpolator-on-interpolator
builder, the mesh-regularity diagnostics (four-corner Jacobians, IPM
violations, the empirical bracket-displacement constant), and the synthetic
warped EGM grid of
the benchmarks and tests.
"""

from __future__ import annotations

import numpy as np
from HARK.interpolation import LinearInterp, LinearInterpOnInterp1D
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

# Engine2DInterp's constructor rejects a mesh that breaks def-ipm, and engine2d
# defines the counter it uses. This module re-exports it beside the other
# mesh-regularity diagnostics.
from engine.engine2d import ipm_violations  # noqa: F401

###############################################################################
# Interpolator Classes (all with same API as Curvilinear2DInterp)
###############################################################################


class Delaunay2DInterp:
    """Delaunay-based 2D interpolator with Curvilinear2DInterp API.

    Inside the convex hull of the nodes it is scipy's ``LinearNDInterpolator``.
    A query outside the hull extrapolates linearly from the hull triangle whose
    boundary edge lies nearest the query: its barycentric weights there, some
    negative, the triangulation's analogue of the curvilinear method's
    exit-sector rule and ENGINE's boundary slope. A node with a non-finite
    coordinate or value is left out of that function's triangulation; NaN
    comes back only when fewer than three finite nodes remain.
    ``off_hull_queries`` counts the extrapolated query points over all calls.
    """

    def __init__(self, f_values, x_values, y_values) -> None:
        """Triangulate the finite nodes of each function in ``f_values``."""
        self.f_values = f_values
        self.x_values = x_values
        self.y_values = y_values
        self.off_hull_queries = 0

        points = np.column_stack([np.ravel(x_values), np.ravel(y_values)])
        coord_valid = np.isfinite(points).all(axis=1)
        self.interpolators = []
        for values in f_values:
            values_flat = np.ravel(values)
            valid = coord_valid & np.isfinite(values_flat)
            if np.sum(valid) < 3:
                self.interpolators.append(None)
                continue
            tri = Delaunay(points[valid])
            self.interpolators.append(
                (
                    LinearNDInterpolator(tri, values_flat[valid], fill_value=np.nan),
                    hull_facets(tri),
                ),
            )

    def __call__(self, x_query, y_query):
        """Return one interpolated array per input function at the query points."""
        x_query = np.asarray(x_query)
        shape = x_query.shape
        q = np.column_stack([x_query.ravel(), np.ravel(y_query)])
        results = []
        n_off = 0
        for entry in self.interpolators:
            if entry is None:
                results.append(np.full(shape, np.nan))
                continue
            interp, facets = entry
            out = interp(q)
            off = np.isnan(out) & np.isfinite(q).all(axis=1)
            if off.any():
                out[off] = extrapolate_from_hull(interp, facets, q[off])
            n_off = max(n_off, int(off.sum()))
            results.append(out.reshape(shape))
        self.off_hull_queries += n_off
        return tuple(results)


def hull_facets(tri):
    """``(simplex, p, q)`` of every convex-hull edge of the triangulation ``tri``:
    the triangle with that edge and the edge's two end points.
    """
    simplex, opposite = np.nonzero(tri.neighbors == -1)
    edge_vertices = np.array(
        [
            np.delete(tri.simplices[s], i)
            for s, i in zip(simplex, opposite, strict=True)
        ],
    )
    return simplex, tri.points[edge_vertices[:, 0]], tri.points[edge_vertices[:, 1]]


def extrapolate_from_hull(interp, facets, q):
    """Values of ``interp`` (a ``LinearNDInterpolator``) at the off-hull points
    ``q``, each extended linearly from the triangle of its nearest hull edge.
    """
    simplex, p0, p1 = facets
    edge = p1 - p0
    t = np.clip(
        np.einsum("qed,ed->qe", q[:, None, :] - p0[None], edge)
        / np.einsum("ed,ed->e", edge, edge),
        0.0,
        1.0,
    )
    nearest = p0[None] + t[..., None] * edge[None]
    owner = simplex[np.argmin(np.sum((q[:, None, :] - nearest) ** 2, axis=-1), axis=1)]
    tri = interp.tri
    transform = tri.transform[owner]
    b = np.einsum("qij,qj->qi", transform[:, :2], q - transform[:, 2])
    weights = np.column_stack([b, 1.0 - b.sum(axis=1)])
    values = interp.values[tri.simplices[owner], 0]
    return np.sum(weights * values, axis=1)


def interp_on_interp(values, grids):
    temp = []
    x, y = grids
    grid = y[0]
    for i in range(grid.size):
        temp.append(LinearInterp(x[:, i], values[:, i]))

    return LinearInterpOnInterp1D(temp, grid)


def corner_jacobians(x, y):
    """Bilinear-map Jacobian determinant at the four corners of every cell
    (``def-fold-free``), shape ``(4, J - 1, K - 1)``.

    Corner ``(s, t)`` pairs the axis-0 edge on row ``k + t`` with the axis-1
    edge on column ``j + s``; the mesh is fold-free iff every entry is positive.
    """
    J, K = x.shape
    dx_j, dy_j = np.diff(x, axis=0), np.diff(y, axis=0)
    dx_k, dy_k = np.diff(x, axis=1), np.diff(y, axis=1)
    dets = []
    for s in (0, 1):
        for t in (0, 1):
            row, col = slice(t, t + K - 1), slice(s, s + J - 1)
            dets.append(dx_j[:, row] * dy_k[col] - dy_j[:, row] * dx_k[col])
    return np.stack(dets)


def cross2(u, v):
    """2-D cross product ``u_x v_y - u_y v_x`` over trailing ``(x, y)`` pairs."""
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


def bilinear_contains(corners, q):
    """Mask of the points ``q`` (shape ``(n, 2)``) inside the bilinear image of
    one cell, ``corners = (p00, p10, p01, p11)`` with ``p_st`` the ``(x, y)``
    vertex at parameter ``(s, t)``.

    Solves the inverse map exactly: eliminating ``s`` from
    ``q = p00 + s a + t b + s t c`` leaves a quadratic in ``t``, whose roots
    take the cancellation-free form (no division by a leading coefficient
    that is zero or rounding noise on a near-parallelogram cell). A point is
    inside when some root has ``(s, t)`` in the closed unit square; on a
    folded cell both roots can qualify.
    """
    p00, p10, p01, p11 = (np.asarray(p, dtype=float) for p in corners)
    a, b, c = p10 - p00, p01 - p00, p11 - p10 - p01 + p00
    d = np.asarray(q, dtype=float) - p00
    quad_a = -cross2(b, c)
    quad_b = cross2(d, c) - cross2(b, a)
    quad_c = cross2(d, a)
    inside = np.zeros(len(d), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        root = np.sqrt(quad_b * quad_b - 4.0 * quad_a * quad_c)
        half = -0.5 * (quad_b + np.copysign(root, quad_b))
        for t in (half / quad_a, quad_c / half):
            v = a + t[:, None] * c
            s = np.sum((d - t[:, None] * b) * v, axis=1) / np.sum(v * v, axis=1)
            inside |= (t >= 0) & (t <= 1) & (s >= 0) & (s <= 1)
    return inside


def reached_folds(x, y, q):
    """Cells breaking ``def-fold-free`` and which of them the points ``q`` reach.

    Returns ``(folded, reached)``: the ``(j, k)`` index pairs of every cell
    with a non-positive corner Jacobian, shape ``(n, 2)``, and a boolean mask
    of length ``n`` marking the cells whose bilinear image contains at least
    one point of ``q``. ``reached.sum()`` is the fold count over the cells
    the query set reaches; ``len(folded)`` is the total fold count.
    """
    folded = np.argwhere(~(corner_jacobians(x, y) > 0).all(axis=0))
    pts = np.stack([x, y], axis=-1)
    reached = np.array(
        [
            bilinear_contains(
                (pts[j, k], pts[j + 1, k], pts[j, k + 1], pts[j + 1, k + 1]),
                q,
            ).any()
            for j, k in folded
        ],
        dtype=bool,
    )
    return folded, reached


def nan_nodes(x, y) -> int:
    """Number of mesh nodes where ``x`` or ``y`` is non-finite."""
    return int(np.sum(~(np.isfinite(x) & np.isfinite(y))))


def empirical_ipo_alpha(fx, n_probe: int = 64) -> int | float:
    """Measure Section 4's bracket-displacement constant alpha ex post.

    Computes the bracket index j*(k; x) in every k-slice for ``n_probe``
    fixed queries spanning the range covered by all slices, and returns the
    maximum bracket displacement between adjacent slices, an ``int``. A mesh
    with any non-finite node has no defined constant: the result is NaN.
    """
    if not np.isfinite(fx).all():
        return float("nan")
    J, K = fx.shape
    x_lo = fx.min(axis=0).max()  # max of per-slice minima: bracketed everywhere
    x_hi = fx.max(axis=0).min()  # min of per-slice maxima
    xq = np.linspace(x_lo, x_hi, n_probe)
    j_star = np.stack(
        [np.clip(np.searchsorted(fx[:, k], xq) - 1, 0, J - 2) for k in range(K)],
    )
    return int(np.abs(np.diff(j_star, axis=0)).max())


def make_warped_egm_grid(J, K, warp_factor=0.1, add_factor=None):
    """Build a synthetic curvilinear EGM-style grid for benchmarks/tests.

    Returns ``(x_endog, y_endog, values)`` where the base axes are
    ``geomspace(0.1, 100)`` with a sinusoidal warp applied. ``add_factor``
    defaults to ``warp_factor`` if not given (the benchmark's convention);
    the test_engine2d fixture passes ``add_factor=0.05`` to preserve its
    original grid. For ``warp_factor < 1`` and ``add_factor >= 0`` both
    coordinates stay positive, so ``values`` is finite.
    """
    if add_factor is None:
        add_factor = warp_factor
    x_base = np.geomspace(0.1, 100, J)
    y_base = np.geomspace(0.1, 100, K)
    xx, yy = np.meshgrid(x_base, y_base, indexing="ij")
    x_endog = xx * (1 + warp_factor * np.sin(yy / 20)) + add_factor * yy
    y_endog = yy * (1 + warp_factor * np.cos(xx / 20)) + add_factor * xx
    values = np.log(x_endog + y_endog + 1)
    return x_endog, y_endog, values
