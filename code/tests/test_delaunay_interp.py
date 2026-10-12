"""Delaunay2DInterp: linear extrapolation outside the convex hull."""

from __future__ import annotations

import numpy as np
from engine.utilities import Delaunay2DInterp
from scipy.interpolate import LinearNDInterpolator


def warped_mesh():
    """A 12 x 9 curvilinear mesh."""
    x, y = np.meshgrid(
        np.geomspace(0.1, 10.0, 12),
        np.linspace(0.0, 5.0, 9),
        indexing="ij",
    )
    return x + 0.1 * np.sin(y), y + 0.05 * x


def linear(x, y):
    return 2.0 + 3.0 * x - 1.5 * y


def test_off_hull_query_on_a_linear_function_is_exact() -> None:
    """Off-hull queries return the linear function's own value (each is the
    barycentric extension of a hull triangle, exact for a linear function), and
    each is counted.
    """
    x, y = warped_mesh()
    interp = Delaunay2DInterp([linear(x, y)], x, y)
    qx = np.array([-1.0, 20.0, 5.0, 1.0])
    qy = np.array([2.0, 3.0, -2.0, 9.0])
    (value,) = interp(qx, qy)
    np.testing.assert_allclose(value, linear(qx, qy), rtol=1e-12)
    assert interp.off_hull_queries == qx.size


def test_inside_the_hull_it_is_linear_nd_interpolator() -> None:
    """In-hull values are scipy's LinearNDInterpolator, bit for bit."""
    x, y = warped_mesh()
    f = np.exp(0.1 * x) * np.cos(y)
    interp = Delaunay2DInterp([f], x, y)
    qx, qy = np.meshgrid(np.linspace(1.0, 9.0, 7), np.linspace(1.0, 4.0, 5))
    reference = LinearNDInterpolator(np.column_stack([x.ravel(), y.ravel()]), f.ravel())
    (value,) = interp(qx, qy)
    np.testing.assert_array_equal(value, reference(qx, qy))
    assert interp.off_hull_queries == 0


def test_non_finite_nodes_are_left_out() -> None:
    """A node with a NaN coordinate or value drops out of the triangulation, and
    queries near it get a finite value from the remaining nodes.
    """
    x, y = warped_mesh()
    f = linear(x, y)
    x[-1, -1] = np.nan
    f[0, 0] = np.nan
    (value,) = Delaunay2DInterp([f], x, y)(np.array([10.0, 0.1]), np.array([5.5, 0.0]))
    np.testing.assert_allclose(
        value,
        linear(np.array([10.0, 0.1]), np.array([5.5, 0.0])),
    )
