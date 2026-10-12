"""Tests for ENGINE 2D interpolation.

Tests the Engine2DInterp class which provides:
- Multi-function interpolation on curvilinear grids
- Scalar, array, and meshgrid queries
- Different axis processing orders
"""

from __future__ import annotations

import numpy as np
import pytest
from engine.engine2d import (
    Engine2DInterp,
    advance_j_walk,
    binary_search_bracket,
    bounded_binary_search,
    engine2d_regrid,
    engine2d_regrid_ipo,
    engine2d_regrid_serial,
    log_engine2d_scalar_multi,
    warmup_jit,
)
from engine.utilities import make_warped_egm_grid

# Initialize random generator for reproducibility
rng = np.random.default_rng(42)


def rect_mesh(x_lo, x_hi, J, y_lo, y_hi, K):
    """``(xx, yy)`` meshgrid, ``indexing="ij"``, of two uniform axes."""
    return np.meshgrid(
        np.linspace(x_lo, x_hi, J),
        np.linspace(y_lo, y_hi, K),
        indexing="ij",
    )


# --- Fixtures ---


@pytest.fixture(scope="module", autouse=True)
def jit_warmup() -> None:
    """Warmup JIT compilation before running tests."""
    warmup_jit()


@pytest.fixture(scope="module")
def regular_grid():
    """Create a regular (non-warped) grid for basic tests."""
    xx, yy = rect_mesh(0.1, 10, 30, 0.1, 5, 40)
    return xx, yy, np.log(xx + yy + 1)


@pytest.fixture(scope="module")
def warped_grid():
    """Create a warped (curvilinear) grid simulating EGM output."""
    return make_warped_egm_grid(
        J=40,
        K=50,
        warp_factor=0.1,
        add_factor=0.05,
    )


@pytest.fixture(scope="module")
def multi_function_grid(warped_grid):
    """Create multiple functions on the same warped grid."""
    x_endog, y_endog, _ = warped_grid
    value_func = np.log(x_endog + y_endog + 1)
    consumption = 0.7 * (x_endog + y_endog)
    investment = 0.3 * x_endog
    return x_endog, y_endog, [value_func, consumption, investment]


@pytest.fixture(scope="module")
def linear_function_interp():
    """Engine2DInterp over ``f(x, y) = 2x + 3y + 1`` on a regular 20x25 grid.

    Bilinear interpolation is exact for an affine function, so this is the
    shared fixture for the linear-accuracy tests in both interpolation and
    regrid.
    """
    xx, yy = rect_mesh(0, 10, 20, 0, 5, 25)
    return Engine2DInterp([2 * xx + 3 * yy + 1], xx, yy)


def assert_curvature_bound_invariant(x, y, values, evaluate) -> None:
    """``evaluate(interp)`` agrees to 1e-10 across ``curvature_bound`` 1, 2, 5."""
    r1, r2, r5 = (
        evaluate(Engine2DInterp([values], x, y, curvature_bound=cb)) for cb in (1, 2, 5)
    )
    np.testing.assert_allclose(r1, r2, rtol=1e-10)
    np.testing.assert_allclose(r2, r5, rtol=1e-10)


@pytest.fixture
def minimal_2x2_grid():
    """The smallest valid Engine2DInterp grid: J=2, K=2."""
    x = np.array([[1.0, 2.0], [1.5, 2.5]])
    y = np.array([[1.0, 1.5], [2.0, 2.5]])
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    return x, y, values


def assert_tuple_of(result, n_funcs, is_ok) -> None:
    """``result`` is a tuple of ``n_funcs`` outputs, each passing ``is_ok``."""
    assert isinstance(result, tuple)
    assert len(result) == n_funcs
    assert all(is_ok(r) for r in result)


# --- Basic Functionality Tests ---


class TestEngine2DInterpBasic:
    """Construction, and the output container for each query shape."""

    @pytest.mark.parametrize(
        ("grid", "n_funcs", "point", "lo", "hi"),
        [
            ("regular_grid", 1, (5.0, 2.5), (1.0, 0.5), (9.0, 4.5)),
            ("multi_function_grid", 3, (50.0, 50.0), (10.0, 10.0), (90.0, 90.0)),
        ],
        ids=["one-function", "three-functions"],
    )
    def test_query_shapes(self, request, grid, n_funcs, point, lo, hi) -> None:
        """A scalar query returns a tuple of floats; 1D and meshgrid queries return
        a tuple of arrays of the query's shape.
        """
        x, y, funcs = request.getfixturevalue(grid)
        interp = Engine2DInterp(funcs if isinstance(funcs, list) else [funcs], x, y)
        assert (interp.n_funcs, interp.J, interp.K) == (n_funcs, *x.shape)

        assert_tuple_of(interp(*point), n_funcs, lambda r: isinstance(r, float))
        line = (np.linspace(lo[0], hi[0], 3), np.linspace(lo[1], hi[1], 3))
        mesh = rect_mesh(lo[0], hi[0], 10, lo[1], hi[1], 12)
        assert_tuple_of(interp(*line), n_funcs, lambda r: r.shape == (3,))
        assert_tuple_of(interp(*mesh), n_funcs, lambda r: r.shape == (10, 12))


# --- Accuracy Tests ---


class TestInterpolationAccuracy:
    """Test interpolation accuracy."""

    def test_exact_at_grid_points(self, regular_grid) -> None:
        """Test interpolation is exact at grid points."""
        xx, yy, values = regular_grid
        interp = Engine2DInterp([values], xx, yy)

        # Query at a grid point
        i, j = 15, 20
        x_pt = xx[i, j]
        y_pt = yy[i, j]
        expected = values[i, j]

        result = interp(x_pt, y_pt)[0]
        assert abs(result - expected) < 1e-10

    def test_linear_function_accuracy(self, linear_function_interp) -> None:
        """Test accuracy on a linear function (should be exact for bilinear)."""
        interp = linear_function_interp

        # Query at random interior points
        x_q = rng.uniform(1, 9, 100)
        y_q = rng.uniform(0.5, 4.5, 100)

        results = interp(x_q, y_q)[0]
        expected = 2 * x_q + 3 * y_q + 1

        np.testing.assert_allclose(results, expected, rtol=1e-10)

    def test_convergence_order(self) -> None:
        """Test O(h^2) convergence for smooth functions."""

        def true_func(x, y):
            return np.sin(x) * np.cos(y)

        errors = []
        grid_sizes = [10, 20, 40, 80]

        # Query points
        xx_q, yy_q = rect_mesh(0.5, 2.5, 50, 0.5, 2.5, 50)
        true_vals = true_func(xx_q, yy_q)

        for N in grid_sizes:
            xx, yy = rect_mesh(0, 3, N, 0, 3, N)
            values = true_func(xx, yy)

            interp = Engine2DInterp([values], xx, yy)
            result = interp(xx_q, yy_q)[0]

            max_err = np.max(np.abs(result - true_vals))
            errors.append(max_err)

        errors = np.array(errors)
        # Check convergence rate (error ratio for 2x grid refinement)
        # For O(h^2), ratio should be ~4
        ratios = errors[:-1] / errors[1:]
        assert np.all(ratios > 2.5), f"Convergence too slow: ratios = {ratios}"


# --- Edge Cases ---


class TestInputValidation:
    """Test input validation and error handling."""

    @pytest.mark.parametrize(
        "line",
        [np.array([[1.0, 2.0]]), np.array([[1.0], [2.0]])],
        ids=["J=1", "K=1"],
    )
    def test_grid_dimension_validation_too_small(self, line) -> None:
        """A grid with J < 2 or K < 2 raises ValueError."""
        with pytest.raises(ValueError, match="Grid must be at least 2x2"):
            Engine2DInterp([line], line, line)

    def test_grid_dimension_validation_minimal_valid(self, minimal_2x2_grid) -> None:
        """A 2x2 grid is valid and interpolates to a finite value."""
        x, y, values = minimal_2x2_grid

        interp = Engine2DInterp([values], x, y)
        assert interp.J == 2
        assert interp.K == 2
        assert np.isfinite(interp(1.3, 1.3)[0])

    @pytest.mark.parametrize(
        "curvature_bound",
        [-1, 0, 2.5],
        ids=["negative", "zero", "float"],
    )
    def test_curvature_bound_validation_invalid(
        self,
        minimal_2x2_grid,
        curvature_bound,
    ) -> None:
        """Test that a non-positive-integer curvature_bound raises ValueError."""
        x, y, values = minimal_2x2_grid

        with pytest.raises(
            ValueError,
            match="curvature_bound must be a positive integer",
        ):
            Engine2DInterp([values], x, y, curvature_bound=curvature_bound)

    def test_curvature_bound_validation_valid(self, minimal_2x2_grid) -> None:
        """Test that valid curvature_bound values work."""
        x, y, values = minimal_2x2_grid

        # Test various valid values
        for cb in [1, 2, 3, 5, 10]:
            interp = Engine2DInterp([values], x, y, curvature_bound=cb)
            assert interp.curvature_bound == cb

    def test_curvature_bound_affects_interpolation(self, warped_grid) -> None:
        """Test that curvature_bound is used in interpolation."""
        x, y, values = warped_grid

        # Different curvature bounds should produce same results on well-behaved grids
        assert_curvature_bound_invariant(x, y, values, lambda f: f(50.0, 50.0)[0])


class TestEdgeCases:
    """Test edge cases and boundary behavior."""

    def test_extrapolation(self, regular_grid) -> None:
        """Test extrapolation beyond grid boundaries."""
        xx, yy, values = regular_grid
        interp = Engine2DInterp([values], xx, yy)

        # Query outside grid
        x_min, x_max = xx.min(), xx.max()
        y_min, y_max = yy.min(), yy.max()

        # Extrapolate below minimum
        result_low = interp(x_min - 1.0, y_min - 0.5)
        assert np.isfinite(result_low[0])

        # Extrapolate above maximum
        result_high = interp(x_max + 1.0, y_max + 0.5)
        assert np.isfinite(result_high[0])


# --- Performance Sanity Tests ---


class TestIPOFallback:
    """Fallback behavior for grids whose bracket displacement exceeds the margin.

    The ENGINE algorithm uses a bounded staircase walk that exploits small
    bracket displacement between adjacent rows. When brackets jump more than
    curvature_bound, the algorithm falls back to binary search. These tests
    verify correct behavior in both cases.
    """

    def test_non_ipo_grid_still_works(self) -> None:
        """Grids with bracket jumps above the margin still produce correct results
        via fallback.
        """
        xx, yy = rect_mesh(0, 10, 20, 0, 5, 25)

        # Aggressive warping: bracket indices can jump by more than curvature_bound.
        x_endog = xx + 2.0 * np.sin(yy * 2)
        y_endog = yy + 2.0 * np.cos(xx * 2)
        values = np.log(np.abs(x_endog) + np.abs(y_endog) + 1)

        # Use small curvature_bound to force fallback
        interp_cb1 = Engine2DInterp([values], x_endog, y_endog, curvature_bound=1)
        # Use large curvature_bound that covers the warping
        interp_cb10 = Engine2DInterp([values], x_endog, y_endog, curvature_bound=10)

        # Query at random points
        x_q = rng.uniform(x_endog.min() * 1.1, x_endog.max() * 0.9, 50)
        y_q = rng.uniform(y_endog.min() * 1.1, y_endog.max() * 0.9, 50)

        # Both should give identical results (fallback to binary search when needed)
        result_cb1 = interp_cb1(x_q, y_q)[0]
        result_cb10 = interp_cb10(x_q, y_q)[0]

        np.testing.assert_allclose(result_cb1, result_cb10, rtol=1e-10)

    def test_both_axis_orders_match(self, warped_grid) -> None:
        """Test both axis orders produce consistent results for various curvature_bounds."""
        x, y, values = warped_grid

        for cb in [1, 2, 5]:
            interp_01 = Engine2DInterp(
                [values],
                x,
                y,
                axes_order=(0, 1),
                curvature_bound=cb,
            )
            interp_10 = Engine2DInterp(
                [values],
                x,
                y,
                axes_order=(1, 0),
                curvature_bound=cb,
            )

            # Random interior points plus the fixed interior point (50, 50).
            x_q = np.append(rng.uniform(x.min() * 1.1, x.max() * 0.9, 100), 50.0)
            y_q = np.append(rng.uniform(y.min() * 1.1, y.max() * 0.9, 100), 50.0)

            result_01 = interp_01(x_q, y_q)[0]
            result_10 = interp_10(x_q, y_q)[0]

            # Results should be close (not identical due to different paths)
            np.testing.assert_allclose(result_01, result_10, rtol=0.05)

    def test_scalar_array_meshgrid_consistency(self, warped_grid) -> None:
        """Test scalar, array, and meshgrid queries give consistent results."""
        x, y, values = warped_grid
        interp = Engine2DInterp([values], x, y, curvature_bound=2)

        # Test point
        x_pt, y_pt = 50.0, 50.0

        # Scalar query
        result_scalar = interp(x_pt, y_pt)[0]

        # Array query (single element)
        result_array = interp(np.array([x_pt]), np.array([y_pt]))[0][0]

        # Meshgrid query (1x1)
        xx_q, yy_q = np.meshgrid([x_pt], [y_pt], indexing="ij")
        result_meshgrid = interp(xx_q, yy_q)[0][0, 0]

        # All paths now use probe-caching algorithm - should match tightly
        np.testing.assert_allclose(result_scalar, result_array, rtol=1e-12)
        np.testing.assert_allclose(result_scalar, result_meshgrid, rtol=1e-12)


class TestPerformance:
    """Basic performance sanity tests."""

    def test_large_grid(self) -> None:
        """Test with large grid doesn't crash."""
        xx, yy = rect_mesh(0, 10, 100, 0, 5, 120)
        interp = Engine2DInterp([np.log(xx + yy + 1)], xx, yy)

        result = interp(*rect_mesh(0.5, 9.5, 50, 0.2, 4.8, 60))
        assert result[0].shape == (50, 60)

    def test_many_functions(self) -> None:
        """Test with many functions on same grid."""
        xx, yy = rect_mesh(0, 10, 30, 0, 5, 40)

        # 10 functions
        funcs = [np.sin(xx * i) * np.cos(yy * i) for i in range(1, 11)]
        interp = Engine2DInterp(funcs, xx, yy)

        result = interp(5.0, 2.5)
        assert len(result) == 10


class TestRegrid:
    """Test efficient regridding to rectangular grids."""

    def test_regrid_basic(self, regular_grid) -> None:
        """Test basic regrid functionality."""
        xx, yy, values = regular_grid
        interp = Engine2DInterp([values], xx, yy)

        # Create rectangular output grid
        x_new = np.linspace(1.0, 9.0, 15)
        y_new = np.linspace(0.5, 4.5, 20)

        result = interp.regrid(x_new, y_new)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert result[0].shape == (15, 20)

    def test_regrid_multi_function(self, multi_function_grid) -> None:
        """Test regrid with multiple functions."""
        x, y, funcs = multi_function_grid
        interp = Engine2DInterp(funcs, x, y)

        x_new = np.linspace(20.0, 80.0, 10)
        y_new = np.linspace(20.0, 80.0, 12)

        result = interp.regrid(x_new, y_new)

        assert len(result) == 3
        assert all(r.shape == (10, 12) for r in result)

    def test_regrid_matches_meshgrid_query(self, regular_grid) -> None:
        """Test that regrid produces same results as meshgrid query."""
        xx, yy, values = regular_grid
        interp = Engine2DInterp([values], xx, yy)

        x_new = np.linspace(1.0, 9.0, 8)
        y_new = np.linspace(0.5, 4.5, 10)

        # Regrid method
        regrid_result = interp.regrid(x_new, y_new)[0]

        # Meshgrid query
        xx_q, yy_q = np.meshgrid(x_new, y_new, indexing="ij")
        meshgrid_result = interp(xx_q, yy_q)[0]

        np.testing.assert_allclose(regrid_result, meshgrid_result, rtol=1e-10)

    def test_regrid_linear_function_exact(self, linear_function_interp) -> None:
        """Test regrid is exact for linear functions."""
        interp = linear_function_interp

        x_new = np.linspace(1.0, 9.0, 20)
        y_new = np.linspace(0.5, 4.5, 25)

        result = interp.regrid(x_new, y_new)[0]

        # Compute expected values
        xx_new, yy_new = np.meshgrid(x_new, y_new, indexing="ij")
        expected = 2 * xx_new + 3 * yy_new + 1

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_regrid_with_different_curvature_bounds(self, warped_grid) -> None:
        """Test regrid works with different curvature_bound settings."""
        x, y, values = warped_grid

        x_new = np.linspace(x.min() * 1.1, x.max() * 0.9, 15)
        y_new = np.linspace(y.min() * 1.1, y.max() * 0.9, 15)
        assert_curvature_bound_invariant(
            x,
            y,
            values,
            lambda f: f.regrid(x_new, y_new)[0],
        )

    def test_regrid_single_point(self, regular_grid) -> None:
        """Test regrid with single point output grids."""
        xx, yy, values = regular_grid
        interp = Engine2DInterp([values], xx, yy)

        x_new = np.array([5.0])
        y_new = np.array([2.5])

        result = interp.regrid(x_new, y_new)[0]
        assert result.shape == (1, 1)

        # Compare with scalar query
        scalar_result = interp(5.0, 2.5)[0]
        np.testing.assert_allclose(result[0, 0], scalar_result, rtol=1e-10)


class TestAxisOrderSemantic:
    """Verify the new axes_order is a memory-layout flag, not a semantic flag.

    Per the P0.2 fix, ``axes_order=(1, 0)`` transposes the stored
    ``fx``/``fy``/``f_values`` AND swaps the kernel's view of the query
    coordinates; the two operations cancel so the user-visible math is
    unchanged. The only effect is the kernel's inner-loop memory access
    pattern.
    """

    def make_axes_order_pair(self, J, K, values_fn):
        """(ref, swap) for a grid built with axes_order (0, 1) and (1, 0).

        Shared setup for the semantic-invariance tests below: a ``J x K``
        grid on the fixed ``[0.5, 5.0] x [0.2, 3.0]`` range, with ``values_fn``
        supplying the field.
        """
        xx, yy = rect_mesh(0.5, 5.0, J, 0.2, 3.0, K)
        values = values_fn(xx, yy)
        ref = Engine2DInterp([values], xx, yy, axes_order=(0, 1))
        swap = Engine2DInterp([values], xx, yy, axes_order=(1, 0))
        return ref, swap

    def test_axes_order_output_invariance(self) -> None:
        """``(0, 1)`` and ``(1, 0)`` produce the same answer to float precision."""
        ref, swap = self.make_axes_order_pair(
            25,
            17,
            lambda xx, yy: np.sin(xx) * np.cos(yy) + 0.3 * xx + 0.7 * yy,
        )

        # Asymmetric query (xq != yq) so a semantic swap would be detectable
        xq, yq = 2.3, 1.4
        np.testing.assert_allclose(swap(xq, yq)[0], ref(xq, yq)[0], rtol=1e-12)
        np.testing.assert_allclose(swap(yq, xq)[0], ref(yq, xq)[0], rtol=1e-12)

        # Array queries too
        x_query = np.linspace(1.0, 4.0, 7)
        y_query = np.linspace(0.5, 2.5, 7)
        np.testing.assert_allclose(
            swap(x_query, y_query)[0],
            ref(x_query, y_query)[0],
            rtol=1e-12,
        )

    def test_axes_order_storage_transposes(self) -> None:
        """``(1, 0)`` transposes the stored grid shapes.

        This is a structural invariant: ``axes_order=(1, 0)`` must swap the
        stored array dimensions. We assert only the shapes, not the exact
        transposed values, to avoid coupling to which attribute holds what
        (a white-box detail). The behavioral guarantee (interpolation output is
        invariant to axis order) is covered by
        ``test_axes_order_regrid_output_invariance``.
        """
        J, K = 25, 17
        ref, swap = self.make_axes_order_pair(
            J,
            K,
            lambda xx, yy: np.sin(xx) * np.cos(yy),
        )

        # Stored arrays have swapped shapes under axis transposition.
        assert ref.fx.shape == (J, K)
        assert swap.fx.shape == (K, J)
        assert ref.fy.shape == (J, K)
        assert swap.fy.shape == (K, J)

    def test_axes_order_regrid_output_invariance(self) -> None:
        """Regrid also produces the same answer under (1, 0)."""
        ref, swap = self.make_axes_order_pair(
            30,
            20,
            lambda xx, yy: np.sin(xx) * np.cos(yy) + 0.3 * xx + 0.7 * yy,
        )

        x_new = np.linspace(0.6, 4.8, 13)
        y_new = np.linspace(0.3, 2.8, 11)

        ref_out = ref.regrid(x_new, y_new)[0]
        swap_out = swap.regrid(x_new, y_new)[0]
        np.testing.assert_allclose(swap_out, ref_out, rtol=1e-12)
        assert swap_out.shape == ref_out.shape == (len(x_new), len(y_new))

    @pytest.mark.parametrize("axes_order", [(0, 0), (2, 0), [0, 1]])
    def test_axes_order_invalid_raises(self, axes_order) -> None:
        """Bad ``axes_order`` values are rejected at construction."""
        xx, yy = rect_mesh(0, 1, 5, 0, 1, 5)

        with pytest.raises(ValueError, match="axes_order"):
            Engine2DInterp([xx + yy], xx, yy, axes_order=axes_order)


class TestFiniteValueValidation:
    """P2.4: __init__ rejects NaN and Inf in any input array."""

    @pytest.mark.parametrize(
        ("arg", "index", "bad", "match"),
        [
            (1, (3, 4), np.nan, "x_values.*NaN or Inf"),
            (1, (0, 0), np.inf, "x_values.*NaN or Inf"),
            (2, (2, 5), np.nan, "y_values.*NaN or Inf"),
            (0, (4, 4), np.nan, r"f_values\[0\].*NaN or Inf"),
            (0, (1, 1), np.inf, r"f_values\[0\].*NaN or Inf"),
        ],
        ids=["nan-x", "inf-x", "nan-y", "nan-f", "inf-f"],
    )
    def test_non_finite_raises(self, regular_grid, arg, index, bad, match) -> None:
        """A NaN or Inf in f, x or y (constructor argument ``arg``) is rejected."""
        xx, yy, values = regular_grid
        args = [values.copy(), xx.copy(), yy.copy()]
        args[arg][index] = bad
        with pytest.raises(ValueError, match=match):
            Engine2DInterp([args[0]], args[1], args[2])

    def test_shape_mismatch_raises(self, regular_grid) -> None:
        xx, yy, values = regular_grid
        wrong_shape = values[:-1, :]
        with pytest.raises(ValueError, match="shape"):
            Engine2DInterp([wrong_shape], xx, yy)


def assert_kernel_matches_parallel(kernel, x_new, y_new, fx, fy, values) -> None:
    """``kernel`` and ``engine2d_regrid`` give bit-identical outputs on one query grid.

    The parallel kernel resets its bracket by binary search per p, so it is the
    exact reference for the serial and ``ipo`` kernels.
    """
    shape = (len(x_new), len(y_new))
    out_par = np.empty(shape, dtype=values.dtype)
    out = np.empty(shape, dtype=values.dtype)
    engine2d_regrid(x_new, y_new, fx, fy, values, out_par)
    kernel(x_new, y_new, fx, fy, values, out)
    np.testing.assert_allclose(out, out_par, rtol=0, atol=0)


class TestRegridSerialParity:
    """Serial and ``ipo`` regrid kernels match the parallel one to float precision."""

    @pytest.mark.parametrize(
        "kernel",
        [engine2d_regrid_serial, engine2d_regrid_ipo],
        ids=["serial", "ipo"],
    )
    def test_kernel_matches_parallel(self, warped_grid, kernel) -> None:
        """Bitwise equality to the parallel kernel, so serial and ``ipo`` also agree."""
        x_new = np.linspace(0.5, 80.0, 23)
        y_new = np.linspace(0.5, 80.0, 19)
        assert_kernel_matches_parallel(kernel, x_new, y_new, *warped_grid)


class TestDegenerateBandConsistency:
    """regrid() and __call__() must agree on a zero-width (degenerate) y-band."""

    def test_regrid_matches_call_on_degenerate_band(self) -> None:
        """A collapsed y-band must use the same fallback in both paths.

        Columns 1 and 2 share identical y-values, so the [1, 2] band has zero
        width at every x. The scalar/__call__ path resolves a degenerate band
        to the lower-column value (t=0); the regrid q-loop must use the same
        convention rather than the band midpoint. Regression for the
        t=0.5-vs-t=0.0 split between regrid_q_loop and the log kernels.
        """
        # (J, K) = (2, 3); each column's x-grid is [0, 10].
        fx = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        # Column y-levels 1, 2, 2 -> band [1, 2] is degenerate (y_1 == y_2).
        fy = np.array([[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]])
        # Distinct column values so midpoint (25) differs from lower value (20).
        fv = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        # The mesh breaks def-ipm, so Engine2DInterp rejects it; the kernels,
        # which callers such as the EGM^n Stage 0 regrid drive directly, take it.
        with pytest.raises(ValueError, match="def-ipm"):
            Engine2DInterp([fv], fx, fy)

        scalar = log_engine2d_scalar_multi(5.0, 2.0, fx, fy, (fv,), 2)[0]
        grid = np.empty((1, 1))
        engine2d_regrid(np.array([5.0]), np.array([2.0]), fx, fy, fv, grid)

        # The scalar kernel resolves the degenerate band to the lower-column value.
        assert scalar == 20.0
        # regrid must match it (was 25.0 = band midpoint before the fix).
        np.testing.assert_allclose(grid[0, 0], scalar, rtol=0, atol=0)


class TestAdvanceJWalk:
    """Direct test of the linear row-advance walk."""

    def test_returns_correct_bracket(self) -> None:
        xp = np.linspace(0.0, 10.0, 11)  # 0, 1, 2, ..., 10
        # Bracket for xq = 3.7 is j = 3 (since xp[3] = 3 <= 3.7 < xp[4] = 4)
        for j_prev in range(9):
            assert advance_j_walk(3.7, xp, j_prev) == 3

    def test_extrapolation_boundaries(self) -> None:
        xp = np.linspace(0.0, 10.0, 11)
        # Below the mesh: returns 0
        assert advance_j_walk(-5.0, xp, 5) == 0
        # Above the mesh: returns n-2 = 9
        assert advance_j_walk(20.0, xp, 5) == 9

    def test_matches_binary_search(self) -> None:
        rng_local = np.random.default_rng(7)
        xp = np.sort(rng_local.uniform(0, 1, 64))
        for _ in range(200):
            xq = rng_local.uniform(-0.1, 1.1)
            j_prev = int(rng_local.integers(0, 63))
            assert advance_j_walk(xq, xp, j_prev) == binary_search_bracket(xq, xp)

    def test_handles_n_lt_2(self) -> None:
        xp = np.array([3.14])
        assert advance_j_walk(2.0, xp, 0) == 0
        empty = np.empty(0)
        assert advance_j_walk(2.0, empty, 0) == 0

    def test_matches_binary_search_boundary_indices(self) -> None:
        """Pin j_prev at the extreme valid indices 0 and n-2 (coverage gap I4)."""
        rng_local = np.random.default_rng(11)
        xp = np.sort(rng_local.uniform(0.0, 1.0, 64))
        n = xp.size
        for j_prev in (0, n - 2):
            for _ in range(100):
                xq = rng_local.uniform(-0.1, 1.1)
                assert advance_j_walk(xq, xp, j_prev) == binary_search_bracket(xq, xp)


class TestRegridIPOEmptyQuery:
    """The ``ipo`` regrid kernel must guard P == 0 (no out-of-bounds read)."""

    def test_ipo_empty_x_query_does_not_crash(self, warped_grid) -> None:
        x_endog, y_endog, values = warped_grid
        x_new = np.empty(0)
        y_new = np.linspace(0.5, 50.0, 7)
        out = np.empty((0, 7), dtype=values.dtype)
        # With P == 0 the kernel must return immediately; before the guard it
        # read x_q_1d[0] out of bounds under boundscheck=False (silent UB).
        engine2d_regrid_ipo(x_new, y_new, x_endog, y_endog, values, out)
        assert out.shape == (0, 7)


class TestIPORegridParity:
    """The ``ipo`` kernel matches parallel/serial to float precision."""

    def test_regrid_mode_ipo(self, warped_grid) -> None:
        """``Engine2DInterp.regrid(mode='ipo')`` matches mode='parallel'."""
        x_endog, y_endog, values = warped_grid
        interp = Engine2DInterp([values], x_endog, y_endog)
        x_new = np.linspace(0.5, 80.0, 17)
        y_new = np.linspace(0.5, 80.0, 13)
        out_par = interp.regrid(x_new, y_new, mode="parallel")[0]
        out_ser = interp.regrid(x_new, y_new, mode="serial")[0]
        out_ipo = interp.regrid(x_new, y_new, mode="ipo")[0]
        np.testing.assert_allclose(out_ser, out_par, rtol=0, atol=0)
        np.testing.assert_allclose(out_ipo, out_par, rtol=0, atol=0)

    def test_regrid_mode_invalid_raises(self, warped_grid) -> None:
        x_endog, y_endog, values = warped_grid
        interp = Engine2DInterp([values], x_endog, y_endog)
        with pytest.raises(ValueError, match="mode"):
            interp.regrid(np.array([1.0, 2.0]), np.array([1.0, 2.0]), mode="bogus")

    def test_ipo_rewind_correctness_random_walk(self) -> None:
        """The ``ipo`` kernel handles non-trivial p,q with state-rewind correctly."""
        rng_local = np.random.default_rng(11)
        J, K = 35, 28
        xx, yy = rect_mesh(0.1, 10.0, J, 0.1, 6.0, K)
        fx = xx * (1 + 0.05 * np.sin(yy)) + 0.02 * yy
        fy = yy * (1 + 0.05 * np.cos(xx)) + 0.02 * xx
        values = (
            np.sin(fx) * np.cos(fy)
            + 0.3 * fx
            + 0.7 * fy
            + 0.01 * rng_local.standard_normal((J, K))
        )

        for _ in range(5):
            P = int(rng_local.integers(5, 40))
            Q = int(rng_local.integers(5, 40))
            x_new = np.sort(rng_local.uniform(0.5, 9.5, P))
            y_new = np.sort(rng_local.uniform(0.5, 5.5, Q))
            assert_kernel_matches_parallel(
                engine2d_regrid_ipo,
                x_new,
                y_new,
                fx,
                fy,
                values,
            )

    def test_ipo_rewind_deep_strong_warp(self) -> None:
        """Force DEEP k-rewinds and confirm ``ipo`` still matches the parallel kernel.

        The rewind branch (``while k_current > 0 and y_q0 < y_k`` in
        ``engine2d_regrid_ipo``) is driven by mesh geometry, not query order:
        the ``ipo`` kernel requires sorted queries (the per-p reset variants handle
        unsorted ones), and the rewind fires when the warped mesh band for a new
        p-column sits above the first y-query, so ``k_current`` must step back.
        ``test_ipo_rewind_correctness_random_walk`` warps the mesh only mildly
        (amplitude 0.05), so the rewinds are shallow. Here a large x-dependent
        vertical shift in ``fy`` (amplitude ~3 over a 6-wide y-range) makes the
        y-band swing several k-steps between adjacent columns, exercising
        multi-step rewinds. The parallel kernel resets its bracket by binary
        search per p, so it is the exact reference: any rewind bug surfaces as a
        mismatch. ``fy`` stays monotone in y (the vertical shift below depends
        only on x) so the warped grid remains well defined.
        """
        xx, yy = rect_mesh(0.1, 10.0, 30, 0.1, 6.0, 26)
        fx = xx
        fy = yy + 3.0 * np.sin(2.0 * np.pi * xx / 10.0)
        values = np.sin(fx) * np.cos(fy) + 0.5 * fx + 0.9 * fy

        x_new = np.linspace(0.5, 9.5, 23)
        y_new = np.linspace(0.5, 5.5, 19)
        assert_kernel_matches_parallel(
            engine2d_regrid_ipo,
            x_new,
            y_new,
            fx,
            fy,
            values,
        )


class TestBoundedSearchGuard:
    """The window-miss guard: an under-set margin costs a search, never a wrong bracket."""

    def test_escapes_a_missed_window_on_both_sides(self) -> None:
        xp = np.linspace(0.0, 10.0, 11)  # true bracket of q is floor(q)
        assert bounded_binary_search(8.5, xp, 2, 4) == 8  # miss above the window
        assert bounded_binary_search(0.5, xp, 5, 7) == 0  # miss below the window
        assert bounded_binary_search(3.5, xp, 2, 4) == 3  # window holds: unchanged

    def test_grid_edge_extrapolation_still_clamps(self) -> None:
        xp = np.linspace(0.0, 10.0, 11)
        assert bounded_binary_search(-1.0, xp, 0, 3) == 0
        assert bounded_binary_search(11.0, xp, 7, 9) == 9


class TestHARKCallPath:
    """Exercise the HARK production call pattern: 3 functions on a curvilinear
    grid, scattered query of shape (J, K) (shock × grid product).

    Mirrors ``ConsHealthSeparableModel.solve_one_period_ConsBasicHealth``'s
    use of ``interpolator([vNvrs, cLvl, nLvl], mLvl, hLvl)`` followed by
    ``solution_next(m_next, h_next)`` where
    ``m_next = R * a + shock * H`` and ``h_next = (1 - shock) * H``.
    The query shape is the (a_grid × H_grid) meshgrid evaluated per shock.
    """

    def hark_like_setup(self):
        J, K = 44, 50  # paper's nominal health-model grid size
        a_grid = np.geomspace(1e-5, 100.0, J)
        H_grid = np.linspace(0.01, 50.0, K)
        aMat, HMat = np.meshgrid(a_grid, H_grid, indexing="ij")

        # Toy three-function "solution": vNvrs, c, n on (mLvl, hLvl) endog grid
        # Construct a curvilinear endog grid via EGM-style inversion shapes
        mLvl = aMat + 0.5 * aMat * (1 + 0.1 * HMat) + 0.05 * HMat
        hLvl = HMat - 0.05 * (HMat / (1 + aMat))  # mild warp
        vNvrs = np.log(mLvl + hLvl + 1)
        cLvl = 0.7 * (mLvl + 0.3 * hLvl)
        nLvl = 0.3 * mLvl + 0.05 * hLvl
        return mLvl, hLvl, [vNvrs, cLvl, nLvl], a_grid, H_grid

    def test_three_func_fast_path_runs(self) -> None:
        """3-function interp with scattered (J, K) query returns (J, K) each."""
        mLvl, hLvl, funcs, _, _ = self.hark_like_setup()
        interp = Engine2DInterp(funcs, mLvl, hLvl)
        # HARK shock x grid query shape
        rng_local = np.random.default_rng(31)
        m_query = 0.5 * np.abs(rng_local.standard_normal(mLvl.shape)) + 1.0
        h_query = 0.5 * np.abs(rng_local.standard_normal(hLvl.shape)) + 1.0

        out = interp(m_query, h_query)
        assert len(out) == 3
        for arr in out:
            assert arr.shape == m_query.shape
            assert np.all(np.isfinite(arr))

    def test_three_func_matches_per_function(self) -> None:
        """3-function batch matches independent 1-function calls."""
        mLvl, hLvl, funcs, _, _ = self.hark_like_setup()
        interp3 = Engine2DInterp(funcs, mLvl, hLvl)
        rng_local = np.random.default_rng(33)
        m_query = 1.0 + rng_local.uniform(0, 50, (8, 6))
        h_query = 0.5 + rng_local.uniform(0, 40, (8, 6))

        out3 = interp3(m_query, h_query)
        for idx, fv in enumerate(funcs):
            interp1 = Engine2DInterp([fv], mLvl, hLvl)
            out1 = interp1(m_query, h_query)[0]
            # 3-func fast path shares row probes; expect ~1e-13 agreement
            np.testing.assert_allclose(out3[idx], out1, rtol=1e-12, atol=1e-12)

    def test_health_model_scalar_call(self) -> None:
        """Scalar query (used by HARK's __call__ scalar dispatch)."""
        mLvl, hLvl, funcs, _, _ = self.hark_like_setup()
        interp = Engine2DInterp(funcs, mLvl, hLvl)
        result = interp(5.0, 3.0)
        assert len(result) == 3
        for v in result:
            assert isinstance(v, float)
            assert np.isfinite(v)


def test_separable_engine_matches_1d_interp() -> None:
    """Separable form: index-aligned mesh reduces ENGINE to exact 1D interp.

    The matched-interpolation claim of the solver benchmark (the separable
    limiting case of Section 4's interpolation-order discussion) is that when
    one mesh coordinate family is index-aligned, the ENGINE regrid IS
    one-dimensional non-uniform linear interpolation along each row, the class
    the SSJ reference solver uses. Verify the reduction to machine precision: on
    a mesh with straight y-lines, the regrid at the mesh's own y-coordinates
    must equal per-row ``np.interp`` exactly.
    """
    rng_local = np.random.default_rng(7)
    J, K = 40, 12
    y_lines = np.linspace(0.0, 4.0, K)
    # Warped x within each y-slice (sorted, non-uniform), straight y family.
    fx = np.sort(rng_local.uniform(0.0, 10.0, (J, K)), axis=0)
    fy = np.tile(y_lines[None, :], (J, 1))
    fv = np.sin(fx) + 0.5 * fy
    x_new = np.linspace(0.5, 9.5, 23)

    out = np.empty((x_new.size, K))
    engine2d_regrid_serial(x_new, y_lines, fx, fy, fv, out)

    for k in range(K):
        xk, vk = fx[:, k], fv[:, k]
        inside = (x_new >= xk[0]) & (x_new <= xk[-1])
        expected = np.interp(x_new[inside], xk, vk)
        np.testing.assert_allclose(out[inside, k], expected, rtol=0, atol=1e-13)
        # Out-of-range queries extend the boundary slope (np.interp clamps,
        # which is NOT the SSJ-class convention; the toolkit extrapolates).
        for x_out in x_new[~inside]:
            if x_out < xk[0]:
                slope = (vk[1] - vk[0]) / (xk[1] - xk[0])
                ref = vk[0] + slope * (x_out - xk[0])
            else:
                slope = (vk[-1] - vk[-2]) / (xk[-1] - xk[-2])
                ref = vk[-1] + slope * (x_out - xk[-1])
            np.testing.assert_allclose(out[x_new == x_out, k], ref, rtol=0, atol=1e-13)


@pytest.mark.parametrize(
    "kernel",
    [engine2d_regrid, engine2d_regrid_serial, engine2d_regrid_ipo],
)
@pytest.mark.parametrize("shape", [(1, 4), (4, 1)])
def test_regrid_kernels_raise_below_2x2(kernel, shape) -> None:
    """Every regrid kernel raises on a mesh below 2 x 2. A silent return there
    would hand back the uninitialized output array.
    """
    fx = np.ones(shape)
    queries = np.array([0.5, 1.5])
    with pytest.raises(ValueError, match="at least 2 x 2"):
        kernel(queries, queries, fx, fx, fx, np.empty((2, 2)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
