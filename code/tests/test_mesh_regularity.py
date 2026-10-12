"""Tests for the mesh-regularity diagnostics and the validator's grid checks."""

from __future__ import annotations

from typing import Never

import numpy as np
import pytest
import validate_claims
from engine.utilities import (
    bilinear_contains,
    corner_jacobians,
    empirical_ipo_alpha,
    ipm_violations,
    nan_nodes,
    reached_folds,
)
from HARKmodels.ConsHealthSeparableModel import (
    SequentialHealthConsumerType,
    paper_timing_params,
)


def index_mesh(x_of, y_of, J=4, K=5):
    """``(x, y)`` built from the index grids ``jj, kk`` by two maps."""
    jj, kk = np.meshgrid(
        np.arange(J, dtype=float),
        np.arange(K, dtype=float),
        indexing="ij",
    )
    return x_of(jj, kk), y_of(jj, kk)


def brute_corner_jacobian(x, y, s, t):
    """Bilinear-map Jacobian ``x_s y_t - x_t y_s`` at corner ``(s, t)`` of every cell."""
    J, K = x.shape
    out = np.empty((J - 1, K - 1))
    for j in range(J - 1):
        for k in range(K - 1):
            d = []
            for a in (x, y):
                d_s = (1 - t) * (a[j + 1, k] - a[j, k]) + t * (
                    a[j + 1, k + 1] - a[j, k + 1]
                )
                d_t = (1 - s) * (a[j, k + 1] - a[j, k]) + s * (
                    a[j + 1, k + 1] - a[j + 1, k]
                )
                d.append((d_s, d_t))
            (x_s, x_t), (y_s, y_t) = d
            out[j, k] = x_s * y_t - x_t * y_s
    return out


@pytest.fixture
def results():
    """Empty the validator's result buckets around one test."""
    for bucket in validate_claims.RESULTS.values():
        bucket.clear()
    yield validate_claims.RESULTS
    for bucket in validate_claims.RESULTS.values():
        bucket.clear()


def test_corner_jacobians_match_bilinear_map() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(5, 7)), rng.normal(size=(5, 7))
    jac = corner_jacobians(x, y)
    assert jac.shape == (4, 4, 6)
    for s in (0, 1):
        for t in (0, 1):
            np.testing.assert_array_equal(
                jac[2 * s + t],
                brute_corner_jacobian(x, y, s, t),
            )


def test_ipm_violations_test_each_coordinate_along_its_own_axis() -> None:
    # Rotated lattice: x falls along axis 1 and y rises along axis 0, both
    # allowed by def-ipm.
    x, y = index_mesh(lambda j, k: j - 0.5 * k, lambda j, k: 0.5 * j + k)
    assert ipm_violations(x, y) == (0, 0)
    y[1, 3] = y[1, 2]
    x[2, 0] = np.nan
    assert ipm_violations(x, y) == (2, 1)


def test_empirical_ipo_alpha_counts_bracket_shift() -> None:
    # Each k-slice shifts the x-nodes up by one node spacing, so every probe's
    # bracket drops by one index per slice.
    x, _ = index_mesh(lambda j, k: j + k, lambda j, k: k, J=12, K=3)
    assert empirical_ipo_alpha(x) == 1


def bilinear_point(corners, s, t):
    """The bilinear map of ``corners = (p00, p10, p01, p11)`` at ``(s, t)``."""
    p00, p10, p01, p11 = (np.asarray(p, dtype=float) for p in corners)
    return (1 - s) * (1 - t) * p00 + s * (1 - t) * p10 + (1 - s) * t * p01 + s * t * p11


# One cell of the 60x50 health mesh's folded k = 0 sliver: IPM holds, the
# (s, t) = (1, 1) corner Jacobian is negative.
SLIVER = ((10.935, -5.286), (34.788, -10.208), (1.224, 1.336), (1.944, 1.081))


def sliver_mesh():
    """``(x, y)`` of the 2x2 mesh whose one cell is ``SLIVER``."""
    pts = np.array(SLIVER).reshape(2, 2, 2).transpose(1, 0, 2)
    return pts[..., 0], pts[..., 1]


@pytest.mark.parametrize(
    "corners",
    [
        ((0, 0), (2, 0.3), (0.2, 1.5), (1.5, 2.5)),
        ((0, 0), (1, 0), (0, 1), (1, 1 + 1e-15)),  # twist at rounding level
        SLIVER,
    ],
)
def test_bilinear_contains_finds_forward_mapped_points(corners) -> None:
    st = np.random.default_rng(1).uniform(size=(400, 2))
    q = np.array([bilinear_point(corners, s, t) for s, t in st])
    assert bilinear_contains(corners, q).all()
    assert not bilinear_contains(corners, q + 10.0).any()


def test_reached_folds_counts_only_cells_a_query_reaches() -> None:
    x, y = sliver_mesh()
    assert ipm_violations(x, y) == (0, 0)
    inside = bilinear_point(SLIVER, 0.05, 0.95)[None, :]
    folded, reached = reached_folds(x, y, inside)
    assert folded.tolist() == [[0, 0]]
    assert reached.tolist() == [True]
    _, reached = reached_folds(x, y, np.array([[100.0, 100.0]]))
    assert reached.tolist() == [False]


def test_grid_regularity_fails_y_not_increasing_along_axis_1(results) -> None:
    # Lower-left Jacobian (1)(-1) - (-1)(2) = 1 > 0 everywhere, x rises along
    # axis 0, but y falls along axis 1: def-ipm fails.
    x, y = index_mesh(lambda j, k: j - k, lambda j, k: 2 * j - k)
    validate_claims.validate_grid_regularity(
        [(x, y)],
        np.array([[100.0, 100.0]]),
        1,
    )
    assert not results["passed"]
    assert len(results["failed"]) == 1


def test_grid_regularity_passes_ipm_mesh_with_x_falling_along_axis_1(results) -> None:
    x, y = index_mesh(lambda j, k: j - 0.5 * k, lambda j, k: 0.5 * j + k)
    validate_claims.validate_grid_regularity([(x, y)], np.array([[1.0, 1.0]]), 2)
    assert not results["failed"]
    assert len(results["passed"]) == 1
    assert "IPM" in results["passed"][0][0]


FAR = np.array([[100.0, 100.0]])  # outside every test cell


def test_grid_regularity_scopes_folds_to_reached_cells(results) -> None:
    x, y = sliver_mesh()
    validate_claims.validate_grid_regularity([(x, y)], FAR, 0)
    assert len(results["passed"]) == 1
    inside = bilinear_point(SLIVER, 0.05, 0.95)[None, :]
    validate_claims.validate_grid_regularity([(x, y)], inside, 0)
    assert len(results["failed"]) == 1


def test_grid_regularity_fails_folded_cell_beside_kink(results) -> None:
    # The sliver is cell row j = 0; no query reaches it, but a fold beside
    # the constructed rows fails on its own.
    x, y = sliver_mesh()
    validate_claims.validate_grid_regularity([(x, y)], FAR, 1)
    assert not results["passed"]
    assert len(results["failed"]) == 1
    assert "beside the kink" in results["failed"][0][1]


def test_kink_cell_rows_from_constrained_row_count() -> None:
    agent = SequentialHealthConsumerType(**paper_timing_params())
    assert validate_claims.kink_cell_rows(agent) == agent.constrained_N + 2


def test_health_solve_crash_exits_inconclusive(results, monkeypatch) -> None:
    def crash() -> Never:
        msg = "health solve failed"
        raise FloatingPointError(msg)

    for name in dir(validate_claims):
        if name.startswith("validate_"):
            monkeypatch.setattr(validate_claims, name, lambda *_: None)
    monkeypatch.setattr(validate_claims, "validate_full_model_speedup", crash)
    monkeypatch.setattr(validate_claims, "warmup_jit", lambda: None)
    assert validate_claims.main() == 2
    assert len(results["errors"]) == 2


def test_fit_r_squared_undefined_for_constant_data() -> None:
    assert np.isnan(validate_claims.fit_r_squared(np.arange(5.0), np.full(5, 3.0)))
    assert validate_claims.fit_r_squared(np.arange(5.0), 2 * np.arange(5.0)) == 1.0


def test_empirical_ipo_alpha_undefined_on_non_finite_mesh() -> None:
    x, _ = index_mesh(lambda j, k: j + k, lambda j, k: k, J=12, K=3)
    x[5, 1] = np.nan
    assert np.isnan(empirical_ipo_alpha(x))
    x[5, 1] = np.inf
    assert np.isnan(empirical_ipo_alpha(x))


def test_nan_nodes_counts_nodes_with_either_coordinate_non_finite() -> None:
    x, y = index_mesh(lambda j, k: j + 0.0 * k, lambda j, k: k + 0.0 * j)
    assert nan_nodes(x, y) == 0
    x[0, 0] = np.nan
    y[0, 0] = np.nan
    y[1, 2] = np.inf
    assert nan_nodes(x, y) == 2
