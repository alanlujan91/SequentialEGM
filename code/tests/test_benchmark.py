"""Benchmark regression: compact EGM$^n$ ``hh`` vs shipped sequence-jacobian.

Stable gates only: the two solvers agree on policies, and the compact solver
converges toward truth under grid refinement.
"""

from __future__ import annotations

import csv
import functools

import numpy as np
import pytest
from repo_paths import FIG_DIR
from ssj import bench_components, bench_interp, bench_ssj_reference, bench_timing
from ssj.bench_common import INTERP_GRIDS, SOLVE_TIME_GRIDS, best_of, solve_egm_k
from ssj.bench_interp import extrapolation_counts, points_in_polygon
from ssj.benchmark import (
    MAXIT,
    assert_mesh_fold_free,
    assert_mesh_ipm,
    backward_args,
    calibration,
    policy_distance,
    robust_euler_stats,
    solve_compact,
    solve_negm,
    solve_shipped,
    stage0_mesh,
)


@functools.cache
def solved(solver, n_b: int, n_a: int) -> dict:
    """``solver`` at the ``(n_b, n_a)`` calibration, solved once per session."""
    return solver(calibration(n_b, n_a))


def assert_policies_agree(sol, shipped, cal, label) -> None:
    """``sol`` and the shipped SSJ solution agree to a 5% median relative
    difference in all three policies.
    """
    for key in ("a", "b", "c"):
        d = policy_distance(sol, cal, shipped, cal, key)
        assert d < 0.05, f"{key}: {label} vs shipped median rel diff {d:.2e}"


def test_compact_agrees_with_shipped() -> None:
    """Shipped SSJ and compact EGM$^n$ produce the same policies (same problem)."""
    cal = calibration(24, 20)
    assert_policies_agree(
        solved(solve_compact, 24, 20),
        solved(solve_shipped, 24, 20),
        cal,
        "compact",
    )


def test_compact_converges_under_refinement() -> None:
    """Compact policies move toward the fine-grid solution (consumption)."""
    coarse, fine = calibration(24, 20), calibration(48, 40)
    kc, kf = solved(solve_compact, 24, 20), solved(solve_compact, 48, 40)
    assert policy_distance(kc, coarse, kf, fine, "c") < 0.1


def test_compact_beats_shipped_on_accuracy() -> None:
    """EGM$^n$'s converged policy is closer to the shipped solver's OWN refined
    solution than the shipped solver's coarse solution is (NEGM covers the
    on-grid residual comparison in test_negm_agrees_with_shipped).
    """
    coarse, fine = calibration(48, 40), calibration(96, 80)
    anchor = solved(solve_shipped, 96, 80)
    dc = policy_distance(solved(solve_compact, 48, 40), coarse, anchor, fine, "c")
    ds = policy_distance(solved(solve_shipped, 48, 40), coarse, anchor, fine, "c")
    assert dc < ds, f"compact {dc:.4f} vs shipped {ds:.4f} to the shipped anchor"
    # Absolute threshold: beating shipped (ds ~ 0.023) is necessary but not sufficient.
    assert dc < 0.02, f"compact distance {dc:.4f} regressed from the 0.0142 level"
    rc = robust_euler_stats(solved(solve_compact, 48, 40), coarse)[0]
    assert rc < 5e-5, f"regular-grid inversion residual {rc:.2e} unexpectedly large"


def test_negm_agrees_with_shipped() -> None:
    """The restored NEGM root-find baseline solves the same household: policies
    agree with shipped, and (lacking the Euler correction) its on-grid residual
    sits above EGM$^n$'s (Section 3).
    """
    cal = calibration(24, 20)
    ng = solved(solve_negm, 24, 20)
    assert_policies_agree(ng, solved(solve_shipped, 24, 20), cal, "NEGM")
    assert (
        robust_euler_stats(ng, cal)[0]
        > robust_euler_stats(solved(solve_compact, 24, 20), cal)[0]
    )


@pytest.mark.parametrize(("n_b", "n_a"), [(40, 60), (80, 120), (160, 240)])
def test_stage0_mesh_fold_free_ipm_and_finite(n_b: int, n_a: int) -> None:
    """Stage 0 is finite, fold-free, and index-monotone at each timing-table refinement.

    The mesh is the kernel's Phase 1 output; the figure generator separately
    checks the displayed 20 x 20 mesh. Together the two asserts cover both of
    ENGINE's mesh conditions, the paper's "checked directly" claim.
    """
    cal = calibration(n_b, n_a)
    mesh = stage0_mesh(solved(solve_compact, n_b, n_a), cal)
    assert np.all(np.isfinite(mesh["b_endo"]))
    assert np.all(np.isfinite(mesh["a_endo"]))
    assert np.all(np.isfinite(mesh["c"]))
    assert_mesh_fold_free(mesh["b_endo"], mesh["a_endo"])  # raises on a fold
    assert_mesh_ipm(mesh["b_endo"], mesh["a_endo"])  # raises on a non-monotone line


def make_monotone_mesh(b_sign: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A (1, 3, 3) mesh with ``b[0, i, j] = b_sign * i`` and ``a[0, i, j] = j``.

    Shared by the fold-free and IPM guard tests below, which each perturb
    this baseline to check the corresponding rejection path.
    """
    b = np.empty((1, 3, 3))
    a = np.empty((1, 3, 3))
    for i in range(3):
        for j in range(3):
            b[0, i, j] = b_sign * i
            a[0, i, j] = float(j)
    return b, a


def test_assert_mesh_fold_free_catches_fold() -> None:
    """The fold check is not a no-op: a punched-in fold is rejected."""
    b, a = make_monotone_mesh()
    assert_mesh_fold_free(b, a)  # monotone mesh: no raise
    a[0, 1, 1] = -5.0
    with pytest.raises(RuntimeError):
        assert_mesh_fold_free(b, a)


def test_assert_mesh_fold_free_rejects_reversed_orientation() -> None:
    """A consistently reversed mesh is rejected despite having no sign changes."""
    b, a = make_monotone_mesh(b_sign=-1.0)
    with pytest.raises(RuntimeError, match="non-positive"):
        assert_mesh_fold_free(b, a)


def test_assert_mesh_ipm_catches_each_axis() -> None:
    """The IPM check is not a no-op: each coordinate's own axis is rejected separately."""
    b, a = make_monotone_mesh()
    assert_mesh_ipm(b, a)  # monotone mesh: no raise
    b_bad = b.copy()
    b_bad[0, 1, 1] = (
        2.5  # column becomes 0 -> 2.5 -> 2: decreasing step along b's own axis
    )
    with pytest.raises(RuntimeError, match="liquid coordinate"):
        assert_mesh_ipm(b_bad, a)
    a_bad = a.copy()
    a_bad[0, 1, 2] = 0.5  # decreasing step along a's own axis
    with pytest.raises(RuntimeError, match="illiquid coordinate"):
        assert_mesh_ipm(b, a_bad)


def test_extrapolation_classifier_includes_polygon_boundary() -> None:
    """Mesh-boundary nodes are interpolation points, not extrapolation points."""
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    points = np.array(
        [
            [0.5, 0.5],
            [0.0, 0.5],
            [1.0, 1.0],
            [-0.1, 0.5],
            [0.5, 1.1],
        ],
    )
    np.testing.assert_array_equal(
        points_in_polygon(points, polygon),
        [True, True, True, False, False],
    )


@pytest.mark.parametrize(("n_b", "n_a"), [(24, 20), (48, 40)], ids=["coarse", "medium"])
def test_stage0_extrapolation_frequency_is_measured(n_b: int, n_a: int) -> None:
    """The coarse and medium production meshes reproduce the published outside
    count: one query in the lowest income state.
    """
    cal = calibration(n_b, n_a)
    by_income, pooled = extrapolation_counts(cal, solved(solve_compact, n_b, n_a))
    assert pooled["n_queries"] == 3 * n_b * n_a
    assert pooled["n_outside"] == sum(row["n_outside"] for row in by_income)
    assert [row["n_outside"] for row in by_income] == [1, 0, 0]
    assert pooled["outside_fraction"] == pytest.approx(1 / (3 * n_b * n_a))


def test_iterate_reports_convergence_contract() -> None:
    """A converged solve carries ``converged=True`` and an iteration count
    strictly below the cap; the published readings rely on this flag.
    """
    out = solved(solve_compact, 24, 20)
    assert out["converged"] is True
    assert out["iter"] < MAXIT


def test_solvers_reject_unconverged_solve() -> None:
    """A capped solve raises, so it never produces a normal-looking timing or
    accuracy row; a caller that records the failure itself passes
    ``require=False`` and gets the unconverged result.
    """
    # Both timing scripts time through the one helper.
    assert bench_timing.best_of is bench_ssj_reference.best_of is best_of
    cal = calibration(24, 20)
    args = backward_args(cal)
    with pytest.raises(RuntimeError, match="failed to converge"):
        solve_egm_k(args, 1, cal, maxit=2)
    assert not solve_egm_k(args, 1, cal, maxit=2, require=False)["converged"]


def test_require_structured_converged_rejects_failed_method() -> None:
    """A failed ENGINE or Curvilinear solve aborts table generation, while
    Delaunay's expected non-convergence stays tolerated.
    """

    def record(reason):
        return {"out": {"converged": reason == "", "reason": reason}}

    records = {}
    for label, _, _ in INTERP_GRIDS:
        records[label, "engine"] = record("")
        records[label, "curvilinear"] = record("")
        records[label, "delaunay"] = record("no_convergence")
    bench_interp.require_structured_converged(records)
    records["Coarse", "engine"] = record("no_convergence")
    with pytest.raises(RuntimeError, match="ENGINE"):
        bench_interp.require_structured_converged(records)
    records["Coarse", "engine"] = record("")
    records["Medium", "delaunay"] = record("nan_policy")
    with pytest.raises(RuntimeError, match="Delaunay"):
        bench_interp.require_structured_converged(records)


def committed_csv(name: str) -> list[dict[str, str]]:
    """Rows of the committed figure CSV ``name``."""
    with (FIG_DIR / name).open() as f:
        return list(csv.DictReader(f))


def assert_regenerates(tmp_path, monkeypatch, module, write, data, table) -> None:
    """``write(data)`` into a scratch ``FIG_DIR`` of ``module`` reproduces the
    committed ``table`` byte for byte.
    """
    monkeypatch.setattr(module, "FIG_DIR", tmp_path)
    write(data)
    assert (tmp_path / table).read_text() == (FIG_DIR / table).read_text()


def test_refinement_bounds_table_roundtrips_from_csv(tmp_path, monkeypatch) -> None:
    """The committed table regenerates byte-for-byte from the committed CSV."""
    rows = [
        [
            r["exercise"],
            r["label"],
            int(r["nB"]),
            int(r["nA"]),
            r["rel_median"],
            r["rel_p95"],
            r["rel_max"],
            r["iters"],
            r["iters_ref"],
        ]
        for r in committed_csv("refinement_bounds_abrs.csv")
    ]
    assert_regenerates(
        tmp_path,
        monkeypatch,
        bench_interp,
        bench_interp.write_refinement_bounds_table,
        rows,
        "tab_refinement_bounds_abrs.md",
    )


def test_extrapolation_table_roundtrips_from_csv(tmp_path, monkeypatch) -> None:
    """The committed table regenerates byte-for-byte from the committed CSV."""
    table_rows = []
    for r in committed_csv("extrap_frequency_abrs.csv"):
        if r["income_state"] == "all":
            pooled = {
                "n_outside": int(r["n_outside"]),
                "n_queries": int(r["n_queries"]),
                "outside_fraction": float(r["outside_fraction"]),
            }
            table_rows.append((r["resolution"], int(r["nB"]), int(r["nA"]), pooled))
    assert_regenerates(
        tmp_path,
        monkeypatch,
        bench_interp,
        bench_interp.write_extrapolation_table,
        table_rows,
        "tab_extrap_frequency_abrs.md",
    )


def test_solvetime_table_roundtrips_from_csv(tmp_path, monkeypatch) -> None:
    """The committed table regenerates byte-for-byte from the committed CSV;
    non-converged rows stay out of the displayed table.
    """
    label_by_lower = {label.lower(): label for label, _, _ in INTERP_GRIDS}
    records = {}
    for rec in committed_csv("solvetime_abrs.csv"):
        ok = rec["status"] == "ok"
        stats = ("log10_median", "log10_p95", "log10_max")
        records[label_by_lower[rec["resolution"]], rec["method"]] = {
            "time": float(rec["solve_seconds"]) if ok else None,
            "out": {"converged": ok},
            "stats": tuple(float(rec[s]) for s in stats) if ok else None,
        }
    assert_regenerates(
        tmp_path,
        monkeypatch,
        bench_interp,
        bench_interp.write_solvetime_table,
        records,
        "tab_solvetime_abrs.md",
    )


def test_ssj_reference_table_roundtrips_from_csv(tmp_path, monkeypatch) -> None:
    """The committed table regenerates byte-for-byte from the committed CSV,
    including the accuracy cells displayed from CSV-persisted precision.
    """
    stored = {}
    for rec in committed_csv("ssj_reference_abrs.csv"):
        g = (int(rec["nB"]), int(rec["nA"]))
        stored[rec["solver"], g, int(rec["threads"])] = rec
    m = {}
    for g in SOLVE_TIME_GRIDS:
        s_ref = stored["reference", g, 1]
        s_e1 = stored["egmn", g, 1]
        s_e16 = stored["egmn", g, 16]
        m["t_ref", g] = float(s_ref["seconds"])
        m["t_egm", g, 1] = float(s_e1["seconds"])
        m["t_egm", g, 16] = float(s_e16["seconds"])
        m["iters", g] = (int(s_e1["iters"]), int(s_ref["iters"]))
        m["acc", g] = (
            float(s_e1["offgrid_log10_med"]),
            float(s_ref["offgrid_log10_med"]),
        )
    assert_regenerates(
        tmp_path,
        monkeypatch,
        bench_ssj_reference,
        bench_ssj_reference.write_table,
        m,
        "tab_ssj_reference.md",
    )


def test_timing_negm_table_roundtrips_from_csv(tmp_path, monkeypatch) -> None:
    """The committed table regenerates byte-for-byte from the committed CSV."""
    t = {}
    solver_key = {"egmn": "egm", "negm": "negm"}
    for rec in committed_csv("timing_negm.csv"):
        g = (int(rec["nB"]), int(rec["nA"]))
        t[solver_key[rec["solver"]], g, int(rec["threads"])] = float(rec["seconds"])
    assert_regenerates(
        tmp_path,
        monkeypatch,
        bench_timing,
        bench_timing.write_negm,
        t,
        "tab_timing_negm.md",
    )


def test_components_table_roundtrips_from_csv(tmp_path, monkeypatch) -> None:
    """The committed table regenerates byte-for-byte from the committed CSV."""
    m = {}
    for rec in committed_csv("components_abrs.csv"):
        g = (int(rec["nB"]), int(rec["nA"]))
        acc, _ = m.setdefault(g, ({}, None))
        acc[rec["component"].replace("_", " ")] = float(rec["seconds"])
        m[g] = (acc, int(rec["iters"]))
    assert_regenerates(
        tmp_path,
        monkeypatch,
        bench_components,
        bench_components.write_table,
        m,
        "tab_components_abrs.md",
    )
