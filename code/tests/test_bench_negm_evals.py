"""The NEGM FOC-evaluation counter reproduces the production solve and counts by
its stated convention.
"""

from __future__ import annotations

import numpy as np
from ssj.bench_negm_evals import counted_solve
from ssj.benchmark import calibration, solve_negm


def test_counted_solve_is_the_production_solve() -> None:
    """At (24, 20) the counted solve converges to the production NEGM fixed point
    bit for bit, and every node's count follows the convention: 1 at the lower
    corner, at least 2 elsewhere, and only bracketed nodes are interior.
    """
    cal = calibration(24, 20)
    out, steps = counted_solve(cal)
    ref = solve_negm(cal)
    assert out["converged"]
    assert out["iter"] == ref["iter"] == len(steps)
    for key in ("Va", "Vb", "a", "b", "c"):
        assert np.array_equal(out[key], ref[key])
    n, interior, _bisected = steps[-1]
    lower = out["a"] == cal["a"][0]
    assert np.all(n[lower & (interior == 0)] == 1)
    assert np.all(n[~lower] >= 2)
    assert np.all(n[interior == 1] >= 2)
