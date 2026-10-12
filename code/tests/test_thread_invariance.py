"""Thread-count invariance of the parallel two-asset backward steps.

The replication footnote in Section 3 claims the converged solver output is
identical across thread counts because the ``@njit(parallel=True)`` backward
step writes each grid cell independently, with no cross-cell reductions. Pin
that claim for both parallel solvers: solve the block at one thread and at
several, and require bit-identical policies, marginal values, and iteration
counts. The in-test ``get_num_threads`` assertion guarantees the two solves
really ran at different thread counts, so the test cannot silently no-op.
"""

from __future__ import annotations

import numba
import pytest
from ssj.benchmark import calibration, solve_compact, solve_negm

_ARRAY_KEYS = ("a", "b", "c", "Va", "Vb")


@pytest.fixture
def restore_threads():
    before = numba.get_num_threads()
    yield
    numba.set_num_threads(before)


@pytest.mark.usefixtures("restore_threads")
@pytest.mark.parametrize("solve", [solve_compact, solve_negm], ids=["egmn", "negm"])
def test_solver_output_thread_invariant(solve) -> None:
    """Converged two-asset output is bit-for-bit identical at 1 and >1 threads."""
    hi = min(numba.config.NUMBA_NUM_THREADS, 16)
    if hi < 2:
        pytest.skip("only one hardware thread available; cannot vary the count")
    cal = calibration(24, 20)
    out = {}
    for nt in (1, hi):
        numba.set_num_threads(nt)
        assert numba.get_num_threads() == nt
        out[nt] = solve(cal)
    assert out[1]["iter"] == out[hi]["iter"]
    for key in _ARRAY_KEYS:
        assert out[1][key].tobytes() == out[hi][key].tobytes(), (
            f"{key} differs between 1 and {hi} threads"
        )
