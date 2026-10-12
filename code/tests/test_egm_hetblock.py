"""Compact EGM$^n$ two-asset @het block tests.

Validates the compact ``hh`` block: the ``@het`` contract, the ``build_l_grid``
hetinput, and closed-form (no root-find) source.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ssj.hh_egm import build_l_grid, hh


def test_compact_hetblock_contract() -> None:
    """The compact ``hh`` exposes the upstream ``@het`` contract."""
    assert sorted(hh.policy) == ["a", "b"]
    back = set(getattr(hh, "back_iter_vars", getattr(hh, "backward", [])))
    assert {"Va", "Vb"} <= back
    assert {"A", "B", "C", "UCE"} <= set(hh.outputs)


def test_build_l_grid_extends_and_is_increasing() -> None:
    """The Stage-1 liquid grid starts at b_grid[0], is strictly increasing,
    and extends past b_grid for withdrawal headroom.
    """
    b = np.array([0.0, 0.5, 2.0, 10.0])
    a = np.array([0.0, 5.0, 50.0])
    e = np.array([0.5, 1.0, 1.5])
    lg = build_l_grid(b, a, e, 0.0075, 0.0125, 0.25, 6.5, 2.0)
    assert lg[0] == b[0]
    assert np.all(np.diff(lg) > 0)
    assert lg[-1] >= b[-1]


def test_no_rootfinding_in_source() -> None:
    """Closed-form inversion only: no solver / NEGM-search primitive in source."""
    text = (Path(__file__).resolve().parents[1] / "ssj" / "hh_egm.py").read_text()
    for banned in ("brentq", "bisect", "newton", "fsolve", "root_scalar", "optimize"):
        assert banned not in text, f"root-finder {banned!r} in hh_egm.py"
    assert "lhs_equals_rhs_interpolate(" not in text
