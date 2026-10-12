"""Regression: ``numerics.require_finite`` raises on non-finite input under any
interpreter flag, where the ``assert`` checks it replaced vanished under ``-O``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numerics import require_finite


def test_require_finite_passes_finite_arrays() -> None:
    require_finite(a=np.arange(4.0), b=np.ones((2, 2)))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_require_finite_names_the_array_and_counts(bad) -> None:
    b = np.array([1.0, bad, bad])
    with pytest.raises(RuntimeError, match=r"^b contains NaN/Inf: 2 bad values$"):
        require_finite(a=np.zeros(3), b=b)
