"""Generate All Figures for ENGINE Paper.

Draws every paper figure cheap enough for one pass, then checks that each paper
figure file exists and that the ones drawn here were rewritten by this run.

To generate all figures, from the repo root:

    PYTHONPATH=code uv run python code/figures/generate_all_paper_figures.py

Each figure script also runs on its own with the same ``PYTHONPATH=code`` prefix.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from cli import start_logging
from figure_styles import FIGURE_DIR
from generate_consumption_labor_figure import (
    generate_consumption_labor_figure,
)
from generate_grid_homeomorphism import generate_grid_homeomorphism
from generate_grid_transformation import generate_grid_transformation
from generate_grid_transformation_abrs import (
    generate_grid_transformation_abrs,
)
from generate_health_model_summary import generate_health_model_summary
from generate_mapping import generate_mapping
from generate_partition_comparison import generate_partition_comparison
from timing import timed

logger = logging.getLogger(__name__)

# (file name, generator) for every paper figure this script draws, by section.
FIGURES = (
    # Section 2: consumption-labor model solution surfaces
    ("consumption_labor_solution", generate_consumption_labor_figure),
    # Section 3: two-asset Stage 0 exogenous vs endogenous mesh
    ("grid_transformation_abrs", generate_grid_transformation_abrs),
    # Online Appendix A: health model exogenous vs endogenous grid
    ("grid_transformation", generate_grid_transformation),
    # Section 4: ENGINE interpolation
    ("grid_homeomorphism", generate_grid_homeomorphism),
    ("engine_interpolation_visual", generate_mapping),
    ("partition_comparison", generate_partition_comparison),
    # Section 4: health model benchmark summary
    ("health_model_summary", generate_health_model_summary),
)

# Paper figures too expensive to draw here: health_model_interpolation_benchmark.py
# and health_warp_sweep.py write them.
SEPARATELY_RUN = ("health_model_interpolation_errors", "health_warp_sweep")


def generate_all_figures():
    """Draw every figure in ``FIGURES``, returning the names drawn and failed."""
    generated, failed = [], []
    total_start = time.perf_counter()
    for name, func in FIGURES:
        try:
            _, elapsed = timed(func)
        except Exception:
            failed.append(name)
            logger.exception("FAILED %s", name)
            continue
        generated.append(name)
        logger.info("generated %s in %.1fs", name, elapsed)

    logger.info(
        "%d figures generated, %d failed, %.1fs total",
        len(generated),
        len(failed),
        time.perf_counter() - total_start,
    )
    return generated, failed


def verify_figure_files(run_start):
    """Whether every paper figure exists as PNG and PDF, those drawn by
    ``FIGURES`` written no earlier than ``run_start`` (epoch seconds).
    """
    fresh = {name for name, _ in FIGURES}
    ok = True
    for name in (*fresh, *SEPARATELY_RUN):
        for path in (Path(FIGURE_DIR) / f"{name}.{ext}" for ext in ("png", "pdf")):
            if not path.exists():
                logger.error("missing %s", path)
                ok = False
            elif name in fresh and path.stat().st_mtime < run_start:
                logger.error("stale %s (not rewritten by this run)", path)
                ok = False
    return ok


if __name__ == "__main__":
    start_logging()
    run_start = time.time()
    _generated, failed = generate_all_figures()
    sys.exit(0 if verify_figure_files(run_start) and not failed else 1)
