"""Shared ENGINE interpolator configuration for the health model.

Health meshes measure a bracket-displacement constant of 10-16 (Online
Appendix A, "Accuracy against Measured Warp"), so the health figure scripts and
validators run ENGINE with its search margin matched to that measurement. The
margin is a speed setting: ENGINE verifies every windowed bracket and falls back
to a full row search on a miss, so the default ``curvature_bound=2`` yields the
same values with more fallback searches.
"""

from __future__ import annotations

import functools

import numpy as np
from engine.engine2d import Engine2DInterp
from HARK.distributions import expected

from HARKmodels.ConsHealthSeparableModel import calc_exp_next, post_decision_mesh

HEALTH_CURVATURE_BOUND = 16

HealthEngine2DInterp = functools.partial(
    Engine2DInterp,
    curvature_bound=HEALTH_CURVATURE_BOUND,
)


def continuation_queries(agent):
    """Every ``(m', h')`` at which ``agent``'s backward step evaluates next
    period's interpolant, shape ``(n, 2)``.

    Records the arguments ``calc_exp_next`` passes to next period's policy
    over the post-decision ``(a, H)`` mesh and every shock atom. The points
    come from the model's own law of motion.
    """
    seen = []

    def record(m, h):
        m, h = np.broadcast_arrays(m, h)
        seen.append(np.column_stack([m.ravel(), h.ravel()]))
        one = np.ones(m.shape)
        return one, one, one

    _, a_mat, h_mat = post_decision_mesh(
        agent.aLvlGrid,
        agent.HLvlGrid,
        agent.ShockDstn[0],
    )
    expected(
        calc_exp_next,
        agent.ShockDstn[0],
        args=(
            a_mat,
            h_mat,
            agent.Rfree[0],
            agent.CRRA,
            agent.HealthProdExp,
            agent.HealthProdFac,
            record,
        ),
    )
    return np.concatenate(seen)
