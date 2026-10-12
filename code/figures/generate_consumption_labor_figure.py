"""Generate the consumption-labor model figure for Section 2.

A publication-quality 1x3 figure showing the Sequential EGM solution to the
consumption-labor model of Section 2. Both stages invert; there is no portfolio
choice and no permanent income, neither of which bears on sequential EGM.

Panels:
- (a): Endogenous grid from the labor-leisure stage, showing curvilinear structure
- (b): Consumption policy function
- (c): Leisure policy function
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from figure_styles import (
    add_right_colorbar,
    apply_engine_style,
    label_panel,
    plot_mesh_lines,
    save_engine_figure,
    scatter_with_colorbar,
)
from HARKmodels.ConsLaborSeparableModel import LaborSeparableConsumerType
from repo_paths import FIG_DIR, write_rows

# MPC plateau window, the cash-on-hand levels whose expected future labor the
# paper quotes, the scan for where it first hits 0, and the simulation size.
_PLATEAU = (20.0, 25.0)
_LABOR_M = (2.0, 7.0, 13.0, 16.0, 20.0)
_ZERO_SCAN = np.arange(14.0, 30.0, 0.25)
_N_PATHS = 4000


def solve_labor_model():
    """Solve the consumption-labor model and return the agent."""
    # Calibration lives in init_labor_separable; see the paper's parameter footnote.
    agent = LaborSeparableConsumerType()
    agent.cycles = 30  # Longer horizon for clearer precautionary savings

    agent.solve()

    return agent


def plot_endogenous_grid_panel(ax, sol) -> None:
    """Panel (a): endogenous grid from the labor-leisure stage, colored by
    the labor policy (already projected onto [0, 1] by the solver).
    """
    grids = sol.labor_leisure.grids
    b_endog = grids["bNrm"]
    theta = grids["tShk"]
    labor = grids["labor"]
    n_m, n_theta = b_endog.shape

    # Curvilinear endogenous grid lines: constant θ (columns, b against m) and
    # constant m (rows, b against θ).
    plot_mesh_lines(
        ax,
        b_endog,
        theta,
        labels=("constant $\\theta_t$", "constant $m_t$"),
        step=(max(1, n_m // 12), max(1, n_theta // 8)),
        linewidth=0.8,
        alpha=0.6,
        zorder=1,
    )

    scatter_with_colorbar(
        ax,
        b_endog.flatten(),
        theta.flatten(),
        labor.flatten(),
        "Labor Supply $n_t$",
        s=6,
    )

    label_panel(
        ax,
        "(a)",
        "Endogenous Grid",
        "Liquid Assets $a_t$",
        "Wage Shock $\\theta_t$",
    )
    ax.legend(loc="center right", frameon=False, fontsize=9)
    # Symlog x: the log-spaced grid packs the region where labor varies into
    # b < ~10; a linear axis to b ~ 550 hides all of the policy variation.
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlim(left=-0.05)
    ax.grid(False)  # No background grid - the curvilinear lines ARE the grid


def plot_consumption_by_age_panel(ax, agent) -> None:
    """Panel (b): consumption function c(m), middle θ, all ages.

    Drops the terminal solution (consumes everything, c = m) and the final
    decision period before it (steep enough to flatten every other age):
    both are finite-horizon artifacts, not features of sequential EGM.
    """
    m_eval = np.linspace(0.5, 30, 100)
    n_periods = len(agent.solution) - 2

    # Color gradient for ALL ages (young=light, old=dark)
    colors_c = plt.cm.Blues(np.linspace(0.25, 0.95, n_periods))

    for t in range(n_periods):
        # consumption_saving.c_func takes m directly
        c_func_t = agent.solution[t].consumption_saving.c_func
        ax.plot(
            m_eval,
            c_func_t(m_eval),
            color=colors_c[t],
            linewidth=1.5,
            alpha=0.8,
        )

    # Add colorbar for age gradient
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(0, n_periods - 1))
    sm.set_array([])
    add_right_colorbar(ax, sm, "Age $t$")

    # Color gradient indicates age; terminal and final decision period excluded
    label_panel(
        ax,
        "(b)",
        "Consumption by Age",
        "Market Resources $m_t$",
        "Consumption $c_t$",
    )
    ax.set_xlim(0, m_eval.max())
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, linewidth=0.5)


def plot_leisure_by_wage_panel(ax, sol) -> None:
    """Panel (c): leisure function z(m, θ), youngest age (t=0), all θ values."""
    # Use the grids from the labor-leisure stage, which have leisure indexed by (m, θ)
    grids = sol.labor_leisure.grids
    m_grid_data = grids["mNrm"]
    theta_grid_data = grids["tShk"]
    leisure_data = grids["leisure"]

    # Select representative θ values
    n_theta_lines = min(6, theta_grid_data.shape[1])
    theta_indices = np.linspace(
        0,
        theta_grid_data.shape[1] - 1,
        n_theta_lines,
        dtype=int,
    )

    # Color gradient for different θ values
    colors_z = plt.cm.Greens(np.linspace(0.4, 0.9, n_theta_lines))

    for idx, k in enumerate(theta_indices):
        m_slice = m_grid_data[:, k]
        z_slice = leisure_data[:, k]
        theta_val = theta_grid_data[0, k]
        ax.plot(
            m_slice,
            z_slice,
            color=colors_z[idx],
            linewidth=2.0,
            label=f"$\\theta = {theta_val:.2f}$",
        )

    # Panel (c) shows period 0 only, with different wage shock realizations
    label_panel(
        ax,
        "(c)",
        "Leisure by Wage Shock (period 0 only)",
        "Market Resources $m_t$",
        "Leisure $z_t$",
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    # Leisure saturates at the full-leisure bound by m ~ 4; a wider window is
    # dead space.
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, linewidth=0.5)


def segment_slopes(c_func):
    """``(m_left, slope)`` of every linear segment of the piecewise-linear
    ``c_func``: its marginal propensity to consume, exactly.
    """
    m = np.asarray(c_func.x_list)
    return m[:-1], np.diff(np.asarray(c_func.y_list)) / np.diff(m)


def mpc_rows(c_func):
    """``(quantity, m, value)`` rows of the age-0 MPC profile: the first segment
    (the limit as ``m -> 0``), the segment at ``m = 0.5``, the trough over
    ``[0.5, 30)`` and the average slope over ``[20, 25]``.
    """
    m_left, slope = segment_slopes(c_func)
    at_half = np.searchsorted(m_left, 0.5, side="right") - 1
    window = np.where((m_left >= 0.5) & (m_left < 30.0), slope, np.inf)
    trough = int(np.argmin(window))
    plateau = (c_func(_PLATEAU[1]) - c_func(_PLATEAU[0])) / (_PLATEAU[1] - _PLATEAU[0])
    return [
        ("mpc_first_segment", m_left[0], slope[0]),
        ("mpc_at_m", 0.5, slope[at_half]),
        ("mpc_trough", m_left[trough], slope[trough]),
        ("mpc_plateau", float(np.mean(_PLATEAU)), plateau),
    ]


def expected_future_labor(agent, m0, n_paths=_N_PATHS, seed=0):
    """Expected labor summed over every later period along the age-0 savings path
    from cash on hand ``m0``, conditional on survival: wage shocks drawn from
    ``IncShkDstn`` over ``n_paths`` paths with a fixed ``seed``.
    """
    rng = np.random.default_rng(seed)
    dstn = agent.IncShkDstn[0]
    wage, rfree = agent.WageRte[0], agent.Rfree[0]
    c0 = agent.solution[0].consumption_saving.c_func
    a = np.full(n_paths, m0 - c0(m0))
    total = 0.0
    for sol in agent.solution[1:-1]:
        theta = rng.choice(dstn.atoms[1], size=n_paths, p=dstn.pmv)
        b = rfree * a
        labor = sol.labor_leisure.labor_func(b, theta)
        total += float(labor.mean())
        m = b + wage * theta * labor
        a = m - sol.consumption_saving.c_func(m)
    return total


def mpc_csv_rows(agent):
    """``(quantity, m, value)`` rows of ``consumption_labor_mpc.csv``: the age-0
    MPC profile, the expected future labor at the cash-on-hand levels the paper
    quotes, and the first ``m`` on a 0.25 grid at which expected future labor
    is exactly 0.
    """
    rows = mpc_rows(agent.solution[0].consumption_saving.c_func)
    rows += [("future_labor", m, expected_future_labor(agent, m)) for m in _LABOR_M]
    zero = next(m for m in _ZERO_SCAN if expected_future_labor(agent, m) == 0.0)
    rows.append(("future_labor_first_zero", zero, 0.0))
    return rows


def write_mpc_csv(agent) -> None:
    """Write ``mpc_csv_rows`` to ``consumption_labor_mpc.csv``."""
    write_rows(
        FIG_DIR / "consumption_labor_mpc.csv",
        ["quantity", "m", "value"],
        [[q, f"{m:.4g}", f"{v:.6g}"] for q, m, v in mpc_csv_rows(agent)],
    )


def generate_consumption_labor_figure():
    """Generate the consumption-labor model figure (1x3: grid, c(m), leisure)
    and its MPC and future-labor readings.
    """
    apply_engine_style()

    agent = solve_labor_model()
    write_mpc_csv(agent)
    # Get solution from an interior period (period 0, not terminal)
    sol = agent.solution[0]

    # Create figure with 1x3 horizontal layout
    fig = plt.figure(figsize=(16, 4.8))
    plot_endogenous_grid_panel(fig.add_subplot(1, 3, 1), sol)
    plot_consumption_by_age_panel(fig.add_subplot(1, 3, 2), agent)
    plot_leisure_by_wage_panel(fig.add_subplot(1, 3, 3), sol)
    plt.tight_layout(pad=1.5)

    return save_engine_figure(fig, "consumption_labor_solution.png")


if __name__ == "__main__":
    generate_consumption_labor_figure()
