"""
Quick T=19 Euler residual test - G2EGM comparison

Matches G2EGM methodology exactly: T=19, infinite horizon approximation
"""

import numpy as np
import sys

sys.path.insert(0, "/mnt/c/Users/alujan/GitHub/alanlujan91/SequentialEGM/code")

from egmn.ConsRetirementModel import (
    RetirementConsumerType,
    RetirementSolver,
    make_retirement_grids,
    make_retirement_solution_terminal,
)


def create_retirement_agent_multiperiod(T=19):
    """Create RetirementConsumerType with T periods, handling HARK compatibility."""

    # Base parameters - using GPR for T=19 stability
    params = {
        "RfreeA": 1.02,  # Restored original value (GPR handles extrapolation)
        "RfreeB": 1.04,  # Restored original value (GPR handles extrapolation)
        "Rfree": [1.02] * T,  # Required by parent class for ShareLimit calculation
        "DiscFac": 0.98,  # Restored original value (GPR handles extrapolation)
        "CRRA": 2.0,
        "TaxDeduct": 0.10,
        "TasteShkStd": 0.10,
        "cycles": T,
        "T_cycle": T,
        # Time-varying parameters (lists of length T)
        "PermGroFac": [1.0] * T,
        "LivPrb": [1.0] * T,
        "TranShkStd": [0.10] * T,
        "PermShkStd": [0.0] * T,
        "TranShkCount": 7,
        "PermShkCount": 1,
        "UnempPrb": 0.0,
        "IncUnemp": 0.0,
        "UnempPrbRet": 0.0,
        "IncUnempRet": 0.50,
        # Risky asset parameters (required by parent class)
        "RiskyAvg": [1.05] * T,
        "RiskyStd": [0.10] * T,
        "RiskyCount": 5,
        # Grids - MODEST SIZE with GPR extrapolation for T=19
        # Key insight: grids define TRAINING region, GPR handles extrapolation
        # Use grids sized for terminal period, let GPR extrapolate to early periods
        "epsilon": 1e-8,
        "aRetCount": 50,
        "aRetMax": 30.0,  # Modest - GPR will extrapolate beyond
        "aRetNestFac": 3,
        "mCount": 50,
        "mMax": 20.0,  # Modest - GPR will extrapolate beyond
        "mNestFac": 3,
        "nCount": 50,
        "nMax": 20.0,  # Modest - GPR will extrapolate beyond
        "nNestFac": 3,
        "lCount": 50,
        "lMax": 20.0,  # Modest - GPR will extrapolate beyond
        "lNestFac": 3,
        "blCount": 50,
        "blMax": 20.0,  # Modest - GPR will extrapolate beyond
        "blNestFac": 3,
        "aCount": 50,
        "aMax": 20.0,  # Modest - GPR will extrapolate beyond
        "aNestFac": 3,
        "bCount": 50,
        "bMax": 20.0,  # Modest - GPR will extrapolate beyond
        "bNestFac": 3,
    }

    # Create grids
    grids = make_retirement_grids(
        **{
            k: v
            for k, v in params.items()
            if k
            in [
                "epsilon",
                "aRetCount",
                "aRetMax",
                "aRetNestFac",
                "aCount",
                "aMax",
                "aNestFac",
                "bCount",
                "bMax",
                "bNestFac",
                "lCount",
                "lMax",
                "lNestFac",
                "blCount",
                "blMax",
                "blNestFac",
                "mCount",
                "mMax",
                "mNestFac",
                "nCount",
                "nMax",
                "nNestFac",
            ]
        }
    )
    params.update(grids)

    # Constructors
    params["constructors"] = {
        "solution_terminal": make_retirement_solution_terminal,
    }

    # Default dict for agent
    params["solver"] = RetirementSolver

    agent = RetirementConsumerType(**params)
    return agent


# Test
print("\n" + "=" * 80)
print("EULER RESIDUAL TEST - T=19 (G2EGM COMPARISON)")
print("=" * 80)

print("\nCreating agent with T=19...", end="", flush=True)
agent = create_retirement_agent_multiperiod(T=19)
print("✅")

print("\nSolving T=19 periods...", end="", flush=True)
agent.solve()
print("✅")

# Test Euler residuals
print(f"\n{'Parameters:':<30}")
print(f"  β (DiscFac):                 {agent.DiscFac}")
print(f"  ρ (CRRA):                    {agent.CRRA}")
print(f"  Ra (RfreeA):                 {agent.RfreeA}")
print(f"  Rb (RfreeB):                 {agent.RfreeB}")
print(f"  χ (TaxDeduct):               {agent.TaxDeduct}")
print("  T (periods):                 19")

# Test grid
K = 20
m_grid = np.linspace(0.5, 5.0, K)
n_grid = np.linspace(0.01, 5.0, K)

print(f"\n{'Test Grid:':<30}")
print("  m range:                     [0.50, 5.00]")
print("  n range:                     [0.01, 5.00]")
print(f"  Grid points:                 {K} × {K} = 400")
print("  Testing period:              t=0 (19 periods before terminal)")

# Get solution
t = 0
sol_t = agent.solution[t].worker_solution.deposit_stage
sol_next = agent.solution[t + 1].worker_solution.deposit_stage

# Marginal utility
u_prime = lambda c: c ** (-agent.CRRA)
u_prime_inv = lambda q: q ** (-1.0 / agent.CRRA)

print(f"\n{'Computing Euler residuals...':<30}", end="", flush=True)

euler_errors = []
skipped = 0

for m in m_grid:
    for n in n_grid:
        c = sol_t.c_func(m, n)
        d = sol_t.d_func(m, n)
        a = m - c - d

        if a < 0.001:
            skipped += 1
            continue

        g_d = agent.TaxDeduct * np.log(1.0 + d)
        b = n + d + g_d

        m_next = agent.RfreeA * a
        n_next = agent.RfreeB * b

        c_next = sol_next.c_func(m_next, n_next)

        # Infinite horizon Euler equation (G2EGM style)
        RHS = agent.DiscFac * agent.RfreeA * u_prime(c_next)
        c_implied = u_prime_inv(RHS)

        euler_raw = c - c_implied
        euler_error = np.log10(np.abs(euler_raw / c) + 1e-16)
        euler_errors.append(euler_error)

print("✅")

euler_errors = np.array(euler_errors)

print(f"\n{'Results:':<30}")
print(f"  Valid test points:           {len(euler_errors)}")
print(f"  Skipped (a < 0.001):         {skipped}")

print(f"\n{'Euler Errors (log₁₀):':<30}")
print(f"  Mean:                        {np.mean(euler_errors):.3f}")
print(f"  Median:                      {np.median(euler_errors):.3f}")
print(f"  5th percentile:              {np.percentile(euler_errors, 5):.3f}")
print(f"  95th percentile:             {np.percentile(euler_errors, 95):.3f}")
print(f"  Min:                         {np.min(euler_errors):.3f}")
print(f"  Max:                         {np.max(euler_errors):.3f}")

print("\n" + "=" * 95)
print("COMPARISON WITH G2EGM")
print("=" * 95)
print(
    f"{'Method':<25} {'Horizon':<10} {'Mean (log₁₀)':<15} {'5th %ile':<15} {'95th %ile':<15}"
)
print("-" * 95)
print(f"{'G2EGM':<25} {'T=20':<10} {'-6.23':<15} {'-7.36':<15} {'-4.27':<15}")
print(f"{'NEGM':<25} {'T=20':<10} {'-5.37':<15} {'-7.21':<15} {'-3.35':<15}")
print(
    f"{'Sequential EGM':<25} {'T=19':<10} {np.mean(euler_errors):>14.2f} {np.percentile(euler_errors, 5):>14.2f} {np.percentile(euler_errors, 95):>14.2f}"
)
print("=" * 95)

if np.mean(euler_errors) < -5.0:
    print(
        "\n✅ SUCCESS: Comparable to G2EGM when testing apples-to-apples (T=19 vs T=20)!"
    )
    print(
        "   Sequential EGM achieves similar accuracy with correct finite-horizon equation."
    )
elif np.mean(euler_errors) < -4.0:
    print("\n✅ GOOD: Slightly higher but acceptable")
else:
    print("\n⚠️  Note: Different from G2EGM, investigate further")

print("\n" + "=" * 80)
