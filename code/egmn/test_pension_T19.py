"""
Test ConsPensionModel with T=19 using GPR for G2EGM comparison.

This tests the PENSION DEPOSIT model (no retirement choice) which is more
directly comparable to G2EGM's model.
"""

import sys

sys.path.insert(0, "/mnt/c/Users/alujan/GitHub/alanlujan91/SequentialEGM/code")

from egmn.ConsPensionModel import (
    PensionConsumerType,
    init_pension_contrib,
    make_pension_grids,
    make_pension_solution_terminal,
)


def create_pension_agent_T19():
    """Create PensionConsumerType with T=19."""

    T = 19

    # Base parameters matching G2EGM closely
    params = init_pension_contrib.copy()
    params["cycles"] = T
    params["T_cycle"] = T

    # Time-varying parameters
    params["Rfree"] = [1.02] * T
    params["RiskyAvg"] = [1.04] * T
    params["RiskyStd"] = [0.0] * T  # No risky asset variance (pure pension)
    params["PermGroFac"] = [1.0] * T
    params["LivPrb"] = [1.0] * T
    params["TranShkStd"] = [0.10] * T
    params["PermShkStd"] = [0.0] * T

    # Other parameters
    params["DiscFac"] = 0.98
    params["CRRA"] = 2.0
    params["TaxDeduct"] = 0.10

    # Grids - modest size for GPR extrapolation
    params["epsilon"] = 1e-8
    params["aCount"] = 50
    params["aMax"] = 20.0
    params["aNestFac"] = 3
    params["bCount"] = 50
    params["bMax"] = 20.0
    params["bNestFac"] = 3
    params["lCount"] = 50
    params["lMax"] = 20.0
    params["lNestFac"] = 3
    params["blCount"] = 50
    params["blMax"] = 20.0
    params["blNestFac"] = 3
    params["mCount"] = 50
    params["mMax"] = 20.0
    params["mNestFac"] = 3
    params["nCount"] = 50
    params["nMax"] = 20.0
    params["nNestFac"] = 3

    # Create grids
    grids = make_pension_grids(
        **{
            k: v
            for k, v in params.items()
            if k
            in [
                "epsilon",
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

    # Add constructors
    if "constructors" not in params:
        params["constructors"] = {}
    params["constructors"]["solution_terminal"] = make_pension_solution_terminal

    # Create agent
    agent = PensionConsumerType(**params)

    return agent


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PENSION MODEL T=19 TEST (for G2EGM comparison)")
    print("=" * 80)

    print("\nCreating agent with T=19...", end="", flush=True)
    agent = create_pension_agent_T19()
    print("✅")

    print("\nSolving T=19 periods...", end="", flush=True)
    agent.solve()
    print("✅")

    print("\n" + "=" * 80)
    print("SUCCESS: ConsPensionModel solves with T=19!")
    print("=" * 80)
