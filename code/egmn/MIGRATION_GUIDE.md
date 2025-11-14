# Migrating Agent Types from HARK 0.13 to HARK 0.16+

## Overview

This guide documents the changes needed to update custom agent types from HARK 0.13 (and earlier) to HARK 0.16+ (main development branch). These changes reflect HARK's shift to a more declarative, constructor-based approach for agent initialization.

## Key Changes in HARK

### 1. Import Path Changes

**Old (HARK 0.13):**
```python
from HARK.distribution import DiscreteDistribution, calc_expectation
from HARK.interpolation import LinearFast
```

**New (HARK 0.16+):**
```python
from HARK.distributions import DiscreteDistribution, calc_expectation  # Added 's'
from HARK.econforgeinterp import LinearFast  # Moved to separate module
```

### 2. Constructor-Based Initialization

HARK now uses a **constructor pattern** where agent attributes are built declaratively through constructor functions rather than imperatively through `update()` methods.

**Old Pattern:**
```python
class MyAgentType(ParentType):
    def __init__(self, **kwds):
        params = init_my_agent.copy()
        params.update(kwds)
        super().__init__(**params)
        self.solve_one_period = make_one_period_oo_solver(MySolver)

    def update(self):
        super().update()
        self.update_solution_terminal()

    def update_solution_terminal(self):
        # Build terminal solution...
        self.solution_terminal = MyTerminalSolution(...)
```

**New Pattern:**
```python
# 1. Define constructor function BEFORE class definition
def make_my_agent_solution_terminal(CRRA, aXtraGrid, ...):
    """Constructor for terminal solution."""
    # Build and return terminal solution
    return MyTerminalSolution(...)

# 2. Set up init dictionary with constructor
init_my_agent = init_parent.copy()
init_my_agent["MyParam"] = value
if "constructors" in init_my_agent:
    init_my_agent["constructors"]["solution_terminal"] = make_my_agent_solution_terminal

# 3. Define agent class AT THE END with default_ dict
class MyAgentType(ParentType):
    time_inv_ = copy(ParentType.time_inv_)
    time_inv_ += ["MyParam"]

    default_ = {
        "params": init_my_agent,
        "solver": make_one_period_oo_solver(MySolver),
    }
```

### 3. Organization Principle

**Define agent classes at the end of the module** after all their dependencies:
1. Imports
2. Utility classes and functions
3. Dataclasses (solution stages)
4. Solver classes
5. Constructor functions
6. Init parameter dictionaries
7. **Agent Type classes** ← At the end with fully-initialized `default_` dicts

This eliminates the need for placeholder `default_` dictionaries that get replaced later.

## Migration Checklist

### Step 1: Update Imports

- [ ] Change `HARK.distribution` → `HARK.distributions` (add 's')
- [ ] Change `LinearFast` import from `HARK.interpolation` → `HARK.econforgeinterp`
- [ ] Verify all other imports are correct for HARK 0.16+

### Step 2: Convert Update Methods to Constructors

For each `update_*()` method that builds agent attributes:

- [ ] Create a standalone constructor function
- [ ] Name it `make_<agent>_<attribute>(params, ...)`
- [ ] Function should take parameters as arguments and **return** the constructed object
- [ ] Add constructor to init dictionary's `"constructors"` dict
- [ ] Remove the `update_*()` method from the class
- [ ] Remove custom `__init__()` if it only calls `update_*()` methods

### Step 3: Reorganize File Structure

- [ ] Move constructor functions before init dictionaries
- [ ] Move init dictionaries before agent class definitions
- [ ] Move agent class definitions to the end of the file
- [ ] Define `default_` dictionary in class body (not after class definition)

### Step 4: Handle Multiple Inheritance

For agent types inheriting from multiple parents:

**Time-varying and time-invariant parameters:**
```python
class MyAgentType(Parent1Type, Parent2Type):
    # Merge time_vary_ from both parents (union)
    time_vary_ = copy(Parent1Type.time_vary_)
    for item in Parent2Type.time_vary_:
        if item not in time_vary_:
            time_vary_.append(item)

    # Merge time_inv_ from both parents (union)
    time_inv_ = copy(Parent1Type.time_inv_)
    for item in Parent2Type.time_inv_:
        if item not in time_inv_:
            time_inv_.append(item)

    # Add your custom parameters
    time_inv_ += ["MyParam1", "MyParam2"]
```

**Constructor dictionaries:**
```python
# Merge constructors from both parents
parent1_constructors = init_parent1.get("constructors", {}).copy()
parent2_constructors = init_parent2.get("constructors", {}).copy()

init_my_agent = init_parent1.copy()
init_my_agent.update(init_parent2)

# Merge constructors - start with one, add from the other
if "constructors" in init_my_agent:
    init_my_agent["constructors"] = parent2_constructors.copy()
    init_my_agent["constructors"]["special_param"] = parent1_constructors["special_param"]
    init_my_agent["constructors"]["solution_terminal"] = make_my_solution_terminal
```

### Step 5: Remove Unnecessary Methods

- [ ] Remove no-op methods (e.g., `update_LbrCost(self): pass`)
- [ ] Remove `construct()` overrides unless you have specific logic beyond calling constructors
- [ ] Remove `update()` overrides that only call parent and custom update methods

## Example: Terminal Solution Constructor

**Old approach (imperative):**
```python
class MyAgentType(ParentType):
    def update_solution_terminal(self):
        # Complex logic to build terminal solution
        stage1 = Stage1(...)
        stage2 = Stage2(...)
        self.solution_terminal = MySolution(stage1=stage1, stage2=stage2)
```

**New approach (declarative):**
```python
def make_my_agent_solution_terminal(CRRA, aXtraGrid, CustomParam):
    """
    Constructs the terminal period solution.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion
    aXtraGrid : np.array
        Extra asset grid points
    CustomParam : float
        Custom parameter for this model

    Returns
    -------
    MySolution
        Terminal period solution object
    """
    # Same complex logic, but returns the solution
    stage1 = Stage1(...)
    stage2 = Stage2(...)
    return MySolution(stage1=stage1, stage2=stage2)
```

## Common Pitfalls

1. **Don't define `default_` as a placeholder**: Define the class after the init dictionary exists, not before.

2. **Incomplete parameter lists**: When inheriting from multiple parents, ensure all time-varying and time-invariant parameters from both parents are included.

3. **Constructor signatures**: Constructor functions must accept all required parameters as arguments. HARK will pass them based on the function signature.

4. **Constructor order**: Constructors are called during the `construct()` phase, which happens during `__init__()` by default (unless `construct=False`).

## Testing Your Migration

After migration, verify:

```python
# Can instantiate
agent = MyAgentType()

# Has all expected attributes
assert hasattr(agent, 'solution_terminal')
assert hasattr(agent, 'all_required_params')

# Can solve
agent.solve()
print("✓ Migration successful!")
```

## Resources

- [HARK Documentation - Gentle Introduction](https://docs.econ-ark.org/examples/Gentle-Intro/Gentle-Intro-To-HARK.html)
- [HARK Documentation - Constructors](https://docs.econ-ark.org/examples/Gentle-Intro/Constructors-Intro.html)
- HARK GitHub: https://github.com/econ-ark/HARK

## Example: ConsLaborSeparableModel

See `ConsLaborSeparableModel.py` (new) vs `ConsLaborSeparableModel_old.py` (old) in this directory for a complete before/after example of migrating a complex agent type with multiple inheritance.
