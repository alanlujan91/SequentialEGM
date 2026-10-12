# Replication Archive: The Sequential Endogenous Grid Method

Implementation and benchmark code for EGM$^n$ (the sequential endogenous grid method) and ENGINE (the warped-grid interpolation kernel), together with the scripts that generate every table, figure, and quoted reading in `content/paper/`.

## Benchmark Environment

The timing and accuracy exhibits in the paper were generated on:

| Component         | Version                                                          |
| ----------------- | ---------------------------------------------------------------- |
| CPU               | AMD Ryzen 9 5900X (12 cores; benchmarks cap Numba at 16 threads) |
| OS                | Linux (WSL2)                                                     |
| Python            | 3.13.11                                                          |
| NumPy             | 2.3.5                                                            |
| Numba             | 0.62.1                                                           |
| SciPy             | 1.16.3                                                           |
| sequence-jacobian | 1.0.0                                                            |
| econ-ark (HARK)   | 0.17.2                                                           |

Every dependency is pinned in the repository's `uv.lock`; `uv sync` reproduces the environment exactly.

Productivity is discretized by the same three-state Rouwenhorst chain in every
two-asset comparison, with persistence $0.966$ and unconditional standard
deviation $0.92$ for log productivity.
Every stationary solve starts from the same `hh_init` marginal values and stops
when the larger of the normalized sup-norm changes in $V_a$ and $V_b$ falls below
$10^{-9}$, subject to an 8,000-iteration cap. The normalization for either
marginal $V$ is $1 + \lVert V_{t-1}\rVert_\infty$.

The timing protocol is identical across solvers: matched grids; the stopping rule
above passed explicitly to every solver (never a solver default); matched linear
interpolation; warmed timings (JIT compilation excluded) reported as best of
three; EGM$^n$ and NEGM timed at 1 and 16 Numba threads via
`numba.set_num_threads`; the SSJ reference is single-threaded as its authors
distribute it. The full-solve interpolator comparison sets 16 Numba threads
explicitly; its component breakdown sets one thread. Every timing cell is
measured, including the minutes-long one-thread NEGM solve at $(160,240)$.

## Layout

| Path                 | Contents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ssj/`               | Two-asset subpackage (Section 3): `hh_egm.py` (EGM$^n$ `@het` block, parallel njit backward step, Stage-0 ENGINE regrid, closed-form regular-grid Euler inversion), `hh_negm.py` (matched NEGM comparator whose safeguarded Stage-0 root-find returns policies directly on the regular grid), `benchmark.py` (shared harness: calibration, solvers, robust Euler metric), `bench_timing.py` and `bench_interp.py` (table generators), `interp_loop.py` (single-difference interpolator harness)                       |
| `engine/`            | ENGINE interpolation kernel: `engine2d.py` (two-pass regrid, user-facing `axes_order` pass-order parameter), `core.py`, `utilities.py`, `bench_regrid_scaling.py`                                                                                                                                                                                                                                                                                                                                             |
| `HARKmodels/`        | Section 2 consumption-labor model (`LaborSeparableConsumerType`) and the health-investment interpolation testbed of Online Appendix A                                                                                                                                                                                                                                                                                                                                                                          |
| `figures/`           | Figure generators, the health-testbed three-way interpolation benchmark, and the warp sweep                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `tests/`             | Pytest suite; run `uv run pytest` from the repository root                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `validate_claims.py` | Section 2 and health-testbed claim checks: model parameters, interpolation convergence order, complexity scaling, empirical bracket-displacement constant, grid regularity                                                                                                                                                                                                                                                                                                                                    |

## Building the Paper PDFs

`make pdf` (`bash code/build_pdfs.sh site`) builds the site's working paper (Typst, from `content/paper/egmn_paper_wp.md`) and the standalone online appendix. `make submission` adds the journal-format paper (`bash code/build_pdfs.sh submission`, from the off-site `content/paper/egmn_paper_only.md`). For each LaTeX document the script repairs the TeX that MyST cannot serialize for a standalone document: references into the other document print as "??" and site-relative links are dead in a PDF, so `code/fix_online_appendix_tex.py` turns both into plain text. The markdown keeps real MyST references for the site. The script fails if a PDF is older than its source, if the LaTeX log shows a fatal error or an undefined citation, if a "??" reaches the page, or if a paper lacks the AI-use declaration.

## Reproducing the Paper's Exhibits

Every quoted reading in the paper is persisted by the script that produces it, in the same run that writes the table, to `content/figures/`.

| Paper exhibit                                                                                      | Command                                                                              | Artifacts                                                                     |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| NEGM timing comparison (`tab-timing-negm`)                                                         | `PYTHONPATH=code uv run python -m ssj.bench_timing`                                  | `tab_timing_negm.md`, `timing_negm.csv`                                       |
| NEGM policy-agreement and Euler accuracy (Section 3 text; on- and off-grid, mesh alpha-hat)        | `PYTHONPATH=code uv run python -m ssj.bench_accuracy_negm`                           | `accuracy_negm.csv`                                                           |
| Interpolation diagnostic (`tab-solvetime-abrs`)                                                    | `PYTHONPATH=code uv run python -m ssj.bench_interp`                                  | `tab_solvetime_abrs.md`, `solvetime_abrs.csv`                                 |
| Per-call Stage-0 regrid timings, ENGINE vs curvilinear (Section 4 text)                            | `PYTHONPATH=code uv run python -m ssj.bench_regrid_percall`                          | `regrid_percall_abrs.csv`                                                     |
| Regrid scaling with mesh size, $N^{0.98}$ and time per output (Section 4 text)                     | `PYTHONPATH=code uv run python -m engine.bench_regrid_scaling`                       | `regrid_scaling.csv`                                                          |
| NEGM inner-tolerance robustness line (Section 3 text)                                              | `PYTHONPATH=code uv run python -m ssj.bench_negm_innertol`                           | `negm_innertol_abrs.csv`                                                      |
| NEGM illiquid-FOC evaluations per node (supplementary reading of Section 3's inner search)         | `PYTHONPATH=code uv run python -m ssj.bench_negm_evals`                              | `negm_foc_evals_abrs.csv`                                                     |
| Component runtime table (`tab-components-abrs`)                                                    | `PYTHONPATH=code uv run python -m ssj.bench_components`                              | `tab_components_abrs.md`, `components_abrs.csv`                               |
| Native-solver reference (`tab-ssj-reference`)                                                      | `PYTHONPATH=code uv run python -m ssj.bench_ssj_reference`                           | `tab_ssj_reference.md`, `ssj_reference_abrs.csv`                              |
| Euler-inversion pass sweep (Online Appendix B: EGM$^n$ $K{\in}\{0,1,2\}$ / NEGM; on- and off-grid) | `PYTHONPATH=code uv run python -m ssj.bench_euler_passes`                            | `euler_passes_abrs.csv`                                                       |
| SSJ accuracy anchor (identical metric, identical queries)                                          | same run                                                                             | `ssj_anchor_abrs.csv`                                                         |
| Refinement against the $192 \times 160$ benchmark; grid-bound sensitivity                          | same `ssj.bench_interp` run, or `-m ssj.bench_interp --validation-only`              | `refinement_bounds_abrs.csv`, `tab_refinement_bounds_abrs.md`                 |
| Converged Stage-0 extrapolation frequency by income state and pooled                               | same `ssj.bench_interp` run, or `-m ssj.bench_interp --validation-only`              | `extrap_frequency_abrs.csv`, `tab_extrap_frequency_abrs.md`                   |
| Reversed interpolation-order reading                                                               | same run                                                                             | `order_reading_abrs.csv`                                                      |
| Health-testbed three-way comparison (figures and policy differences)                               | `PYTHONPATH=code uv run python code/figures/health_model_interpolation_benchmark.py` | `health_interp_policy_diff.csv`, `health_solver_equivalence.csv`, figure PDFs |
| Warp sweep (measured bracket-displacement constants, margin sensitivity)                           | `PYTHONPATH=code uv run python code/figures/health_warp_sweep.py`                    | `health_warp_sweep.csv`, `health_warp_sweep.pdf`                              |
| Paper figures the benchmarks above do not draw, and a check that every figure file exists          | `PYTHONPATH=code uv run python code/figures/generate_all_paper_figures.py`           | figure PDFs and PNGs, `health_model_summary.csv`                              |
| Section 2 and testbed claim checks                                                                 | `PYTHONPATH=code uv run python code/validate_claims.py`                              | log output; nonzero exit on any failed claim                                  |

`solvetime_abrs.csv` records the Delaunay triangulation's two-asset rows (status `fails_no_convergence`), which `tab_solvetime_abrs.md` leaves out because that comparator runs to the 8,000-iteration cap without converging on the two-asset model.

## Tests

`uv run pytest` runs the full suite. Beyond unit coverage, the suite pins the claims the paper makes about this archive:

- `tests/test_thread_invariance.py`: the converged two-asset output (policies, marginal values, iteration counts) is bit-identical across Numba thread counts, for both EGM$^n$ and NEGM, so parallelism is a pure speed transform.
- `tests/test_engine2d.py::test_separable_engine_matches_1d_interp`: on index-aligned meshes the ENGINE regrid reduces to one-dimensional non-uniform linear interpolation, verified to machine precision (the separable-ENGINE remark of Section 4).
- `tests/test_engine2d.py::test_kernel_serial_vs_parallel`: the serial and parallel ENGINE kernels agree.
- `tests/test_interp_loop.py`: the interpolator harness's ENGINE configuration reproduces the production solver, so the interpolation diagnostic differs from the solver benchmark in the regrid primitive only.
- `tests/test_benchmark.py`: EGM$^n$, NEGM, and the upstream SSJ reference solve the same problem (policy agreement, refinement convergence, Euler-residual ordering, fold-free Stage 0 mesh); it also checks the Stage-0 extrapolation classifier and pins the published coarse-grid outside count.

Speedup ratios depend on hardware, JIT state and problem configuration. They move from run to run, so the paper states them as ranges. Accuracy readings reproduce to their printed precision.
