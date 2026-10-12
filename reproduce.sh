#!/usr/bin/env bash
# Reproduce every table, figure and quoted reading of the paper and online appendix.
# Each command below is a row of the exhibit table in code/README.md and writes its
# results to content/figures/. The timing benchmarks dominate the run time and measure
# this machine, so speedups move from run to run; run them on an otherwise idle machine.
# reproduce_min.sh is the quick check.
set -euo pipefail
cd "$(dirname "$0")"
uv sync --locked

PYTHONPATH=code uv run python -m ssj.bench_timing
PYTHONPATH=code uv run python -m ssj.bench_accuracy_negm
PYTHONPATH=code uv run python -m ssj.bench_interp
PYTHONPATH=code uv run python -m ssj.bench_regrid_percall
PYTHONPATH=code uv run python -m engine.bench_regrid_scaling
PYTHONPATH=code uv run python -m ssj.bench_negm_innertol
PYTHONPATH=code uv run python -m ssj.bench_negm_evals
PYTHONPATH=code uv run python -m ssj.bench_components
PYTHONPATH=code uv run python -m ssj.bench_ssj_reference
PYTHONPATH=code uv run python -m ssj.bench_euler_passes
PYTHONPATH=code uv run python code/figures/health_model_interpolation_benchmark.py
PYTHONPATH=code uv run python code/figures/health_warp_sweep.py
PYTHONPATH=code uv run python code/figures/generate_all_paper_figures.py
PYTHONPATH=code uv run python code/validate_claims.py
