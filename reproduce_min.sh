#!/usr/bin/env bash
# Quick check, a few minutes: the test suite, then the website with every example
# notebook executed (make html fails on a notebook page without outputs or with an
# error). reproduce.sh regenerates the paper's results.
set -euo pipefail
cd "$(dirname "$0")"
uv sync --locked
make test
make html
