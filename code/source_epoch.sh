#!/usr/bin/env bash
# Print the SOURCE_DATE_EPOCH for a PDF built from the given source paths, so that
# rebuilding unchanged sources gives byte-identical PDFs (xdvipdfmx stamps the
# /CreationDate and trailer /ID from it). Run from the repository root:
#     SOURCE_DATE_EPOCH=$(bash code/source_epoch.sh PATH...)
# A preset SOURCE_DATE_EPOCH wins; else the paths' last commit; else, in an
# unpacked archive whose mtimes are the commit time, their newest mtime.
set -euo pipefail

epoch=${SOURCE_DATE_EPOCH:-}
if [ -z "$epoch" ] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    epoch=$(git log -1 --format=%ct -- "$@")
fi
if [ -z "$epoch" ]; then
    epoch=$(find "$@" -type f -printf '%T@\n' | sort -n | tail -1 | cut -d. -f1)
fi
echo "$epoch"
