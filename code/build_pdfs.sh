#!/usr/bin/env bash
# Build the PDFs of one artifact set and repair their cross-document TeX; see
# code/README.md, "Building the Paper PDFs". Run from anywhere:
#     bash code/build_pdfs.sh site         working paper + online appendix
#     bash code/build_pdfs.sh submission   JEDC submission paper
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:?usage: build_pdfs.sh site|submission}

# Byte-reproducible PDFs (code/source_epoch.sh); FORCE_SOURCE_DATE=1 points
# \today at the same epoch.
SOURCE_DATE_EPOCH=$(bash code/source_epoch.sh content/paper content/figures \
    content/references.bib myst.yml code/fix_online_appendix_tex.py)
export SOURCE_DATE_EPOCH FORCE_SOURCE_DATE=1

# The PDF's text on one line. Capture it before grepping: piping pdftotext into
# `grep -q` under pipefail fails the pipeline on SIGPIPE exactly when grep finds
# a match early, so a "??" check would pass the PDFs that contain one.
pdf_text() {
    pdftotext "$1" - 2>/dev/null | tr '\n' ' '
}

# A printed "??" is an unresolved reference.
fail_on_unresolved() {
    local text
    text=$(pdf_text "$1")
    if grep -q '??' <<<"$text"; then
        echo "ERROR: unresolved reference ('??') printed in $1" >&2
        exit 1
    fi
}

# A renamed template part drops the AI-use declaration without an error.
require_declaration() {
    local text
    text=$(pdf_text "$1")
    if ! grep -q 'During the preparation of this work' <<<"$text"; then
        echo "ERROR: $1 lacks the AI-use declaration" >&2
        exit 1
    fi
}

fail_on_log() { # pattern, description, matches to show, log
    if grep -qa "$1" "$4"; then
        echo "ERROR: $2 in $4:" >&2
        grep -m "$3" -a "$1" "$4" >&2
        exit 1
    fi
}

# Fix one exported document's TeX, recompile it (dropping latexmk's database,
# which the fixer's edits stale), and gate the PDF on freshness (a failed compile
# keeps the old PDF), a clean log and a "??"-free page.
build_tex() { # fixer document, TeX stem
    local dir=content/public/${2}_pdf_tex
    uv run python code/fix_online_appendix_tex.py "$1" --inplace
    (
        cd "$dir"
        rm -f "$2.fdb_latexmk"
        latexmk -xelatex -interaction=batchmode -file-line-error -shell-escape \
            "$2.tex" >/dev/null 2>&1 || true
    )
    if [ "$dir/$2.pdf" -ot "$dir/$2.tex" ]; then
        echo "ERROR: $dir/$2.pdf is older than its .tex; the recompile did not complete." >&2
        exit 1
    fi
    fail_on_log '^!' "LaTeX errors" 3 "$dir/$2.log"
    fail_on_log 'Missing character' "glyphs missing from a font (garbled math or text)" 3 "$dir/$2.log"
    fail_on_log 'Citation .* undefined' "undefined citations" 5 "$dir/$2.log"
    fail_on_unresolved "$dir/$2.pdf"
    cp "$dir/$2.pdf" "content/public/$2.pdf"
}

# Each export starts from an empty TeX bundle: myst copies template and figure
# files into an existing bundle without refreshing or pruning them, so a stale
# class file or content-hashed figures from an earlier build would survive.
case $mode in
    site)
        # MyST exits 0 when the Typst working paper fails to compile, so a marker
        # touched before the build exposes its stale or missing PDF.
        started=$(mktemp)
        trap 'rm -f "$started"' EXIT
        rm -rf content/public/SequentialEGMn_online_appendix_pdf_tex/
        uv run myst build --all
        wp=content/public/SequentialEGMn_wp.pdf
        if ! [ "$wp" -nt "$started" ]; then
            echo "ERROR: $wp is missing or older than this build; the Typst export failed." >&2
            exit 1
        fi
        fail_on_unresolved "$wp"
        require_declaration "$wp"
        build_tex online SequentialEGMn_online_appendix
        ;;
    submission)
        # The submission article is off the site's toc, so `--all` skips it.
        rm -rf content/public/SequentialEGMn_pdf_tex/
        uv run myst build content/paper/egmn_paper_only.md --pdf
        build_tex paper SequentialEGMn
        require_declaration content/public/SequentialEGMn.pdf
        ;;
    *)
        echo "usage: build_pdfs.sh site|submission" >&2
        exit 2
        ;;
esac

echo "$mode PDFs rebuilt; cross-document references unlinked"
