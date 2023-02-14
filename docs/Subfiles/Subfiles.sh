#!/bin/bash

for f in *.tex; do
    pdflatex -file-line-error  -interaction=nonstopmode "$f"
done

for ext in aux dep log out upa upb gz fls fdb_latexmk; do
    rm -f *.$ext
done
