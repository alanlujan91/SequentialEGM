.PHONY: help pdf submission html test clean

# Default target
help:
	@echo "Available targets:"
	@echo "  make pdf        - Build the site's working-paper and online-appendix PDFs"
	@echo "  make submission - Also build the journal-format paper PDF"
	@echo "  make html       - Build the website"
	@echo "  make test       - Run the test suite"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make help       - Show this help message"

# Runs the Python tools in the locked uv environment; `make html RUN=` uses the active one.
RUN ?= uv run

# Build the PDFs via code/build_pdfs.sh, which removes each TeX bundle first. The online
# appendix numbers the paper's tables from the paper's .aux, which a fresh clone lacks.
pdf:
	test -f content/public/SequentialEGMn_pdf_tex/SequentialEGMn.aux || bash code/build_pdfs.sh submission
	bash code/build_pdfs.sh site

submission: pdf
	bash code/build_pdfs.sh submission

# Execute every notebook afresh (MyST's cache keys on notebook text, not the code it
# imports) and fail if a notebook page has no outputs, an error, or stderr text.
html:
	$(RUN) myst clean --execute --site --html --yes
	$(RUN) myst build --execute --html
	$(RUN) python code/check_notebook_outputs.py

# Run the test suite (paths from pyproject's testpaths)
test:
	$(RUN) pytest -q

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf content/public/SequentialEGMn_pdf_tex/
	rm -rf content/public/SequentialEGMn_online_appendix_pdf_tex/
	rm -f content/public/SequentialEGMn.pdf
	rm -f content/public/SequentialEGMn_online_appendix.pdf
	rm -f content/public/SequentialEGMn_wp.pdf
	uv run myst clean

# Maintainer targets (letters, submission archive, git hooks), absent from public clones.
-include maintain.mk
