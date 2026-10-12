"""Fail unless every notebook page of the built site shows executed outputs.

``myst build`` without ``--execute`` renders a notebook page with its code and
no outputs, and ``myst build --execute`` exits 0 even when a cell raises. This
check reads each page MyST built into ``_build/site/content`` and fails when a
notebook page has no outputs, an error output, or text on stderr.

Run after ``myst build --execute --html``:
``uv run python code/check_notebook_outputs.py``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from cli import start_logging
from repo_paths import ROOT

logger = logging.getLogger(__name__)

SITE_CONTENT = ROOT / "_build" / "site" / "content"


def outputs(node):
    """Every ``output`` node's Jupyter payload under ``node``, depth first."""
    if isinstance(node, dict):
        if node.get("type") == "output":
            yield node.get("jupyter_data", {})
        for child in node.get("children", []):
            yield from outputs(child)


def page_problems(page) -> list[str]:
    """What is wrong with one built page: nothing unless it is a notebook."""
    if page.get("kind") != "Notebook":
        return []
    found = list(outputs(page.get("mdast", {})))
    problems = [] if found else ["no executed outputs"]
    errors = [o for o in found if o.get("output_type") == "error"]
    problems += [f"error output: {o.get('ename')}: {o.get('evalue')}" for o in errors]
    if any(o.get("name") == "stderr" for o in found):
        problems.append("text on stderr")
    return problems


def check(content_dir=SITE_CONTENT) -> dict[str, list[str]]:
    """``{page: problems}`` for every built page with a problem; raises when
    the directory is empty of built pages, so a missing build never passes.
    """
    pages = sorted(Path(content_dir).glob("*.json"))
    if not pages:
        msg = f"no built pages in {content_dir}; run myst build --execute --html first"
        raise FileNotFoundError(msg)
    report = {p.stem: page_problems(json.loads(p.read_text())) for p in pages}
    return {name: problems for name, problems in report.items() if problems}


def main() -> int:
    start_logging()
    failures = check()
    for name, problems in failures.items():
        for problem in problems:
            logger.error("%s: %s", name, problem)
    if failures:
        return 1
    logger.info("every notebook page shows executed outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
