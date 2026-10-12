r"""Repair cross-document references in the TeX of the paper or the online appendix.

The paper and the online appendix cross-reference each other, but each PDF is
built alone. MyST's TeX serializer emits a cross-document reference to a
*section-level* label as ``Kind~\ref{label}``, undefined in the other document,
which the PDF prints as ``Kind ??``. It emits a *page-level* reference as a
root-relative ``\href{/page}{text}``, a dead link inside a PDF. The markdown
keeps real MyST references for the site's links, and this script turns each
such reference in one document's TeX into plain text. The text is the markdown
link text where the source used ``[text](#label)``, else the other document's
``.aux`` number where it used an ``@label`` reference. Each PDF is a separate
file of one submission, which a reader finds by its name.

Run AFTER MyST has exported the document's TeX. ``code/build_pdfs.sh``
orchestrates this script and the recompile:

    uv run python code/fix_online_appendix_tex.py paper|online --inplace
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from cli import start_logging
from repo_paths import ROOT

PUBLIC = ROOT / "content/public"
PAPER_DIR = ROOT / "content/paper"
# Per document: its TeX, the markdown sources holding its references' link
# texts, and the other document's .aux, which numbers the rest.
DOCS = {
    "paper": (
        PUBLIC / "SequentialEGMn_pdf_tex/SequentialEGMn.tex",
        sorted(PAPER_DIR.glob("[1-7]_*.md")),
        PUBLIC
        / "SequentialEGMn_online_appendix_pdf_tex/SequentialEGMn_online_appendix.aux",
    ),
    "online": (
        PUBLIC
        / "SequentialEGMn_online_appendix_pdf_tex/SequentialEGMn_online_appendix.tex",
        [PAPER_DIR / "8_online_appendix.md"],
        PUBLIC / "SequentialEGMn_pdf_tex/SequentialEGMn.aux",
    ),
}

REF = re.compile(
    r"(Section|Figure|Table|Proposition|Definition|Theorem|Corollary|Remark|Algorithm)"
    r"~\\ref\{([^}]+)\}",
)
SITE_LINK = re.compile(r"\\href\{/[^}]*\}\{([^{}]*)\}")
MD_LINK = re.compile(r"\[([^\]]+)\]\(#([\w:.-]+)\)")
NEWLABEL = re.compile(r"\\newlabel\{([^}]+)\}\{\{([^{}]*)\}")

logger = logging.getLogger(__name__)


def repair(
    tex: str,
    md_text: dict[str, str],
    aux_num: dict[str, str],
) -> tuple[str, list[str]]:
    r"""Return the repaired TeX and any labels that could not be resolved.

    Each ``Kind~\\ref{label}`` whose label the document does not define becomes
    its markdown link text, else ``Kind~<number>`` from the other document's
    ``.aux``.
    """
    defined = set(re.findall(r"\\label\{([^}]+)\}", tex))
    missing: list[str] = []

    def repl(m: re.Match[str]) -> str:
        kind, label = m.group(1), m.group(2)
        if label in defined:
            return m.group(0)
        if label in md_text:
            return md_text[label]
        if aux_num.get(label):
            return f"{kind}~{aux_num[label]}"
        missing.append(label)
        return m.group(0)

    return REF.sub(repl, tex), missing


def unlink_site(tex: str) -> tuple[str, int]:
    r"""Replace each root-relative ``\\href{/...}{text}`` by its ``text``."""
    return SITE_LINK.subn(r"\1", tex)


def md_link_texts(*paths: Path) -> dict[str, str]:
    """Map label -> first markdown link text across the given sources."""
    texts: dict[str, str] = {}
    for path in paths:
        for text, label in MD_LINK.findall(path.read_text()):
            texts.setdefault(label, text)
    return texts


def aux_numbers(path: Path) -> dict[str, str]:
    """Map label -> printed number from a ``.aux``, empty when it is not built."""
    if not path.exists():
        return {}
    return {m.group(1): m.group(2) for m in NEWLABEL.finditer(path.read_text())}


def fix(doc: str) -> tuple[str, str] | None:
    """``(original, repaired)`` TeX of ``doc``, or None after logging a failure."""
    tex_path, md_paths, other_aux = DOCS[doc]
    if not tex_path.exists():
        logger.error("missing %s; export the document with MyST first", tex_path)
        return None
    original = tex_path.read_text()
    fixed, missing = repair(original, md_link_texts(*md_paths), aux_numbers(other_aux))
    if missing:
        logger.error(
            "unresolvable labels (not in %s, markdown, or aux): %s", doc, missing
        )
        return None
    fixed, n_links = unlink_site(fixed)
    if r"\href{/" in fixed:
        logger.error("a root-relative link in %s has braces in its text", doc)
        return None
    n_refs = len(REF.findall(original)) - len(REF.findall(fixed))
    logger.info(
        "%s: unlinked %d cross-document references, %d site links", doc, n_refs, n_links
    )
    return original, fixed


def main() -> int:
    start_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc", choices=sorted(DOCS))
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()
    result = fix(args.doc)
    if result is None:
        return 1
    if args.inplace:
        DOCS[args.doc][0].write_text(result[1])
    else:
        sys.stdout.write(result[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
