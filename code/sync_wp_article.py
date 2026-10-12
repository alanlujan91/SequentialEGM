"""Regenerate the working-paper article from the submission article.

``egmn_paper_wp.md`` is ``egmn_paper_only.md`` minus any exports, plus the
working paper's own ``exports`` block and ``keypoints`` part, the two fields
authored in the working-paper file. Run from anywhere:
    uv run python code/sync_wp_article.py           rewrite a stale copy
    uv run python code/sync_wp_article.py --check   exit 1 when stale
"""

from __future__ import annotations

import argparse
import logging
import re
import sys

from cli import start_logging
from repo_paths import ROOT

PAPER = ROOT / "content" / "paper" / "egmn_paper_only.md"
WP = ROOT / "content" / "paper" / "egmn_paper_wp.md"

logger = logging.getLogger(__name__)

# A top-level frontmatter key, or a ``parts:`` entry, through its indented body.
_EXPORTS = re.compile(r"^exports:\n(?:[ ]+.*\n)+", re.MULTILINE)
_PARTS = re.compile(r"^parts:", re.MULTILINE)
_KEYPOINTS = re.compile(r"^  keypoints:\n(?:    .*\n)+", re.MULTILINE)
_ACKNOWLEDGEMENT = re.compile(r"^  acknowledgement:", re.MULTILINE)


def one_match(pattern, text, what):
    """The single match of ``pattern`` in ``text``, with ``what`` naming it in errors."""
    found = list(pattern.finditer(text))
    if len(found) != 1:
        msg = f"expected one {what}, found {len(found)}"
        raise ValueError(msg)
    return found[0]


def insert_before(text, anchor, block, what):
    """``text`` with ``block`` inserted at the single match of ``anchor``."""
    at = one_match(anchor, text, what).start()
    return text[:at] + block + text[at:]


def wp_text(paper, wp):
    """The working-paper article: ``paper`` without any exports block, plus the
    exports block of ``wp`` (if any) before the parts and its keypoints block
    before the acknowledgement part.
    """
    body = _EXPORTS.sub("", paper)
    wp_exports = "".join(m.group(0) for m in _EXPORTS.finditer(wp))
    body = insert_before(body, _PARTS, wp_exports, "parts block")
    keypoints = one_match(_KEYPOINTS, wp, "keypoints part in the WP article")
    return insert_before(
        body, _ACKNOWLEDGEMENT, keypoints.group(0), "acknowledgement part"
    )


def main() -> int:
    start_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 when stale")
    args = parser.parse_args()
    current = WP.read_text()
    wanted = wp_text(PAPER.read_text(), current)
    if current == wanted:
        return 0
    if args.check:
        logger.error("%s is stale; run code/sync_wp_article.py", WP.relative_to(ROOT))
        return 1
    WP.write_text(wanted)
    logger.info("rewrote %s", WP.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
