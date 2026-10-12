"""The working-paper article is generated from the submission article."""

import pytest
from sync_wp_article import PAPER, WP, wp_text

PAPER_TEXT = """---
title: "T"
exports:
  - format: cff
    output: ../../CITATION.cff
parts:
  abstract: >-
    A.
  acknowledgement: >-
    K.
---
Body.
"""

WP_TEXT = """---
title: "T"
parts:
  abstract: >-
    A.
  keypoints:
    - One.
    - Two.
  acknowledgement: >-
    K.
---
Body.
"""


def test_wp_is_paper_minus_exports_plus_keypoints():
    assert wp_text(PAPER_TEXT, WP_TEXT) == WP_TEXT


@pytest.mark.parametrize(
    ("old", "new"),
    [('title: "T"', 'title: "U"'), ("    A.", "    B."), ("Body.", "Other.")],
)
def test_paper_edit_reaches_wp(old, new):
    assert wp_text(PAPER_TEXT.replace(old, new), WP_TEXT) == WP_TEXT.replace(old, new)


def test_wp_keeps_its_own_exports():
    wp = WP_TEXT.replace('title: "T"\n', 'title: "T"\nexports:\n  - id: wp-pdf\n')
    assert wp_text(PAPER_TEXT, wp) == wp


def test_paper_without_exports():
    paper = PAPER_TEXT.replace(
        "exports:\n  - format: cff\n    output: ../../CITATION.cff\n", ""
    )
    wp = WP_TEXT.replace('title: "T"\n', 'title: "T"\nexports:\n  - id: wp-pdf\n')
    assert wp_text(paper, WP_TEXT) == WP_TEXT
    assert wp_text(paper, wp) == wp


def test_missing_keypoints_raises():
    with pytest.raises(ValueError, match="keypoints"):
        wp_text(PAPER_TEXT, PAPER_TEXT)


def test_committed_wp_article_is_in_sync():
    assert WP.read_text() == wp_text(PAPER.read_text(), WP.read_text())
