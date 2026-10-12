"""The built-site output check passes an executed notebook page and rejects
pages without outputs, with an error output, or with stderr text."""

from __future__ import annotations

import json

import pytest
from check_notebook_outputs import check


def page(kind, *jupyter_outputs):
    """A built MyST page whose mdast holds one ``output`` node per payload."""
    nodes = [{"type": "output", "jupyter_data": o} for o in jupyter_outputs]
    return {
        "kind": kind,
        "mdast": {"type": "root", "children": [{"type": "block", "children": nodes}]},
    }


def write_pages(tmp_path, **pages):
    for name, content in pages.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(content))
    return tmp_path


def test_executed_notebook_and_article_pass(tmp_path) -> None:
    stream = {"output_type": "stream", "name": "stdout", "text": "ok"}
    write_pages(tmp_path, nb=page("Notebook", stream), paper=page("Article"))
    assert check(tmp_path) == {}


@pytest.mark.parametrize(
    ("payload", "problem"),
    [
        (None, "no executed outputs"),
        (
            {"output_type": "error", "ename": "ValueError", "evalue": "x"},
            "error output: ValueError: x",
        ),
        ({"output_type": "stream", "name": "stderr", "text": "warn"}, "text on stderr"),
    ],
)
def test_bad_notebook_pages_fail(tmp_path, payload, problem) -> None:
    content = page("Notebook") if payload is None else page("Notebook", payload)
    write_pages(tmp_path, nb=content)
    assert check(tmp_path) == {"nb": [problem]}


def test_missing_build_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        check(tmp_path)
