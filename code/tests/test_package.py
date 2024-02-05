from __future__ import annotations

import importlib.metadata

import egmn as m


def test_version():
    assert importlib.metadata.version("egmn") == m.__version__
