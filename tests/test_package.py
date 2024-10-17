from __future__ import annotations

import importlib.metadata

import sequentialegm as m


def test_version():
    assert importlib.metadata.version("sequentialegm") == m.__version__
