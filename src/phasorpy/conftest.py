"""Pytest configuration."""

import pytest

from .datasets import fetch


@pytest.fixture(autouse=True)
def add_fetch(doctest_namespace):
    """Add datasets.fetch to doctest namespace."""
    doctest_namespace['fetch'] = fetch
