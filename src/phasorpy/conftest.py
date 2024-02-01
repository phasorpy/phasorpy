"""Pytest configuration."""

import pytest
import numpy

from .datasets import fetch


@pytest.fixture(autouse=True)
def add_fetch(doctest_namespace):
    """Add datasets.fetch to doctest namespace."""
    doctest_namespace['fetch'] = fetch


@pytest.fixture(autouse=True)
def set_printoptions():
    """Adjust numpy array print options for use with `# doctest: +NUMBER`."""
    numpy.set_printoptions(
        # precision=3,
        threshold=5,
        formatter={'float': lambda x: f'{x:.4g}'},  # remove whitespace
    )
