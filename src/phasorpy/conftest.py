"""Pytest configuration."""

import math

import numpy
import pytest

from .datasets import fetch


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace):
    """Add common modules and functions to doctest namespace."""
    doctest_namespace['fetch'] = fetch
    doctest_namespace['math'] = math
    doctest_namespace['numpy'] = numpy


@pytest.fixture(autouse=True)
def set_printoptions():
    """Adjust numpy array print options for use with `# doctest: +NUMBER`."""
    numpy.set_printoptions(
        # precision=3,
        threshold=5,
        formatter={'float': lambda x: f'{x:.4g}'},  # remove whitespace
    )
