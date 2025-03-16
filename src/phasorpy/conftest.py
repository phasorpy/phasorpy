"""Pytest configuration."""

from __future__ import annotations

__all__: list[str] = []

import math
from typing import TYPE_CHECKING

import numpy
import pytest

from .datasets import fetch

if TYPE_CHECKING:
    from ._typing import Any

# numpy 2.0 changed the scalar type representation,
# causing many doctests to fail.
numpy.set_printoptions(legacy='1.21')


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace: dict[str, Any]) -> None:
    """Add common modules and functions to doctest namespace."""
    doctest_namespace['fetch'] = fetch
    doctest_namespace['math'] = math
    doctest_namespace['numpy'] = numpy


@pytest.fixture(autouse=True)
def set_printoptions() -> None:
    """Adjust numpy array print options for use with `# doctest: +NUMBER`."""
    numpy.set_printoptions(
        # precision=3,
        threshold=5,
        formatter={'float': lambda x: f'{x:.4g}'},  # remove whitespace
    )
