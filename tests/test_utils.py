"""Test the phasorpy.utils module."""

import os

import numpy

from phasorpy.utils import logger, number_threads

numpy.random.seed(42)


def test_logger():
    """Test logger function."""
    import logging

    assert logger() is logging.getLogger('phasorpy')


def test_number_threads():
    """Test number_threads function."""
    assert number_threads() == 1
    assert number_threads(None, 0) == 1
    assert number_threads(1) == 1
    assert number_threads(-1) == 1
    assert number_threads(-1, 2) == 1
    assert number_threads(6) == 6
    assert number_threads(100) == 100
    assert number_threads(6, 5) == 5
    num_threads = number_threads(0)
    assert num_threads >= 1
    if num_threads > 4:
        assert number_threads(0, 4) == 4
        os.environ['PHASORPY_NUM_THREADS'] = '4'
        assert number_threads(0) == 4
        assert number_threads(6) == 6
        del os.environ['PHASORPY_NUM_THREADS']


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
