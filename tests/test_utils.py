"""Tests for the phasorpy.utils module."""

import os

import numpy
import pytest

from phasorpy.utils import (
    anscombe_transformation,
    anscombe_transformation_inverse,
    number_threads,
)

numpy.random.seed(42)


def test_number_threads():
    """Test `number_threads` function."""
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


@pytest.mark.parametrize('dtype', ['float32', 'uint16'])
def test_anscombe_transformation(dtype):
    """Test anscombe_transformation and inverse functions."""
    x = numpy.random.poisson(10, 100000).astype(dtype)
    if dtype == 'float32':
        x[0] = numpy.nan
    z = anscombe_transformation(x)
    numpy.testing.assert_allclose(numpy.std(z[1:]), 1.0, atol=0.01)

    x2 = anscombe_transformation_inverse(z)
    numpy.testing.assert_allclose(x2[1:], x[1:], atol=0.01)

    x3 = anscombe_transformation_inverse(z, approx=True)
    numpy.testing.assert_allclose(x3[1:], x[1:], atol=1.0)

    if dtype == 'float32':
        assert numpy.isnan(z[0])
        assert numpy.isnan(x2[0])
        assert numpy.isnan(x3[0])


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
