"""Test the phasorpy.experimental module."""

import numpy
import pytest
from numpy.testing import assert_allclose

from phasorpy.experimental import (
    anscombe_transform,
    anscombe_transform_inverse,
)

numpy.random.seed(42)


@pytest.mark.parametrize('dtype', ['float32', 'uint16'])
def test_anscombe_transform(dtype):
    """Test anscombe_transform and inverse functions."""
    x = numpy.random.poisson(10, 100000).astype(dtype)
    if dtype == 'float32':
        x[0] = numpy.nan
    z = anscombe_transform(x)
    assert_allclose(numpy.std(z[1:]), 1.0, atol=0.01)

    x2 = anscombe_transform_inverse(z)
    assert_allclose(x2[1:], x[1:], atol=0.01)

    x3 = anscombe_transform_inverse(z, approx=True)
    assert_allclose(x3[1:], x[1:], atol=1.0)

    if dtype == 'float32':
        assert numpy.isnan(z[0])
        assert numpy.isnan(x2[0])
        assert numpy.isnan(x3[0])


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
