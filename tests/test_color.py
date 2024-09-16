"""Tests for the phasorpy.color module."""

import numpy
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from phasorpy.color import float2int, wavelength2rgb


def test_wavelength2rgb():
    """Test wavelength2rgb function."""
    rgb = wavelength2rgb(517)
    assert isinstance(rgb, tuple)
    assert_almost_equal(rgb, (0.0, 0.749744, 0.0))

    assert wavelength2rgb(517.2, numpy.uint8) == (0, 191, 0)

    assert_array_equal(
        wavelength2rgb([517, 566], 'uint8'),
        numpy.array([[0, 191, 0], [133, 190, 0]], 'uint8'),
    )

    assert_array_equal(
        wavelength2rgb([517, 566], 'float16'),
        numpy.array([[0, 0.749744, 0], [0.52133, 0.744288, 0]], 'float16'),
    )


def test_float2int():
    """Test float2int function."""
    assert_array_equal(float2int([0.0, 0.5, 1.0]), [0, 128, 255])
    with pytest.raises(ValueError):
        float2int(500.0, numpy.float32)  # not an integer dtype
    with pytest.raises(ValueError):
        float2int(500, numpy.uint8)  # not a floating-point array


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
