# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""Test the phasorpy.color module."""

import numpy
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from phasorpy.color import float2int, wavelength2rgb


def test_wavelength2rgb() -> None:
    """Test wavelength2rgb function."""
    rgb = wavelength2rgb(517)
    assert isinstance(rgb, tuple)
    assert_almost_equal(rgb, (0.0, 0.749744, 0.0))

    assert wavelength2rgb(517.2, numpy.uint8) == (0, 191, 0)

    assert_array_equal(
        wavelength2rgb([517, 566], numpy.uint8),
        numpy.array([[0, 191, 0], [133, 190, 0]], numpy.uint8),
    )

    assert_array_equal(
        wavelength2rgb([517, 566], numpy.float16),
        numpy.array([[0, 0.749744, 0], [0.52133, 0.744288, 0]], numpy.float16),
    )

    # integer dtype other than uint8
    rgb16 = wavelength2rgb([517], numpy.uint16)
    assert rgb16.dtype == numpy.uint16
    assert_array_equal(rgb16, [[0, 49134, 0]])

    # small integer wavelength types are clipped
    assert_array_equal(
        wavelength2rgb(numpy.array([200, 255], numpy.uint8)),
        [wavelength2rgb(360)] * 2,
    )

    # NaN wavelength
    assert numpy.isnan(wavelength2rgb(numpy.nan)).all()
    assert wavelength2rgb(numpy.nan, numpy.uint8) == (0, 0, 0)


def test_float2int() -> None:
    """Test float2int function."""
    assert_array_equal(float2int([0.0, 0.5, 1.0]), [0, 128, 255])
    with pytest.raises(ValueError):
        float2int(500.0, numpy.float32)  # not an integer type
    with pytest.raises(ValueError):
        float2int(500, numpy.uint8)  # not a floating-point array
