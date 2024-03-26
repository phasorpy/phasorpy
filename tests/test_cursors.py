"""Tests for the phasorpy.cursors module."""

import numpy
import pytest

from numpy.testing import (
    assert_array_equal,
)

from phasorpy.cursors import (
    circular_cursor,
    range_cursor,
)


def test_circular_cursor():
    # real, imag, center, radius=radius, components=4
    real = numpy.array([-0.5, -0.5, 0.5, 0.5])
    imag = numpy.array([-0.5, 0.5, -0.5, 0.5])
    center = numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radius = [0.1, 0.1, 0.1, 0.1]
    mask = circular_cursor(real, imag, center, radius=radius, components=4)
    assert_array_equal(mask, [1., 2., 3., 4.])


def test_range_cursor():
    values = numpy.array([[3.3, 6, 8], [15, 20, 7]]) 
    ranges = numpy.array([(2, 8), (10, 15), (20, 25)])
    mask = range_cursor(values, ranges)
    assert_array_equal(mask, [[1, 1, 1], [2, 3, 1]])

