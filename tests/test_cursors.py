"""Tests for the phasorpy.cursors module."""

import numpy
import pytest
from numpy.testing import assert_array_equal

from phasorpy.cursors import *


def test_mask_from_circular_cursor():
    # real, imag, center, radius=radius
    real = numpy.array([-0.5, -0.5, 0.5, 0.5])
    imag = numpy.array([-0.5, 0.5, -0.5, 0.5])
    center = (numpy.array([[-0.5, -0.5]]),)
    radius = 0.1
    mask = mask_from_circular_cursor(real, imag, center, radius=radius)
    assert_array_equal(mask, [True, False, False, False])


def test_mask_from_circular_cursor():
    # Test ValueErrors
    real = numpy.array([-0.5, -0.5, 0.5, 0.5])
    imag = numpy.array([-0.5, 0.5, -0.5, 0.5])
    center = (numpy.array([[-0.5, -0.5]]),)
    radius = 0.1
    with pytest.raises(ValueError):
        mask_from_circular_cursor(real, imag, center, radius=-0.1)
    with pytest.raises(ValueError):
        mask_from_circular_cursor(
            numpy.array([-0.5, -0.5, 0.5]), imag, center, radius=radius
        )
    with pytest.raises(ValueError):
        mask_from_circular_cursor(real, imag, numpy.array([[-0.5]]), radius)


def test_mask_from_cursor():
    xarray = [[337, 306, 227], [21, 231, 235], [244, 328, 116]]
    yarray = [[0.22, 0.40, 0.81], [0.33, 0.43, 0.36], [0.015, 0.82, 0.58]]
    mask = mask_from_cursor(
        xarray=xarray, yarray=yarray, xrange=[0, 270], yrange=[0, 0.5]
    )
    assert_array_equal(
        mask, [[False, False, False], [True, True, True], [True, False, False]]
    )


def test_mask_from_cursor_erros():
    # Test ValueErrors
    xarray = [[337, 306], [21, 231, 235], [244, 328, 116]]
    yarray = [[0.22, 0.40, 0.81], [0.33, 0.43, 0.36], [0.015, 0.82, 0.58]]
    with pytest.raises(ValueError):
        mask_from_cursor(
            xarray=xarray, yarray=yarray, xrange=[0, 270], yrange=[0, 0.5]
        )
    xarray = [[337, 306, 227], [21, 231, 235], [244, 328, 116]]
    with pytest.raises(ValueError):
        mask_from_cursor(
            xarray=xarray, yarray=yarray, xrange=[0], yrange=[0, 0.5]
        )
        mask_from_cursor(
            xarray=xarray, yarray=yarray, xrange=[0], yrange=[0, 0.5]
        )


def test_join_arrays():
    out = join_arrays([[1, 1], [2, 3]])
    assert_array_equal(out, [[1, 2], [1, 3]])


def test_segmentate_with_cursors():
    segmented = segmentate_with_cursors(
        [True, False, False], [255, 0, 0], [0, 128, 255]
    )
    assert_array_equal(
        segmented,
        [[255.0, 0.0, 0.0], [128.0, 128.0, 128.0], [255.0, 255.0, 255.0]],
    )


def test_segmentate_with_cursors_errors():
    with pytest.raises(ValueError):
        segmentate_with_cursors([True, False, False], [255, 0, 0], [0, 128])
    with pytest.raises(ValueError):
        segmentate_with_cursors(
            [[True, False, False], [True, False, False]],
            [255, 0, 0],
            [[0, 128], [0, 128]],
        )
