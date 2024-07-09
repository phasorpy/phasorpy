"""Tests for the phasorpy.cursors module."""

import numpy
import pytest
from numpy.testing import assert_array_equal

from phasorpy.cursors import mask_from_circular_cursor, mask_from_cursor, segmentate_with_cursors


def test_mask_from_circular_cursor():
    """Test mask_from_circular_cursor function."""
    real = [-0.5, -0.5, 0.5, 0.5]
    imag = [-0.5, 0.5, -0.5, 0.5]
    center = ([[-0.5, -0.5]],)
    radius = 0.1
    assert_array_equal(mask_from_circular_cursor(real, imag, center, radius), [True, False, False, False])
    with pytest.raises(ValueError):
        mask_from_circular_cursor(real, imag, center, -radius)
    with pytest.raises(ValueError):
        mask_from_circular_cursor(numpy.array([-0.5, -0.5, 0.5]), imag, center, radius)
    with pytest.raises(ValueError):
        mask_from_circular_cursor(real, imag, numpy.array([[-0.5]]), radius)


def test_mask_from_cursor():
    """Test mask_from_cursor function."""
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
