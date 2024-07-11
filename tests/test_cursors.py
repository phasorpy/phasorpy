"""Tests for the phasorpy.cursors module."""

import numpy
import pytest
from numpy.testing import assert_array_equal

from phasorpy.cursors import mask_from_circular_cursor, mask_from_polar_cursor, pseudo_color


@pytest.mark.parametrize(
    "real, imag, center_real, center_imag, radius, expected",
    [
        (-0.5, -0.5, -0.5, -0.5, 0.1, [[True]]), # single phasor inside single cursor
        (-0.5, -0.5, -0.5, 0.5, 0.1, [[False]]), # single phasor outside single cursor
        (-0.5, -0.5, [-0.5, 0.5], [-0.5, 0.5], 0.1, [[True, False]]), # single phasor inside with two cursors
        ([-0.5, 0.5], [-0.5, 0.5], -0.5, -0.5, 0.1, [[True], [False]]), # two phasors and one cursor
        ([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], 0.1, [[True, False, False], [False, True, False]]), # two phasors and three cursors
        ([[-0.5, 0.5],[-0.5, 0.5]], [[-0.5, 0.5],[-0.5, 0.5]], -0.5, -0.5, 0.1, [[True], [False]]), # two phasors and one cursor
    ],
)
def test_mask_from_circular_cursor(real, imag, center_real, center_imag, radius, expected):
    """Test mask_from_circular_cursor function."""
    assert_array_equal(mask_from_circular_cursor(real, imag, center_real, center_imag, radius=radius), expected)

@pytest.mark.parametrize(
    "real, imag, center_real, center_imag, radius",
    [
        ([-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, -0.5, 0.5], -0.5, -0.5, -0.1),
        ([-0.5, -0.5, 0.5], [-0.5, 0.5, -0.5, 0.5], -0.5, -0.5, 0.1),
    ],
)   
def test_mask_from_circular_cursor_errors(real, imag, center_real, center_imag, radius):
    """Test errors for mask_from_circular_cursor function."""
    with pytest.raises(ValueError):
        mask_from_circular_cursor(real, imag, center_real, center_imag, radius=radius)


# def test_mask_from_polar_cursor():
#     """Test mask_from_cursor function."""
#     xarray = [[337, 306, 227], [21, 231, 235], [244, 328, 116]]
#     yarray = [[0.22, 0.40, 0.81], [0.33, 0.43, 0.36], [0.015, 0.82, 0.58]]
#     mask = mask_from_polar_cursor(
#         xarray=xarray, yarray=yarray, xrange=[0, 270], yrange=[0, 0.5]
#     )
#     assert_array_equal(
#         mask, [[False, False, False], [True, True, True], [True, False, False]]
#     )


# def test_mask_from_polar_cursor_errors():
#     # Test ValueErrors
#     xarray = [[337, 306], [21, 231, 235], [244, 328, 116]]
#     yarray = [[0.22, 0.40, 0.81], [0.33, 0.43, 0.36], [0.015, 0.82, 0.58]]
#     with pytest.raises(ValueError):
#         mask_from_polar_cursor(
#             xarray=xarray, yarray=yarray, xrange=[0, 270], yrange=[0, 0.5]
#         )
#     xarray = [[337, 306, 227], [21, 231, 235], [244, 328, 116]]
#     with pytest.raises(ValueError):
#         mask_from_polar_cursor(
#             xarray=xarray, yarray=yarray, xrange=[0], yrange=[0, 0.5]
#         )
#         mask_from_polar_cursor(
#             xarray=xarray, yarray=yarray, xrange=[0], yrange=[0, 0.5]
#         )


# def test_pseudo_color():
#     segmented = pseudo_color(
#         [True, False, False], [255, 0, 0], [0, 128, 255]
#     )
#     assert_array_equal(
#         segmented,
#         [[255.0, 0.0, 0.0], [128.0, 128.0, 128.0], [255.0, 255.0, 255.0]],
#     )


# def test_pseudo_color_errors():
#     with pytest.raises(ValueError):
#         pseudo_color([True, False, False], [255, 0, 0], [0, 128])
#     with pytest.raises(ValueError):
#         pseudo_color(
#             [[True, False, False], [True, False, False]],
#             [255, 0, 0],
#             [[0, 128], [0, 128]],
#         )
