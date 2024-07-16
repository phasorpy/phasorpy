"""Tests for the phasorpy.cursors module."""

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phasorpy.color import CATEGORICAL
from phasorpy.cursors import (
    mask_from_circular_cursor,
    mask_from_polar_cursor,
    pseudo_color,
)


@pytest.mark.parametrize(
    "real, imag, center_real, center_imag, radius, axis, expected",
    [
        (
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            0.1,
            0,
            [True],
        ),  # single phasor inside single cursor
        (
            -0.5,
            -0.5,
            -0.5,
            0.5,
            0.1,
            0,
            [False],
        ),  # single phasor outside single cursor
        (
            -0.5,
            -0.5,
            [-0.5, 0.5],
            [-0.5, 0.5],
            0.1,
            0,
            [[True], [False]],
        ),  # single phasor inside one of two cursors
        (
            [-0.5, 0.5],
            [-0.5, 0.5],
            -0.5,
            -0.5,
            0.1,
            0,
            [True, False],
        ),  # two phasors and one cursor
        (
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
            0.1,
            0,
            [[True, False], [False, True], [False, False]],
        ),  # two phasors and three cursors
        (
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.5, 0.5], [-0.5, 0.5]],
            -0.5,
            -0.5,
            0.1,
            0,
            [[True, False], [True, False]],
        ),  # 2D phasor and one cursor
        (
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.5, 0.5], [-0.5, 0.5]],
            [-0.5, 0.0],
            [-0.5, 0.0],
            0.1,
            0,
            [[[True, False], [True, False]], [[False, False], [False, False]]],
        ),  # 2D phasor and two cursors
        # TODO: add tests for axis parameter
        # TODO: add tests for multiple radius
    ],
)
def test_mask_from_circular_cursor(
    real, imag, center_real, center_imag, radius, axis, expected
):
    """Test mask_from_circular_cursor function."""
    assert_array_equal(
        mask_from_circular_cursor(
            real, imag, center_real, center_imag, radius=radius, axis=axis
        ),
        expected,
    )


@pytest.mark.parametrize(
    "real, imag, center_real, center_imag, radius, axis",
    [
        ([0], [0, 0], 0, 0, 0.1, 0),
        ([0, 0], [0], 0, 0, 0.1, 0),
        (0, 0, 0, [0, 0], 0.1, 0),
        (0, 0, [0, 0], 0, 0.1, 0),
        (0, 0, 0, [0, 0], 0.1, 0),
        (0, 0, [[0, 0], [0, 0]], [[0, 0], [0, 0]], 0.1, 2),
    ],
)
def test_mask_from_circular_cursor_errors(
    real, imag, center_real, center_imag, radius, axis
):
    """Test errors for mask_from_circular_cursor function."""
    with pytest.raises(ValueError):
        mask_from_circular_cursor(
            real, imag, center_real, center_imag, radius=radius, axis=axis
        )


@pytest.mark.parametrize(
    "phase, modulation, phase_range, modulation_range, expected",
    [
        (
            10,
            0.5,
            [0, 20],
            [0.4, 0.6],
            [True],
        ),  # single polar point inside single cursor
        (
            10,
            0.5,
            [15, 20],
            [0.4, 0.6],
            [False],
        ),  # single polar point outside phase range single cursor
        (
            10,
            0.5,
            [0, 20],
            [0.2, 0.2],
            [False],
        ),  # single polar point outside phase range single cursor
        (
            10,
            0.5,
            [[0, 20], [0, 30]],
            [[0.4, 0.6], [0.6, 0.8]],
            [True, False],
        ),  # single polar point inside one of two cursors
        (
            [10, 40],
            [0.2, 0.4],
            [0, 20],
            [0.1, 0.5],
            [True, False],
        ),  # two polar points and one cursor
        (
            [10, 40],
            [0.2, 0.4],
            [[0, 20], [0, 40], [50, 100]],
            [[0.1, 0.5], [0.3, 0.5], [0.2, 0.4]],
            [[True, False], [False, True], [False, False]],
        ),  # two polar points and three cursors
        (
            [[10, 40], [20, 30]],
            [[0.2, 0.4], [0.6, 0.8]],
            [0, 20],
            [0.1, 0.5],
            [[True, False], [False, False]],
        ),  # 2D polar points and one cursor
        (
            [[10, 40], [20, 30]],
            [[0.2, 0.4], [0.6, 0.8]],
            [[0, 20], [0, 30]],
            [[0.4, 0.6], [0.6, 0.8]],
            [[[False, False], [True, False]], [[False, False], [True, True]]],
        ),  # 2D polar points and two cursors
        # TODO: add tests for axis parameter
    ],
)
def test_mask_from_polar_cursor(
    phase, modulation, phase_range, modulation_range, expected
):
    """Test mask_from_cursor function."""
    assert_array_equal(
        mask_from_polar_cursor(
            phase, modulation, phase_range, modulation_range
        ),
        expected,
    )


@pytest.mark.parametrize(
    "phase, modulation, phase_range, modulation_range, axis",
    [
        (
            [0],
            [0, 0],
            [0, 0],
            [0, 0],
            0,
        ),  # phase and modulation are not the same shape
        ([0, 0], [0], [0, 0], [0, 0], 0),
        (
            0,
            0,
            [[0, 0], [0, 0]],
            [0, 0],
            0,
        ),  # range arrays are not the same shape
        (0, 0, [0, 0], [[0, 0], [0, 0]], 0),
        (
            0,
            0,
            [0, 0, 0],
            [0, 0, 0],
            0,
        ),  # last dimensions of range arrays is not 2
        (0, 0, [[0, 0], [0, 0]], [[0, 0], [0, 0]], 2),  # axis out of bounds
    ],
)
def test_mask_from_polar_cursor_errors(
    phase, modulation, phase_range, modulation_range, axis
):
    """Test errors for mask_from_polar_cursor function."""
    with pytest.raises(ValueError):
        mask_from_polar_cursor(
            phase, modulation, phase_range, modulation_range, axis=axis
        )


@pytest.mark.parametrize(
    "mean, masks, colors, axis, expected",
    [
        (0, [True], 'CATEGORICAL', 0, CATEGORICAL[0]),  # single value true
        (0, [False], 'CATEGORICAL', 0, [0, 0, 0]),  # single value false
        (
            [0, 0],
            [True, True],
            'CATEGORICAL',
            0,
            numpy.asarray([CATEGORICAL[0], CATEGORICAL[0]]),
        ),  # 1D array
        (
            [0, 0],
            [True, False],
            'CATEGORICAL',
            0,
            numpy.asarray([CATEGORICAL[0], [0, 0, 0]]),
        ),  # 1D array with false
        (
            [[0, 0], [0, 0]],
            [[True, True], [False, False]],
            'CATEGORICAL',
            0,
            numpy.asarray(
                [[CATEGORICAL[0], CATEGORICAL[0]], [[0, 0, 0], [0, 0, 0]]]
            ),
        ),  # 2D array
        (
            0,
            [True, True],
            'CATEGORICAL',
            0,
            CATEGORICAL[1],
        ),  # single value with two masks
        (
            [0, 0],
            [[True, False], [False, True]],
            'CATEGORICAL',
            0,
            numpy.asarray([CATEGORICAL[0], CATEGORICAL[1]]),
        ),  # 1D array with two masks
        (
            [0, 0],
            [[True, False], [True, True]],
            'CATEGORICAL',
            0,
            numpy.asarray([CATEGORICAL[1], CATEGORICAL[1]]),
        ),  # 1D array with two masks all true
        (
            0,
            [True],
            [[0, 0, 128]],
            0,
            [0, 0, 128],
        ),  # single value true with custom color
        (
            0,
            [False],
            [[0, 0, 128]],
            0,
            [0, 0, 0],
        ),  # single value false with custom color
        (
            [0, 0],
            [True, False],
            [[0, 0, 128]],
            0,
            [[0, 0, 128], [0, 0, 0]],
        ),  # 1D array with custom color
        # TODO: add tests for axis parameter
    ],
)
def test_pseudo_color(mean, masks, colors, axis, expected):
    """Test pseudo_color function."""
    assert_allclose(
        pseudo_color(mean, masks, colors=colors, axis=axis),
        expected,
    )


@pytest.mark.parametrize(
    "mean, masks, colors, axis",
    [
        (
            [[0], [0]],
            [[True, True], [True, False]],
            'CATEGORICAL',
            0,
        ),  # masks an mean same dimensions, incompatible shape
        (
            0,
            [[True, True], [True, False]],
            'CATEGORICAL',
            0,
        ),  # masks shape along axis not compatible with mean shape
        (0, True, [0, 0, 0], 0),  # colors is not 2D
        (0, True, [[0, 0]], 0),  # colors last dimension is not 3
        (0, [True, True], 'CATEGORICAL', 1),  # axis out of bounds
        (
            [0, 0],
            [[True, True], [True, False]],
            'CATEGORICAL',
            2,
        ),  # axis out of bounds
    ],
)
def test_pseudo_color_errors(mean, masks, colors, axis):
    with pytest.raises(ValueError):
        pseudo_color(mean, masks, colors=colors, axis=axis)
