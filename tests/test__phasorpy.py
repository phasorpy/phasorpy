"""Test the phasorpy._phasorpy module."""

import math

import pytest
from numpy import nan
from numpy.testing import assert_allclose, assert_array_equal

from phasorpy._phasorpy import _is_near_segment  # same as _is_inside_stadium
from phasorpy._phasorpy import (
    _distance_from_line,
    _distance_from_point,
    _distance_from_segment,
    _fraction_on_line,
    _fraction_on_segment,
    _intersection_circle_circle,
    _intersection_circle_line,
    _is_inside_circle,
    _is_inside_ellipse,
    _is_inside_ellipse_,
    _is_inside_polar_rectangle,
    _is_inside_range,
    _is_inside_rectangle,
    _is_inside_stadium,
    _is_near_line,
    _point_on_line,
    _point_on_segment,
    _segment_direction_and_length,
)

LINE = 0.2, 0.4, 0.9, 0.3
POINTS = [0.4, 0.86, 0.82], [0.38, 0.4, 0.4]


def test_is_inside_circle():
    """Test _is_inside_circle function."""
    circle = 0.8, 0.4, 0.05
    assert_array_equal(
        _is_inside_circle(*POINTS, *circle, True).astype(bool),
        [False, False, True],
    )
    assert_array_equal(
        _is_inside_circle(*POINTS, *circle, False).astype(bool),
        [False, False, False],
    )


def test_is_inside_ellipse():
    """Test _is_inside_ellipse function."""
    # compare to circle
    circle = 0.8, 0.4, 0.05
    assert_array_equal(
        _is_inside_ellipse(*POINTS, *circle, 0.05, 1e-3, True),
        _is_inside_circle(*POINTS, *circle, True),
    )
    assert_array_equal(
        _is_inside_ellipse_(*POINTS, *circle, 0.05, 1e-3, 1e-3, True),
        _is_inside_circle(*POINTS, *circle, True),
    )
    # ellipse
    ellipse = 0.8, 0.4, 0.05, 0.1
    angle = math.pi / 4
    assert_array_equal(
        _is_inside_ellipse(*POINTS, *ellipse, angle, True).astype(bool),
        [False, True, True],
    )
    assert_array_equal(
        _is_inside_ellipse_(
            *POINTS, *ellipse, math.sin(angle), math.cos(angle), True
        ).astype(bool),
        [False, True, True],
    )


def test_is_inside_range():
    """Test _is_inside_range function."""
    assert_array_equal(
        _is_inside_range(*POINTS, 0.3, 0.5, 0.35, 0.5, True).astype(bool),
        [True, False, False],
    )


def test_is_inside_rectangle():
    """Test _is_inside_rectangle function."""
    assert_array_equal(
        _is_inside_rectangle(*POINTS, 0.4, 0.38, 0.83, 0.4, 0.1, True).astype(
            bool
        ),
        [True, False, True],
    )


def test_is_inside_polar_rectangle():
    """Test _is_inside_polar_rectangle function."""
    assert_array_equal(
        _is_inside_polar_rectangle(
            *POINTS, math.pi / 3, math.pi / 5, 0.5, 0.8, True
        ).astype(bool),
        [True, False, False],
    )


def test_is_inside_stadium():
    """Test _is_inside_stadium function."""
    stadium = 0.8, 0.4, 0.042, 0.2, 0.025
    assert_allclose(
        _is_inside_stadium([0.4, 0.84], [0.38, 0.4], *stadium, True).astype(
            bool
        ),
        [False, False],
    )
    assert_allclose(
        _is_inside_stadium([0.4, 0.82], [0.38, 0.4], *stadium, True).astype(
            bool
        ),
        [False, True],
    )
    assert_allclose(
        _is_inside_stadium(0.8, 0.4, *stadium, True).astype(bool),
        [True],
    )
    assert_allclose(
        _is_inside_stadium(0.9, 0.4, *stadium, True).astype(bool),
        [False],
    )
    assert _is_near_segment is _is_near_segment


def test_is_near_line():
    """Test _is_near_line function."""
    assert_array_equal(
        _is_near_line(*POINTS, 0.4, 0.38, 0.83, 0.4, 0.001, True).astype(bool),
        [True, False, True],
    )


def test_distance_from_point():
    """Test _distance_from_point function."""
    assert_allclose(
        _distance_from_point(*POINTS, 0.8, 0.4),
        [0.4005, 0.06, 0.02],
        atol=1e-6,
    )


def test_distance_from_line():
    """Test _distance_from_line function."""
    assert_allclose(
        _distance_from_line(*POINTS, 0.4, 0.38, 0.83, 0.4),
        [0.0, 0.001394, 0.000465],
        atol=1e-6,
    )


def test_distance_from_segment():
    """Test _distance_from_segment function."""
    assert_allclose(
        _distance_from_segment(*POINTS, 0.4, 0.38, 0.83, 0.4),
        [0.0, 0.03, 0.000465],
        atol=1e-6,
    )


def test_fraction_on_line():
    """Test _fraction_on_line function."""
    assert_allclose(
        _fraction_on_line(*POINTS, 0.4, 0.38, 0.83, 0.4),
        [1.0, -0.069617, 0.023206],
        atol=1e-6,
    )


def test_fraction_on_segment():
    """Test _fraction_on_segment function."""
    assert_allclose(
        _fraction_on_segment(*POINTS, 0.4, 0.38, 0.83, 0.4),
        [1.0, 0.0, 0.023206],
        atol=1e-6,
    )


def test_point_on_line():
    """Test _point_on_line function."""
    assert_allclose(
        _point_on_line([0.7, 0.5, 0.3], [0.3, 0.4, 0.3], *LINE),
        [[0.704, 0.494, 0.312], [0.328, 0.358, 0.384]],
    )
    # beyond endpoints
    assert_allclose(
        _point_on_line([0.1, 1.0], [0.5, 0.5], *LINE),
        [[0.088, 0.97], [0.416, 0.29]],
    )
    assert_allclose(
        _point_on_line(*POINTS, 0.4, 0.38, 0.83, 0.4),
        [[0.4, 0.859935, 0.820022], [0.38, 0.401392, 0.399536]],
        atol=1e-6,
    )


def test_point_on_segment():
    """Test _point_on_segment function."""
    assert_allclose(
        _point_on_segment([0.7, 0.5, 0.3], [0.3, 0.4, 0.3], *LINE),
        [[0.704, 0.494, 0.312], [0.328, 0.358, 0.384]],
    )
    # beyond endpoints
    assert_allclose(
        _point_on_segment([0.1, 1.0], [0.5, 0.5], *LINE),
        [[0.2, 0.9], [0.4, 0.3]],
    )
    assert_allclose(
        _point_on_segment(*POINTS, 0.4, 0.38, 0.83, 0.4),
        [[0.4, 0.83, 0.820022], [0.38, 0.4, 0.399536]],
        atol=1e-6,
    )


@pytest.mark.parametrize(
    'segment, expected',
    [
        (LINE, [0.98994949, -0.14142136, 0.70710678]),
        ((0.9, 0.3, 0.2, 0.4), [-0.98994949, 0.14142136, 0.70710678]),
    ],
)
def test_segment_direction_and_length(segment, expected):
    """Test _segment_direction_and_length function."""
    assert_allclose(_segment_direction_and_length(*segment), expected)


def test_intersection_circle_circle():
    """Test _intersection_circle_circle function."""
    assert_allclose(
        _intersection_circle_circle(
            0.0, 0.0, math.hypot(0.6, 0.4), 0.6, 0.4, 0.2
        ),
        [0.686791, 0.219813, 0.467055, 0.549418],
        1e-3,
    )
    assert_array_equal(
        _intersection_circle_circle(0.0, 0.0, 1.0, 0.6, 0.4, 0.2),
        [nan, nan, nan, nan],
    )


def test__intersection_circle_line():
    """Test _intersection_circle_line function."""
    assert_allclose(
        _intersection_circle_line(0.6, 0.4, 0.2, 0.0, 0.0, 0.6, 0.4),
        [0.76641, 0.51094, 0.43359, 0.28906],
        1e-3,
    )
    assert_array_equal(
        _intersection_circle_line(0.6, 0.4, 0.2, 0.0, 0.0, 0.6, 0.1),
        [nan, nan, nan, nan],
    )
