"""Test the phasorpy._utils module."""

import math

from numpy.testing import assert_allclose, assert_array_equal

from phasorpy._utils import (
    circle_circle_intersection,
    circle_line_intersection,
    kwargs_notnone,
    parse_kwargs,
    scale_matrix,
    sort_coordinates,
    update_kwargs,
)


def test_parse_kwargs():
    """Test parse_kwargs function."""
    kwargs = {'one': 1, 'two': 2, 'four': 4}
    kwargs2 = parse_kwargs(kwargs, 'two', 'three', four=None, five=5)
    assert kwargs == {'one': 1}
    assert kwargs2 == {'two': 2, 'four': 4, 'five': 5}

    kwargs = {'one': 1, 'two': 2, 'four': 4}
    kwargs2 = parse_kwargs(
        kwargs, 'two', 'three', four=None, five=5, _del=False
    )
    assert kwargs == {'one': 1, 'two': 2, 'four': 4}
    assert kwargs2 == {'two': 2, 'four': 4, 'five': 5}


def test_update_kwargs():
    """Test update_kwargs function."""
    kwargs = {
        'one': 1,
    }
    update_kwargs(kwargs, one=None, two=2)
    assert kwargs == {'one': 1, 'two': 2}


def test_kwargs_notnone():
    """Test kwargs_notnone function."""
    assert kwargs_notnone(one=1, none=None) == {'one': 1}


def test_scale_matrix():
    """Test scale_matrix function."""
    assert_allclose(
        scale_matrix(1.1, (0.0, 0.5)),
        [[1.1, 0, -0], [0, 1.1, -0.05], [0, 0, 1]],
        1e-6,
    )


def test_sort_coordinates():
    """Test sort_coordinates function."""
    x, y = sort_coordinates([0, 1, 2, 3], [0, 1, -1, 0])
    assert_allclose(x, [2, 3, 1, 0])
    assert_allclose(y, [-1, 0, 1, 0])


def test_circle_line_intersection():
    """Test circle_line_intersection function."""
    assert_allclose(
        circle_circle_intersection(
            0.0, 0.0, math.hypot(0.6, 0.4), 0.6, 0.4, 0.2
        ),
        ((0.6868, 0.2198), (0.4670, 0.5494)),
        1e-3,
    )
    assert not circle_circle_intersection(0.0, 0.0, 1.0, 0.6, 0.4, 0.2)


def test_circle_circle_intersection():
    """Test circle_circle_intersection function."""
    assert_allclose(
        circle_line_intersection(0.6, 0.4, 0.2, 0.0, 0.0, 0.6, 0.4),
        ((0.7664, 0.5109), (0.4335, 0.2890)),
        1e-3,
    )
    assert not circle_line_intersection(0.6, 0.4, 0.2, 0.0, 0.0, 0.6, 0.1)
