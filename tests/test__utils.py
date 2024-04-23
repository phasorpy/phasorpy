"""Test the phasorpy._utils module."""

import math

import pytest
from numpy.testing import assert_allclose

from phasorpy._utils import (
    circle_circle_intersection,
    circle_line_intersection,
    kwargs_notnone,
    parse_kwargs,
    phasor_from_polar_scalar,
    phasor_to_polar_scalar,
    project_phasor_to_line,
    scale_matrix,
    sort_coordinates,
    update_kwargs,
)


def test_phasor_to_polar_scalar():
    """Test phasor_to_polar_scalar function."""
    assert phasor_to_polar_scalar(0.0, 0.0) == (0.0, 0.0)
    assert_allclose(
        phasor_to_polar_scalar(0.8, 0.4, degree=True, percent=True),
        (26.565051, 89.442719),
        atol=1e-6,
    )
    assert_allclose(
        phasor_to_polar_scalar(1.0, 0.0, degree=True, percent=True),
        (0.0, 100.0),
        atol=1e-6,
    )


def test_phasor_from_polar_scalar():
    """Test phasor_from_polar_scalar function."""
    assert phasor_from_polar_scalar(0.0, 0.0) == (0.0, 0.0)
    assert_allclose(
        phasor_from_polar_scalar(
            26.565051, 89.442719, degree=True, percent=True
        ),
        (0.8, 0.4),
        atol=1e-6,
    )
    assert_allclose(
        phasor_from_polar_scalar(0.0, 100.0, degree=True, percent=True),
        (1.0, 0.0),
        atol=1e-6,
    )
    # roundtrip
    assert_allclose(
        phasor_from_polar_scalar(
            *phasor_to_polar_scalar(-0.4, -0.2, degree=True, percent=True),
            degree=True,
            percent=True,
        ),
        (-0.4, -0.2),
        atol=1e-6,
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

    x, y = sort_coordinates([0, 1, 2], [0, 1, -1])
    assert_allclose(x, [0, 1, 2])
    assert_allclose(y, [0, 1, -1])

    with pytest.raises(ValueError):
        sort_coordinates([0, 1, 2, 3], [0, 1, -1])


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


def test_project_phasor_to_line():
    """Test project_phasor_to_line function."""
    assert_allclose(
        project_phasor_to_line(
            [0.7, 0.5, 0.3], [0.3, 0.4, 0.3], [0.2, 0.9], [0.4, 0.3]
        ),
        (
            [0.704, 0.494, 0.312],
            [0.328, 0.358, 0.384],
        ),
    )
    assert_allclose(
        project_phasor_to_line([0.1, 1.0], [0.5, 0.5], [0.2, 0.9], [0.4, 0.3]),
        ([0.2, 0.9], [0.4, 0.3]),
    )
    assert_allclose(
        project_phasor_to_line(
            [0.1, 1.0], [0.5, 0.5], [0.2, 0.9], [0.4, 0.3], clip=False
        ),
        ([0.088, 0.97], [0.416, 0.29]),
    )
    with pytest.raises(ValueError):
        project_phasor_to_line([0], [0], [0.1, 0.2], [0.1, 0.2])
    with pytest.raises(ValueError):
        project_phasor_to_line([0], [0], [0.3], [0.1, 0.2])
    with pytest.raises(ValueError):
        project_phasor_to_line([0], [0], [0.1, 0.2], [0.3])
    with pytest.raises(ValueError):
        project_phasor_to_line([0], [0], [0.1], [0.3])
    with pytest.raises(ValueError):
        project_phasor_to_line([0], [0], [0.1, 0.1, 0, 1], [0.1, 0, 2])
