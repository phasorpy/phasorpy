"""Test the phasorpy._phasorpy module."""

import math
import sys

import numpy
import pytest
from numpy import nan
from numpy.testing import assert_allclose, assert_array_equal

from phasorpy._phasorpy import _is_near_segment  # same as _is_inside_stadium
from phasorpy._phasorpy import (
    _blend_darken,
    _blend_lighten,
    _blend_multiply,
    _blend_normal,
    _blend_overlay,
    _blend_screen,
    _distance_from_line,
    _distance_from_point,
    _distance_from_segment,
    _distance_from_semicircle,
    _fraction_on_line,
    _fraction_on_segment,
    _intersect_circle_circle,
    _intersect_circle_line,
    _is_inside_circle,
    _is_inside_ellipse,
    _is_inside_ellipse_,
    _is_inside_polar_rectangle,
    _is_inside_range,
    _is_inside_rectangle,
    _is_inside_semicircle,
    _is_inside_stadium,
    _is_near_line,
    _is_near_semicircle,
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
        _is_inside_circle(*POINTS, *circle).astype(bool),
        [False, False, True],
    )


def test_is_inside_ellipse():
    """Test _is_inside_ellipse function."""
    # compare to circle
    circle = 0.8, 0.4, 0.05
    assert_array_equal(
        _is_inside_ellipse(*POINTS, *circle, 0.05, 1e-3),
        _is_inside_circle(*POINTS, *circle),
    )
    assert_array_equal(
        _is_inside_ellipse_(*POINTS, *circle, 0.05, 1e-3, 1e-3),
        _is_inside_circle(*POINTS, *circle),
    )
    # ellipse
    ellipse = 0.8, 0.4, 0.05, 0.1
    angle = math.pi / 4
    assert_array_equal(
        _is_inside_ellipse(*POINTS, *ellipse, angle).astype(bool),
        [False, True, True],
    )
    assert_array_equal(
        _is_inside_ellipse_(
            *POINTS, *ellipse, math.sin(angle), math.cos(angle)
        ).astype(bool),
        [False, True, True],
    )


def test_is_inside_range():
    """Test _is_inside_range function."""
    assert_array_equal(
        _is_inside_range(*POINTS, 0.3, 0.5, 0.35, 0.5).astype(bool),
        [True, False, False],
    )


def test_is_inside_rectangle():
    """Test _is_inside_rectangle function."""
    assert_array_equal(
        _is_inside_rectangle(*POINTS, 0.4, 0.38, 0.83, 0.4, 0.1).astype(bool),
        [True, False, True],
    )


def test_is_inside_polar_rectangle():
    """Test _is_inside_polar_rectangle function."""
    assert_array_equal(
        _is_inside_polar_rectangle(
            *POINTS, math.pi / 3, math.pi / 5, 0.5, 0.8
        ).astype(bool),
        [True, False, False],
    )


def test_is_inside_stadium():
    """Test _is_inside_stadium function."""
    stadium = 0.8, 0.4, 0.042, 0.2, 0.025
    assert_allclose(
        _is_inside_stadium([0.4, 0.84], [0.38, 0.4], *stadium).astype(bool),
        [False, False],
    )
    assert_allclose(
        _is_inside_stadium([0.4, 0.82], [0.38, 0.4], *stadium).astype(bool),
        [False, True],
    )
    assert_allclose(
        _is_inside_stadium(0.8, 0.4, *stadium).astype(bool),
        [True],
    )
    assert_allclose(
        _is_inside_stadium(0.9, 0.4, *stadium).astype(bool),
        [False],
    )
    assert _is_near_segment is _is_near_segment


def test_is_inside_semicircle():
    """Test _is_inside_semicircle function."""
    real = [0.0, 0.5, 1.0, 0.5, -0.01, 1.01, 0.5, -0.015, math.nan]
    imag = [0.0, 0.5, 0.0, 0.25, -0.01, -0.01, -1.0, -0.015, 0.0]
    assert_array_equal(
        _is_inside_semicircle(real, imag, 0.02).astype(bool),
        [True, True, True, True, True, True, False, False, False],
    )
    assert_array_equal(
        _is_inside_semicircle(real, imag, 0.0).astype(bool),
        [True, True, True, True, False, False, False, False, False],
    )
    assert_array_equal(
        _is_inside_semicircle(real, imag, -0.1).astype(bool),
        [False, False, False, False, False, False, False, False, False],
    )


def test_is_near_semicircle():
    """Test _is_near_semicircle function."""
    real = [0.0, 0.5, 1.0, 0.5, -0.01, 1.01, 0.5, -0.015, math.nan]
    imag = [0.0, 0.5, 0.0, 0.25, -0.01, -0.01, -1.0, -0.015, 0.0]
    assert_array_equal(
        _is_near_semicircle(real, imag, 0.02).astype(bool),
        [True, True, True, False, True, True, False, False, False],
    )
    assert_array_equal(
        _is_near_semicircle(real, imag, 0.0).astype(bool),
        [True, True, True, False, False, False, False, False, False],
    )
    assert_array_equal(
        _is_near_semicircle(real, imag, -0.1).astype(bool),
        [False, False, False, False, False, False, False, False, False],
    )


def test_is_near_line():
    """Test _is_near_line function."""
    assert_array_equal(
        _is_near_line(*POINTS, 0.4, 0.38, 0.83, 0.4, 0.001).astype(bool),
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


def test_distance_from_semicircle():
    """Test _distance_from_semicircle function."""
    assert_allclose(
        _distance_from_semicircle(
            [0.0, 0.5, 1.0, 0.5, 0.5, 0.0, -0.5, 1.0, nan],
            [0.0, 0.5, 0.0, 0.25, 0.0, -0.5, 0.0, -0.5, 0.0],
        ),
        [0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5, nan],
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


def test_intersect_circle_circle():
    """Test _intersect_circle_circle function."""
    assert_allclose(
        _intersect_circle_circle(
            0.0, 0.0, math.hypot(0.6, 0.4), 0.6, 0.4, 0.2
        ),
        [0.686791, 0.219813, 0.467055, 0.549418],
        1e-3,
    )
    assert_array_equal(
        _intersect_circle_circle(0.0, 0.0, 1.0, 0.6, 0.4, 0.2),
        [nan, nan, nan, nan],
    )


def test_intersect_circle_line():
    """Test _intersect_circle_line function."""
    assert_allclose(
        _intersect_circle_line(0.6, 0.4, 0.2, 0.0, 0.0, 0.6, 0.4),
        [0.76641, 0.51094, 0.43359, 0.28906],
        1e-3,
    )
    assert_array_equal(
        _intersect_circle_line(0.6, 0.4, 0.2, 0.0, 0.0, 0.6, 0.1),
        [nan, nan, nan, nan],
    )


def test_geometric_ufunc_on_grid():
    """Plot geometric ufuncs used on grid of points."""
    from math import pi

    from matplotlib import pyplot

    show = False  # enable to see figure

    def plot_mask(real, imag, mask, **kwargs):
        show = 'ax' not in kwargs
        ax = kwargs.pop('ax') if not show else pyplot.subplot()
        mask = mask.astype(bool)
        ax.set(
            aspect='equal',
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xticks=[],
            yticks=[],
            **kwargs,
        )
        ax.plot(real[mask], imag[mask], ',')
        if show:
            pyplot.show()

    def plot_points(real, imag, **kwargs):
        show = 'ax' not in kwargs
        ax = kwargs.pop('ax') if not show else pyplot.subplot()
        ax.set(
            aspect='equal',
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            xticks=[],
            yticks=[],
            **kwargs,
        )
        ax.plot(real, imag, ',', color='tab:blue')
        if show:
            pyplot.show()

    def plot_image(values, **kwargs):
        show = 'ax' not in kwargs
        ax = kwargs.pop('ax') if not show else pyplot.subplot()
        ax.set(xticks=[], yticks=[], **kwargs)
        _ = ax.imshow(values, origin='lower', interpolation='nearest')
        # ax.figure.colorbar(_, ax=ax)
        if show:
            pyplot.show()

    line = (0.25, 0.75, 0.75, 0.25)
    coords = numpy.linspace(-0.05, 1.05, 501)
    real, imag = numpy.meshgrid(coords, coords)

    _, ax = pyplot.subplots(9, 2, figsize=(4, 13), layout='constrained')

    # plot_points(real, imag, title='grid', ax=ax[0, 0])

    distance = _distance_from_point(real, imag, 0.5, 0.5)
    plot_image(distance, title='_distance_from_point', ax=ax[0, 0])

    distance = _distance_from_semicircle(real, imag)
    plot_image(distance, title='_distance_from_semicircle', ax=ax[0, 1])

    distance = _distance_from_line(real, imag, *line)
    plot_image(distance, title='_distance_from_line', ax=ax[1, 0])

    distance = _distance_from_segment(real, imag, *line)
    plot_image(distance, title='_distance_from_segment', ax=ax[1, 1])

    fraction = _fraction_on_line(real, imag, *line)
    plot_image(fraction, title='_fraction_on_line', ax=ax[2, 0])

    fraction = _fraction_on_segment(real, imag, *line)
    plot_image(fraction, title='_fraction_on_segment', ax=ax[2, 1])

    re, im = _point_on_line(real, imag, *line)
    plot_points(re, im, title='_point_on_line', ax=ax[3, 0])

    re, im = _point_on_segment(real, imag, *line)
    plot_points(re, im, title='_point_on_segment', ax=ax[3, 1])

    mask = _is_near_line(real, imag, *line, 0.1)
    plot_mask(real, imag, mask, title='_is_near_line', ax=ax[4, 0])

    mask = _is_inside_stadium(real, imag, *line, 0.1)
    plot_mask(real, imag, mask, title='_is_inside_stadium', ax=ax[4, 1])

    mask = _is_inside_circle(real, imag, 0.5, 0.5, 0.1)
    plot_mask(real, imag, mask, title='_is_inside_circle', ax=ax[5, 0])

    mask = _is_inside_ellipse(real, imag, 0.5, 0.5, 0.05, 0.15, pi / 4)
    plot_mask(real, imag, mask, title='_is_inside_ellipse', ax=ax[5, 1])

    mask = _is_inside_polar_rectangle(
        real, imag, pi / 5, pi / 3 + 4 * pi, 0.6071, 0.8071
    )
    plot_mask(
        real, imag, mask, title='_is_inside_polar_rectangle', ax=ax[6, 0]
    )

    mask = _is_inside_rectangle(real, imag, *line, 0.1)
    plot_mask(real, imag, mask, title='_is_inside_rectangle', ax=ax[6, 1])

    mask = _is_inside_range(real, imag, 0.4, 0.6, 0.45, 0.55)
    plot_mask(real, imag, mask, title='_is_inside_range', ax=ax[7, 0])

    plot_points([], [], title='', ax=ax[7, 1])

    mask = _is_near_semicircle(real, imag, 0.02)
    plot_mask(real, imag, mask, title='_is_near_semicircle', ax=ax[8, 0])

    mask = _is_inside_semicircle(real, imag, 0.02)
    plot_mask(real, imag, mask, title='_is_inside_semicircle', ax=ax[8, 1])

    if show:
        pyplot.show()
    else:
        pyplot.close()


@pytest.mark.parametrize(
    'a, b, expected',
    [(0.1, 0.6, 0.6), (0.6, 0.1, 0.1), (0.1, nan, 0.1), (nan, 0.6, 0.6)],
)
def test_blend_normal(a, b, expected):
    """Test _blend_normal function."""
    assert_allclose(_blend_normal(a, b), expected)


@pytest.mark.parametrize(
    'a, b, expected',
    [(0.1, 0.6, 0.06), (0.6, 0.1, 0.06), (0.1, nan, 0.1), (nan, 0.6, nan)],
)
def test_blend_multiply(a, b, expected):
    """Test _blend_multiply function."""
    assert_allclose(_blend_multiply(a, b), expected)


@pytest.mark.parametrize(
    'a, b, expected',
    [(0.1, 0.6, 0.64), (0.6, 0.1, 0.64), (0.1, nan, 0.1), (nan, 0.6, nan)],
)
def test_blend_screen(a, b, expected):
    """Test _blend_screen function."""
    assert_allclose(_blend_screen(a, b), expected)


@pytest.mark.parametrize(
    'a, b, expected',
    [(0.1, 0.6, 0.12), (0.6, 0.1, 0.28), (0.1, nan, 0.1), (nan, 0.6, nan)],
)
def test_blend_overlay(a, b, expected):
    """Test _blend_overlay function."""
    assert_allclose(_blend_overlay(a, b), expected)


@pytest.mark.skipif(sys.platform == 'darwin', reason='github #115')
@pytest.mark.parametrize(
    'a, b, expected',
    [(0.1, 0.6, 0.1), (0.6, 0.1, 0.1), (0.1, nan, 0.1), (nan, 0.6, nan)],
)
def test_blend_darken(a, b, expected):
    """Test _blend_darken function."""
    assert_allclose(_blend_darken(a, b), expected)


@pytest.mark.skipif(sys.platform == 'darwin', reason='github #115')
@pytest.mark.parametrize(
    'a, b, expected',
    [(0.1, 0.6, 0.6), (0.6, 0.1, 0.6), (0.1, nan, 0.1), (nan, 0.6, nan)],
)
def test_blend_lighten(a, b, expected):
    """Test _blend_lighten function."""
    assert_allclose(_blend_lighten(a, b), expected)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
