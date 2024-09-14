"""Tests for the phasorpy.cursors module."""

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phasorpy.color import CATEGORICAL
from phasorpy.cursors import (
    mask_from_circular_cursor,
    mask_from_elliptic_cursor,
    mask_from_polar_cursor,
    pseudo_color,
)
from phasorpy.phasor import phasor_from_polar


@pytest.mark.parametrize(
    'func', [mask_from_circular_cursor, mask_from_elliptic_cursor]
)
@pytest.mark.parametrize(
    'real, imag, center_real, center_imag, radius, expected',
    [
        (
            -0.5,
            -0.5,
            -0.5,
            -0.5,
            0.1,
            [True],
        ),  # single phasor inside single cursor
        (
            -0.5,
            -0.5,
            -0.5,
            0.5,
            0.1,
            [False],
        ),  # single phasor outside single cursor
        (
            -0.5,
            -0.5,
            [-0.5, 0.5],
            [-0.5, 0.5],
            0.1,
            [True, False],
        ),  # single phasor inside one of two cursors
        (
            [-0.5, 0.5],
            [-0.5, 0.5],
            -0.5,
            -0.5,
            0.1,
            [True, False],
        ),  # two phasors and one cursor
        (
            [-0.5, 0.5],
            [-0.5, 0.5],
            [-0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
            0.1,
            [[True, False], [False, True], [False, False]],
        ),  # two phasors and three cursors
        (
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.5, 0.5], [-0.5, 0.5]],
            -0.5,
            -0.5,
            0.1,
            [[True, False], [True, False]],
        ),  # 2D phasor and one cursor
        (
            [[-0.5, 0.5], [-0.5, 0.5]],
            [[-0.5, 0.5], [-0.5, 0.5]],
            [-0.5, 0.0],
            [-0.5, 0.0],
            0.1,
            [[[True, False], [True, False]], [[False, False], [False, False]]],
        ),  # 2D phasor and two cursors
        # TODO: add tests for multiple radius
    ],
)
def test_mask_from_circular_cursor(
    func, real, imag, center_real, center_imag, radius, expected
):
    """Test mask_from_circular/elliptic_cursor functions."""
    mask = func(real, imag, center_real, center_imag, radius=radius)
    assert_array_equal(mask, expected)


@pytest.mark.parametrize(
    'func', [mask_from_circular_cursor, mask_from_elliptic_cursor]
)
@pytest.mark.parametrize(
    'real, imag, center_real, center_imag, radius',
    [
        ([0.0], [0, 0], 0, 0, 0.1),
        ([0.0, 0.0], [0], 0, 0, 0.1),
        (0.0, 0.0, [[0, 0], [0, 0]], [[0, 0], [0, 0]], 0.1),
    ],
)
def test_mask_from_circular_cursor_errors(
    func, real, imag, center_real, center_imag, radius
):
    """Test errors for mask_from_circular/elliptic_cursor functions."""
    with pytest.raises(ValueError):
        func(real, imag, center_real, center_imag, radius=radius)


@pytest.mark.parametrize(
    'radius, radius_minor, angle, align_semicircle, expected',
    [
        ([0.1, 0.05], 0.15, None, None, [[True, False], [False, True]]),
        (0.1, [0.15, 0.1], None, None, [[True, False], [False, True]]),
        ([0.1, 0.05], [0.15, 0.1], None, True, [[True, False], [False, True]]),
        ([0.1, 0.05], [0.15, 0.1], 3.1, None, [[True, False], [False, True]]),
        (
            [0.1, 0.05],
            [0.15, 0.1],
            [3.1, 1.6],
            None,
            [[True, False], [False, True]],
        ),
        (0.5, 0.5, 0.0, None, [[True, True], [True, True]]),
    ],
)
def test_mask_from_elliptic_cursor(
    radius, radius_minor, angle, align_semicircle, expected
):
    """Test mask_from_elliptic_cursor function."""
    # the function is also tested in test_mask_from_circular_cursor
    mask = mask_from_elliptic_cursor(
        [0.2, 0.5],
        [0.4, 0.5],
        [0.2, 0.5],
        [0.4, 0.5],
        radius=radius,
        radius_minor=radius_minor,
        angle=angle,
        align_semicircle=align_semicircle,
    )
    assert_array_equal(mask, expected)


@pytest.mark.parametrize(
    'phase, modulation, '
    'phase_min, phase_max, '
    'modulation_min, modulation_max, '
    'expected',
    [
        (
            10,
            0.5,
            0,
            20,
            0.4,
            0.6,
            [True],
        ),  # single polar point inside single cursor
        (
            10,
            0.5,
            15,
            20,
            0.4,
            0.6,
            [False],
        ),  # single polar point outside phase range single cursor
        (
            10,
            0.5,
            0,
            20,
            0.2,
            0.2,
            [False],
        ),  # single polar point outside phase range single cursor
        (
            10,
            0.5,
            [0, 0],
            [20, 30],
            [0.4, 0.6],
            [0.6, 0.8],
            [True, False],
        ),  # single polar point inside one of two cursors
        (
            [10, 40],
            [0.2, 0.4],
            0,
            20,
            0.1,
            0.5,
            [True, False],
        ),  # two polar points and one cursor
        (
            [10, 40],
            [0.2, 0.4],
            [0, 0, 50],
            [20, 40, 100],
            [0.1, 0.3, 0.2],
            [0.5, 0.5, 0.4],
            [[True, False], [False, True], [False, False]],
        ),  # two polar points and three cursors
        (
            [[10, 40], [20, 30]],
            [[0.2, 0.4], [0.6, 0.8]],
            0,
            20,
            0.1,
            0.5,
            [[True, False], [False, False]],
        ),  # 2D polar points and one cursor
        (
            [[10, 40], [20, 30]],
            [[0.2, 0.4], [0.6, 0.8]],
            [0, 0],
            [20, 30],
            [0.4, 0.6],
            [0.6, 0.8],
            [[[False, False], [True, False]], [[False, False], [True, True]]],
        ),  # 2D polar points and two cursors
        # TODO: add tests for axis parameter
    ],
)
def test_mask_from_polar_cursor(
    phase,
    modulation,
    phase_min,
    phase_max,
    modulation_min,
    modulation_max,
    expected,
):
    """Test mask_from_cursor function."""
    real, imag = phasor_from_polar(numpy.deg2rad(phase), modulation)
    phase_min = numpy.deg2rad(phase_min)
    phase_max = numpy.deg2rad(phase_max)

    assert_array_equal(
        mask_from_polar_cursor(
            real, imag, phase_min, phase_max, modulation_min, modulation_max
        ),
        expected,
    )


@pytest.mark.parametrize(
    'real, imag, phase_range, modulation_range',
    [
        (
            [0],
            [0, 0],
            [0, 0],
            [0, 0],
        ),  # phase and modulation are not the same shape
        ([0, 0], [0], [0, 0], [0, 0]),
        (
            0,
            0,
            [[[0]], [[0]]],
            [0, 0],
        ),  # range array is 2D
    ],
)
def test_mask_from_polar_cursor_errors(
    real, imag, phase_range, modulation_range
):
    """Test errors for mask_from_polar_cursor function."""
    with pytest.raises(ValueError):
        mask_from_polar_cursor(real, imag, *phase_range, *modulation_range)


def test_cursors_on_grid():
    """Plot cursor functions on grid of points."""
    from math import pi

    from matplotlib import pyplot

    show = False  # enable to see figure

    def plot_mask(real, imag, mask, **kwargs):
        show = 'ax' not in kwargs
        ax = kwargs.pop('ax') if not show else pyplot.subplot()
        mask = mask.astype(bool)
        ax.set(
            aspect='equal',
            xlim=[0, 1],
            ylim=[0, 1],
            xticks=[],
            yticks=[],
            **kwargs,
        )
        ax.plot(real[mask], imag[mask], ',')
        if show:
            pyplot.show()

    coords = numpy.linspace(0.0, 1.0, 501)
    real, imag = numpy.meshgrid(coords, coords)

    _, ax = pyplot.subplots(4, 1, figsize=(3.2, 9), layout='constrained')

    mask = mask_from_circular_cursor(real, imag, 0.5, 0.5, radius=0.1)
    plot_mask(real, imag, mask, title='mask_from_circular_cursor', ax=ax[0])
    assert_array_equal(
        mask, mask_from_elliptic_cursor(real, imag, 0.5, 0.5, radius=0.1)
    )

    mask = mask_from_elliptic_cursor(
        real, imag, 0.5, 0.5, radius=0.15, radius_minor=0.05  # , angle=pi / 4
    )
    plot_mask(real, imag, mask, title='mask_from_elliptic_cursor', ax=ax[1])
    assert_array_equal(
        mask,
        mask_from_elliptic_cursor(
            real,
            imag,
            0.5,
            0.5,
            radius=0.15,
            radius_minor=0.05,
            angle=numpy.pi / 4,
        ),
    )

    mask = mask_from_elliptic_cursor(
        real,
        imag,
        0.5,
        0.5,
        radius=0.15,
        radius_minor=0.05,
        align_semicircle=True,
    )
    plot_mask(real, imag, mask, title='align_semicircle=True', ax=ax[2])
    assert_array_equal(
        mask,
        mask_from_elliptic_cursor(
            real,
            imag,
            0.5,
            0.5,
            radius=0.15,
            radius_minor=0.05,
            angle=numpy.pi / 2,
        ),
    )

    mask = mask_from_polar_cursor(
        real, imag, pi / 5, pi / 3 + 4 * pi, 0.6071, 0.8071
    )
    plot_mask(real, imag, mask, title='mask_from_polar_cursor', ax=ax[3])

    if show:
        pyplot.show()
    else:
        pyplot.close()


@pytest.mark.parametrize(
    'masks, mean, colors, expected',
    [
        ([True], None, None, CATEGORICAL[0]),  # single value true
        ([False], None, None, [0, 0, 0]),  # single value false
        (
            [[True, True]],
            None,
            None,
            numpy.asarray([CATEGORICAL[0], CATEGORICAL[0]]),
        ),  # 1D array
        (
            [[True, False]],
            None,
            None,
            numpy.asarray([CATEGORICAL[0], [0, 0, 0]]),
        ),  # 1D array with false
        (
            [[[True, True], [False, False]]],
            None,
            None,
            numpy.asarray(
                [[CATEGORICAL[0], CATEGORICAL[0]], [[0, 0, 0], [0, 0, 0]]]
            ),
        ),  # 2D array
        (
            [True, True],
            None,
            None,
            CATEGORICAL[1],
        ),  # single value with two masks
        (
            [[True, False], [False, True]],
            None,
            None,
            numpy.asarray([CATEGORICAL[0], CATEGORICAL[1]]),
        ),  # 1D array with two masks
        (
            [[True, False], [True, True]],
            None,
            None,
            numpy.asarray([CATEGORICAL[1], CATEGORICAL[1]]),
        ),  # 1D array with two masks all true
        (
            [True],
            None,
            [[0, 0, 0.5]],
            [0, 0, 0.5],
        ),  # single value true with custom color
        (
            [False],
            None,
            [[0, 0, 0.5]],
            [0, 0, 0],
        ),  # single value false with custom color
        (
            [[True, False]],
            None,
            [[0, 0, 0.5]],
            [[0, 0, 0.5], [0, 0, 0]],
        ),  # 1D array with custom color
    ],
)
def test_pseudo_color(masks, mean, colors, expected):
    """Test pseudo_color function."""
    assert_allclose(
        pseudo_color(*masks, intensity=mean, colors=colors),
        expected,
    )


def test_pseudo_color_overlay():
    """Test pseudo_color function with intensity."""
    assert_allclose(pseudo_color(True, intensity=1.0), [1, 1, 1])
    assert_allclose(pseudo_color(True, intensity=0.5), CATEGORICAL[0])
    assert_allclose(pseudo_color(False, intensity=0.4), [0.4, 0.4, 0.4])
    assert_allclose(
        pseudo_color(
            [True, True, True],
            intensity=[-0.1, 0.5, 1.1],
            vmin=None,
            vmax=None,
        ),
        [[0.0, 0.0, 0.0], [0.825397, 0.095238, 0.126984], [1.0, 1.0, 1.0]],
    )
    assert_allclose(
        pseudo_color(
            [True, True, True], intensity=[-0.1, 0.5, 1.1], vmin=-0.1, vmax=1.1
        ),
        [[0.0, 0.0, 0.0], [0.825397, 0.095238, 0.126984], [1.0, 1.0, 1.0]],
    )
    assert_allclose(
        pseudo_color(
            [True, True, True], intensity=[-0.1, 0.5, 1.1], vmin=0.0, vmax=1.0
        ),
        [[0.0, 0.0, 0.0], [0.825397, 0.095238, 0.126984], [1.0, 1.0, 1.0]],
    )


def test_pseudo_color_errors():
    """Test errors for pseudo_color function."""
    # no masks
    with pytest.raises(TypeError):
        pseudo_color()
    # masks shape mismatch
    with pytest.raises(ValueError):
        pseudo_color([0], [[0]])
    # colors not float
    with pytest.raises(ValueError):
        pseudo_color(0, colors=[[0, 0, 0]])
    # colors not 2D
    with pytest.raises(ValueError):
        pseudo_color(0, colors=[0.0, 0, 0])
    # colors last dimension not 3
    with pytest.raises(ValueError):
        pseudo_color(0, colors=[[0.0, 0]])


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
