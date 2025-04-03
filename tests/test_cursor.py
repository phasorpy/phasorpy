"""Test the phasorpy._cursor module."""

import numpy
import pytest
from numpy.testing import assert_array_equal

from phasorpy import (
    phasor_from_polar,
    phasor_mask_circular,
    phasor_mask_elliptic,
    phasor_mask_polar,
)


@pytest.mark.parametrize('func', [phasor_mask_circular, phasor_mask_elliptic])
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
def test_phasor_mask_circular(
    func, real, imag, center_real, center_imag, radius, expected
):
    """Test phasor_mask_circular/elliptic functions."""
    mask = func(real, imag, center_real, center_imag, radius=radius)
    assert_array_equal(mask, expected)


@pytest.mark.parametrize('func', [phasor_mask_circular, phasor_mask_elliptic])
@pytest.mark.parametrize(
    'real, imag, center_real, center_imag, radius',
    [
        ([0.0], [0, 0], 0, 0, 0.1),
        ([0.0, 0.0], [0], 0, 0, 0.1),
        (0.0, 0.0, [[0, 0], [0, 0]], [[0, 0], [0, 0]], 0.1),
    ],
)
def test_phasor_mask_circular_errors(
    func, real, imag, center_real, center_imag, radius
):
    """Test errors for phasor_mask_circular/elliptic functions."""
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
def test_phasor_mask_elliptic(
    radius, radius_minor, angle, align_semicircle, expected
):
    """Test phasor_mask_elliptic function."""
    # the function is also tested in test_phasor_mask_circular
    mask = phasor_mask_elliptic(
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
def test_phasor_mask_polar(
    phase,
    modulation,
    phase_min,
    phase_max,
    modulation_min,
    modulation_max,
    expected,
):
    """Test phasor_mask_polar function."""
    real, imag = phasor_from_polar(numpy.deg2rad(phase), modulation)
    phase_min = numpy.deg2rad(phase_min)
    phase_max = numpy.deg2rad(phase_max)

    assert_array_equal(
        phasor_mask_polar(
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
def test_phasor_mask_polar_errors(real, imag, phase_range, modulation_range):
    """Test errors for phasor_mask_polar function."""
    with pytest.raises(ValueError):
        phasor_mask_polar(real, imag, *phase_range, *modulation_range)


def test_mask_on_grid():
    """Plot mask functions on grid of points."""
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

    mask = phasor_mask_circular(real, imag, 0.5, 0.5, radius=0.1)
    plot_mask(real, imag, mask, title='phasor_mask_circular', ax=ax[0])
    assert_array_equal(
        mask, phasor_mask_elliptic(real, imag, 0.5, 0.5, radius=0.1)
    )

    mask = phasor_mask_elliptic(
        real, imag, 0.5, 0.5, radius=0.15, radius_minor=0.05  # , angle=pi / 4
    )
    plot_mask(real, imag, mask, title='phasor_mask_elliptic', ax=ax[1])
    assert_array_equal(
        mask,
        phasor_mask_elliptic(
            real,
            imag,
            0.5,
            0.5,
            radius=0.15,
            radius_minor=0.05,
            angle=numpy.pi / 4,
        ),
    )

    mask = phasor_mask_elliptic(
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
        phasor_mask_elliptic(
            real,
            imag,
            0.5,
            0.5,
            radius=0.15,
            radius_minor=0.05,
            angle=numpy.pi / 2,
        ),
    )

    mask = phasor_mask_polar(
        real, imag, pi / 5, pi / 3 + 4 * pi, 0.6071, 0.8071
    )
    plot_mask(real, imag, mask, title='phasor_mask_polar', ax=ax[3])

    if show:
        pyplot.show()
    else:
        pyplot.close()


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
