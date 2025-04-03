"""Test the phasorpy.color module."""

import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)

from phasorpy.color import CATEGORICAL, float2int, pseudo_color, wavelength2rgb


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


def test_wavelength2rgb():
    """Test wavelength2rgb function."""
    rgb = wavelength2rgb(517)
    assert isinstance(rgb, tuple)
    assert_almost_equal(rgb, (0.0, 0.749744, 0.0))

    assert wavelength2rgb(517.2, numpy.uint8) == (0, 191, 0)

    assert_array_equal(
        wavelength2rgb([517, 566], 'uint8'),
        numpy.array([[0, 191, 0], [133, 190, 0]], 'uint8'),
    )

    assert_array_equal(
        wavelength2rgb([517, 566], 'float16'),
        numpy.array([[0, 0.749744, 0], [0.52133, 0.744288, 0]], 'float16'),
    )


def test_float2int():
    """Test float2int function."""
    assert_array_equal(float2int([0.0, 0.5, 1.0]), [0, 128, 255])
    with pytest.raises(ValueError):
        float2int(500.0, numpy.float32)  # not an integer dtype
    with pytest.raises(ValueError):
        float2int(500, numpy.uint8)  # not a floating-point array


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
