"""Test the phasorpy._utils module."""

import numpy
import pytest
from numpy.testing import assert_allclose

from phasorpy._utils import (
    chunk_iter,
    dilate_coordinates,
    kwargs_notnone,
    parse_harmonic,
    parse_kwargs,
    parse_signal_axis,
    parse_skip_axis,
    phasor_from_polar_scalar,
    phasor_to_polar_scalar,
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
    x, y, i = sort_coordinates([0, 1, 2, 3], [0, 1, -1, 0])
    assert_allclose(x, [2, 3, 1, 0])
    assert_allclose(y, [-1, 0, 1, 0])
    assert_allclose(i, [2, 3, 1, 0])

    x, y, i = sort_coordinates([0, 1, 2], [0, 1, -1])
    assert_allclose(x, [0, 1, 2])
    assert_allclose(y, [0, 1, -1])
    assert_allclose(i, [0, 1, 2])

    with pytest.raises(ValueError):
        sort_coordinates([0, 1, 2, 3], [0, 1, -1])


def test_dilate_coordinates():
    """Test dilate_coordinates function."""
    x, y = dilate_coordinates([0, 1, 2, 3], [0, 1, -1, 0], 0.05)
    assert_allclose(x, [-0.048507, 1.0, 2.0, 3.048507], atol=1e-3)
    assert_allclose(y, [-0.012127, 1.05, -1.05, 0.012127], atol=1e-3)

    x, y = dilate_coordinates([0, 1, 2], [0, 1, -1], -0.05)
    assert_allclose(x, [0.05, 1.0, 1.964645], atol=1e-3)
    assert_allclose(y, [0.0, 0.95, -0.964645], atol=1e-3)

    with pytest.raises(ValueError):
        dilate_coordinates([0, 1, 2, 3], [0, 1, -1], 0.05)


def test_parse_harmonic():
    """Test parse_harmonic function."""
    assert parse_harmonic(None) == ([1], False)
    assert parse_harmonic(None, 1) == ([1], False)
    assert parse_harmonic(1) == ([1], False)
    assert parse_harmonic(1, 1) == ([1], False)
    assert parse_harmonic(numpy.int32(1), 1) == ([1], False)
    assert parse_harmonic([1], 1) == ([1], True)
    assert parse_harmonic((1,), 1) == ([1], True)
    assert parse_harmonic([numpy.int32(1)], 1) == (  # type: ignore[list-item]
        [1],
        True,
    )
    assert parse_harmonic([1, 2], 2) == ([1, 2], True)
    assert parse_harmonic([2, 1], 2) == ([2, 1], True)
    assert parse_harmonic((1, 2), 2) == ([1, 2], True)
    assert parse_harmonic(numpy.array([1, 2]), 2) == ([1, 2], True)
    assert parse_harmonic('all', 1) == ([1], True)
    assert parse_harmonic('all', 2) == ([1, 2], True)

    with pytest.raises(ValueError):
        parse_harmonic(1, 0)
    with pytest.raises(IndexError):
        parse_harmonic(0, 1)
    with pytest.raises(IndexError):
        parse_harmonic(2, 1)
    with pytest.raises(IndexError):
        parse_harmonic([1, 2], 1)
    with pytest.raises(TypeError):
        parse_harmonic([[1]], 1)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        parse_harmonic([], 1)
    with pytest.raises(ValueError):
        parse_harmonic([1, 1], 1)
    with pytest.raises(ValueError):
        parse_harmonic('alles', 1)
    with pytest.raises(TypeError):
        parse_harmonic(1.0, 1)
    with pytest.raises(TypeError):
        parse_harmonic('all')


def test_parse_signal_axis():
    """Test parse_signal_axis function."""

    class DataArray:
        dims = ('T', 'C', 'H', 'Y', 'X')

    assert parse_signal_axis(DataArray()) == (2, 'H')
    assert parse_signal_axis(DataArray(), 'C') == (1, 'C')
    assert parse_signal_axis(DataArray(), -3) == (-3, 'H')
    assert parse_signal_axis([]) == (-1, '')
    assert parse_signal_axis([], 2) == (2, '')

    DataArray.dims = ('T', 'A', 'B', 'Y', 'X')
    assert parse_signal_axis(DataArray()) == (-1, 'X')

    with pytest.raises(ValueError):
        parse_signal_axis([], 'H')

    with pytest.raises(ValueError):
        parse_signal_axis(DataArray(), 'not found')


def test_parse_skip_axis():
    """Test parse_skip_axis function."""
    assert parse_skip_axis(None, 0) == ((), ())
    assert parse_skip_axis(None, 1) == ((), (0,))
    assert parse_skip_axis((), 1) == ((), (0,))
    assert parse_skip_axis(0, 1) == ((0,), ())
    assert parse_skip_axis(0, 2) == ((0,), (1,))
    assert parse_skip_axis(-1, 2) == ((1,), (0,))
    assert parse_skip_axis((1, -2), 5) == ((1, 3), (0, 2, 4))
    with pytest.raises(ValueError):
        parse_skip_axis(0, -1)
    with pytest.raises(IndexError):
        parse_skip_axis(0, 0)
    with pytest.raises(IndexError):
        parse_skip_axis(1, 1)
    with pytest.raises(IndexError):
        parse_skip_axis(-2, 1)


def test_chunk_iter():
    """test chunk_iter function."""

    assert list(chunk_iter((), ())) == [((), '', False)]
    assert list(chunk_iter((), (), '')) == [((), '', False)]
    assert list(chunk_iter((2,), ())) == [
        ((0,), '_0', False),
        ((1,), '_1', False),
    ]
    assert list(chunk_iter((2,), (), 'X')) == [
        ((0,), '_X0', False),
        ((1,), '_X1', False),
    ]
    assert list(chunk_iter((2,), (2,), 'X')) == [
        ((slice(0, 2, 1),), '_X0', False)
    ]
    assert list(chunk_iter((2,), (2,), 'X', squeeze=True)) == [
        ((slice(0, 2, 1),), '', False)
    ]
    assert list(chunk_iter((2,), (1,), 'X')) == [
        ((slice(0, 1, 1),), '_X0', False),
        ((slice(1, 2, 1),), '_X1', False),
    ]
    assert list(chunk_iter((2,), (2,), pattern='_X{}')) == [
        ((slice(0, 2, 1),), '_X0', False)
    ]
    assert list(chunk_iter((2, 2), (2,), 'YX')) == [
        ((0, slice(0, 2, 1)), '_Y0_X0', False),
        ((1, slice(0, 2, 1)), '_Y1_X0', False),
    ]
    assert list(chunk_iter((2, 2), (1, 2), 'YX')) == [
        ((slice(0, 1, 1), slice(0, 2, 1)), '_Y0_X0', False),
        ((slice(1, 2, 1), slice(0, 2, 1)), '_Y1_X0', False),
    ]
    assert list(chunk_iter((2, 2), (3,), 'YX', squeeze=True)) == [
        ((0, slice(0, 3, 1)), '_Y0', True),
        ((1, slice(0, 3, 1)), '_Y1', True),
    ]
    assert list(chunk_iter((1, 2, 3, 4), (2, 2), 'TZYX'))[-2] == (
        (0, 1, slice(2, 4, 1), slice(0, 2, 1)),
        '_T0_Z1_Y1_X0',
        True,
    )
    assert list(chunk_iter((1, 2, 3, 4), (2, 2), 'TZYX', use_index=True))[
        -2
    ] == (
        (0, 1, slice(2, 4, 1), slice(0, 2, 1)),
        '_T0_Z1_Y2_X0',
        True,
    )
    assert list(chunk_iter((3, 255), (2, 128), 'YX', use_index=True))[0] == (
        (slice(0, 2, 1), slice(0, 128, 1)),
        '_Y0_X000',
        False,
    )

    with pytest.raises(ValueError):
        list(chunk_iter((2,), (), 'YX'))

    with pytest.raises(ValueError):
        assert list(chunk_iter((-1,), (2,)))

    with pytest.raises(ValueError):
        assert list(chunk_iter((2,), (0,)))

    with pytest.raises(ValueError):
        assert list(chunk_iter((2,), (1,), pattern='{}{}'))

    with pytest.raises(ValueError):
        list(chunk_iter((2,), (1, 2)))


def test_init_module():
    """Test init_module function."""
    from phasorpy._utils import init_module  # noqa: F401
    from phasorpy.io import phasor_from_ometiff

    assert phasor_from_ometiff.__module__ == 'phasorpy.io'


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
