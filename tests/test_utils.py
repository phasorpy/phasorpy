"""Tests for the phasorpy.utils module."""

import os

import numpy
import pytest
from numpy.testing import assert_allclose

from phasorpy.utils import number_threads, phasor_filter


def test_number_threads():
    """Test `number_threads` function."""
    assert number_threads() == 1
    assert number_threads(None, 0) == 1
    assert number_threads(1) == 1
    assert number_threads(-1) == 1
    assert number_threads(-1, 2) == 1
    assert number_threads(6) == 6
    assert number_threads(100) == 100
    assert number_threads(6, 5) == 5
    num_threads = number_threads(0)
    assert num_threads >= 1
    if num_threads > 4:
        assert number_threads(0, 4) == 4
        os.environ['PHASORPY_NUM_THREADS'] = '4'
        assert number_threads(0) == 4
        assert number_threads(6) == 6
        del os.environ['PHASORPY_NUM_THREADS']


@pytest.mark.parametrize(
    "real, imag, method, repeat, kwargs, expected",
    [
        ([0], [0], 'median', 1, {}, ([0], [0])),  # single element
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            'median',
            1,
            {},
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ),
        ),  # all equal
        (
            [[0.5, 0.5, 2.0], [1.0, 2.0, 3.0], [10.0, 5.0, 1.0]],
            [[5.0, 6.0, 2.0], [10.0, 4.0, 8.0], [0.0, 7.0, 8.0]],
            'median',
            1,
            {},
            (
                [[0.5, 1.0, 2.0], [1.0, 2.0, 2.0], [5.0, 3.0, 2.0]],
                [[5.0, 5.0, 4.0], [5.0, 6.0, 7.0], [4.0, 7.0, 8.0]],
            ),
        ),  # random float values
        (
            numpy.arange(25).reshape(5, 5),
            numpy.arange(25, 50).reshape(5, 5),
            'median',
            1,
            {},
            (
                [
                    [1, 2, 3, 4, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 20, 21, 22, 23],
                ],
                [
                    [26, 27, 28, 29, 29],
                    [30, 31, 32, 33, 34],
                    [35, 36, 37, 38, 39],
                    [40, 41, 42, 43, 44],
                    [45, 45, 46, 47, 48],
                ],
            ),
        ),  # 5x5 array with 3x3 filter
        (
            numpy.arange(25).reshape(5, 5),
            numpy.arange(25, 50).reshape(5, 5),
            'median',
            5,
            {},
            (
                [
                    [4, 4, 4, 4, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 20, 20, 20, 20],
                ],
                [
                    [29, 29, 29, 29, 29],
                    [30, 31, 32, 33, 34],
                    [35, 36, 37, 38, 39],
                    [40, 41, 42, 43, 44],
                    [45, 45, 45, 45, 45],
                ],
            ),
        ),  # 5x5 array with 3x3 filter repeated 5 times
        (
            numpy.arange(27).reshape(3, 3, 3),
            numpy.arange(10, 37).reshape(3, 3, 3),
            'median',
            3,
            {'axes': (1, 2)},
            (
                [
                    [[2, 2, 2], [3, 4, 5], [6, 6, 6]],
                    [[11, 11, 11], [12, 13, 14], [15, 15, 15]],
                    [[20, 20, 20], [21, 22, 23], [24, 24, 24]],
                ],
                [
                    [[12, 12, 12], [13, 14, 15], [16, 16, 16]],
                    [[21, 21, 21], [22, 23, 24], [25, 25, 25]],
                    [[30, 30, 30], [31, 32, 33], [34, 34, 34]],
                ],
            ),
        ),  # 3x3x3 array with 3x3 filter repeated 3 times along axes 1 and 2
    ],
)
def test_phasor_filter(real, imag, method, repeat, kwargs, expected):
    """Test `phasor_filter` function."""
    assert_allclose(
        phasor_filter(real, imag, method=method, repeat=repeat, **kwargs),
        expected,
    )


def test_phasor_filter_errors():
    """Test `phasor_filter` function errors."""
    with pytest.raises(ValueError):
        phasor_filter(
            [0], [0], method='error', repeat=1
        )  # method not supported
    with pytest.raises(ValueError):
        phasor_filter([[0]], [0], repeat=1)  # shape mismatch
    with pytest.raises(ValueError):
        phasor_filter([0], [[0]], repeat=1)  # shape mismatch
    with pytest.raises(ValueError):
        phasor_filter([[0]], [[0]], repeat=0)  # repeat = 0
    with pytest.raises(ValueError):
        phasor_filter([[0]], [[0]], repeat=-3)  # repeat < 1
