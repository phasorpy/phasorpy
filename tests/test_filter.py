"""Tests for the phasorpy.filter module."""

import math
import os

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phasorpy.datasets import fetch
from phasorpy.filter import (
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_threshold,
    signal_filter_ncpca,
    signal_filter_svd,
)
from phasorpy.io import signal_from_imspector_tiff, signal_from_lsm
from phasorpy.phasor import phasor_from_polar, phasor_from_signal

SKIP_FETCH = os.environ.get('SKIP_FETCH', False)
NAN = math.nan

numpy.random.seed(42)


@pytest.mark.parametrize(
    'real, imag, use_scipy, repeat, size, skip_axis, kwargs, expected',
    [
        # single element
        ([0], [0], False, 1, 3, None, {}, ([0], [0])),
        # all equal
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            False,
            1,
            3,
            None,
            {},
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ),
        ),
        # random float values
        (
            [[0.5, 0.5, 2.0], [1.0, 2.0, 3.0], [10.0, 5.0, 1.0]],
            [[5.0, 6.0, 2.0], [10.0, 4.0, 8.0], [0.0, 7.0, 8.0]],
            False,
            1,
            3,
            None,
            {},
            (
                [[0.5, 1.0, 2.0], [1.0, 2.0, 2.0], [5.0, 3.0, 2.0]],
                [[5.0, 5.0, 4.0], [5.0, 6.0, 7.0], [4.0, 7.0, 8.0]],
            ),
        ),
        # 5x5 array with 3x3 filter
        (
            numpy.arange(25).reshape(5, 5),
            numpy.arange(25, 50).reshape(5, 5),
            False,
            1,
            3,
            None,
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
        ),
        # 5x5 array with 3x3 filter repeated 5 times
        (
            numpy.arange(25).reshape(5, 5),
            numpy.arange(25, 50).reshape(5, 5),
            False,
            5,
            3,
            None,
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
        ),
        # 5x5 array with 5x5 filter repeated 1 time
        (
            numpy.arange(25).reshape(5, 5),
            numpy.arange(25, 50).reshape(5, 5),
            False,
            1,
            5,
            None,
            {},
            (
                [
                    [2, 3, 4, 4, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 20, 20, 21, 22],
                ],
                [
                    [27, 28, 29, 29, 29],
                    [30, 31, 32, 33, 34],
                    [35, 36, 37, 38, 39],
                    [40, 41, 42, 43, 44],
                    [45, 45, 45, 46, 47],
                ],
            ),
        ),
        # 5x5 array with float32 dtype values
        (
            numpy.arange(25, dtype=numpy.float32).reshape(5, 5),
            numpy.arange(25, 50, dtype=numpy.float32).reshape(5, 5),
            False,
            5,
            3,
            None,
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
        ),
        # 3x3x3 array with 3x3 filter repeated 3 with first axis skipped
        (
            numpy.arange(27).reshape(3, 3, 3),
            numpy.arange(10, 37).reshape(3, 3, 3),
            False,
            3,
            3,
            0,
            {},
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
        ),
        # 'median_scipy' method with axes as kwarg
        (
            numpy.arange(27).reshape(3, 3, 3),
            numpy.arange(10, 37).reshape(3, 3, 3),
            True,
            3,
            3,
            0,  # None,
            {},  # {'axes': (1, 2)},
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
        ),
        # same output for methods from 2D array without NaN
        (
            numpy.arange(25).reshape(5, 5),
            numpy.arange(25, 50).reshape(5, 5),
            False,
            1,
            3,
            None,
            {},
            phasor_filter_median(
                numpy.ones((5, 5)),
                numpy.arange(25).reshape(5, 5),
                numpy.arange(25, 50).reshape(5, 5),
                use_scipy=True,
            )[1:],
        ),
        # same output for methods from 3D array without NaN
        (
            numpy.arange(27).reshape(3, 3, 3),
            numpy.arange(10, 37).reshape(3, 3, 3),
            False,
            1,
            3,
            None,
            {},
            phasor_filter_median(
                numpy.ones((3, 3, 3)),
                numpy.arange(27).reshape(3, 3, 3),
                numpy.arange(10, 37).reshape(3, 3, 3),
                use_scipy=True,
            )[1:],
        ),
        # same output for methods from 3D array without NaN and skip axes
        (
            numpy.arange(27).reshape(3, 3, 3),
            numpy.arange(10, 37).reshape(3, 3, 3),
            False,
            1,
            3,
            0,
            {},
            phasor_filter_median(
                numpy.ones((3, 3, 3)),
                numpy.arange(27).reshape(3, 3, 3),
                numpy.arange(10, 37).reshape(3, 3, 3),
                use_scipy=True,
                skip_axis=0,
            )[1:],
        ),
        # non-contiguos axes for 2D filtering
        (
            numpy.arange(81).reshape(3, 3, 3, 3),
            numpy.arange(10, 91).reshape(3, 3, 3, 3),
            False,
            1,
            3,
            [0, 2],
            {},
            phasor_filter_median(
                numpy.ones((3, 3, 3, 3)),
                numpy.arange(81).reshape(3, 3, 3, 3),
                numpy.arange(10, 91).reshape(3, 3, 3, 3),
                use_scipy=True,
                skip_axis=[0, 2],
            )[1:],
        ),
        # repeat = 0
        (
            numpy.arange(9).reshape(3, 3),
            numpy.arange(10, 19).reshape(3, 3),
            False,
            0,
            3,
            None,
            {},
            (
                numpy.arange(9).reshape(3, 3),
                numpy.arange(10, 19).reshape(3, 3),
            ),
        ),
        # size = 1
        (
            numpy.arange(9).reshape(3, 3),
            numpy.arange(10, 19).reshape(3, 3),
            False,
            1,
            1,
            None,
            {},
            (
                numpy.arange(9).reshape(3, 3),
                numpy.arange(10, 19).reshape(3, 3),
            ),
        ),
    ],
)
def test_phasor_filter_median(
    real, imag, use_scipy, repeat, size, skip_axis, kwargs, expected
):
    """Test phasor_filter_median function."""
    assert_allclose(
        phasor_filter_median(
            numpy.full_like(real, 1),
            real,
            imag,
            use_scipy=use_scipy,
            repeat=repeat,
            size=size,
            skip_axis=skip_axis,
            **kwargs,
        )[1:],
        expected,
    )


def test_phasor_filter_median_errors():
    """Test phasor_filter_median function errors."""
    # shape mismatch
    with pytest.raises(ValueError):
        phasor_filter_median([[0]], [0], [0], repeat=1)
    # shape mismatch
    with pytest.raises(ValueError):
        phasor_filter_median([0], [0], [[0]], repeat=1)
    # repeat < 0
    with pytest.raises(ValueError):
        phasor_filter_median([[0]], [[0]], [[0]], repeat=-3)
    # size < 1
    with pytest.raises(ValueError):
        phasor_filter_median([[0]], [[0]], [[0]], size=0)


@pytest.mark.parametrize(
    'mean, real, imag, sigma, levels, harmonic, skip_axis, expected',
    [
        # single element
        (
            [1],
            [[1], [1]],
            [[1], [1]],
            2,
            1,
            None,
            None,
            ([1], [[1], [1]], [[1], [1]]),
        ),
        # 2D arrays with 2 harmonics not specified
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            2,
            1,
            None,
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                    ],
                    [
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                    ],
                ],
                [
                    [
                        [0.275, 0.275, 0.275],
                        [0.3, 0.3, 0.3],
                        [0.325, 0.325, 0.325],
                    ],
                    [
                        [0.1375, 0.1375, 0.1375],
                        [0.15, 0.15, 0.15],
                        [0.1625, 0.1625, 0.1625],
                    ],
                ],
            ),
        ),
        # 2D arrays with 1 and 2 harmonics specified
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            2,
            1,
            [1, 2],
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                    ],
                    [
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                    ],
                ],
                [
                    [
                        [0.275, 0.275, 0.275],
                        [0.3, 0.3, 0.3],
                        [0.325, 0.325, 0.325],
                    ],
                    [
                        [0.1375, 0.1375, 0.1375],
                        [0.15, 0.15, 0.15],
                        [0.1625, 0.1625, 0.1625],
                    ],
                ],
            ),
        ),
        # 2D arrays with 2 and 4 harmonics specified
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            2,
            1,
            [1, 2],
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                    ],
                    [
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                    ],
                ],
                [
                    [
                        [0.275, 0.275, 0.275],
                        [0.3, 0.3, 0.3],
                        [0.325, 0.325, 0.325],
                    ],
                    [
                        [0.1375, 0.1375, 0.1375],
                        [0.15, 0.15, 0.15],
                        [0.1625, 0.1625, 0.1625],
                    ],
                ],
            ),
        ),
        # 2D arrays with 1, 2 and 4 harmonics specified
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                [[0.05, 0.125, 0.2], [0.05, 0.125, 0.2], [0.05, 0.125, 0.2]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                [[0.05, 0.05, 0.05], [0.075, 0.075, 0.075], [0.1, 0.1, 0.1]],
            ],
            2,
            1,
            [1, 2, 4],
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                    ],
                    [
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                    ],
                    [
                        [0.10625, 0.125, 0.14375],
                        [0.10625, 0.125, 0.14375],
                        [0.10625, 0.125, 0.14375],
                    ],
                ],
                [
                    [
                        [0.275, 0.275, 0.275],
                        [0.3, 0.3, 0.3],
                        [0.325, 0.325, 0.325],
                    ],
                    [
                        [0.1375, 0.1375, 0.1375],
                        [0.15, 0.15, 0.15],
                        [0.1625, 0.1625, 0.1625],
                    ],
                    [
                        [0.06875, 0.06875, 0.06875],
                        [0.075, 0.075, 0.075],
                        [0.08125, 0.08125, 0.08125],
                    ],
                ],
            ),
        ),
        # 2D arrays with 1, 2, 4 and 8 harmonics specified
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                [[0.05, 0.125, 0.2], [0.05, 0.125, 0.2], [0.05, 0.125, 0.2]],
                [
                    [0.025, 0.0625, 0.1],
                    [0.025, 0.0625, 0.1],
                    [0.025, 0.0625, 0.1],
                ],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                [[0.05, 0.05, 0.05], [0.075, 0.075, 0.075], [0.1, 0.1, 0.1]],
                [
                    [0.025, 0.025, 0.025],
                    [0.0375, 0.0375, 0.0375],
                    [0.05, 0.05, 0.05],
                ],
            ],
            2,
            1,
            [1, 2, 4, 8],
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                        [0.425, 0.5, 0.575],
                    ],
                    [
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                        [0.2125, 0.25, 0.2875],
                    ],
                    [
                        [0.10625, 0.125, 0.14375],
                        [0.10625, 0.125, 0.14375],
                        [0.10625, 0.125, 0.14375],
                    ],
                    [
                        [0.053125, 0.0625, 0.071875],
                        [0.053125, 0.0625, 0.071875],
                        [0.053125, 0.0625, 0.071875],
                    ],
                ],
                [
                    [
                        [0.275, 0.275, 0.275],
                        [0.3, 0.3, 0.3],
                        [0.325, 0.325, 0.325],
                    ],
                    [
                        [0.1375, 0.1375, 0.1375],
                        [0.15, 0.15, 0.15],
                        [0.1625, 0.1625, 0.1625],
                    ],
                    [
                        [0.06875, 0.06875, 0.06875],
                        [0.075, 0.075, 0.075],
                        [0.08125, 0.08125, 0.08125],
                    ],
                    [
                        [0.034375, 0.034375, 0.034375],
                        [0.0375, 0.0375, 0.0375],
                        [0.040625, 0.040625, 0.040625],
                    ],
                ],
            ),
        ),
        # levels == 0 (no filtering)
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            2,
            0,
            None,
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                ],
                [
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                ],
            ),
        ),
        # sigma == 0 (no filtering)
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            0,
            1,
            None,
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                ],
                [
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                ],
            ),
        ),
        # sigma < 0 (no filtering)
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [
                [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
            ],
            [
                [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
            ],
            -2,
            1,
            None,
            None,
            (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                ],
                [
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                ],
            ),
        ),
        # skip_axis
        (
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            ],
            [
                [
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                    [[0.2, 0.5, 0.8], [0.2, 0.5, 0.8], [0.2, 0.5, 0.8]],
                ],
                [
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                    [[0.1, 0.25, 0.4], [0.1, 0.25, 0.4], [0.1, 0.25, 0.4]],
                ],
            ],
            [
                [
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                    [[0.2, 0.2, 0.2], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4]],
                ],
                [
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                    [[0.1, 0.1, 0.1], [0.15, 0.15, 0.15], [0.2, 0.2, 0.2]],
                ],
            ],
            2,
            1,
            None,
            0,
            (
                [
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                ],
                [
                    [
                        [
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                        ],
                        [
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                        ],
                        [
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                        ],
                        [
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                            [0.425, 0.5, 0.575],
                        ],
                    ],
                    [
                        [
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                        ],
                        [
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                        ],
                        [
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                        ],
                        [
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                            [0.2125, 0.25, 0.2875],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [0.275, 0.275, 0.275],
                            [0.3, 0.3, 0.3],
                            [0.325, 0.325, 0.325],
                        ],
                        [
                            [0.275, 0.275, 0.275],
                            [0.3, 0.3, 0.3],
                            [0.325, 0.325, 0.325],
                        ],
                        [
                            [0.275, 0.275, 0.275],
                            [0.3, 0.3, 0.3],
                            [0.325, 0.325, 0.325],
                        ],
                        [
                            [0.275, 0.275, 0.275],
                            [0.3, 0.3, 0.3],
                            [0.325, 0.325, 0.325],
                        ],
                    ],
                    [
                        [
                            [0.1375, 0.1375, 0.1375],
                            [0.15, 0.15, 0.15],
                            [0.1625, 0.1625, 0.1625],
                        ],
                        [
                            [0.1375, 0.1375, 0.1375],
                            [0.15, 0.15, 0.15],
                            [0.1625, 0.1625, 0.1625],
                        ],
                        [
                            [0.1375, 0.1375, 0.1375],
                            [0.15, 0.15, 0.15],
                            [0.1625, 0.1625, 0.1625],
                        ],
                        [
                            [0.1375, 0.1375, 0.1375],
                            [0.15, 0.15, 0.15],
                            [0.1625, 0.1625, 0.1625],
                        ],
                    ],
                ],
            ),
        ),
        # NANs
        # zeroes in mean
    ],
)
def test_phasor_filter_pawflim(
    mean, real, imag, sigma, levels, harmonic, skip_axis, expected
):
    """Test phasor_filter_pawflim function."""
    mean, real, imag = phasor_filter_pawflim(
        mean,
        real,
        imag,
        sigma=sigma,
        levels=levels,
        harmonic=harmonic,
        skip_axis=skip_axis,
    )
    assert_allclose(mean, expected[0])
    assert_allclose(real, expected[1])
    assert_allclose(imag, expected[2])


def test_phasor_filter_pawflim_errors():
    """Test phasor_filter_pawflim function errors."""
    # shape mismatch between real and imag
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1]], [[1], [1]])
    # shape mismatch between real and imag
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1], [1]], [[1]])
    # shape mismatch between mean and real
    with pytest.raises(ValueError):
        phasor_filter_pawflim(
            [[1, 1, 1], [1, 1, 1]],
            [[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
            [[[1, 1], [1, 1]], [[1, 1], [1, 1]]],
        )
    # levels < 1
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1], [1]], [[1], [1]], levels=-1)
    # not all harmonics with corresponding double/half
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1], [1]], [[1], [1]], harmonic=[1, 3])
    # not all harmonics with corresponding double/half
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1], [1]], [[1], [1]], harmonic=[2, 8])
    # no harmonic axis
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [1], [1])
    # no harmonic axis
    with pytest.raises(ValueError):
        phasor_filter_pawflim([[1], [1]], [[1], [1]], [[1], [1]])
    # no harmonic axis 2D
    with pytest.raises(ValueError):
        phasor_filter_pawflim([[1], [1]], [[1], [1]], [[1], [1]])
    # less than two harmonics
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1]], [[1]])
    # number of harmonics does not match first axis of `real` and `imag`
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1], [1]], [[1], [1]], harmonic=[1, 2, 3])


def test_phasor_threshold():
    """Test phasor_threshold function."""
    # no threshold
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, 0.5], [0.3, 0.5]),
        ([0.5, 0.4], [0.2, 0.5], [0.3, 0.5]),
    )
    # lower mean threshold
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, 0.5], [0.3, 0.5], 0.5),
        ([0.5, NAN], [0.2, NAN], [0.3, NAN]),
    )
    # NAN in mean propagated to real and imag
    assert_allclose(
        phasor_threshold([0.5, NAN], [0.2, 0.5], [0.3, 0.5], 0.5),
        ([0.5, NAN], [0.2, NAN], [0.3, NAN]),
    )
    # NAN in real propagated to mean and imag
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, NAN], [0.3, 0.5], 0.5),
        ([0.5, NAN], [0.2, NAN], [0.3, NAN]),
    )
    # NAN in imag propagated to real and mean
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, 0.5], [0.3, NAN], 0.5),
        ([0.5, NAN], [0.2, NAN], [0.3, NAN]),
    )
    # 2D array with lower mean threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            0.5,
        ),
        (
            [[0.5, NAN], [0.8, 0.6]],
            [[0.1, NAN], [0.3, 0.4]],
            [[0.5, NAN], [0.7, 0.8]],
        ),
    )
    # 2D array with lower and upper mean threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            0.5,
            0.7,
        ),
        (
            [[0.5, NAN], [NAN, 0.6]],
            [[0.1, NAN], [NAN, 0.4]],
            [[0.5, NAN], [NAN, 0.8]],
        ),
    )
    # 2D array with lower mean and real threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            0.5,
            real_min=0.3,
            real_max=0.35,
        ),
        (
            [[NAN, NAN], [0.8, NAN]],
            [[NAN, NAN], [0.3, NAN]],
            [[NAN, NAN], [0.7, NAN]],
        ),
    )
    # 2D array with lower mean and imag threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            0.5,
            imag_min=0.7,
            imag_max=0.75,
        ),
        (
            [[NAN, NAN], [0.8, NAN]],
            [[NAN, NAN], [0.3, NAN]],
            [[NAN, NAN], [0.7, NAN]],
        ),
    )
    # 3D array with different real threshold for first dimension
    assert_allclose(
        phasor_threshold(
            [[[0.4, 0.5]], [[0.8, 0.9]]],
            [[[0.1, 0.2]], [[0.5, 0.6]]],
            [[[0.5, 0.3]], [[0.6, 0.7]]],
            real_min=[[[0.2]], [[0.6]]],
        ),
        (
            [[[NAN, 0.5]], [[NAN, 0.9]]],
            [[[NAN, 0.2]], [[NAN, 0.6]]],
            [[[NAN, 0.3]], [[NAN, 0.7]]],
        ),
    )
    # 2D array with lower and upper phase threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            phase_min=1.2,
            phase_max=1.3,
        ),
        (
            [[NAN, 0.4], [NAN, NAN]],
            [[NAN, 0.2], [NAN, NAN]],
            [[NAN, 0.6], [NAN, NAN]],
        ),
    )
    # 2D array with lower and upper modulation threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            modulation_min=0.7,
            modulation_max=0.8,
        ),
        (
            [[NAN, NAN], [0.8, NAN]],
            [[NAN, NAN], [0.3, NAN]],
            [[NAN, NAN], [0.7, NAN]],
        ),
    )
    # 2D array with lower and upper phase and modulation threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            phase_min=1.2,
            phase_max=1.3,
            modulation_min=0.7,
            modulation_max=0.8,
        ),
        (
            [[NAN, NAN], [NAN, NAN]],
            [[NAN, NAN], [NAN, NAN]],
            [[NAN, NAN], [NAN, NAN]],
        ),
    )
    # 2D array with open interval, lower and upper mean threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            0.5,
            0.8,
            open_interval=True,
        ),
        (
            [[NAN, NAN], [NAN, 0.6]],
            [[NAN, NAN], [NAN, 0.4]],
            [[NAN, NAN], [NAN, 0.8]],
        ),
    )
    # 2D array with open interval, lower and upper real threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            real_min=0.2,
            real_max=0.4,
            open_interval=True,
        ),
        (
            [[NAN, NAN], [0.8, NAN]],
            [[NAN, NAN], [0.3, NAN]],
            [[NAN, NAN], [0.7, NAN]],
        ),
    )
    # 2D array with open interval, lower and upper imag threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            imag_min=0.6,
            imag_max=0.8,
            open_interval=True,
        ),
        (
            [[NAN, NAN], [0.8, NAN]],
            [[NAN, NAN], [0.3, NAN]],
            [[NAN, NAN], [0.7, NAN]],
        ),
    )
    # 2D array with open interval, lower and upper phase threshold
    real, imag = phasor_from_polar(
        [[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]
    )
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            real,
            imag,
            phase_min=0.2,
            phase_max=0.4,
            open_interval=True,
        ),
        (
            [[NAN, NAN], [0.8, NAN]],
            [[NAN, NAN], [real[1][0], NAN]],
            [[NAN, NAN], [imag[1][0], NAN]],
        ),
    )
    # 2D array with open interval, lower and upper modulation threshold
    assert_allclose(
        phasor_threshold(
            [[0.5, 0.4], [0.8, 0.6]],
            real,
            imag,
            modulation_min=0.6,
            modulation_max=0.8,
            open_interval=True,
        ),
        (
            [[NAN, NAN], [0.8, NAN]],
            [[NAN, NAN], [real[1][0], NAN]],
            [[NAN, NAN], [imag[1][0], NAN]],
        ),
    )


@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64])
@pytest.mark.parametrize(
    'kwargs',
    [
        {},
        {'mean_min': 0.5},
        {'real_min': 0.5, 'phase_max': 0.7},
    ],
)
def test_phasor_threshold_dtype(dtype, kwargs):
    """Test phasor_threshold function preserves dtype."""
    mean, real, imag = phasor_threshold(
        numpy.asarray([0.5, 0.4], dtype=dtype),
        numpy.asarray([0.2, 0.5], dtype=dtype),
        numpy.asarray([0.3, 0.5], dtype=dtype),
        **kwargs,
    )
    assert mean.dtype == dtype
    assert real.dtype == dtype
    assert imag.dtype == dtype


def test_phasor_threshold_harmonic():
    """Test phasor_threshold function with multiple harmonics."""
    data = numpy.random.random((3, 2, 8))
    data[0, 0, 0] = NAN
    data[1, 0, 1] = NAN
    data[1, 1, 2] = NAN
    data[2, 0, 3] = NAN
    data[2, 1, 4] = NAN
    mean, real, imag = data
    mean_copy, real_copy, imag_copy = data.copy()

    # NaNs should propagate to all dimensions
    result = data.copy()
    result[:, :, :5] = NAN
    mean_, real_, imag_ = result

    # detect harmonic axis
    mean1, real1, imag1 = phasor_threshold(mean[0], real, imag)
    assert_array_equal(mean, mean_copy)
    assert_array_equal(real, real_copy)
    assert_array_equal(imag, imag_copy)
    assert_allclose(mean1, mean_[0])
    assert_allclose(real1, real_)
    assert_allclose(imag1, imag_)

    # scalar
    mean, real, imag = data[..., 0]
    mean_, real_, imag_ = result[..., 0]

    mean1, real1, imag1 = phasor_threshold(mean[0], real, imag)
    assert_allclose(mean1, mean_[0])
    assert_allclose(real1, real_)
    assert_allclose(imag1, imag_)

    # use ufunc: disable detect harmonic axis
    mean, real, imag = data
    result = data.copy()
    result[0, 1, :] = result[0, 0, :]
    result[0, 0, 1] = NAN
    result[0, 1, 2] = NAN
    result[0, 0, 3] = NAN
    result[0, 1, 4] = NAN
    result[1, 0, 0] = NAN
    result[1, 0, 3] = NAN
    result[1, 1, 0] = NAN
    result[1, 1, 4] = NAN
    result[2, 0, 0] = NAN
    result[2, 0, 1] = NAN
    result[2, 1, 0] = NAN
    result[2, 1, 2] = NAN
    mean_, real_, imag_ = result

    mean1, real1, imag1 = phasor_threshold(
        mean[0], real, imag, detect_harmonics=False
    )
    assert_allclose(mean1, mean_)
    assert_allclose(real1, real_)
    assert_allclose(imag1, imag_)


@pytest.mark.parametrize('dtype', [None, 'float32'])
@pytest.mark.parametrize('spectral_vector', [None, True])
def test_signal_filter_svd(dtype, spectral_vector):
    """Test signal_filter_svd function."""
    # TODO: test synthetic data

    signal = signal_from_lsm(fetch('paramecium.lsm')).data[:, ::16, ::16]
    if dtype is not None:
        signal = signal.astype(dtype)
    mean, real, imag = phasor_from_signal(signal, axis=0)

    if spectral_vector is not None:
        spectral_vector = numpy.moveaxis(numpy.stack([real, imag]), 0, -1)

    denoised = signal_filter_svd(
        signal,
        spectral_vector,
        axis=0,
        sigma=0.1,
        vmin=None,
        harmonic=None,
        dtype=dtype,
        num_threads=1,
    )

    mean1, real1, imag1 = phasor_from_signal(denoised, axis=0)
    assert_allclose(mean, mean1, atol=1e-3)
    assert_allclose(signal, denoised, atol=22)
    assert denoised.dtype == dtype


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_filter_svd_nan():
    """Test signal_filter_svd function NaN handling."""
    signal = signal_from_lsm(fetch('paramecium.lsm')).data[:, ::16, ::16]
    signal = signal.astype(numpy.float64)
    signal[0, 0, 0] = numpy.nan

    mean, real, imag = phasor_from_signal(signal, axis=0)
    spectral_vector = numpy.moveaxis(numpy.stack([real, imag]), 0, -1)
    spectral_vector[0, 1] = numpy.nan
    assert numpy.all(numpy.isnan(spectral_vector[0, 0]))

    denoised = signal_filter_svd(signal, spectral_vector, vmin=20, axis=0)

    assert_allclose(signal, denoised, atol=22)
    # spectral_vector is NaN
    assert_allclose(denoised[:, 0, 1], signal[:, 0, 1], atol=1e-3)
    # signal < vmin
    assert_allclose(denoised[:, -1, 0], signal[:, -1, 0], atol=1e-3)
    # no signal
    assert_allclose(denoised[:, -1, -1], signal[:, -1, -1], atol=1e-3)
    # signal is NaN
    assert numpy.isnan(denoised[0, 0, 0])

    mean1, real1, imag1 = phasor_from_signal(denoised, axis=0)
    assert_allclose(mean, mean1, atol=1e-3)


def test_signal_filter_svd_exceptions():
    """Test signal_filter_svd function exceptions."""
    signal = numpy.random.randint(0, 255, (16, 8, 16)).astype(numpy.float32)
    spectral_vector = numpy.random.random((16, 16, 2))

    signal_filter_svd(signal, spectral_vector, axis=1)

    with pytest.raises(ValueError):
        signal_filter_svd(signal, spectral_vector, axis=1, dtype=numpy.uint8)

    with pytest.raises(ValueError):
        signal_filter_svd(signal, spectral_vector[:15], axis=1)


@pytest.mark.parametrize('dtype', [None, 'float32'])
@pytest.mark.parametrize('n_components', [None, 8, 'mle'])
def test_signal_filter_ncpca(dtype, n_components):
    """Test signal_filter_ncpca function."""
    # TODO: test synthetic data

    signal = signal_from_imspector_tiff(fetch('Embryo.tif')).data
    if dtype is not None:
        signal = signal.astype(dtype)
        signal[:, 0, 0] = numpy.nan
    signal_copy = signal.copy()
    denoised = signal_filter_ncpca(signal, n_components, axis=0)

    assert_array_equal(signal, signal_copy)
    assert_allclose(numpy.nanmean(denoised), numpy.nanmean(signal), atol=1e-4)
    assert_allclose(
        denoised, signal, atol=1e-3 if n_components is None else 30
    )
    if dtype is not None:
        assert numpy.isnan(denoised[0, 0, 0])
    assert denoised.dtype == dtype

    mean, real, imag = phasor_from_signal(signal, axis=0)
    mean1, real1, imag1 = phasor_from_signal(denoised, axis=0)
    assert_allclose(mean1, mean, atol=1e-3 if n_components is None else 0.1)


def test_signal_filter_ncpca_exceptions():
    """Test signal_filter_ncpca function exceptions."""
    signal = numpy.random.randint(0, 255, (16, 8, 2)).astype(numpy.float32)

    signal_filter_ncpca(signal, axis=1)

    with pytest.raises(ValueError):
        signal_filter_ncpca(signal, axis=1, dtype=numpy.uint8)

    with pytest.raises(ValueError):
        signal_filter_ncpca(signal, axis=2)

    with pytest.raises(ValueError):
        signal_filter_ncpca(signal, n_components=9, axis=1)

    with pytest.raises(ValueError):
        signal_filter_ncpca([], axis=0)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, unreachable"
