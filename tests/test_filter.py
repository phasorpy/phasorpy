"""Test the phasorpy._filter module."""

import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
)

from phasorpy import (
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_from_polar,
    phasor_threshold,
)


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
    nan = numpy.nan
    # no threshold
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, 0.5], [0.3, 0.5]),
        ([0.5, 0.4], [0.2, 0.5], [0.3, 0.5]),
    )
    # lower mean threshold
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, 0.5], [0.3, 0.5], 0.5),
        ([0.5, nan], [0.2, nan], [0.3, nan]),
    )
    # nan in mean propagated to real and imag
    assert_allclose(
        phasor_threshold([0.5, nan], [0.2, 0.5], [0.3, 0.5], 0.5),
        ([0.5, nan], [0.2, nan], [0.3, nan]),
    )
    # nan in real propagated to mean and imag
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, nan], [0.3, 0.5], 0.5),
        ([0.5, nan], [0.2, nan], [0.3, nan]),
    )
    # nan in imag propagated to real and mean
    assert_allclose(
        phasor_threshold([0.5, 0.4], [0.2, 0.5], [0.3, nan], 0.5),
        ([0.5, nan], [0.2, nan], [0.3, nan]),
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
            [[0.5, nan], [0.8, 0.6]],
            [[0.1, nan], [0.3, 0.4]],
            [[0.5, nan], [0.7, 0.8]],
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
            [[0.5, nan], [nan, 0.6]],
            [[0.1, nan], [nan, 0.4]],
            [[0.5, nan], [nan, 0.8]],
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
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [0.3, nan]],
            [[nan, nan], [0.7, nan]],
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
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [0.3, nan]],
            [[nan, nan], [0.7, nan]],
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
            [[[nan, 0.5]], [[nan, 0.9]]],
            [[[nan, 0.2]], [[nan, 0.6]]],
            [[[nan, 0.3]], [[nan, 0.7]]],
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
            [[nan, 0.4], [nan, nan]],
            [[nan, 0.2], [nan, nan]],
            [[nan, 0.6], [nan, nan]],
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
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [0.3, nan]],
            [[nan, nan], [0.7, nan]],
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
            [[nan, nan], [nan, nan]],
            [[nan, nan], [nan, nan]],
            [[nan, nan], [nan, nan]],
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
            [[nan, nan], [nan, 0.6]],
            [[nan, nan], [nan, 0.4]],
            [[nan, nan], [nan, 0.8]],
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
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [0.3, nan]],
            [[nan, nan], [0.7, nan]],
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
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [0.3, nan]],
            [[nan, nan], [0.7, nan]],
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
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [real[1][0], nan]],
            [[nan, nan], [imag[1][0], nan]],
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
            [[nan, nan], [0.8, nan]],
            [[nan, nan], [real[1][0], nan]],
            [[nan, nan], [imag[1][0], nan]],
        ),
    )


def test_phasor_threshold_harmonic():
    """Test phasor_threshold function with multiple harmonics."""
    nan = numpy.nan
    data = numpy.random.random((3, 2, 8))
    data[0, 0, 0] = nan
    data[1, 0, 1] = nan
    data[1, 1, 2] = nan
    data[2, 0, 3] = nan
    data[2, 1, 4] = nan
    mean, real, imag = data
    mean_copy, real_copy, imag_copy = data.copy()

    # NaNs should propagate to all dimensions
    result = data.copy()
    result[:, :, :5] = nan
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
    result[0, 0, 1] = nan
    result[0, 1, 2] = nan
    result[0, 0, 3] = nan
    result[0, 1, 4] = nan
    result[1, 0, 0] = nan
    result[1, 0, 3] = nan
    result[1, 1, 0] = nan
    result[1, 1, 4] = nan
    result[2, 0, 0] = nan
    result[2, 0, 1] = nan
    result[2, 1, 0] = nan
    result[2, 1, 2] = nan
    mean_, real_, imag_ = result

    mean1, real1, imag1 = phasor_threshold(
        mean[0], real, imag, detect_harmonics=False
    )
    assert_allclose(mean1, mean_)
    assert_allclose(real1, real_)
    assert_allclose(imag1, imag_)


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, unreachable"
