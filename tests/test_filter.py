"""Test the phasorpy.filter module."""

import os
from math import nan

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phasorpy._typing import Any, ArrayLike, DTypeLike, Sequence
from phasorpy.datasets import fetch
from phasorpy.filter import (
    phasor_filter_gaussian,
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_threshold,
    signal_filter_gaussian,
    signal_filter_median,
    signal_filter_ncpca,
    signal_filter_svd,
)
from phasorpy.io import signal_from_imspector_tiff, signal_from_lsm
from phasorpy.phasor import (
    phasor_from_polar,
    phasor_from_signal,
    phasor_normalize,
)

SKIP_FETCH = bool(os.environ.get('SKIP_FETCH', ''))

rng = numpy.random.default_rng(42)


@pytest.mark.parametrize(
    ('sigma', 'repeat', 'skip_axis'),
    [
        (1.0, 1, None),
        (2.0, 1, None),
        (1.0, 2, None),
        (1.0, 0, None),
        (1.0, 1, 0),  # 3-D with skip_axis
    ],
)
def test_phasor_filter_gaussian(
    sigma: float, repeat: int, skip_axis: int | None
) -> None:
    """Test phasor_filter_gaussian function."""
    if skip_axis is None:
        real = rng.uniform(0.1, 0.9, (5, 5))
        imag = rng.uniform(0.1, 0.9, (5, 5))
        mean = numpy.ones((5, 5))
    else:
        real = rng.uniform(0.1, 0.9, (3, 5, 5))
        imag = rng.uniform(0.1, 0.9, (3, 5, 5))
        mean = numpy.ones((3, 5, 5))

    mean_out, real_out, imag_out = phasor_filter_gaussian(
        mean,
        real,
        imag,
        sigma=sigma,
        repeat=repeat,
        skip_axis=skip_axis,
    )
    # mean is unchanged
    assert_array_equal(mean_out, mean)
    assert real_out.shape == real.shape
    assert imag_out.shape == imag.shape
    # no NaN introduced on finite input
    assert not numpy.any(numpy.isnan(real_out))
    assert not numpy.any(numpy.isnan(imag_out))


def test_phasor_filter_gaussian_nan() -> None:
    """Test that NaN values in phasor_filter_gaussian are preserved."""
    real = numpy.array(
        [[0.2, 0.3, 0.2], [0.4, numpy.nan, 0.4], [0.2, 0.3, 0.2]]
    )
    imag = numpy.array(
        [[0.5, 0.6, 0.5], [0.7, 0.8, 0.7], [numpy.nan, 0.6, 0.5]]
    )
    mean = numpy.ones((3, 3))

    _, real_out, imag_out = phasor_filter_gaussian(mean, real, imag, sigma=1.0)
    # NaN positions are preserved
    assert numpy.isnan(real_out[1, 1])
    assert numpy.isnan(imag_out[2, 0])
    # non-NaN positions are finite
    assert numpy.isfinite(real_out[0, 0])
    assert numpy.isfinite(imag_out[0, 0])


def test_phasor_filter_gaussian_mean_nan() -> None:
    """Test NaN handling in phasor_filter_gaussian."""
    # NaN in mean: propagates to real/imag outputs (nan_safe=True default)
    mean = numpy.array([[1.0, numpy.nan], [3.0, 4.0]])
    real = numpy.zeros((2, 2))
    imag = numpy.zeros((2, 2))
    mean_out, real_out, imag_out = phasor_filter_gaussian(
        mean, real, imag, sigma=1.0
    )
    assert numpy.isnan(mean_out[0, 1])
    assert numpy.isnan(real_out[0, 1])
    assert numpy.isnan(imag_out[0, 1])
    assert numpy.isfinite(mean_out[0, 0])

    # NaN in real propagates to mean_out when nan_safe=True,
    # but not when nan_safe=False
    mean2 = numpy.array([[1.0, 1.0], [1.0, 1.0]])
    real2 = numpy.array([[0.0, numpy.nan], [0.3, 0.4]])
    imag2 = numpy.zeros((2, 2))
    mean_out2, _, _ = phasor_filter_gaussian(
        mean2, real2, imag2, sigma=1.0, nan_safe=True
    )
    assert numpy.isnan(mean_out2[0, 1])  # NaN propagated from real

    mean_out3, _, _ = phasor_filter_gaussian(
        mean2, real2, imag2, sigma=1.0, nan_safe=False
    )
    assert numpy.isfinite(mean_out3[0, 1])  # NaN not propagated to mean


def test_phasor_filter_gaussian_scipy_equivalence() -> None:
    """phasor_filter_gaussian matches scipy.ndimage.convolve1d with NaNmask."""
    from scipy.ndimage import convolve1d

    real = rng.uniform(0.1, 0.9, (7, 7))
    imag = rng.uniform(0.1, 0.9, (7, 7))
    mean = numpy.ones((7, 7))
    from scipy.signal.windows import gaussian as scipy_gaussian

    size = 3
    sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8  # OpenCV formula, default
    kernel = scipy_gaussian(size, std=sigma)

    _, real_out, imag_out = phasor_filter_gaussian(mean, real, imag)

    for arr, out in [(real, real_out), (imag, imag_out)]:
        filled = arr.copy()
        weights = numpy.ones_like(arr)
        for ax in range(arr.ndim):
            filled = convolve1d(filled, kernel, axis=ax, mode='nearest')
            weights = convolve1d(weights, kernel, axis=ax, mode='nearest')
        expected = filled / weights
        assert_allclose(out, expected, atol=1e-12)


def test_phasor_filter_gaussian_intensity_weighted() -> None:
    """Test that phasor_filter_gaussian applies intensity weighting."""
    from scipy.ndimage import convolve1d
    from scipy.signal.windows import gaussian as scipy_gaussian

    real = rng.uniform(0.1, 0.9, (7, 7))
    imag = rng.uniform(0.1, 0.9, (7, 7))
    mean = rng.uniform(0.5, 5.0, (7, 7))

    size = 3
    sigma = 1.0
    kernel = scipy_gaussian(size, std=sigma)

    _, real_out, imag_out = phasor_filter_gaussian(
        mean, real, imag, size=size, sigma=sigma
    )

    for arr, out in [(real, real_out), (imag, imag_out)]:
        filled_w = arr * mean
        filled_m = mean.copy()
        for ax in range(arr.ndim):
            filled_w = convolve1d(filled_w, kernel, axis=ax, mode='nearest')
            filled_m = convolve1d(filled_m, kernel, axis=ax, mode='nearest')
        expected = filled_w / filled_m
        assert_allclose(out, expected, atol=1e-12)


def test_phasor_filter_gaussian_float32() -> None:
    """Test phasor_filter_gaussian preserves float32 dtype."""
    real = rng.uniform(0.1, 0.9, (5, 5)).astype(numpy.float32)
    imag = rng.uniform(0.1, 0.9, (5, 5)).astype(numpy.float32)
    mean = numpy.ones((5, 5), dtype=numpy.float32)

    _, real_out, imag_out = phasor_filter_gaussian(mean, real, imag, sigma=1.0)
    assert real_out.dtype == numpy.float32
    assert imag_out.dtype == numpy.float32


def test_phasor_filter_gaussian_noop() -> None:
    """Test phasor_filter_gaussian no-op modes."""
    real = rng.uniform(0.1, 0.9, (5, 5))
    imag = rng.uniform(0.1, 0.9, (5, 5))
    mean = numpy.ones((5, 5))

    _, real_out, imag_out = phasor_filter_gaussian(
        mean, real, imag, sigma=1.0, repeat=0
    )
    assert_array_equal(real_out, real)
    assert_array_equal(imag_out, imag)


def test_phasor_filter_gaussian_harmonics() -> None:
    """Test phasor_filter_gaussian with a harmonics axis (prepend_axis)."""
    # mean has shape (H, W), real/imag have shape (N, H, W)
    # axis 0 of real/imag is the harmonics axis and must not be filtered
    mean = rng.uniform(0.5, 5.0, (5, 5))
    real = rng.uniform(0.1, 0.9, (2, 5, 5))
    imag = rng.uniform(0.1, 0.9, (2, 5, 5))

    mean_out, real_out, imag_out = phasor_filter_gaussian(
        mean, real, imag, sigma=1.0
    )
    assert mean_out.shape == mean.shape
    assert real_out.shape == real.shape
    assert imag_out.shape == imag.shape
    assert not numpy.any(numpy.isnan(real_out))
    assert not numpy.any(numpy.isnan(imag_out))
    # each harmonic must equal the single-harmonic result
    for h in range(2):
        _, real_h, imag_h = phasor_filter_gaussian(
            mean, real[h], imag[h], sigma=1.0
        )
        assert_allclose(real_out[h], real_h)
        assert_allclose(imag_out[h], imag_h)


def test_phasor_filter_gaussian_unweighted_equivalence() -> None:
    """Test weighted=False + phasor_normalize matches weighted=True workflow.

    Filtering unnormalized phasor coordinates (from phasor_from_signal with
    normalize=False) using an unweighted Gaussian and then normalizing with
    phasor_normalize is mathematically equivalent to filtering normalized
    phasor coordinates with intensity weighting (the default).
    """
    rng_local = numpy.random.default_rng(123)
    samples = 32
    signal = rng_local.uniform(0.5, 2.0, (5, 5, samples))

    # Workflow 1: normalized phasor + intensity-weighted filter (default)
    mean1, real1, imag1 = phasor_from_signal(signal, normalize=True)
    mean1, real1, imag1 = phasor_filter_gaussian(
        mean1, real1, imag1, sigma=1.0, nan_safe=False
    )

    # Workflow 2: unnormalized phasor + unweighted filter + normalize
    mean2, real2, imag2 = phasor_from_signal(signal, normalize=False)
    mean2, real2, imag2 = phasor_filter_gaussian(
        mean2, real2, imag2, sigma=1.0, weighted=False, nan_safe=False
    )
    mean2, real2, imag2 = phasor_normalize(
        mean2, real2, imag2, samples=samples
    )

    assert_allclose(mean1, mean2, atol=1e-10)
    assert_allclose(real1, real2, atol=1e-10)
    assert_allclose(imag1, imag2, atol=1e-10)


def test_phasor_filter_gaussian_signal_prefilter_equivalence() -> None:
    """Test pre- and post-gaussian filtering produce identical results."""
    # Because the Gaussian filter is linear and commutes with the phasor
    # transform, filtering the signal before computing phasor coordinates is
    # mathematically equivalent to computing phasor coordinates first and then
    # applying the intensity-weighted Gaussian filter (weighted=True,
    # the default).
    rng_local = numpy.random.default_rng(456)
    signal = rng_local.uniform(0.5, 2.0, (5, 5, 32))

    # filter signal first, then compute phasor coordinates
    signal_filtered = signal_filter_gaussian(signal, skip_axis=-1, sigma=1.0)
    mean1, real1, imag1 = phasor_from_signal(signal_filtered)

    # compute phasor coordinates first, then intensity-weighted filter
    mean2, real2, imag2 = phasor_from_signal(signal)
    mean2, real2, imag2 = phasor_filter_gaussian(
        mean2, real2, imag2, sigma=1.0, nan_safe=False
    )

    assert_allclose(mean1, mean2, atol=1e-6)
    assert_allclose(real1, real2, atol=1e-6)
    assert_allclose(imag1, imag2, atol=1e-6)


def test_phasor_filter_gaussian_errors() -> None:
    """Test phasor_filter_gaussian function errors."""
    # shape mismatch mean vs real
    with pytest.raises(ValueError):
        phasor_filter_gaussian([[0]], [0], [0])
    # shape mismatch real vs imag
    with pytest.raises(ValueError):
        phasor_filter_gaussian([0], [0], [[0]])
    # repeat < 0
    with pytest.raises(ValueError):
        phasor_filter_gaussian([[0]], [[0]], [[0]], repeat=-1)
    # sigma <= 0
    with pytest.raises(ValueError):
        phasor_filter_gaussian([[0]], [[0]], [[0]], sigma=0.0)
    # size < 1
    with pytest.raises(ValueError):
        phasor_filter_gaussian([[0]], [[0]], [[0]], size=0)
    # size is even
    with pytest.raises(ValueError):
        phasor_filter_gaussian([[0]], [[0]], [[0]], size=2)


@pytest.mark.parametrize(
    (
        'real',
        'imag',
        'use_scipy',
        'repeat',
        'size',
        'skip_axis',
        'kwargs',
        'expected',
    ),
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
            numpy.arange(25).reshape((5, 5)),
            numpy.arange(25, 50).reshape((5, 5)),
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
            numpy.arange(25).reshape((5, 5)),
            numpy.arange(25, 50).reshape((5, 5)),
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
            numpy.arange(25).reshape((5, 5)),
            numpy.arange(25, 50).reshape((5, 5)),
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
            numpy.arange(25, dtype=numpy.float32).reshape((5, 5)),
            numpy.arange(25, 50, dtype=numpy.float32).reshape((5, 5)),
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
        # 3x3x3 array with 3x3 filter repeated 3x with first axis skipped
        (
            numpy.arange(27).reshape((3, 3, 3)),
            numpy.arange(10, 37).reshape((3, 3, 3)),
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
            numpy.arange(27).reshape((3, 3, 3)),
            numpy.arange(10, 37).reshape((3, 3, 3)),
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
            numpy.arange(25).reshape((5, 5)),
            numpy.arange(25, 50).reshape((5, 5)),
            False,
            1,
            3,
            None,
            {},
            phasor_filter_median(
                numpy.ones((5, 5)),
                numpy.arange(25).reshape((5, 5)),
                numpy.arange(25, 50).reshape((5, 5)),
                use_scipy=True,
            )[1:],
        ),
        # same output for methods from 3D array without NaN
        (
            numpy.arange(27).reshape((3, 3, 3)),
            numpy.arange(10, 37).reshape((3, 3, 3)),
            False,
            1,
            3,
            None,
            {},
            phasor_filter_median(
                numpy.ones((3, 3, 3)),
                numpy.arange(27).reshape((3, 3, 3)),
                numpy.arange(10, 37).reshape((3, 3, 3)),
                use_scipy=True,
            )[1:],
        ),
        # same output for methods from 3D array without NaN and skip axes
        (
            numpy.arange(27).reshape((3, 3, 3)),
            numpy.arange(10, 37).reshape((3, 3, 3)),
            False,
            1,
            3,
            0,
            {},
            phasor_filter_median(
                numpy.ones((3, 3, 3)),
                numpy.arange(27).reshape((3, 3, 3)),
                numpy.arange(10, 37).reshape((3, 3, 3)),
                use_scipy=True,
                skip_axis=0,
            )[1:],
        ),
        # non-contiguous axes for 2D filtering
        (
            numpy.arange(81).reshape((3, 3, 3, 3)),
            numpy.arange(10, 91).reshape((3, 3, 3, 3)),
            False,
            1,
            3,
            [0, 2],
            {},
            phasor_filter_median(
                numpy.ones((3, 3, 3, 3)),
                numpy.arange(81).reshape((3, 3, 3, 3)),
                numpy.arange(10, 91).reshape((3, 3, 3, 3)),
                use_scipy=True,
                skip_axis=[0, 2],
            )[1:],
        ),
        # repeat = 0
        (
            numpy.arange(9).reshape((3, 3)),
            numpy.arange(10, 19).reshape((3, 3)),
            False,
            0,
            3,
            None,
            {},
            (
                numpy.arange(9).reshape((3, 3)),
                numpy.arange(10, 19).reshape((3, 3)),
            ),
        ),
        # size = 1
        (
            numpy.arange(9).reshape((3, 3)),
            numpy.arange(10, 19).reshape((3, 3)),
            False,
            1,
            1,
            None,
            {},
            (
                numpy.arange(9).reshape((3, 3)),
                numpy.arange(10, 19).reshape((3, 3)),
            ),
        ),
    ],
)
def test_phasor_filter_median(
    *,
    real: ArrayLike,
    imag: ArrayLike,
    use_scipy: bool,
    repeat: int,
    size: int,
    skip_axis: bool | None,
    kwargs: Any,
    expected: tuple[ArrayLike, ArrayLike],
) -> None:
    """Test phasor_filter_median function."""
    mean = numpy.full_like(real, 1)
    mean_out, real_out, imag_out = phasor_filter_median(
        mean,
        real,
        imag,
        use_scipy=use_scipy,
        repeat=repeat,
        size=size,
        skip_axis=skip_axis,
        **kwargs,
    )
    assert_array_equal(mean_out, mean)
    assert_allclose((real_out, imag_out), numpy.asarray(expected))


def test_phasor_filter_median_mean_nan() -> None:
    """Test that NaN in mean is returned unchanged by phasor_filter_median."""
    mean = numpy.array([[1.0, numpy.nan], [3.0, 4.0]])
    real = numpy.zeros((2, 2))
    imag = numpy.zeros((2, 2))
    mean_out, _, _ = phasor_filter_median(mean, real, imag)
    assert_array_equal(mean_out, mean)


def test_phasor_filter_median_harmonics() -> None:
    """Test phasor_filter_median with a harmonics axis (prepend_axis)."""
    # mean has shape (H, W), real/imag have shape (N, H, W)
    # axis 0 of real/imag is the harmonics axis and must not be filtered
    mean = numpy.full((5, 5), 1.0)
    real = rng.uniform(0.1, 0.9, (2, 5, 5))
    imag = rng.uniform(0.1, 0.9, (2, 5, 5))

    mean_out, real_out, imag_out = phasor_filter_median(mean, real, imag)
    assert mean_out.shape == mean.shape
    assert_array_equal(mean_out, mean)
    assert real_out.shape == real.shape
    assert imag_out.shape == imag.shape
    assert not numpy.any(numpy.isnan(real_out))
    assert not numpy.any(numpy.isnan(imag_out))
    # each harmonic must equal the single-harmonic result
    for h in range(2):
        _, real_h, imag_h = phasor_filter_median(mean, real[h], imag[h])
        assert_allclose(real_out[h], real_h)
        assert_allclose(imag_out[h], imag_h)


def test_phasor_filter_median_errors() -> None:
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


@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64])
def test_signal_filter_median(dtype: DTypeLike) -> None:
    """Test signal_filter_median function."""
    signal = numpy.array(
        [
            [[0.0, 10.0, 0.0], [1.0, 2.0, 3.0], [10.0, 5.0, 1.0]],
            [[5.0, 6.0, 2.0], [10.0, 4.0, 8.0], [0.0, 7.0, 8.0]],
        ],
        dtype=dtype,
    )

    filtered = signal_filter_median(
        signal, skip_axis=0, size=3, repeat=1, num_threads=1
    )
    expected = numpy.array(
        [
            [[1.0, 1.0, 2.0], [2.0, 2.0, 2.0], [5.0, 3.0, 2.0]],
            [[5.0, 5.0, 4.0], [5.0, 6.0, 7.0], [4.0, 7.0, 8.0]],
        ],
        dtype=dtype,
    )
    assert_allclose(filtered, expected)
    assert filtered.dtype == dtype

    # sequence of skip_axis should produce same result as scalar
    filtered_seq = signal_filter_median(
        signal, skip_axis=[0], size=3, repeat=1, num_threads=1
    )
    assert_allclose(filtered_seq, expected)


def test_signal_filter_median_scipy_equivalence() -> None:
    """Test signal_filter_median against scipy with finite input."""
    from scipy.ndimage import median_filter

    signal = numpy.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(numpy.float64)

    expected = median_filter(signal, size=3, axes=(0, 2), mode='nearest')
    filtered = signal_filter_median(signal, skip_axis=1, size=3)
    assert_allclose(filtered, expected)


def test_signal_filter_median_nan() -> None:
    """Test signal_filter_median function NaN handling."""
    signal = numpy.array(
        [
            [[1.0, nan, 5.0], [2.0, 3.0, 4.0], [9.0, 8.0, 7.0]],
            [[5.0, 6.0, 7.0], [8.0, 9.0, 1.0], [2.0, 3.0, 4.0]],
        ]
    )

    filtered = signal_filter_median(signal, skip_axis=0, size=3)
    assert numpy.isnan(filtered[0, 0, 1])
    assert not numpy.isnan(filtered[0, 1, 1])


def test_signal_filter_median_noop() -> None:
    """Test signal_filter_median no-op modes."""
    signal = numpy.arange(8).reshape((2, 4))

    assert_array_equal(signal_filter_median(signal, repeat=0), signal)
    assert_array_equal(signal_filter_median(signal, size=1), signal)


def test_signal_filter_median_skip_axis_none() -> None:
    """Test signal_filter_median with skip_axis=None filters all axes."""
    signal = numpy.array(
        [[1.0, 0.0, 1.0], [0.0, numpy.nan, 0.0], [1.0, 0.0, 1.0]]
    )

    filtered = signal_filter_median(signal, skip_axis=None, size=3)
    assert filtered.shape == signal.shape
    assert numpy.isnan(filtered[1, 1])


def test_signal_filter_median_scipy() -> None:
    """Test signal_filter_median with use_scipy=True."""
    from scipy.ndimage import median_filter

    signal = numpy.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(numpy.float64)

    expected = median_filter(signal, size=3, axes=(0, 2))
    filtered = signal_filter_median(
        signal, skip_axis=1, size=3, use_scipy=True
    )
    assert_allclose(filtered, expected)


def test_signal_filter_median_ndim() -> None:
    """Test signal_filter_median numpy nD filter when len(axes) != 2."""
    from scipy.ndimage import median_filter

    signal = numpy.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(numpy.float64)

    # 3 filtered axes (len(axes) == 3, numpy nD path)
    expected = median_filter(signal, size=3)
    filtered = signal_filter_median(signal, skip_axis=None, size=3)
    assert_allclose(filtered, expected)

    # 1 filtered axis (len(axes) == 1, numpy nD path)
    expected = median_filter(signal, size=3, axes=(2,))
    filtered = signal_filter_median(signal, skip_axis=[0, 1], size=3)
    assert_allclose(filtered, expected)


def test_signal_filter_median_errors() -> None:
    """Test signal_filter_median function errors."""
    with pytest.raises(ValueError):
        signal_filter_median([[0]], repeat=-1)

    with pytest.raises(ValueError):
        signal_filter_median([[0]], size=0)

    with pytest.raises(IndexError):
        signal_filter_median([[0]], skip_axis=2)


@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64])
def test_signal_filter_gaussian(dtype: DTypeLike) -> None:
    """Test signal_filter_gaussian function."""
    signal = numpy.array(
        [
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ],
        dtype=dtype,
    )
    filtered = signal_filter_gaussian(signal, skip_axis=0, size=3, repeat=1)
    assert filtered.shape == signal.shape
    assert filtered.dtype == dtype
    # constant slices must remain constant
    assert_allclose(filtered[1], signal[1], atol=1e-6)


def test_signal_filter_gaussian_nan() -> None:
    """Test signal_filter_gaussian NaN handling."""
    signal = numpy.array(
        [
            [[1.0, nan, 5.0], [2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0], [8.0, 9.0, 1.0]],
        ]
    )
    filtered = signal_filter_gaussian(signal, skip_axis=0, size=3)
    assert numpy.isnan(filtered[0, 0, 1])
    assert not numpy.isnan(filtered[0, 1, 1])


def test_signal_filter_gaussian_equivalence() -> None:
    """Test signal_filter_gaussian matches normalized convolve1d (no NaNs)."""
    from scipy.ndimage import convolve1d
    from scipy.signal.windows import gaussian as scipy_gaussian

    signal = numpy.arange(3 * 4 * 5, dtype=numpy.float64).reshape((3, 4, 5))
    size = 3
    sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    kernel = scipy_gaussian(size, std=sigma)
    kernel_norm = kernel / kernel.sum()  # _gaussian_filter divides by weights

    filtered = signal_filter_gaussian(signal, skip_axis=1, size=size)

    # apply normalized kernel along axes (0, 2)
    expected = signal.copy()
    for ax in (0, 2):
        expected = convolve1d(expected, kernel_norm, axis=ax, mode='nearest')
    assert_allclose(filtered, expected, atol=1e-12)


def test_signal_filter_gaussian_noop() -> None:
    """Test signal_filter_gaussian no-op modes."""
    signal = numpy.arange(8, dtype=numpy.float64).reshape((2, 4))
    assert_array_equal(signal_filter_gaussian(signal, repeat=0), signal)
    assert_array_equal(signal_filter_gaussian(signal, size=1), signal)


def test_signal_filter_gaussian_errors() -> None:
    """Test signal_filter_gaussian error handling."""
    with pytest.raises(ValueError):
        signal_filter_gaussian([[0]], repeat=-1)
    with pytest.raises(ValueError):
        signal_filter_gaussian([[0]], size=0)
    with pytest.raises(ValueError):
        signal_filter_gaussian([[0]], size=2)
    with pytest.raises(ValueError):
        signal_filter_gaussian([[0]], sigma=0.0)
    with pytest.raises(IndexError):
        signal_filter_gaussian([[0]], skip_axis=2)


@pytest.mark.parametrize(
    (
        'mean',
        'real',
        'imag',
        'sigma',
        'levels',
        'harmonic',
        'skip_axis',
        'expected',
    ),
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
        # TODO: 2D arrays with 2 and 4 harmonics specified
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
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    sigma: float,
    levels: int,
    harmonic: Sequence[int] | None,
    skip_axis: int,
    expected: list[ArrayLike],
) -> None:
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
    assert_allclose(mean, numpy.asarray(expected[0]))
    assert_allclose(real, numpy.asarray(expected[1]))
    assert_allclose(imag, numpy.asarray(expected[2]))


def test_phasor_filter_pawflim_mean_nan() -> None:
    """Test that NaN in mean is returned unchanged by phasor_filter_pawflim."""
    mean = numpy.array([[1.0, numpy.nan], [3.0, 4.0]])
    real = numpy.ones((2, 2, 2)) * 0.5
    imag = numpy.ones((2, 2, 2)) * 0.4
    mean_out, _, _ = phasor_filter_pawflim(mean, real, imag, harmonic=[1, 2])
    assert_array_equal(mean_out, mean)


def test_phasor_filter_pawflim_errors() -> None:
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
    # sigma < 0
    with pytest.raises(ValueError):
        phasor_filter_pawflim([1], [[1], [1]], [[1], [1]], sigma=-1)
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


def test_phasor_threshold() -> None:
    """Test phasor_threshold function."""
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


@pytest.mark.parametrize('dtype', [numpy.float32, numpy.float64])
@pytest.mark.parametrize(
    'kwargs',
    [
        {},
        {'mean_min': 0.5},
        {'real_min': 0.5, 'phase_max': 0.7},
    ],
)
def test_phasor_threshold_dtype(dtype: DTypeLike, kwargs: Any) -> None:
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


def test_phasor_threshold_harmonic() -> None:
    """Test phasor_threshold function with multiple harmonics."""
    data = rng.random((3, 2, 8))
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


def test_phasor_threshold_returns_copies() -> None:
    """Test phasor_threshold always returns new arrays, never views."""
    mean = numpy.ones((3, 3))
    real = numpy.full((3, 3), 0.5)
    imag = numpy.full((3, 3), 0.3)

    kwargs: Any
    for kwargs in (
        {},  # NaN-propagation only
        {'mean_min': 0.5},  # mean threshold
        {'real_min': 0.1, 'imag_max': 0.9},  # coordinate thresholds
    ):
        m2, r2, i2 = phasor_threshold(mean, real, imag, **kwargs)
        assert not numpy.shares_memory(mean, m2)
        assert not numpy.shares_memory(real, r2)
        assert not numpy.shares_memory(imag, i2)


@pytest.mark.parametrize('dtype', [None, 'float32'])
@pytest.mark.parametrize('spectral_vector', [None, True])
def test_signal_filter_svd(*, dtype: DTypeLike, spectral_vector: Any) -> None:
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

    mean1, _real1, _imag1 = phasor_from_signal(denoised, axis=0)
    assert_allclose(mean, mean1, atol=1e-3)
    assert_allclose(signal, denoised, atol=22)
    assert denoised.dtype == dtype


@pytest.mark.skipif(SKIP_FETCH, reason='fetch is disabled')
def test_signal_filter_svd_nan() -> None:
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

    mean1, _real1, _imag1 = phasor_from_signal(denoised, axis=0)
    assert_allclose(mean, mean1, atol=1e-3)


def test_signal_filter_svd_exceptions() -> None:
    """Test signal_filter_svd function exceptions."""
    signal = rng.integers(0, 255, (16, 8, 16)).astype(numpy.float32)
    spectral_vector = rng.random((16, 16, 2))

    signal_filter_svd(signal, spectral_vector, axis=1)

    with pytest.raises(ValueError):
        signal_filter_svd(signal, spectral_vector, axis=1, dtype=numpy.uint8)

    with pytest.raises(ValueError):
        signal_filter_svd(signal, spectral_vector[:15], axis=1)

    # sigma <= 0
    with pytest.raises(ValueError):
        signal_filter_svd(signal, axis=1, sigma=0)

    with pytest.raises(ValueError):
        signal_filter_svd(signal, axis=1, sigma=-0.1)

    # samples < 3
    signal_small = rng.integers(0, 255, (16, 2, 16)).astype(numpy.float32)
    with pytest.raises(ValueError):
        signal_filter_svd(signal_small, axis=1)


@pytest.mark.parametrize('dtype', [None, 'float32'])
@pytest.mark.parametrize('n_components', [None, 8, 'mle'])
def test_signal_filter_ncpca(*, dtype: DTypeLike, n_components: Any) -> None:
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

    mean, _real, _imag = phasor_from_signal(signal, axis=0)
    mean1, _real1, _imag1 = phasor_from_signal(denoised, axis=0)
    assert_allclose(mean1, mean, atol=1e-3 if n_components is None else 0.1)


def test_signal_filter_ncpca_exceptions() -> None:
    """Test signal_filter_ncpca function exceptions."""
    signal = rng.integers(0, 255, (16, 8, 2)).astype(numpy.float32)

    signal_filter_ncpca(signal, axis=1)

    with pytest.raises(ValueError):
        signal_filter_ncpca(signal, axis=1, dtype=numpy.uint8)

    with pytest.raises(ValueError):
        signal_filter_ncpca(signal, axis=2)

    with pytest.raises(ValueError):
        signal_filter_ncpca(signal, n_components=9, axis=1)

    with pytest.raises(ValueError):
        signal_filter_ncpca([], axis=0)
