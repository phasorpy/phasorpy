"""Filter signals and phasor coordinates.

The ``phasorpy.filter`` module provides functions to filter

- phasor coordinates:

  - :py:func:`phasor_filter_gaussian` (not implemented yet)
  - :py:func:`phasor_filter_median`
  - :py:func:`phasor_filter_pawflim`
  - :py:func:`phasor_threshold`

- signals:

  - :py:func:`signal_filter_ncpca`
    (noise-corrected principal component analysis)
  - :py:func:`signal_filter_svd` (spectral vector denoise)
  - :py:func:`signal_filter_median` (not implemented yet)

"""

from __future__ import annotations

__all__ = [
    # 'signal_filter_gaussian',
    'phasor_filter_median',
    'phasor_filter_pawflim',
    'phasor_threshold',
    # 'signal_filter_median',
    'signal_filter_ncpca',
    'signal_filter_svd',
]

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, NDArray, ArrayLike, DTypeLike, Literal

import numpy

from ._phasorpy import (
    _median_filter_2d,
    _phasor_from_signal_vector,
    _phasor_threshold_closed,
    _phasor_threshold_mean_closed,
    _phasor_threshold_mean_open,
    _phasor_threshold_nan,
    _phasor_threshold_open,
    _signal_denoise_vector,
)
from ._utils import parse_harmonic, parse_skip_axis
from .utils import number_threads


def phasor_filter_median(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    repeat: int = 1,
    size: int = 3,
    skip_axis: int | Sequence[int] | None = None,
    use_scipy: bool = False,
    num_threads: int | None = None,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return median-filtered phasor coordinates.

    By default, apply a NaN-aware median filter independently to the real
    and imaginary components of phasor coordinates once with a kernel size of 3
    multiplied by the number of dimensions of the input arrays. Return the
    intensity unchanged.

    Parameters
    ----------
    mean : array_like
        Intensity of phasor coordinates.
    real : array_like
        Real component of phasor coordinates to be filtered.
    imag : array_like
        Imaginary component of phasor coordinates to be filtered.
    repeat : int, optional
        Number of times to apply median filter. The default is 1.
    size : int, optional
        Size of median filter kernel. The default is 3.
    skip_axis : int or sequence of int, optional
        Axes in `mean` to exclude from filter.
        By default, all axes except harmonics are included.
    use_scipy : bool, optional
        Use :py:func:`scipy.ndimage.median_filter`.
        This function has undefined behavior if the input arrays contain
        NaN values but is faster when filtering more than 2 dimensions.
        See `issue #87 <https://github.com/phasorpy/phasorpy/issues/87>`_.
    num_threads : int, optional
        Number of OpenMP threads to use for parallelization.
        Applies to filtering in two dimensions when not using scipy.
        By default, multi-threading is disabled.
        If zero, up to half of logical CPUs are used.
        OpenMP may not be available on all platforms.
    **kwargs
        Optional arguments passed to :py:func:`scipy.ndimage.median_filter`.

    Returns
    -------
    mean : ndarray
        Unchanged intensity of phasor coordinates.
    real : ndarray
        Filtered real component of phasor coordinates.
    imag : ndarray
        Filtered imaginary component of phasor coordinates.

    Raises
    ------
    ValueError
        If `repeat` is less than 0.
        If `size` is less than 1.
        The array shapes of `mean`, `real`, and `imag` do not match.

    Examples
    --------
    Apply three times a median filter with a kernel size of three:

    >>> mean, real, imag = phasor_filter_median(
    ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ...     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]],
    ...     [[0.3, 0.3, 0.3], [0.6, math.nan, 0.6], [0.4, 0.4, 0.4]],
    ...     size=3,
    ...     repeat=3,
    ... )
    >>> mean
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> real
    array([[0, 0, 0],
           [0.2, 0.2, 0.2],
           [0.2, 0.2, 0.2]])
    >>> imag
    array([[0.3, 0.3, 0.3],
           [0.4, nan, 0.4],
           [0.4, 0.4, 0.4]])

    """
    if repeat < 0:
        raise ValueError(f'{repeat=} < 0')
    if size < 1:
        raise ValueError(f'{size=} < 1')
    if size == 1:
        # no need to filter
        repeat = 0

    mean = numpy.asarray(mean)
    if use_scipy or repeat == 0:  # or using nD numpy filter
        real = numpy.asarray(real)
    elif isinstance(real, numpy.ndarray) and real.dtype == numpy.float32:
        real = real.copy()
    else:
        real = numpy.asarray(real, dtype=numpy.float64, copy=True)
    if use_scipy or repeat == 0:  # or using nD numpy filter
        imag = numpy.asarray(imag)
    elif isinstance(imag, numpy.ndarray) and imag.dtype == numpy.float32:
        imag = imag.copy()
    else:
        imag = numpy.asarray(imag, dtype=numpy.float64, copy=True)

    if mean.shape != real.shape[-mean.ndim if mean.ndim else 1 :]:
        raise ValueError(f'{mean.shape=} != {real.shape=}')
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')

    prepend_axis = mean.ndim + 1 == real.ndim
    _, axes = parse_skip_axis(skip_axis, mean.ndim, prepend_axis)

    # in case mean is also filtered
    # if prepend_axis:
    #     mean = numpy.expand_dims(mean, axis=0)
    # ...
    # if prepend_axis:
    #     mean = numpy.asarray(mean[0])

    if repeat == 0:
        # no need to call filter
        return mean, real, imag

    if use_scipy:
        # use scipy NaN-unaware fallback
        from scipy.ndimage import median_filter

        kwargs.pop('axes', None)

        for _ in range(repeat):
            real = median_filter(real, size=size, axes=axes, **kwargs)
            imag = median_filter(imag, size=size, axes=axes, **kwargs)

        return mean, numpy.asarray(real), numpy.asarray(imag)

    if len(axes) != 2:
        # n-dimensional median filter using numpy
        from numpy.lib.stride_tricks import sliding_window_view

        kernel_shape = tuple(
            size if i in axes else 1 for i in range(real.ndim)
        )
        pad_width = [
            (s // 2, s // 2) if s > 1 else (0, 0) for s in kernel_shape
        ]
        axis = tuple(range(-real.ndim, 0))

        nan_mask = numpy.isnan(real)
        for _ in range(repeat):
            real = numpy.pad(real, pad_width, mode='edge')
            real = sliding_window_view(real, kernel_shape)
            real = numpy.nanmedian(real, axis=axis)
            real = numpy.where(nan_mask, numpy.nan, real)

        nan_mask = numpy.isnan(imag)
        for _ in range(repeat):
            imag = numpy.pad(imag, pad_width, mode='edge')
            imag = sliding_window_view(imag, kernel_shape)
            imag = numpy.nanmedian(imag, axis=axis)
            imag = numpy.where(nan_mask, numpy.nan, imag)

        return mean, real, imag

    # 2-dimensional median filter using optimized Cython implementation
    num_threads = number_threads(num_threads)

    buffer = numpy.empty(
        tuple(real.shape[axis] for axis in axes), dtype=real.dtype
    )

    for index in numpy.ndindex(
        *[real.shape[ax] for ax in range(real.ndim) if ax not in axes]
    ):
        index_list: list[int | slice] = list(index)
        for ax in axes:
            index_list = index_list[:ax] + [slice(None)] + index_list[ax:]
        full_index = tuple(index_list)

        _median_filter_2d(real[full_index], buffer, size, repeat, num_threads)
        _median_filter_2d(imag[full_index], buffer, size, repeat, num_threads)

    return mean, real, imag


def phasor_filter_pawflim(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    sigma: float = 2.0,
    levels: int = 1,
    harmonic: Sequence[int] | None = None,
    skip_axis: int | Sequence[int] | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return pawFLIM wavelet-filtered phasor coordinates.

    This function must only be used with calibrated, unprocessed phasor
    coordinates obtained from FLIM data. The coordinates must not be filtered,
    thresholded, or otherwise pre-processed.

    The pawFLIM wavelet filter is described in [1]_.

    Parameters
    ----------
    mean : array_like
        Intensity of phasor coordinates.
    real : array_like
        Real component of phasor coordinates to be filtered.
        Must have at least two harmonics in the first axis.
    imag : array_like
        Imaginary component of phasor coordinates to be filtered.
        Must have at least two harmonics in the first axis.
    sigma : float, optional
        Significance level to test difference between two phasors.
        Given in terms of the equivalent 1D standard deviations.
        sigma=2 corresponds to ~95% (or 5%) significance.
    levels : int, optional
        Number of levels for wavelet decomposition.
        Controls the maximum averaging area, which has a length of
        :math:`2^level`.
    harmonic : sequence of int or None, optional
        Harmonics included in first axis of `real` and `imag`.
        If None (default), the first axis of `real` and `imag` contains lower
        harmonics starting at and increasing by one.
        All harmonics must have a corresponding half or double harmonic.
    skip_axis : int or sequence of int, optional
        Axes in `mean` to exclude from filter.
        By default, all axes except harmonics are included.

    Returns
    -------
    mean : ndarray
        Unchanged intensity of phasor coordinates.
    real : ndarray
        Filtered real component of phasor coordinates.
    imag : ndarray
        Filtered imaginary component of phasor coordinates.

    Raises
    ------
    ValueError
        If `level` is less than 0.
        The array shapes of `mean`, `real`, and `imag` do not match.
        If `real` and `imag` have no harmonic axis.
        Number of harmonics in `harmonic` is less than 2 or does not match
        the first axis of `real` and `imag`.
        Not all harmonics in `harmonic` have a corresponding half
        or double harmonic.

    References
    ----------
    .. [1] Silberberg M, and Grecco H. `pawFLIM: reducing bias and
      uncertainty to enable lower photon count in FLIM experiments
      <https://doi.org/10.1088/2050-6120/aa72ab>`_.
      *Methods Appl Fluoresc*, 5(2): 024016 (2017)

    Examples
    --------
    Apply a pawFLIM wavelet filter with four significance levels (sigma)
    and three decomposition levels:

    >>> mean, real, imag = phasor_filter_pawflim(
    ...     [[1, 1], [1, 1]],
    ...     [[[0.5, 0.8], [0.5, 0.8]], [[0.2, 0.4], [0.2, 0.4]]],
    ...     [[[0.5, 0.4], [0.5, 0.4]], [[0.4, 0.5], [0.4, 0.5]]],
    ...     sigma=4,
    ...     levels=3,
    ...     harmonic=[1, 2],
    ... )
    >>> mean
    array([[1, 1],
           [1, 1]])
    >>> real
    array([[[0.65, 0.65],
            [0.65, 0.65]],
           [[0.3, 0.3],
            [0.3, 0.3]]])
    >>> imag
    array([[[0.45, 0.45],
            [0.45, 0.45]],
           [[0.45, 0.45],
            [0.45, 0.45]]])

    """
    from pawflim import pawflim  # type: ignore[import-untyped]

    if levels < 0:
        raise ValueError(f'{levels=} < 0')
    if levels == 0:
        return numpy.asarray(mean), numpy.asarray(real), numpy.asarray(imag)

    mean = numpy.asarray(mean, dtype=numpy.float64, copy=True)
    real = numpy.asarray(real, dtype=numpy.float64, copy=True)
    imag = numpy.asarray(imag, dtype=numpy.float64, copy=True)

    if mean.shape != real.shape[-mean.ndim if mean.ndim else 1 :]:
        raise ValueError(f'{mean.shape=} != {real.shape=}')
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')

    has_harmonic_axis = mean.ndim + 1 == real.ndim
    if not has_harmonic_axis:
        raise ValueError('no harmonic axis')
    if harmonic is None:
        harmonics, _ = parse_harmonic('all', real.shape[0])
    else:
        harmonics, _ = parse_harmonic(harmonic, None)
    if len(harmonics) < 2:
        raise ValueError(
            'at least two harmonics required, ' f'got {len(harmonics)}'
        )
    if len(harmonics) != real.shape[0]:
        raise ValueError(
            'number of harmonics does not match first axis of real and imag'
        )

    mean = numpy.asarray(numpy.nan_to_num(mean, copy=False))
    real = numpy.asarray(numpy.nan_to_num(real, copy=False))
    imag = numpy.asarray(numpy.nan_to_num(imag, copy=False))
    real *= mean
    imag *= mean

    mean_expanded = numpy.broadcast_to(mean, real.shape).copy()
    original_mean_expanded = mean_expanded.copy()
    real_filtered = real.copy()
    imag_filtered = imag.copy()

    _, axes = parse_skip_axis(skip_axis, mean.ndim, True)

    for index in numpy.ndindex(
        *(
            real.shape[ax]
            for ax in range(real.ndim)
            if ax not in axes and ax != 0
        )
    ):
        index_list: list[int | slice] = list(index)
        for ax in axes:
            index_list = index_list[:ax] + [slice(None)] + index_list[ax:]
        full_index = tuple(index_list)

        processed_harmonics = set()

        for h in harmonics:
            if h in processed_harmonics and (
                h * 4 in harmonics or h * 2 not in harmonics
            ):
                continue
            if h * 2 not in harmonics:
                raise ValueError(
                    f'harmonic {h} does not have a corresponding half '
                    f'or double harmonic in {harmonics}'
                )
            n = harmonics.index(h)
            n2 = harmonics.index(h * 2)

            complex_phasor = numpy.empty(
                (3, *original_mean_expanded[n][full_index].shape),
                dtype=numpy.complex128,
            )
            complex_phasor[0] = original_mean_expanded[n][full_index]
            complex_phasor[1] = real[n][full_index] + 1j * imag[n][full_index]
            complex_phasor[2] = (
                real[n2][full_index] + 1j * imag[n2][full_index]
            )

            complex_phasor = pawflim(
                complex_phasor, n_sigmas=sigma, levels=levels
            )

            for i, idx in enumerate([n, n2]):
                if harmonics[idx] in processed_harmonics:
                    continue
                mean_expanded[idx][full_index] = complex_phasor[0].real
                real_filtered[idx][full_index] = complex_phasor[i + 1].real
                imag_filtered[idx][full_index] = complex_phasor[i + 1].imag

            processed_harmonics.add(h)
            processed_harmonics.add(h * 2)

    with numpy.errstate(divide='ignore', invalid='ignore'):
        real = numpy.asarray(numpy.divide(real_filtered, mean_expanded))
        imag = numpy.asarray(numpy.divide(imag_filtered, mean_expanded))

    return mean, real, imag


def phasor_threshold(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    mean_min: ArrayLike | None = None,
    mean_max: ArrayLike | None = None,
    *,
    real_min: ArrayLike | None = None,
    real_max: ArrayLike | None = None,
    imag_min: ArrayLike | None = None,
    imag_max: ArrayLike | None = None,
    phase_min: ArrayLike | None = None,
    phase_max: ArrayLike | None = None,
    modulation_min: ArrayLike | None = None,
    modulation_max: ArrayLike | None = None,
    open_interval: bool = False,
    detect_harmonics: bool = True,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return phasor coordinates with values outside interval replaced by NaN.

    Interval thresholds can be set for mean intensity, real and imaginary
    coordinates, and phase and modulation.
    Phasor coordinates smaller than minimum thresholds or larger than maximum
    thresholds are replaced with NaN.
    No threshold is applied by default.
    NaNs in `mean` or any `real` and `imag` harmonic are propagated to
    `mean` and all harmonics in `real` and `imag`.

    Parameters
    ----------
    mean : array_like
        Intensity of phasor coordinates.
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    mean_min : array_like, optional
        Lower threshold for mean intensity.
    mean_max : array_like, optional
        Upper threshold for mean intensity.
    real_min : array_like, optional
        Lower threshold for real coordinates.
    real_max : array_like, optional
        Upper threshold for real coordinates.
    imag_min : array_like, optional
        Lower threshold for imaginary coordinates.
    imag_max : array_like, optional
        Upper threshold for imaginary coordinates.
    phase_min : array_like, optional
        Lower threshold for phase angle.
    phase_max : array_like, optional
        Upper threshold for phase angle.
    modulation_min : array_like, optional
        Lower threshold for modulation.
    modulation_max : array_like, optional
        Upper threshold for modulation.
    open_interval : bool, optional
        If true, the interval is open, and the threshold values are
        not included in the interval.
        If false (default), the interval is closed, and the threshold values
        are included in the interval.
    detect_harmonics : bool, optional
        By default, detect presence of multiple harmonics from array shapes.
        If false, no harmonics are assumed to be present, and the function
        behaves like a numpy universal function.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    mean : ndarray
        Thresholded intensity of phasor coordinates.
    real : ndarray
        Thresholded real component of phasor coordinates.
    imag : ndarray
        Thresholded imaginary component of phasor coordinates.

    Examples
    --------
    Set phasor coordinates to NaN if mean intensity is smaller than 1.1:

    >>> phasor_threshold([1, 2, 3], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], 1.1)
    (array([nan, 2, 3]), array([nan, 0.2, 0.3]), array([nan, 0.5, 0.6]))

    Set phasor coordinates to NaN if real component is smaller than 0.15 or
    larger than 0.25:

    >>> phasor_threshold(
    ...     [1.0, 2.0, 3.0],
    ...     [0.1, 0.2, 0.3],
    ...     [0.4, 0.5, 0.6],
    ...     real_min=0.15,
    ...     real_max=0.25,
    ... )
    (array([nan, 2, nan]), array([nan, 0.2, nan]), array([nan, 0.5, nan]))

    Apply NaNs to other input arrays:

    >>> phasor_threshold(
    ...     [numpy.nan, 2, 3], [0.1, 0.2, 0.3], [0.4, 0.5, numpy.nan]
    ... )
    (array([nan, 2, nan]), array([nan, 0.2, nan]), array([nan, 0.5, nan]))

    """
    threshold_mean_only = None
    if mean_min is None:
        mean_min = numpy.nan
    else:
        threshold_mean_only = True
    if mean_max is None:
        mean_max = numpy.nan
    else:
        threshold_mean_only = True
    if real_min is None:
        real_min = numpy.nan
    else:
        threshold_mean_only = False
    if real_max is None:
        real_max = numpy.nan
    else:
        threshold_mean_only = False
    if imag_min is None:
        imag_min = numpy.nan
    else:
        threshold_mean_only = False
    if imag_max is None:
        imag_max = numpy.nan
    else:
        threshold_mean_only = False
    if phase_min is None:
        phase_min = numpy.nan
    else:
        threshold_mean_only = False
    if phase_max is None:
        phase_max = numpy.nan
    else:
        threshold_mean_only = False
    if modulation_min is None:
        modulation_min = numpy.nan
    else:
        threshold_mean_only = False
    if modulation_max is None:
        modulation_max = numpy.nan
    else:
        threshold_mean_only = False

    if detect_harmonics:
        mean = numpy.asarray(mean)
        real = numpy.asarray(real)
        imag = numpy.asarray(imag)

        shape = numpy.broadcast_shapes(mean.shape, real.shape, imag.shape)
        ndim = len(shape)

        has_harmonic_axis = (
            # detect multi-harmonic in axis 0
            mean.ndim + 1 == ndim
            and real.shape == shape
            and imag.shape == shape
            and mean.shape == shape[-mean.ndim if mean.ndim else 1 :]
        )
    else:
        has_harmonic_axis = False

    if threshold_mean_only is None:
        mean, real, imag = _phasor_threshold_nan(mean, real, imag, **kwargs)

    elif threshold_mean_only:
        mean_func = (
            _phasor_threshold_mean_open
            if open_interval
            else _phasor_threshold_mean_closed
        )
        mean, real, imag = mean_func(
            mean, real, imag, mean_min, mean_max, **kwargs
        )

    else:
        func = (
            _phasor_threshold_open
            if open_interval
            else _phasor_threshold_closed
        )
        mean, real, imag = func(
            mean,
            real,
            imag,
            mean_min,
            mean_max,
            real_min,
            real_max,
            imag_min,
            imag_max,
            phase_min,
            phase_max,
            modulation_min,
            modulation_max,
            **kwargs,
        )

    mean = numpy.asarray(mean)
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    if has_harmonic_axis and mean.ndim > 0:
        # propagate NaN to all dimensions
        mean = numpy.mean(mean, axis=0, keepdims=True)
        mask = numpy.where(numpy.isnan(mean), numpy.nan, 1.0)
        numpy.multiply(real, mask, out=real)
        numpy.multiply(imag, mask, out=imag)
        # remove harmonic dimension created by broadcasting
        mean = numpy.asarray(numpy.asarray(mean)[0])

    return mean, real, imag


def signal_filter_svd(
    signal: ArrayLike,
    /,
    spectral_vector: ArrayLike | None = None,
    *,
    axis: int = -1,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
    sigma: float = 0.05,
    vmin: float | None = None,
    dtype: DTypeLike | None = None,
    num_threads: int | None = None,
) -> NDArray[Any]:
    """Return spectral-vector-denoised signal.

    The spectral vector denoising algorithm is based on a Gaussian weighted
    average calculation, with weights obtained in n-dimensional Chebyshev or
    Fourier space [2]_.

    Parameters
    ----------
    signal : array_like
        Hyperspectral data to be denoised.
        A minimum of three samples are required along `axis`.
        The samples must be uniformly spaced.
    spectral_vector : array_like, optional
        Spectral vector.
        For example, phasor coordinates, PCA projected phasor coordinates,
        or Chebyshev coefficients.
        Must be of the same shape as `signal` with `axis` removed and an axis
        containing spectral space appended.
        If None (default), phasor coordinates are calculated at specified
        `harmonic`.
    axis : int, optional, default: -1
        Axis over which `spectral_vector` is computed if not provided.
        The default is the last axis (-1).
    harmonic : int, sequence of int, or 'all', optional
        Harmonics to include in calculating `spectral_vector`.
        If `'all'`, include all harmonics for `signal` samples along `axis`.
        Else, harmonics must be at least one and no larger than half the
        number of `signal` samples along `axis`.
        The default is the first harmonic (fundamental frequency).
        A minimum of `harmonic * 2 + 1` samples are required along `axis`
        to calculate correct phasor coordinates at `harmonic`.
    sigma : float, default: 0.05
        Width of Gaussian filter in spectral vector space.
        Weighted averages are calculated using the spectra of signal items
        within a spectral vector Euclidean distance of `3 * sigma` and
        intensity above `vmin`.
    vmin : float, optional
        Signal intensity along `axis` below which spectra are excluded from
        denoising.
    dtype : dtype_like, optional
        Data type of output arrays. Either float32 or float64.
        The default is float64 unless the `signal` is float32.
    num_threads : int, optional
        Number of OpenMP threads to use for parallelization.
        By default, multi-threading is disabled.
        If zero, up to half of logical CPUs are used.
        OpenMP may not be available on all platforms.

    Returns
    -------
    ndarray
        Denoised signal of `dtype`.
        Spectra with integrated intensity below `vmin` are unchanged.

    References
    ----------
    .. [2] Harman RC, Lang RT, Kercher EM, Leven P, and Spring BQ.
       `Denoising multiplexed microscopy images in n-dimensional spectral space
       <https://doi.org/10.1364/BOE.463979>`_.
       *Biomed Opt Express*, 13(8): 4298-4309 (2022)

    Notes
    -----
    This implementation is considered experimental. It is not validated
    against the reference implementation and may not be practical with
    real-world data. See discussion in `issue #142
    <https://github.com/phasorpy/phasorpy/issues/142#issuecomment-2499421491>`_.

    Examples
    --------
    Denoise a hyperspectral image with a Gaussian filter width of 0.1 in
    spectral vector space using first and second harmonic:

    >>> signal = numpy.random.randint(0, 255, (8, 16, 16))
    >>> signal_filter_svd(signal, axis=0, sigma=0.1, harmonic=[1, 2])
    array([[[...]]])

    """
    num_threads = number_threads(num_threads)

    signal = numpy.asarray(signal)
    if axis == -1 or axis == signal.ndim - 1:
        axis = -1
    else:
        signal = numpy.moveaxis(signal, axis, -1)
    shape = signal.shape
    samples = shape[-1]

    if harmonic is None:
        harmonic = 1
    harmonic, _ = parse_harmonic(harmonic, samples // 2)
    num_harmonics = len(harmonic)

    if vmin is None or vmin < 0.0:
        vmin = 0.0

    signal = numpy.ascontiguousarray(signal).reshape(-1, samples)
    size = signal.shape[0]

    if dtype is None:
        if signal.dtype.char == 'f':
            dtype = signal.dtype
        else:
            dtype = numpy.float64
    dtype = numpy.dtype(dtype)
    if dtype.char not in {'d', 'f'}:
        raise ValueError('dtype is not floating point')

    if spectral_vector is None:
        sincos = numpy.empty((num_harmonics, samples, 2))
        for i, h in enumerate(harmonic):
            phase = numpy.linspace(
                0,
                h * math.pi * 2.0,
                samples,
                endpoint=False,
                dtype=numpy.float64,
            )
            sincos[i, :, 0] = numpy.cos(phase)
            sincos[i, :, 1] = numpy.sin(phase)
        spectral_vector = numpy.zeros((size, num_harmonics * 2), dtype=dtype)

        _phasor_from_signal_vector(
            spectral_vector, signal, sincos, num_threads
        )
    else:
        spectral_vector = numpy.ascontiguousarray(spectral_vector, dtype=dtype)
        if spectral_vector.shape[:-1] != shape[:-1]:
            raise ValueError('signal and spectral_vector shape mismatch')
        spectral_vector = spectral_vector.reshape(
            -1, spectral_vector.shape[-1]
        )

    if dtype == signal.dtype:
        denoised = signal.copy()
    else:
        denoised = numpy.zeros(signal.shape, dtype=dtype)
        denoised[:] = signal
    integrated = numpy.zeros(size, dtype=dtype)
    _signal_denoise_vector(
        denoised, integrated, signal, spectral_vector, sigma, vmin, num_threads
    )

    denoised = denoised.reshape(shape)  # type: ignore[assignment]
    if axis != -1:
        denoised = numpy.moveaxis(denoised, -1, axis)
    return denoised


def signal_filter_ncpca(
    signal: ArrayLike,
    /,
    n_components: int | float | str | None = 3,
    *,
    axis: int = -1,
    dtype: DTypeLike | None = None,
    **kwargs: Any,
) -> NDArray[Any]:
    """Return signal filtered by noise-corrected principal component analysis.

    Apply noise-corrected Principal Component Analysis (NC-PCA) to denoise
    signal containing shot noise. The signal is Poisson-normalized,
    its dimensionality reduced by PCA with a specified number of components,
    and then reconstructed according to [3]_.

    Parameters
    ----------
    signal : array_like
        Data containing Poisson noise to be filtered.
        Must have at least 3 samples along the specified `axis`.
    n_components : int, float, or str, optional, default: 3
        Number of principal components to retain.
        The default is 3, matching the reference implementation.
        If None, all components are kept (no denoising).
        If 'mle', use Minka's MLE to guess the dimension.
        If 0 < n_components < 1 and svd_solver == 'full', select the number
        of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.
        See `sklearn.decomposition.PCA
        <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
        for more details.
    axis : int, optional, default: -1
        Axis containing PCA features, for example, FLIM histogram bins.
        The default is the last axis (-1).
        Other axes are flattened and used as PCA samples.
    dtype : dtype_like, optional
        Data type of computation and output arrays. Either float32 or float64.
        The default is float64 unless the input `signal` is float32.
    **kwargs
        Optional arguments passed to :py:class:`sklearn.decomposition.PCA`.

    Returns
    -------
    ndarray
        Denoised signal of specified `dtype`. Values may be negative.
        Values, where the signal mean along `axis` is zero, are set to NaN.

    Raises
    ------
    ValueError
        If `dtype` is not a floating point type.
        If `signal` has fewer than 3 samples along specified axis.
        If `n_components` is invalid for the data size.

    References
    ----------
    .. [3] Soltani S, Paulson J, Fong E, Mumenthaler, S, and Armani A.
       `Denoising of fluorescence lifetime imaging data via principal
       component analysis <https://doi.org/10.21203/rs.3.rs-7143126/v1>`_.
       *Preprint*, (2025)

    Notes
    -----
    Intensities of the reconstructed signal may be negative.
    Hence, the phasor coordinates calculated from the reconstructed signal
    may be outside the unit circle.
    Consider thresholding low intensities for further analysis.

    Examples
    --------
    Denoise FLIM data using 3 principal components:

    >>> signal = numpy.random.poisson(100, (32, 32, 64))
    >>> denoised = signal_filter_ncpca(signal, n_components=3)
    >>> denoised.shape
    (32, 32, 64)

    """
    from sklearn.decomposition import PCA

    if (
        dtype is None
        and isinstance(signal, numpy.ndarray)
        and signal.dtype == numpy.float32
    ):
        signal = signal.copy()
        dtype = signal.dtype
    else:
        dtype = numpy.dtype(dtype)
        if dtype.char not in {'f', 'd'}:
            raise ValueError(f'{dtype=} is not a floating point type')
        signal = numpy.asarray(signal, dtype=dtype, copy=True)

    if axis == -1 or axis == signal.ndim - 1:
        axis = -1
    else:
        signal = numpy.moveaxis(signal, axis, -1)

    shape = signal.shape

    if signal.size == 0:
        raise ValueError('signal array is empty')
    if signal.shape[-1] < 3:
        raise ValueError(f'{signal.shape[-1]=} < 3')

    # flatten sample dimensions
    signal = signal.reshape(-1, shape[-1])

    # poisson-normalize signal
    scale = numpy.sqrt(numpy.nanmean(signal, axis=0, keepdims=True))
    with numpy.errstate(divide='ignore', invalid='ignore'):
        signal /= scale

    # replace NaN and infinite values
    mask = numpy.logical_or(
        numpy.any(numpy.isnan(signal), axis=-1),
        numpy.any(numpy.isinf(signal), axis=-1),
    )
    assert isinstance(signal, numpy.ndarray)  # for Mypy
    signal[mask] = 0.0

    # PCA transform and reconstruction with n_components
    pca = PCA(n_components, **kwargs)
    signal = pca.inverse_transform(pca.fit_transform(signal))
    del pca

    # restore original scale and shape
    assert isinstance(signal, numpy.ndarray)  # for Mypy
    signal[mask] = numpy.nan
    signal *= scale
    signal = signal.reshape(shape)
    if axis != -1:
        signal = numpy.moveaxis(signal, -1, axis)

    return signal
