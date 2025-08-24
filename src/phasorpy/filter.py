"""Filter signals and phasor coordinates.

The ``phasorpy.filter`` module provides functions to filter

- phasor coordinates:

  - :py:func:`phasor_filter_gaussian` (not implemented yet)
  - :py:func:`phasor_filter_median`
  - :py:func:`phasor_filter_pawflim`
  - :py:func:`phasor_threshold`

- signals:

  - :py:func:`signal_filter_median` (not implemented yet)
  - :py:func:`signal_filter_pca` (not implemented yet)
  - :py:func:`signal_filter_svd` (not implemented yet)

"""

from __future__ import annotations

__all__ = [
    # 'signal_filter_gaussian',
    'phasor_filter_median',
    'phasor_filter_pawflim',
    'phasor_threshold',
    # 'signal_filter_median',
    # 'signal_filter_pca',
    # 'signal_filter_svd',
]

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, NDArray, ArrayLike

import numpy

from ._phasorpy import (
    _median_filter_2d,
    _phasor_threshold_closed,
    _phasor_threshold_mean_closed,
    _phasor_threshold_mean_open,
    _phasor_threshold_nan,
    _phasor_threshold_open,
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
        real = numpy.array(real, numpy.float64, copy=True)
    if use_scipy or repeat == 0:  # or using nD numpy filter
        imag = numpy.asarray(imag)
    elif isinstance(imag, numpy.ndarray) and imag.dtype == numpy.float32:
        imag = imag.copy()
    else:
        imag = numpy.array(imag, numpy.float64, copy=True)

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

    mean = numpy.asarray(mean)
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)

    if levels < 0:
        raise ValueError(f'{levels=} < 0')
    if levels == 0:
        return mean, real, imag

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

    mean = numpy.asarray(numpy.nan_to_num(mean), dtype=float)
    real = numpy.asarray(numpy.nan_to_num(real * mean), dtype=float)
    imag = numpy.asarray(numpy.nan_to_num(imag * mean), dtype=float)

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
                dtype=complex,
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
