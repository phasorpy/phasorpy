"""Utility functions.

The ``phasorpy.utils`` module provides auxiliary and convenience functions
that do not naturally fit into other modules.

"""

from __future__ import annotations

__all__ = ['number_threads', 'phasor_filter']

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import numpy
from scipy.ndimage import median_filter


def number_threads(
    num_threads: int | None = None,
    max_threads: int | None = None,
    /,
) -> int:
    """Return number of threads for parallel computations on CPU cores.

    This function is used to parse ``num_threads`` parameters.

    Parameters
    ----------
    num_threads : int, optional
        Number of threads to use for parallel computations on CPU cores.
        If None (default), return 1, disabling multi-threading.
        If greater than zero, return value up to `max_threads` if set.
        If zero, return the value of the ``PHASORPY_NUM_THREADS`` environment
        variable if set, else half the CPU cores up to `max_threads` or 32.
    max_threads : int, optional
        Maximum number of threads to return.

    Examples
    --------
    >>> number_threads()
    1
    >>> number_threads(0)  # doctest: +SKIP
    8

    """
    if num_threads is None or num_threads < 0:
        # disable multi-threading by default
        return 1
    if num_threads == 0:
        # return default number of threads
        if max_threads is None:
            max_threads = 32
        else:
            max_threads = max(max_threads, 1)
        if 'PHASORPY_NUM_THREADS' in os.environ:
            return min(
                max_threads, max(1, int(os.environ['PHASORPY_NUM_THREADS']))
            )
        cpu_count: int | None
        try:
            cpu_count = len(os.sched_getaffinity(0))  # type: ignore
        except AttributeError:
            # sched_getaffinity not available on Windows
            cpu_count = os.cpu_count()
        if cpu_count is None:
            return 1
        return min(max_threads, max(1, cpu_count // 2))
    # return num_threads up to max_threads
    if max_threads is None:
        return num_threads
    return min(num_threads, max(max_threads, 1))


def phasor_filter(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    method: str = 'median',
    repeat: int = 1,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Apply a filter to phasor coordinates.

    By default a median filter is applied to the real and imaginary
    components of phasor coordinates once with a kernel size of 3
    multiplied by the number of dimensions of the input arrays.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates to be filtered.
    imag : array_like
        Imaginary component of phasor coordinates to be filtered.
    method : str, optional
        Method used for filtering:

        - ``'median'``: Spatial median of phasor coordinates.

    repeat : int, optional
        The number of times to apply the median filter. Default is 1.
    **kwargs
        Optional arguments passed to :py:func:`scipy.ndimage.median_filter`.

    Returns
    -------
    real : ndarray
        Filtered real component of phasor coordinates.
    imag : ndarray
        Filtered imaginary component of phasor coordinates.

    Raises
    ------
    ValueError
        If the specified method is not supported.
        The array shapes of `real` and `imag` do not match.
        If `repeat` is less than 1.

    Notes
    -----
    For now only the median filter method is implemented.
    Additional filtering methods may be added in the future.

    Examples
    --------
    Apply once a median filter with a kernel size of 3:

    >>> phasor_filter(
    ...     [[2, 2, 2], [1, 1, 1], [3, 3, 3]],
    ...     [[4, 4, 4], [6, 6, 6], [5, 5, 5]],
    ... )
    (array([[2, 2, 2],
            [2, 2, 2],
            [3, 3, 3]]),
    array([[4, 4, 4],
            [5, 5, 5],
            [5, 5, 5]]))

    Apply 3 times a median filter with a kernel size of 3:

    >>> phasor_filter(
    ...     [[0, 0, 0], [5, 5, 5], [2, 2, 2]],
    ...     [[3, 3, 3], [6, 6, 6], [4, 4, 4]],
    ...     size=3,
    ...     repeat=3,
    ... )
    (array([[0, 0, 0],
            [2, 2, 2],
            [2, 2, 2]]),
    array([[3, 3, 3],
            [4, 4, 4],
            [4, 4, 4]]))

    """
    supported_methods = ['median']
    if method not in supported_methods:
        raise ValueError(
            f"Method not supported, supported methods are: "
            f"{', '.join(supported_methods)}"
        )
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if repeat < 1:
        raise ValueError(f'{repeat=} < 1')

    return {
        'median': _median_filter,
    }[
        method
    ](real, imag, repeat, **kwargs)


def _median_filter(
    real: ArrayLike,
    imag: ArrayLike,
    repeat: int = 1,
    size: int = 3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the phasor coordinates after applying a median filter.

    Convenience wrapper around :py:func:`scipy.ndimage.median_filter`.

    Parameters
    ----------
    real : numpy.ndarray
        Real components of the phasor coordinates.
    imag : numpy.ndarray
        Imaginary components of the phasor coordinates.
    repeat : int, optional
        The number of times to apply the median filter. Default is 1.
    size : int, optional
        The size of the median filter kernel. Default is 3.
    **kwargs
        Optional arguments passed to :py:func:`numpy.median`.

    Returns
    -------
    real : ndarray
        Filtered real component of phasor coordinates.
    imag : ndarray
        Filtered imaginary component of phasor coordinates.

    Examples
    --------
    >>> _median_filter(
    ...     [[2, 2, 2], [1, 1, 1], [3, 3, 3]],
    ...     [[4, 4, 4], [6, 6, 6], [5, 5, 5]],
    ... )
    (array([[2, 2, 2],
            [2, 2, 2],
            [3, 3, 3]]),
    array([[4, 4, 4],
            [5, 5, 5],
            [5, 5, 5]]))

    """
    for _ in range(repeat):
        real = median_filter(real, size=size, **kwargs)
        imag = median_filter(imag, size=size, **kwargs)

    return numpy.asarray(real), numpy.asarray(imag)
