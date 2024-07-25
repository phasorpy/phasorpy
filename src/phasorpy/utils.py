"""Utility functions.

The ``phasorpy.utils`` module provides auxiliary and convenience functions
that do not naturally fit into other modules.

"""

from __future__ import annotations

__all__ = ['number_threads', 'median_filter']

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import numpy

from ._phasorpy import _apply_2D_filter


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


def median_filter(
    image: ArrayLike,
    /,
    *,
    kernel_size: int = 3,
    reflect: bool = False,
    num_iter: int = 1,
) -> NDArray[Any]:
    """Apply a median filter to an image using Cython.

    Parameters
    ----------
    image : array_like
        Input image.
    kernel_size : int, optional
        Size of the kernel.
    reflect : bool, optional
        If True, the image is padded by reflection.
        If False, the image borders are kept intact. Default is False.
    num_iter : int, optional
        Number of iterations to apply the filter. Default is 1.

    Returns
    -------
    filtered_image : ndarray
        Filtered image.

    Raises
    ------
    ValueError
        If `kernel_size` is not an odd number.
        If `image` is not a 2D array.
        If `num_iter` is less than 1.

    Examples
    --------
    Apply a median filter with a kernel size of 3 and keeping intact borders:

    >>> median_filter([[2, 2, 2], [1, 1, 1], [3, 3, 3]], kernel_size=3)
    array([[2, 2, 2],
        [1, 2, 1],
        [3, 3, 3]])

    Apply a median filter with a kernel size of 3 using 'reflect' padding:

    >>> median_filter([[2, 2, 2], [1, 1, 1], [3, 3, 3]], reflect=True)
    array([[2, 2, 2],
        [2, 2, 2],
        [3, 3, 3]])

    """
    image = numpy.asarray(image)

    if image.ndim != 2:
        raise ValueError(f'{image.ndim=} != 2')
    if kernel_size % 2 == 0:
        raise ValueError("The kernel size must be an odd number.")
    if num_iter < 1:
        raise ValueError(f'{num_iter=} < 1')

    filtered_image = numpy.copy(image)
    for _ in range(num_iter):
        _apply_2D_filter(image, filtered_image, kernel_size, reflect)

    return filtered_image
