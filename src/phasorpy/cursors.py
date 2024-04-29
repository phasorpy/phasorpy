""" Select phasor coordinates.
    The ``phasorpy.cursors`` module provides functions to:

- create labels for region of interests in the phasor space:

  - :py:func:`circular_cursor`
  - :py:func:`range_cursor`

"""

from __future__ import annotations

__all__ = [
    'label_from_phasor_circular',
    'label_from_ranges',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import (
        Any,
        ArrayLike,
        NDArray,
    )

import warnings

import numpy


def label_from_phasor_circular(
    real,
    imag,
    center: ArrayLike,
    radius: ArrayLike,
) -> tuple[NDArray[Any]]:
    r"""Return indices of circle to which each phasor coordinate belongs.
    Phasor coordinates that do not fall in a circle have an index of zero.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    center : array_like, shape (M, 2)
        Phasor coordinates of circle centers.
    radius : array_like, shape (M,)
        Radii of circles.

    Returns
    -------
    label : ndarray
        Indices of circle to which each phasor coordinate belongs.

    Raises
    ------
    ValueError
        `real` and `imag` must have the same dimensions.
        `radius` must be positive.

    Examples
    --------
    Compute label array for four circles:

    >>> label_from_phasor_circular(numpy.array([-0.5, -0.5, 0.5, 0.5]),
    ...     numpy.array([-0.5, 0.5, -0.5, 0.5]),
    ...     numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]),
    ...     radius=[0.1, 0.1, 0.1, 0.1])
    array([1, 2, 3, 4])
    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    center = numpy.asarray(center)
    radius = numpy.asarray(radius)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if numpy.any(radius < 0):
        raise ValueError('radius is < 0')
    label = numpy.zeros(real.shape, dtype=numpy.int8)
    for i in range(len(center)):
        condition = (
            (real - center[i][0]) ** 2
            + (imag - center[i][1]) ** 2
            - radius[i] ** 2
        )
        label = numpy.where(
            condition > 0, label, numpy.ones(label.shape) * (i + 1)
        )
    return label


def label_from_ranges(values, ranges: ArrayLike) -> tuple[NDArray[Any]]:
    r"""Return indices of range to which each value belongs.
    Values that do not fall in any range have an index of zero.

    Parameters
    ----------
    values : array_like, shape (M, 2)
    ranges : array_like
        Start and stop values of ranges.

    Returns
    -------
    label : ndarray
        A mask indicating the index of the range each value belongs to.

    Raises
    ------
    Warning:
        Overlapping ranges not recommended.

    Examples
    --------
    Compute the range cursor:

    >>> label_from_ranges(numpy.array([[3.3, 6, 8], [15, 20, 7]]),
    ...     numpy.array([(2, 8), (10, 15), (20, 25)]))
    array([[1, 1, 1], [2, 3, 1]])
    """
    values = numpy.asarray(values)
    ranges = numpy.asarray(ranges)

    if _overlapping_ranges(ranges):
        warnings.warn("Overlapping ranges", UserWarning)
    label = numpy.zeros_like(values, dtype=int)
    # Iterate over each value in the array
    for index, value in numpy.ndenumerate(values):
        # Iterate over each range
        for range_index, (start, end) in enumerate(ranges):
            # Check if the value falls within the current range
            if start <= value <= end:
                # Set the index of the current range
                label[index] = range_index + 1
                break
    return label


def _overlapping_ranges(ranges) -> bool:
    r"""Check if there are overlapping ranges in an array of ranges.

    Parameters
    ----------
        ranges : array_like
            Start and stop values of ranges.

    Returns
    -------
        bool: True if there are overlapping ranges, False otherwise.

    Example
    -------
    Compute for some range with overlapping.
        >>> _overlapping_ranges([(1, 5), (3, 8), (6, 10), (9, 12)])
        True
    """
    ranges = numpy.asarray(ranges)
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            # Check if the ranges overlap
            if ranges[i][0] <= ranges[j][1] and ranges[j][0] <= ranges[i][1]:
                return True  # Ranges overlap
    return False  # No overlaps found
