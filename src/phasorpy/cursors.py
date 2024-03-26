"""cursors.

The ``phasorpy.cursors`` module provides functions to:

- use cursors to select region of interest in the phasor:

  - :py:func:`circular_cursor`
  - :py:func:`range_cursor`


"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    'circular_cursor',
    'range_cursor',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import ArrayLike

import warnings

import numpy


def circular_cursor(
        real: ArrayLike,
        imag: ArrayLike,
        center: ArrayLike,
        radius: ArrayLike,
        components: int):
    """Return labeled mask with components for each circle.

    Parameters
    ----------
    real : ndarray
        Real component of phasor coordinates along axis.
    imag : ndarray
        Imaginary component of phasor coordinates along axis.
    center : float
        Circle center.
    radius : ndarray
        Radius for each circle.
    components : int
        Amount of components, default 2. 

    Returns
    -------
    label : ndarray
        Labeled matrix for all components. 

    Raises
    ------
    ValueError
        real and imag must have the same dimensions.
    ValueError
        radius must be greater than zero.
    ValueError
        components must be at least 1.

    Examples
    --------
    >>> circular_cursor(numpy.array([-0.5, -0.5, 0.5, 0.5]),
    ...     numpy.array([-0.5, 0.5, -0.5, 0.5]), 
    ...     numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]), 
    ...     radius=[0.1, 0.1, 0.1, 0.1], components=4)
    array([1., 2., 3., 4.])
    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    center = numpy.asarray(center)
    radius = numpy.asarray(radius)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    for i in range(len(radius)):
        if radius[i] < 0:
            raise ValueError(f'radius is < 0 for index {i}')
    if components < 1:
        raise ValueError(f'components is {components}, must be at least 1')
    if len(center) == components:
        label = numpy.zeros(real.shape)
        for i in range (components):
            condition = (real - center[i][0]) ** 2 + (imag - center[i][1]) ** 2 - radius[i] ** 2
            label = numpy.where(condition > 0, label, numpy.ones(label.shape) * (i + 1))
        return label
    else: 
        raise ValueError(f'center length array and components must be equal')


def range_cursor(
        values: ArrayLike,
        ranges: ArrayLike):

    """Return the labeled mask for each range.

        Parameters
        ----------
        values : ndarray
            An n-dimensional array of values.
        ranges : ndarray
            Represents ranges as (start, end).  

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
        >>> range_cursor(numpy.array([[3.3, 6, 8], [15, 20, 7]]), 
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
                label[index] = range_index + 1  # Set the index of the current range
                break
    return label


def _overlapping_ranges(
        ranges: ArrayLike):
    
    """Check if there are overlapping ranges in an array of ranges.

    Parameters
    ----------
        ranges : ndarray
            Represents ranges as (start, end).

    Returns
    -------
        bool: True if there are overlapping ranges, False otherwise.

    Example
    -------
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
