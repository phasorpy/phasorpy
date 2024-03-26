"""cursors.

The ``phasorpy.cursors`` module provides functions to:

- use cursors to select region of interest in the phasor:

  - :py:func:`circular_cursor`
  - :py:func:`range_cursor`


"""

from __future__ import annotations

__all__ = [
    'circular_cursor',
    'range_cursor',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import ArrayLike

import numpy

import warnings

def circular_cursor(
        real: ArrayLike,
        imag: ArrayLike,
        center: ArrayLike,
        *,
        radius: ArrayLike=[0.1, 0.1],
        components: int = 2):
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
    real = numpy.array([-0.5, -0.5, 0.5, 0.5])
    imag = numpy.array([-0.5, 0.5, -0.5, 0.5])
    center = numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])
    radius = [0.1, 0.1, 0.1, 0.1]
    >>> circular_cursor(real, imag, center, radius=radius, components=4)
    array([1., 2., 3., 4.])  
    """
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
        raise ValueError(f'center lenth array and components must be equal')


def range_cursor(
        value: ArrayLike,
        range: ArrayLike):

    """Return the labeled mask for each range.

        Parameters
        ----------
        value : ndarray
            An n-dimensional array of values.
        range : ndarray
            Represents ranges as (start, end).  

        Returns
        -------
        label : ndarray
            A mask indicating the index of the range each value belongs to.

        Raises
        ------
        Warning:
            Overlapping ranges not recomended.  

        Examples
        --------
        Compute the range cursor: 
        values = numpy.array([[3.3, 6, 8], [15, 20, 7]]) 
        ranges = numpy.array([(2, 8), (10, 15), (20, 25)])
        >>> range_cursor(values, ranges)
        array([[1 1 1] [2 3 1]])
    """
    if _overlapping_ranges(range):
        warnings.warn("Overlapping ranges", UserWarning)
    mask = numpy.zeros_like(value, dtype=int)
    # Iterate over each value in the array
    for index, value in numpy.ndenumerate(value):
        # Iterate over each range
        for range_index, (start, end) in enumerate(range):
            # Check if the value falls within the current range
            if start <= value <= end:
                mask[index] = range_index + 1  # Set the index of the current range
                break
    return mask


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
        ranges = [(1, 5), (3, 8), (6, 10), (9, 12)]
        >>> _overlapping_ranges(ranges)
        True
    """
    # Sort the ranges based on their start values
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    # Iterate through the sorted ranges and check for overlaps
    for i in range(len(sorted_ranges) - 1):
        if sorted_ranges[i][1] >= sorted_ranges[i + 1][0]:
            # Ranges overlap
            return True
    # No overlaps found
    return False
