""" Select phasor coordinates.

    The ``phasorpy.cursors`` module provides functions to:

- create labels for region of interests in the phasor space:

  - :py:func:`label_from_phasor_circular`
  - :py:func:`create_lut`
  - :py:func:`label_from_lut` 

"""

from __future__ import annotations

__all__ = [
    'label_from_phasor_circular',
    'create_lut',
    'label_from_lut',
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
    real: ArrayLike,
    imag: ArrayLike,
    center: ArrayLike,
    radius: ArrayLike,
) -> NDArray[Any]:
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

    >>> label_from_phasor_circular(
    ...     numpy.array([-0.5, -0.5, 0.5, 0.5]),
    ...     numpy.array([-0.5, 0.5, -0.5, 0.5]),
    ...     numpy.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]),
    ...     radius=[0.1, 0.1, 0.1, 0.1])
    array([1, 2, 3, 4], dtype=uint8)
    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    center = numpy.asarray(center)
    radius = numpy.asarray(radius)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if center.ndim != 2 or center.shape[1] != 2:
        raise ValueError(f'invalid {center.shape=}')
    if radius.ndim != 1 or radius.shape != (center.shape[0],):
        raise ValueError(f'invalid {radius.shape=}')
    if numpy.any(radius < 0):
        raise ValueError('radius is < 0')
    dtype = numpy.uint8 if len(center) < 256 else numpy.uint16
    label = numpy.zeros(real.shape, dtype=dtype)
    for i in range(len(center)):
        condition = (
            numpy.square(real - center[i][0])
            + numpy.square(imag - center[i][1])
            - numpy.square(radius[i])
        )
        label = numpy.where(
            condition > 0, label, numpy.full(label.shape, i + 1, dtype=dtype)
        )
    return label


#############################################
#############################################


def create_lut(
    min_vals1: NDArray,
    max_vals1: NDArray,
    min_vals2: NDArray,
    max_vals2: NDArray,
) -> dict:
    """
    Create a Lookup Table (LUT) with two pairs of minimum and maximum values.

    Parameters
    ----------
    - min_vals1: NDArray
        Array of minimum values to binarize data1.
    - max_vals1: NDArray
        Array of maximum values to binarize data1.
    - min_vals2: NDArray
        Array of minimum values to binarize data2.
    - max_vals2: NDArray
        Array of maximum values to binarize data2.

    Returns
    -------
    - dict
        Lookup Table (LUT) mapping input values to binarized output values.

    Raises
    ------
    ValueError
        'Input array must have same shapes'
    """
    if (
        min_vals1.shape
        == max_vals1.shape
        == min_vals2.shape
        == max_vals2.shape
    ):
        # Initialize the Lookup Table (LUT)
        lut = {}
        # Define the binning ranges and their corresponding binarized values
        for i, (min1, max1) in enumerate(zip(min_vals1, max_vals1)):
            for j, (min2, max2) in enumerate(zip(min_vals2, max_vals2)):
                lut[((min1, max1), (min2, max2))] = i * len(min_vals2) + j + 1
        return lut
    else:
        raise ValueError('Input array must have same shapes')


def label_from_lut(arr1: NDArray, arr2: NDArray, lut: dict) -> NDArray[Any]:
    """
    Binarize two arrays based on a Lookup Table (LUT).

    Parameters
    ----------
    - data1: NDArray
        The first data array.
    - data2: NDArray
        The second data array.
    - lut: dict
        Lookup Table (LUT) mapping input values to binarized output values.

    Returns
    -------
    - label: NDArray:
        The binarized array.

    Raises
    ------
    ValueError
        'Input arrays must have same shapes'

    """
    # Check if the input arrays have compatible shapes
    if arr1.shape != arr2.shape:
        raise ValueError('Input arrays must have same shapes')
    label = numpy.zeros(arr1.shape, dtype=int)
    # Loop through the Lookup Table (LUT) and binarize the data
    for ((min1, max1), (min2, max2)), binarized_val in lut.items():
        label += numpy.where(
            (min1 <= arr1) & (arr1 <= max1) & (min2 <= arr2) & (arr2 <= max2),
            binarized_val,
            0,
        )
    return label


#### EXAMPLES
# Examples
# --------
# Create a LUT based on ranges values:

# >>> create_lut(
# ...     min_vals1 = numpy.array([0, 3, 6]),
# ...     max_vals1 = numpy.array([2, 5, 8]),
# ...     min_vals2 = numpy.array([1, 4, 7]),
# ...     max_vals2 = numpy.array([3, 6, 9]))
# {((0, 2), (1, 3)): 1, ((0, 2), (4, 6)): 2, ((0, 2), (7, 9)): 3,
# ... ((3, 5), (1, 3)): 4, ((3, 5), (4, 6)): 5, ((3, 5), (7, 9)): 6,
# ... ((6, 8), (1, 3)): 7, ((6, 8), (4, 6)): 8, ((6, 8), (7, 9)): 9}

# Example
# -------
# >>> arr1 = numpy.array([[1.2, 2.4, 3.5], [4.7, 5.1, 6.9], [7.3, 8.6, 9.0]])
# >>> arr2 = numpy.array([[0.8, 2.1, 3.9], [4.2, 5.7, 6.3],[7.5, 8.2, 9.5]])
# >>> lut = {((0, 2), (1, 3)): 1, ((0, 2), (4, 6)): 2, ((0, 2), (7, 9)): 3,
# ... ((3, 5), (1, 3)): 4, ((3, 5), (4, 6)): 5, ((3, 5), (7, 9)): 6,
# ... ((6, 8), (1, 3)): 7, ((6, 8), (4, 6)): 8, ((6, 8), (7, 9)): 9}
# >>> label = label_from_lut(arr1, arr2, lut)
# array([[0, 0, 0], [5, 0, 0], [9, 0, 0]])
