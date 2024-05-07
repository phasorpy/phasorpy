""" Select phasor coordinates.

    The ``phasorpy.cursors`` module provides functions to:

- create labels for region of interests in the phasor space:

  - :py:func:`label_from_phasor_circular`
  - :py:func:`label_from_ranges`

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


def label_from_ranges(values: ArrayLike, /, ranges: ArrayLike) -> NDArray[Any]:
    r"""Return indices of range to which each value belongs.
     Values that do not fall in any range have an index of zero.

     Parameters
     ----------
     values : array_like
         Values to be labeled.
     ranges : array_like, Array of bins. It has to be 1-dimensional and monotonic.

     Returns
     -------
     label : ndarray
         A mask indicating the index of the range each value belongs to.

     Examples
     --------
     Compute label array for three ranges:

    >>> label_from_ranges([[0, 3.3, 6, 8], [21, 15, 20, 7]], 
    ...     ranges=[2, 8, 15, 20, 25])
     array([[0, 1, 1, 2],
       [4, 3, 4, 1]])
    """
    return numpy.digitize(values, ranges)
