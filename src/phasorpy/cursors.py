""" Select phasor coordinates.

    The ``phasorpy.cursors`` module provides functions to:

- create labels for region of interests in the phasor space:

  - :py:func:`label_from_phasor_circular`
  - :py:func:`mask_from_cursor`

"""

from __future__ import annotations

__all__ = [
    'label_from_phasor_circular',
    'mask_from_cursor',
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
    ...     radius=[0.1, 0.1, 0.1, 0.1],
    ... )
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


def mask_from_cursor(
    xarray: NDArray,
    yarray: NDArray,
    xrange: NDArray,
    yrange: NDArray,
) -> NDArray[Any]:
    """
    Create mask for a cursor.
    Create mask for a cursor.

    Parameters
    ----------
    - xarray: NDArray
        x-coordinates.
    - yarray: NDArray
        y-coordinates.
    - xrange: NDArray
        x range to be binned.
    - yrange: NDArray
        y range to be binned.
    - xarray: NDArray
        x-coordinates.
    - yarray: NDArray
        y-coordinates.
    - xrange: NDArray
        x range to be binned.
    - yrange: NDArray
        y range to be binned.

    Returns
    -------
    - mask: NDArray:
        cursor mask.
    - mask: NDArray:
        cursor mask.

    Raises
    ------
    ValueError
        `xarray` and `yarray` must be same shape.
    ValueError
        `xrange` and y `range` must be the same length.

    Example
    -------
    Creat mask from cursor.
    >>> phase = [[337, 306, 227], [21, 231, 235], [244, 328, 116]]
    >>> mod = [[0.22, 0.40, 0.81], [0.33, 0.43, 0.36], [0.015, 0.82, 0.58]]
    >>> mask_from_cursor(
    ...     xarray=phase, yarray=mod, xrange=[0, 270], yrange=[0, 0.5]
    ... )
    array([[False, False, False],
            [ True,  True,  True],
            [ True, False, False]])
    """
    xarray = numpy.asarray(xarray)
    yarray = numpy.asarray(yarray)
    if xarray.shape != yarray.shape:
        raise ValueError('xarray and yarray must have same shape')
    if len(xrange) != len(yrange):
        raise ValueError('xrange and y range must be the same length')
    xmask = (xarray >= xrange[0]) & (xarray <= xrange[1])
    ymask = (yarray >= yrange[0]) & (yarray <= yrange[1])
    return xmask & ymask
