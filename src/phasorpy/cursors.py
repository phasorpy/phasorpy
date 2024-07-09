""" Select phasor coordinates.

    The ``phasorpy.cursors`` module provides functions to:

- create labels for region of interests in the phasor space:

  - :py:func:`mask_from_circular_cursor`
  - :py:func:`mask_from_cursor`
  - :py:func:`segmentate_with_cursors`

"""

from __future__ import annotations

__all__ = [
    'mask_from_circular_cursor',
    'mask_from_polar_cursor',
    'segmentate_with_cursors',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import (
        Any,
        ArrayLike,
        NDArray,
    )

import numpy


def mask_from_circular_cursor(
    real: ArrayLike,
    imag: ArrayLike,
    centers: ArrayLike,
    radius: float,
) -> NDArray[Any]:
    """Return masks for circular cursors of phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    centers : array_like, shape (..., 2)
        Phasor coordinates of circle centers.
    radius : float
        Radii of circles.

    Returns
    -------
    masks : ndarray
        Indices of circle to which each phasor coordinate belongs.

    Raises
    ------
    ValueError
        `real` and `imag` must have the same dimensions.
        'centers' second dimension must be 2.
        `radius` must be positive.

    Examples
    --------
    Compute mask for one circular cursor:

    >>> mask_from_circular_cursor(
    ...     [0.0, 0.0],
    ...     [0.0, 0.5],
    ...     [0.0, 0.5],
    ...     0.1,
    ... )
    array([ False, True])

    Compute masks for three circular cursors:

    >>> mask_from_circular_cursor(
    ...     [0.0, 0.0],
    ...     [0.0, 0.5],
    ...     [[0.0, 0.5],[0.0, 0.0]],
    ...     0.1,
    ... )
    array([ False, True], [True, False])

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    centers = numpy.asarray(centers)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if centers.shape[1] != 2:
        raise ValueError(f'invalid {centers.shape=}')
    if numpy.any(radius < 0):
        raise ValueError('radius is < 0')

    centers = centers[:, numpy.newaxis, :]
    distance_sq = numpy.square(real - centers[..., 0]) + numpy.square(imag - centers[..., 1])
    masks = distance_sq <= numpy.square(radius)
    return masks


def mask_from_polar_cursor(
    xarray: NDArray,
    yarray: NDArray,
    xrange: NDArray,
    yrange: NDArray,
) -> NDArray[Any]:
    """
    Create mask for a cursor.

    Parameters
    ----------
    - xarray: NDArray
        x-coordinates.
    - yarray: NDArray
        y-coordinates.
    - xarray: NDArray
        x-coordinates.
    - yarray: NDArray
        y-coordinates.

    Returns
    -------
    - mask: NDArray:
        cursor mask.

    Raises
    ------
    ValueError
        `xarray` and `yarray` must be same shape.

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


def segmentate_with_cursors(
    mask: NDArray, cursors_color: NDArray, mean: NDArray
) -> NDArray[Any]:
    """
    Create the segmented image with cursors.

    Parameters
    ----------
    - mask: NDArray
        masks for each cursor.
    - cursors_color: NDArray
        cursor color to match with the segmented region.
    - mean: NDArray
        grayscale image to overwrite with segmented areas.

    Returns
    -------
    - Segmented image: NDArray:
        Segmented image with cursors.

    Raises
    ------
    ValueError
        `xarray` and `yarray` must be same shape.

    Example
    -------
    Segment an image with cursors.
    >>> segmentate_with_cursors(
    ...     [True, False, False], [255, 0, 0], [0, 128, 255]
    ... )
    array([[255,   0,   0],
           [128, 128, 128],
           [255, 255, 255]])
    """

    mask = numpy.asarray(mask)
    mean = numpy.asarray(mean)

    if mask.ndim == 1 and mean.ndim == 1:
        if mask.shape[0] != mean.shape[0]:
            raise ValueError('mask and mean first dimension must be equal')
        else:
            imcolor = numpy.zeros([mask.shape[0], 3])
            mean = numpy.stack([mean, mean, mean], axis=-1)
            imcolor[:, 0] = cursors_color[0]
            imcolor[:, 1] = cursors_color[1]
            imcolor[:, 2] = cursors_color[2]
            segmented = numpy.where(
                numpy.stack([mask, mask, mask], -1), imcolor, mean
            )
            return segmented
    else:
        if mask.shape[0] != mean.shape[0] or mask.shape[1] != mean.shape[1]:
            raise ValueError(
                'mask and mean first and second dimension\n' 'must be the same'
            )
        else:
            imcolor = numpy.zeros([mask.shape[0], mask.shape[1], 3])
            mean = numpy.stack([mean, mean, mean], -1)
            segmented = numpy.copy(mean)
            for i in range(len(cursors_color)):
                imcolor[:, :, 0] = cursors_color[i][0]
                imcolor[:, :, 1] = cursors_color[i][1]
                imcolor[:, :, 2] = cursors_color[i][2]
                segmented = numpy.where(
                    numpy.stack(
                        [mask[:, :, i], mask[:, :, i], mask[:, :, i]], -1
                    ),
                    imcolor,
                    segmented,
                )
            return segmented
