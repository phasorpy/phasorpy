""" Select phasor coordinates.

    The ``phasorpy.cursors`` module provides functions to:

- create labels for region of interests in the phasor space:

  - :py:func:`mask_from_circular_cursor`
  - :py:func:`mask_from_polar_cursor`
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
    centers_real: ArrayLike,
    centers_imag: ArrayLike,
    /,
    *,
    radius: ArrayLike = 0.05,
) -> NDArray[numpy.bool_]:
    """Return masks for circular cursors of phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    centers_real : array_like, shape (n,)
        Real coordinates of circle centers.
    centers_imag : array_like, shape (n,)
        Imaginary coordinates of circle centers.
    radius : array_like, shape (n,)
        Radii of circles.

    Returns
    -------
    masks : ndarray
        Phasor coordinates masked for each circular cursor.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag`, or `centers_real` and
        `centers_imag` do not match.
        Center's coordinates and/or the radii are not one-dimensional.
        Any of the radii is negative.

    Examples
    --------
    Create mask for a single circular cursor:

    >>> mask_from_circular_cursor([0.0, 0.0], [0.0, 0.5], 0.0, 0.5, radius=0.1)
    array([ False, True])

    Create masks for two circular cursors with different radius:

    >>> mask_from_circular_cursor(
    ...     [0.0, 1.0],
    ...     [0.0, 0.5],
    ...     [0.0, 1.0],
    ...     [0.0, 0.4],
    ...     radius=[0.1, 0.05]
    ... )
    array([ False, True], [False, False])

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    centers_real = numpy.asarray(centers_real)
    centers_imag = numpy.asarray(centers_imag)
    radius = numpy.atleast_1d(radius)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if centers_real.shape != centers_imag.shape:
        raise ValueError(f'{centers_real.shape=} != {centers_imag.shape=}')
    if centers_real.ndim != 1:
        raise ValueError(
            'center coordinates are not one-dimensional: '
            f'{centers_real.ndim} dimensions found'
        )
    if radius.ndim != 1:
        raise ValueError(
            f'radius must be one dimensional: {radius.ndim} dimensions found'
        )
    if numpy.min(radius) < 0:
        raise ValueError('all radii must be positive')

    distance = numpy.square(
        real[..., numpy.newaxis] - centers_real
    ) + numpy.square(imag[..., numpy.newaxis] - centers_imag)
    masks = distance <= numpy.square(radius)
    return masks


def mask_from_polar_cursor(
    phase: ArrayLike,
    modulation: ArrayLike,
    phase_range: ArrayLike,
    modulation_range: ArrayLike,
    /,
) -> NDArray[numpy.bool_]:
    """Return mask for polar cursor of polar coordinates.

    Parameters
    ----------
    - phase: array_like
        Angular component of polar coordinates in radians.
    - modulation: array_like
        Radial component of polar coordinates.
    - phase_range: array_like, shape (..., 2)
        Angular range of the cursors in radians.
        The start and end of the range must be in the last dimension.
    - modulation_range: array_like, shape (..., 2)
        Radial range of the cursors.
        The start and end of the range must be in the last dimension.

    Returns
    -------
    - masks: ndarray
        Polar coordinates masked for each polar cursor.

    Raises
    ------
    ValueError
        The array shapes of `phase` and `modulation`, or `phase_range` and
        `modulation_range` do not match.
        The last dimension of `phase_range` and `modulation_range` is not 2.

    Example
    -------
    Create mask from a single polar cursor:

    >>> mask_from_polar_cursor([5,100], [0.2, 0.4], [50, 150], [0.2, 0.5])
    array([False,  True])

    Create masks for two polar cursors with different ranges:

    >>> mask_from_polar_cursor(
    ...     [5,100],
    ...     [0.2, 0.4],
    ...     [[50, 150], [0, 270]],
    ...     [[0.2, 0.5], [0, 0.3]]
    ... )
    array([[False,  True],
           [ True, False]])

    """
    phase = numpy.asarray(phase)
    modulation = numpy.asarray(modulation)
    phase_range = numpy.asarray(phase_range)
    modulation_range = numpy.asarray(modulation_range)

    if phase.shape != modulation.shape:
        raise ValueError(f'{phase.shape=} != {modulation.shape=}')
    if phase_range.shape != modulation_range.shape:
        raise ValueError(f'{phase_range.shape=} != {modulation_range.shape=}')
    if phase_range.shape[-1] != 2:
        raise ValueError(f'The last dimension of the range must be 2: {phase_range.shape[-1]} found')
    
    if numpy.ndim(phase_range) > 1:
        axes = tuple(range(1, numpy.ndim(phase) + 1))
        phase = numpy.expand_dims(phase, axis=0)
        modulation = numpy.expand_dims(modulation, axis=0)
        phase_range = numpy.expand_dims(phase_range, axis=axes)
        modulation_range = numpy.expand_dims(modulation_range, axis=axes)
    phase_mask = (phase >= phase_range[..., 0]) & (phase <= phase_range[..., 1])
    modulation_mask = (modulation >= modulation_range[..., 0]) & (modulation <= modulation_range[..., 1])
    return phase_mask & modulation_mask


def segmentate_with_cursors(
    mask: NDArray, cursors_color: NDArray, mean: NDArray, /,
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
