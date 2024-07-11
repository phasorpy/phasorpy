"""Select regions of interest (cursors) from phasor coordinates.

    The ``phasorpy.cursors`` module provides functions to:

- create masks for regions of interests in the phasor space:

  - :py:func:`mask_from_circular_cursor`

- create masks for regions of interests in the polar space:

  - :py:func:`mask_from_polar_cursor`

- create a pseudo-color array of average signal from cursor regions:

  - :py:func:`pseudo_color`

"""

from __future__ import annotations

__all__ = [
    'mask_from_circular_cursor',
    'mask_from_polar_cursor',
    'pseudo_color',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import (
        Any,
        ArrayLike,
        NDArray,
    )

import numpy

from phasorpy.color import CATEGORICAL


def mask_from_circular_cursor(
    real: ArrayLike,
    imag: ArrayLike,
    center_real: ArrayLike,
    center_imag: ArrayLike,
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
    center_real : array_like, shape (n,)
        Real coordinates of circle centers.
    center_imag : array_like, shape (n,)
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
        The array shapes of `real` and `imag`, or `center_real` and
        `center_imag` do not match.
        Center's coordinates and/or the radii are not one-dimensional.
        Any of the radii is negative.

    Examples
    --------
    Create mask for a single circular cursor:

    >>> mask_from_circular_cursor([0.0, 0.0], [0.0, 0.5], 0.0, 0.5, radius=0.1)
    array([[False],
           [ True]])

    Create masks for two circular cursors with different radius:

    >>> mask_from_circular_cursor(
    ...     [0.0, 1.0], [0.0, 0.5], [0.0, 1.0], [0.0, 0.4], radius=[0.1, 0.05]
    ... )
    array([[ True, False],
           [False, False]])

    """
    real = numpy.atleast_1d(real)
    imag = numpy.atleast_1d(imag)
    center_real = numpy.atleast_1d(center_real)
    center_imag = numpy.atleast_1d(center_imag)
    radius = numpy.atleast_1d(radius)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if center_real.shape != center_imag.shape:
        raise ValueError(f'{center_real.shape=} != {center_imag.shape=}')
    if center_real.ndim != 1:
        raise ValueError(
            'center coordinates are not one-dimensional: '
            f'{center_real.ndim} dimensions found'
        )
    if radius.ndim != 1:
        raise ValueError(
            f'radius must be one dimensional: {radius.ndim} dimensions found'
        )
    if numpy.min(radius) < 0:
        raise ValueError('all radii must be positive')

    distance = numpy.square(
        real[..., numpy.newaxis] - center_real
    ) + numpy.square(imag[..., numpy.newaxis] - center_imag)
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

    >>> mask_from_polar_cursor([5, 100], [0.2, 0.4], [50, 150], [0.2, 0.5])
    array([False,  True])

    Create masks for two polar cursors with different ranges:

    >>> mask_from_polar_cursor(
    ...     [5, 100], [0.2, 0.4], [[50, 150], [0, 270]], [[0.2, 0.5], [0, 0.3]]
    ... )
    array([[False,  True],
           [ True, False]])

    """
    phase = numpy.atleast_1d(phase)
    modulation = numpy.atleast_1d(modulation)
    phase_range = numpy.atleast_1d(phase_range)
    modulation_range = numpy.atleast_1d(modulation_range)

    if phase.shape != modulation.shape:
        raise ValueError(f'{phase.shape=} != {modulation.shape=}')
    if phase_range.shape != modulation_range.shape:
        raise ValueError(f'{phase_range.shape=} != {modulation_range.shape=}')
    if phase_range.shape[-1] != 2:
        raise ValueError(
            'The last dimension of the range must be 2: '
            f'{phase_range.shape[-1]} found'
        )

    if numpy.ndim(phase_range) > 1:
        axes = tuple(range(1, numpy.ndim(phase) + 1))
        phase = numpy.expand_dims(phase, axis=0)
        modulation = numpy.expand_dims(modulation, axis=0)
        phase_range = numpy.expand_dims(phase_range, axis=axes)
        modulation_range = numpy.expand_dims(modulation_range, axis=axes)
    phase_mask = (phase >= phase_range[..., 0]) & (
        phase <= phase_range[..., 1]
    )
    modulation_mask = (modulation >= modulation_range[..., 0]) & (
        modulation <= modulation_range[..., 1]
    )
    return phase_mask & modulation_mask


def pseudo_color(
    mean: NDArray,
    masks: NDArray,
    /,
    *,
    colors: ArrayLike = CATEGORICAL,
    mask_axis: int = -1,
) -> NDArray[Any]:
    """Return the average of signal colored for each cursor.

    Parameters
    ----------
    - mean: NDArray
        Average of signal (zero harmonic).
    - masks: NDArray
        Masks for each cursor.
    - colors: array_like, optional, shape (..., 3)
        Colors assigned to each cursor. Last dimension must contain the
        RGB values. Default is `CATEGORICAL` from ``phasorpy.color`` module.
    - mask_axis: int, optional
        Axis along which the masks are applied.

    Returns
    -------
    - pseudocolor: ndarray
        Average of signal replaced by colors for each cursor.

    Raises
    ------
    ValueError
        'mean' has more than 2 dimensions.
        'colors' last dimension is not of size 3 (must contain RGB values).
        'masks' shape (except for the mask axis) does not match 'mean' shape.

    Example
    -------
    Pseudo-color for a single mask.

    >>> pseudo_color([0, 1, 2], [True, False, True])  # doctest: +NUMBER
    array([[0.825, 0.095, 0.127], [0, 0, 0], [0.825, 0.095, 0.127]])

    Pseudo-color for two masks.

    >>> pseudo_color(
    ...     [0, 1], [[True, False], [False, True]]
    ... )  # doctest: +NUMBER
    array([[0.825, 0.095, 0.127], [0.095, 0.413, 1]])

    """
    mean = numpy.atleast_1d(mean)
    masks = numpy.atleast_1d(masks)
    colors = numpy.asarray(colors)

    if mean.ndim > 2:
        raise ValueError('only 1D and 2D mean arrays are supported')
    if colors.shape[-1] != 3:
        raise ValueError(f'{colors.shape[-1]=} != 3')

    pseudocolor = numpy.zeros(mean.shape + (3,))
    if mean.ndim == masks.ndim:
        if mean.shape != masks.shape:
            raise ValueError(f'{mean.shape=} != {masks.shape=}')
        pseudocolor[masks] = colors[0]
    else:
        if mean.shape != masks.shape[mask_axis:]:
            raise ValueError(f'{mean.shape=} != {masks.shape[mask_axis:]=}')
        for i, mask in enumerate(numpy.rollaxis(masks, mask_axis)):
            pseudocolor[mask] = colors[i]

    return pseudocolor
