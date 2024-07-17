"""Select regions of interest (cursors) from phasor coordinates.

The ``phasorpy.cursors`` module provides functions to:

- create masks for regions of interests in the phasor space:

  - :py:func:`mask_from_circular_cursor`
  - :py:func:`mask_from_polar_cursor`

- create pseudo-color image from average signal and cursor masks:

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
    axis: int = 0,
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
    axis : int, optional
        Axis along which the masks are returned. Default is 0.

    Returns
    -------
    masks : ndarray
        Phasor coordinates masked for each circular cursor.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag`, or `center_real` and
        `center_imag` do not match.
        The axis is out of bounds.

    See Also
    --------
    :ref:`sphx_glr_tutorials_phasorpy_cursors.py`

    Examples
    --------
    Create mask for a single circular cursor:

    >>> mask_from_circular_cursor([0.0, 0.0], [0.0, 0.5], 0.0, 0.5, radius=0.1)
    array([False,  True])

    Create masks for two circular cursors with different radius:

    >>> mask_from_circular_cursor(
    ...     [0.0, 1.0], [0.0, 0.5], [0.0, 1.0], [0.0, 0.4], radius=[0.1, 0.05]
    ... )
    array([[ True, False],
           [False, False]])

    """
    real = numpy.atleast_1d(real)
    imag = numpy.atleast_1d(imag)
    center_real = numpy.asarray(center_real)
    center_imag = numpy.asarray(center_imag)
    radius = numpy.asarray(radius)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if center_real.shape != center_imag.shape:
        raise ValueError(f'{center_real.shape=} != {center_imag.shape=}')

    if numpy.ndim(center_real) or numpy.ndim(radius) > 0:
        real = numpy.expand_dims(real, axis=-1)
        imag = numpy.expand_dims(imag, axis=-1)
    distance = numpy.square(real - center_real) + numpy.square(
        imag - center_imag
    )
    masks = distance <= numpy.square(radius)
    if numpy.ndim(center_real) or numpy.ndim(radius) > 1:
        max_axis = masks.ndim - 1
        if not (-max_axis - 1 <= axis <= max_axis):
            raise ValueError(
                f'Invalid axis {axis=}.'
                f'Must be between {-max_axis - 1} and {max_axis}.'
            )
        masks = numpy.moveaxis(masks, -1, axis)
    # TODO: handle radius dimension > 1
    return masks


def mask_from_polar_cursor(
    phase: ArrayLike,
    modulation: ArrayLike,
    phase_range: ArrayLike,
    modulation_range: ArrayLike,
    /,
    *,
    axis: int = 0,
) -> NDArray[numpy.bool_]:
    """Return mask for polar cursor of polar coordinates.

    Parameters
    ----------
    phase: array_like
        Angular component of polar coordinates in radians.
    modulation: array_like
        Radial component of polar coordinates.
    phase_range: array_like, shape (..., 2)
        Angular range of the cursors in radians.
        The start and end of the range must be in the last dimension.
    modulation_range: array_like, shape (..., 2)
        Radial range of the cursors.
        The start and end of the range must be in the last dimension.
    axis: int, optional
        Axis along which the masks are returned. Default is 0.

    Returns
    -------
    masks: ndarray
        Polar coordinates masked for each polar cursor.

    Raises
    ------
    ValueError
        The array shapes of `phase` and `modulation`, or `phase_range` and
        `modulation_range` do not match.
        The last dimension of `phase_range` and `modulation_range` is not 2.
        The axis is out of bounds.

    See Also
    --------
    :ref:`sphx_glr_tutorials_phasorpy_cursors.py`

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
    phase = numpy.asarray(phase)
    modulation = numpy.asarray(modulation)
    phase_range = numpy.atleast_1d(phase_range)
    modulation_range = numpy.atleast_1d(modulation_range)

    if phase.shape != modulation.shape:
        raise ValueError(f'{phase.shape=} != {modulation.shape=}')
    if phase_range.shape != modulation_range.shape:
        raise ValueError(f'{phase_range.shape=} != {modulation_range.shape=}')
    if phase_range.shape[-1] != 2:
        raise ValueError(f'{phase_range.shape[-1]=} != 2')
    # TODO: check if angles are between -pi and pi

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
    masks = phase_mask & modulation_mask
    if numpy.ndim(phase_range) > 1:
        max_axis = masks.ndim - 1
        if not (-max_axis - 1 <= axis <= max_axis):
            raise ValueError(
                f'Invalid axis {axis=}.'
                f'Must be between {-max_axis - 1} and {max_axis}.'
            )
        masks = numpy.moveaxis(masks, 0, axis)
    return masks


def pseudo_color(
    mean: NDArray,
    masks: NDArray,
    /,
    *,
    colors: ArrayLike | None = None,
    axis: int = 0,
) -> NDArray[Any]:
    """Return the average of signal pseudo-colored for each cursor.

    Parameters
    ----------
    mean: ndarray
        Average of signal (zero harmonic).
    masks: ndarray
        Masks for each cursor.
    colors: array_like, optional, shape (N, 3)
        Colors assigned to each cursor. Last dimension must contain the
        RGB values. Default is :py:data:`phasorpy.color.CATEGORICAL`.
    axis: int, optional
        Axis with masks. Default is 0.

    Returns
    -------
    pseudocolor: ndarray
        Average of signal replaced by colors for each cursor.

    Raises
    ------
    ValueError
        The `colors` array is not 2 dimensional and/or the last dimension is
        not of size 3 (must contain RGB values).
        The `masks` shape along axis does not match `mean` shape.
        Axis is out of bounds.

    See Also
    --------
    :ref:`sphx_glr_tutorials_phasorpy_cursors.py`

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
    mean = numpy.asarray(mean)
    masks = numpy.atleast_1d(masks)
    if colors is None:
        colors = CATEGORICAL
    else:
        colors = numpy.asarray(colors)
        if colors.ndim != 2:
            raise ValueError(f'{colors.ndim=} != 2')
        if colors.shape[-1] != 3:
            raise ValueError(f'{colors.shape[-1]=} != 3')
    if not (-masks.ndim <= axis < masks.ndim):
        raise ValueError(
            f'Invalid {axis=}.'
            f'Must be between {-masks.ndim} and {masks.ndim - 1}.'
        )
    # TODO: add support for matplotlib colors

    pseudocolor = numpy.zeros(mean.shape + (3,))
    if mean.ndim == masks.ndim:
        if mean.shape != masks.shape:
            raise ValueError(f'{mean.shape=} != {masks.shape=}')
        pseudocolor[masks] = colors[0]
    else:
        if mean.shape != masks[axis].shape:
            raise ValueError(
                'shapes of the mask along axis must match `mean` shape'
            )
        for i, mask in enumerate(numpy.rollaxis(masks, axis)):
            pseudocolor[mask] = colors[i]

    return pseudocolor
