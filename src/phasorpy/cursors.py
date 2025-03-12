"""Select regions of interest (cursors) from phasor coordinates.

The ``phasorpy.cursors`` module provides functions to:

- create masks for regions of interests in the phasor space:

  - :py:func:`mask_from_circular_cursor`
  - :py:func:`mask_from_elliptic_cursor`
  - :py:func:`mask_from_polar_cursor`

- create pseudo-color image from average signal and cursor masks:

  - :py:func:`pseudo_color`

"""

from __future__ import annotations

__all__ = [
    'mask_from_circular_cursor',
    'mask_from_elliptic_cursor',
    'mask_from_polar_cursor',
    'pseudo_color',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import (
        ArrayLike,
        NDArray,
    )

import numpy

from phasorpy.color import CATEGORICAL

from ._phasorpy import (
    _blend_normal,
    _blend_overlay,
    _is_inside_circle,
    _is_inside_ellipse_,
    _is_inside_polar_rectangle,
)


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
    radius : array_like, optional, shape (n,)
        Radii of circles.

    Returns
    -------
    masks : ndarray
        Boolean array of shape `(n, *real.shape)`.
        The first dimension is omitted if `center_*` and `radius` are scalars.
        Values are True if phasor coordinates are inside circular cursor,
        else False.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag` do not match.
        The array shapes of `center_*` or `radius` have more than
        one dimension.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_cursors.py`

    Examples
    --------
    Create mask for a single circular cursor:

    >>> mask_from_circular_cursor([0.2, 0.5], [0.4, 0.5], 0.2, 0.4, radius=0.1)
    array([ True, False])

    Create masks for two circular cursors with different radius:

    >>> mask_from_circular_cursor(
    ...     [0.2, 0.5], [0.4, 0.5], [0.2, 0.5], [0.4, 0.5], radius=[0.1, 0.05]
    ... )
    array([[ True, False],
           [False,  True]])

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    center_real = numpy.asarray(center_real)
    center_imag = numpy.asarray(center_imag)
    radius = numpy.asarray(radius)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if center_real.ndim > 1 or center_imag.ndim > 1 or radius.ndim > 1:
        raise ValueError(
            f'{center_real.ndim=}, {center_imag.ndim=}, or {radius.ndim=} > 1'
        )

    moveaxis = False
    if real.ndim > 0 and (
        center_real.ndim > 0 or center_imag.ndim > 0 or radius.ndim > 0
    ):
        moveaxis = True
        real = numpy.expand_dims(real, axis=-1)
        imag = numpy.expand_dims(imag, axis=-1)

    mask = _is_inside_circle(real, imag, center_real, center_imag, radius)
    if moveaxis:
        mask = numpy.moveaxis(mask, -1, 0)
    return mask.astype(numpy.bool_)  # type: ignore[no-any-return]


def mask_from_elliptic_cursor(
    real: ArrayLike,
    imag: ArrayLike,
    center_real: ArrayLike,
    center_imag: ArrayLike,
    /,
    *,
    radius: ArrayLike = 0.05,
    radius_minor: ArrayLike | None = None,
    angle: ArrayLike | None = None,
    align_semicircle: bool = False,
) -> NDArray[numpy.bool_]:
    """Return masks for elliptic cursors of phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    center_real : array_like, shape (n,)
        Real coordinates of ellipses centers.
    center_imag : array_like, shape (n,)
        Imaginary coordinates of ellipses centers.
    radius : array_like, optional, shape (n,)
        Radii of ellipses along semi-major axis.
    radius_minor : array_like, optional, shape (n,)
        Radii of ellipses along semi-minor axis.
        By default, the ellipses are circular.
    angle : array_like, optional, shape (n,)
        Rotation angles of semi-major axes of ellipses in radians.
        By default, the ellipses are automatically oriented depending on
        `align_semicircle`.
    align_semicircle : bool, optional
        Determines orientation of ellipses if `angle` is not provided.
        If true, align the minor axes of the ellipses with the closest tangent
        on the universal semicircle, else the unit circle (default).

    Returns
    -------
    masks : ndarray
        Boolean array of shape `(n, *real.shape)`.
        The first dimension is omitted if `center*`, `radius*`, and `angle`
        are scalars.
        Values are True if phasor coordinates are inside elliptic cursor,
        else False.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag` do not match.
        The array shapes of `center*`, `radius*`, or `angle` have more than
        one dimension.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_cursors.py`

    Examples
    --------
    Create mask for a single elliptic cursor:

    >>> mask_from_elliptic_cursor([0.2, 0.5], [0.4, 0.5], 0.2, 0.4, radius=0.1)
    array([ True, False])

    Create masks for two elliptic cursors with different radii:

    >>> mask_from_elliptic_cursor(
    ...     [0.2, 0.5],
    ...     [0.4, 0.5],
    ...     [0.2, 0.5],
    ...     [0.4, 0.5],
    ...     radius=[0.1, 0.05],
    ...     radius_minor=[0.15, 0.1],
    ...     angle=[math.pi, math.pi / 2],
    ... )
    array([[ True, False],
           [False,  True]])

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    center_real = numpy.asarray(center_real)
    center_imag = numpy.asarray(center_imag)
    radius_a = numpy.asarray(radius)
    if radius_minor is None:
        radius_b = radius_a  # circular by default
        angle = 0.0
    else:
        radius_b = numpy.asarray(radius_minor)
    if angle is None:
        # TODO: vectorize align_semicircle?
        if align_semicircle:
            angle = numpy.arctan2(center_imag, center_real - 0.5)
        else:
            angle = numpy.arctan2(center_imag, center_real)

    angle_sin = numpy.sin(angle)
    angle_cos = numpy.cos(angle)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if (
        center_real.ndim > 1
        or center_imag.ndim > 1
        or radius_a.ndim > 1
        or radius_b.ndim > 1
        or angle_sin.ndim > 1
    ):
        raise ValueError(
            f'{center_real.ndim=}, {center_imag.ndim=}, '
            f'radius.ndim={radius_a.ndim}, '
            f'radius_minor.ndim={radius_b.ndim}, or '
            f'angle.ndim={angle_sin.ndim}, > 1'
        )

    moveaxis = False
    if real.ndim > 0 and (
        center_real.ndim > 0
        or center_imag.ndim > 0
        or radius_a.ndim > 0
        or radius_b.ndim > 0
        or angle_sin.ndim > 0
    ):
        moveaxis = True
        real = numpy.expand_dims(real, axis=-1)
        imag = numpy.expand_dims(imag, axis=-1)

    mask = _is_inside_ellipse_(
        real,
        imag,
        center_real,
        center_imag,
        radius_a,
        radius_b,
        angle_sin,
        angle_cos,
    )
    if moveaxis:
        mask = numpy.moveaxis(mask, -1, 0)
    return mask.astype(numpy.bool_)  # type: ignore[no-any-return]


def mask_from_polar_cursor(
    real: ArrayLike,
    imag: ArrayLike,
    phase_min: ArrayLike,
    phase_max: ArrayLike,
    modulation_min: ArrayLike,
    modulation_max: ArrayLike,
    /,
) -> NDArray[numpy.bool_]:
    """Return mask for polar cursor of polar coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    phase_min : array_like, shape (n,)
        Lower bound of angular range of cursors in radians.
        Values should be in range [-pi, pi].
    phase_max : array_like, shape (n,)
        Upper bound of angular range of cursors in radians.
        Values should be in range [-pi, pi].
    modulation_min : array_like, shape (n,)
        Lower bound of radial range of cursors.
    modulation_max : array_like, shape (n,)
        Upper bound of radial range of cursors.

    Returns
    -------
    masks : ndarray
        Boolean array of shape `(n, *real.shape)`.
        The first dimension is omitted if `phase_*` and `modulation_*`
        are scalars.
        Values are True if phasor coordinates are inside polar range cursor,
        else False.

    Raises
    ------
    ValueError
        The array shapes of `phase` and `modulation`, or `phase_range` and
        `modulation_range` do not match.
        The array shapes of `phase_*` or `modulation_*` have more than
        one dimension.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_cursors.py`

    Example
    -------
    Create mask from a single polar cursor:

    >>> mask_from_polar_cursor([0.2, 0.5], [0.4, 0.5], 1.1, 1.2, 0.4, 0.5)
    array([ True, False])

    Create masks for two polar cursors with different ranges:

    >>> mask_from_polar_cursor(
    ...     [0.2, 0.5],
    ...     [0.4, 0.5],
    ...     [1.1, 0.7],
    ...     [1.2, 0.8],
    ...     [0.4, 0.7],
    ...     [0.5, 0.8],
    ... )
    array([[ True, False],
           [False,  True]])

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    phase_min = numpy.asarray(phase_min)
    phase_max = numpy.asarray(phase_max)
    modulation_min = numpy.asarray(modulation_min)
    modulation_max = numpy.asarray(modulation_max)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if (
        phase_min.ndim > 1
        or phase_max.ndim > 1
        or modulation_min.ndim > 1
        or modulation_max.ndim > 1
    ):
        raise ValueError(
            f'{phase_min.ndim=}, {phase_max.ndim=}, '
            f'{modulation_min.ndim=}, or {modulation_max.ndim=} > 1'
        )
    # TODO: check if angles are in range [-pi and pi]

    moveaxis = False
    if real.ndim > 0 and (
        phase_min.ndim > 0
        or phase_max.ndim > 0
        or modulation_min.ndim > 0
        or modulation_max.ndim > 0
    ):
        moveaxis = True
        real = numpy.expand_dims(real, axis=-1)
        imag = numpy.expand_dims(imag, axis=-1)

    mask = _is_inside_polar_rectangle(
        real, imag, phase_min, phase_max, modulation_min, modulation_max
    )
    if moveaxis:
        mask = numpy.moveaxis(mask, -1, 0)
    return mask.astype(numpy.bool_)  # type: ignore[no-any-return]


def pseudo_color(
    *masks: ArrayLike,
    intensity: ArrayLike | None = None,
    colors: ArrayLike | None = None,
    vmin: float | None = 0.0,
    vmax: float | None = None,
) -> NDArray[numpy.float32]:
    """Return pseudo-colored image from cursor masks.

    Parameters
    ----------
    *masks : array_like
        Boolean mask for each cursor.
    intensity : array_like, optional
        Intensity used as base layer to blend cursor colors in "overlay" mode.
        If None, cursor masks are blended using "screen" mode.
    vmin : float, optional
        Minimum value to normalize `intensity`.
        If None, the minimum value of `intensity` is used.
    vmax : float, optional
        Maximum value to normalize `intensity`.
        If None, the maximum value of `intensity` is used.
    colors : array_like, optional, shape (N, 3)
        RGB colors assigned to each cursor.
        The last dimension contains the normalized RGB floating point values.
        The default is :py:data:`phasorpy.color.CATEGORICAL`.

    Returns
    -------
    ndarray
        Pseudo-colored image of shape ``(*masks[0].shape, 3)``.

    Raises
    ------
    ValueError
        `colors` is not a (n, 3) shaped floating point array.
        The shapes of `masks` or `mean` cannot broadcast.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_cursors.py`

    Example
    -------
    Create pseudo-color image from single mask:

    >>> pseudo_color([True, False, True])  # doctest: +NUMBER
    array([[0.8254, 0.09524, 0.127],
           [0, 0, 0],
           [0.8254, 0.09524, 0.127]]...)

    Create pseudo-color image from two masks and intensity image:

    >>> pseudo_color(
    ...     [True, False], [False, True], intensity=[0.4, 0.6], vmax=1.0
    ... )  # doctest: +NUMBER
    array([[0.6603, 0.07619, 0.1016],
           [0.2762, 0.5302, 1]]...)

    """
    if len(masks) == 0:
        raise TypeError(
            "pseudo_color() missing 1 required positional argument: 'masks'"
        )

    if colors is None:
        colors = CATEGORICAL
    else:
        colors = numpy.asarray(colors)
        if colors.ndim != 2:
            raise ValueError(f'{colors.ndim=} != 2')
        if colors.shape[-1] != 3:
            raise ValueError(f'{colors.shape[-1]=} != 3')
        if colors.dtype.kind != 'f':
            raise ValueError('colors is not a floating point array')
    # TODO: add support for matplotlib colors

    shape = numpy.asarray(masks[0]).shape

    if intensity is not None:
        # normalize intensity to range [0, 1]
        intensity = numpy.array(
            intensity, dtype=numpy.float32, ndmin=1, copy=True
        )
        if intensity.size > 1:
            if vmin is None:
                vmin = numpy.nanmin(intensity)
            if vmax is None:
                vmax = numpy.nanmax(intensity)
            if vmin != 0.0:
                intensity -= vmin
            scale = vmax - vmin
            if scale != 0.0 and scale != 1.0:
                intensity /= scale
        numpy.clip(intensity, 0.0, 1.0, out=intensity)
        if intensity.shape == shape:
            intensity = intensity[..., numpy.newaxis]
        pseudocolor = numpy.full((*shape, 3), intensity, dtype=numpy.float32)
    else:
        pseudocolor = numpy.zeros((*shape, 3), dtype=numpy.float32)

    # TODO: support intensity or RGB input in addition to masks
    blend = numpy.empty_like(pseudocolor)
    for i, mask_ in enumerate(masks):
        mask = numpy.asarray(mask_)
        if mask.shape != shape:
            raise ValueError(f'masks[{i}].shape={mask.shape} != {shape}')
        blend.fill(numpy.nan)
        blend[mask] = colors[i]
        if intensity is None:
            # TODO: replace by _blend_screen?
            _blend_normal(pseudocolor, blend, out=pseudocolor)
        else:
            _blend_overlay(pseudocolor, blend, out=pseudocolor)

    pseudocolor.clip(0.0, 1.0, out=pseudocolor)
    return pseudocolor
