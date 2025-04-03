"""Select regions of interest (cursors) in phasor coordinates."""

from __future__ import annotations

__all__ = [
    'phasor_mask_circular',
    'phasor_mask_elliptic',
    'phasor_mask_polar',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import (
        ArrayLike,
        NDArray,
    )

import numpy

from ._phasorpy import (
    _is_inside_circle,
    _is_inside_ellipse_,
    _is_inside_polar_rectangle,
)


def phasor_mask_circular(
    real: ArrayLike,
    imag: ArrayLike,
    center_real: ArrayLike,
    center_imag: ArrayLike,
    /,
    *,
    radius: ArrayLike = 0.05,
) -> NDArray[numpy.bool_]:
    """Return mask for phasor coordinates within circle(s).

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
        Values are True if phasor coordinates are inside circle, else False.

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
    Create mask for a single circle:

    >>> phasor_mask_circular([0.2, 0.5], [0.4, 0.5], 0.2, 0.4, radius=0.1)
    array([ True, False])

    Create masks for two circles with different radius:

    >>> phasor_mask_circular(
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


def phasor_mask_elliptic(
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
    """Return mask for phasor coordinates within ellipse(s).

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
        Values are True if phasor coordinates are inside ellipse, else False.

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
    Create mask for a single ellipse:

    >>> phasor_mask_elliptic([0.2, 0.5], [0.4, 0.5], 0.2, 0.4, radius=0.1)
    array([ True, False])

    Create masks for two ellipses with different radii:

    >>> phasor_mask_elliptic(
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


def phasor_mask_polar(
    real: ArrayLike,
    imag: ArrayLike,
    phase_min: ArrayLike,
    phase_max: ArrayLike,
    modulation_min: ArrayLike,
    modulation_max: ArrayLike,
    /,
) -> NDArray[numpy.bool_]:
    """Return mask for polar range of polar coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    phase_min : array_like, shape (n,)
        Lower bound of angular range in radians.
        Values should be in range [-pi, pi].
    phase_max : array_like, shape (n,)
        Upper bound of angular range in radians.
        Values should be in range [-pi, pi].
    modulation_min : array_like, shape (n,)
        Lower bound of radial range.
    modulation_max : array_like, shape (n,)
        Upper bound of radial range.

    Returns
    -------
    masks : ndarray
        Boolean array of shape `(n, *real.shape)`.
        The first dimension is omitted if `phase_*` and `modulation_*`
        are scalars.
        Values are True if phasor coordinates are inside polar range,
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
    Create mask from a single polar range:

    >>> phasor_mask_polar([0.2, 0.5], [0.4, 0.5], 1.1, 1.2, 0.4, 0.5)
    array([ True, False])

    Create masks for two polar ranges:

    >>> phasor_mask_polar(
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
