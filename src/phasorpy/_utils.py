"""Private utility functions.

The ``phasorpy._utils`` module provides private auxiliary and convenience
functions.

"""

from __future__ import annotations

__all__: list[str] = [
    'parse_kwargs',
    'update_kwargs',
    'kwargs_notnone',
    'scale_matrix',
    'sort_coordinates',
    'phasor_to_polar_scalar',
    'phasor_from_polar_scalar',
    'circle_line_intersection',
    'circle_circle_intersection',
    'project_phasor_to_line',
    'line_from_components',
    'mask_cursor',
    'mask_segment',
]

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, Sequence, ArrayLike, NDArray

import numpy


def parse_kwargs(
    kwargs: dict[str, Any],
    /,
    *keys: str,
    _del: bool = True,
    **keyvalues: Any,
) -> dict[str, Any]:
    """Return dict with keys from keys|keyvals and values from kwargs|keyvals.

    If `_del` is true (default), existing keys are deleted from `kwargs`.

    >>> kwargs = {'one': 1, 'two': 2, 'four': 4}
    >>> kwargs2 = parse_kwargs(kwargs, 'two', 'three', four=None, five=5)
    >>> kwargs == {'one': 1}
    True
    >>> kwargs2 == {'two': 2, 'four': 4, 'five': 5}
    True

    """
    result = {}
    for key in keys:
        if key in kwargs:
            result[key] = kwargs[key]
            if _del:
                del kwargs[key]
    for key, value in keyvalues.items():
        if key in kwargs:
            result[key] = kwargs[key]
            if _del:
                del kwargs[key]
        else:
            result[key] = value
    return result


def update_kwargs(kwargs: dict[str, Any], /, **keyvalues: Any) -> None:
    """Update dict with keys and values if keys do not already exist.

    >>> kwargs = {'one': 1}
    >>> update_kwargs(kwargs, one=None, two=2)
    >>> kwargs == {'one': 1, 'two': 2}
    True

    """
    for key, value in keyvalues.items():
        if key not in kwargs:
            kwargs[key] = value


def kwargs_notnone(**kwargs: Any) -> dict[str, Any]:
    """Return dict of kwargs which values are not None.

    >>> kwargs_notnone(one=1, none=None)
    {'one': 1}

    """
    return dict(item for item in kwargs.items() if item[1] is not None)


def scale_matrix(factor: float, origin: Sequence[float]) -> NDArray[Any]:
    """Return matrix to scale homogeneous coordinates by factor around origin.

    Parameters
    ----------
    factor: float
        Scale factor.
    origin: (float, float)
        Coordinates of point around which to scale.

    Returns
    -------
    matrix: ndarray
        A 3x3 homogeneous transformation matrix.

    Examples
    --------
    >>> scale_matrix(1.1, (0.0, 0.5))
    array([[1.1, 0, -0],
           [0, 1.1, -0.05],
           [0, 0, 1]])

    """
    mat = numpy.diag((factor, factor, 1.0))
    mat[:2, 2] = origin[:2]
    mat[:2, 2] *= 1.0 - factor
    return mat


def sort_coordinates(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    origin: tuple[float, float] | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return cartesian coordinates sorted counterclockwise around origin.

    Parameters
    ----------
    real, imag : array_like
        Coordinates to be sorted.
    origin : (float, float)
        Coordinates around which to sort by angle.

    Returns
    -------
    real, imag : ndarray
        Coordinates sorted by angle.

    Examples
    --------
    >>> sort_coordinates([0, 1, 2, 3], [0, 1, -1, 0])
    (array([2, 3, 1, 0]), array([-1,  0,  1,  0]))

    """
    x = numpy.asanyarray(real)
    y = numpy.asanyarray(imag)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f'invalid {x.shape=} or {y.shape=}')
    if x.size < 4:
        return x, y
    if origin is None:
        origin = x.mean(), y.mean()
    indices = numpy.argsort(numpy.arctan2(y - origin[1], x - origin[0]))
    return x[indices], y[indices]


def phasor_to_polar_scalar(
    real: float,
    imag: float,
    /,
    *,
    degree: bool = False,
    percent: bool = False,
) -> tuple[float, float]:
    """Return polar from scalar phasor coordinates.

    >>> phasor_to_polar_scalar(1.0, 0.0, degree=True, percent=True)
    (0.0, 100.0)

    """
    phi = math.atan2(imag, real)
    mod = math.hypot(imag, real)
    if degree:
        phi = math.degrees(phi)
    if percent:
        mod *= 100.0
    return phi, mod


def phasor_from_polar_scalar(
    phase: float,
    modulation: float,
    /,
    *,
    degree: bool = False,
    percent: bool = False,
) -> tuple[float, float]:
    """Return phasor from scalar polar coordinates.

    >>> phasor_from_polar_scalar(0.0, 100.0, degree=True, percent=True)
    (1.0, 0.0)

    """
    if degree:
        phase = math.radians(phase)
    if percent:
        modulation /= 100.0
    real = modulation * math.cos(phase)
    imag = modulation * math.sin(phase)
    return real, imag


def circle_circle_intersection(
    x0: float, y0: float, r0: float, x1: float, y1: float, r1: float, /
) -> tuple[tuple[float, float], ...]:
    """Return coordinates of intersection points of two circles.

    >>> circle_circle_intersection(
    ...     0.0, 0.0, math.hypot(0.6, 0.4), 0.6, 0.4, 0.2
    ... )  # doctest: +NUMBER
    ((0.6868, 0.2198), (0.4670, 0.5494))
    >>> circle_circle_intersection(0.0, 0.0, 1.0, 0.6, 0.4, 0.2)
    ()

    """
    dx = x1 - x0
    dy = y1 - y0
    dr = math.hypot(dx, dy)
    ll = (r0 * r0 - r1 * r1 + dr * dr) / (dr + dr)
    dd = r0 * r0 - ll * ll
    if dd < 0.0 or dr < 1e-16:
        return tuple()  # no solution
    hd = math.sqrt(dd) / dr
    ld = ll / dr
    return (
        (ld * dx + hd * dy + x0, ld * dy - hd * dx + y0),
        (ld * dx - hd * dy + x0, ld * dy + hd * dx + y0),
    )


def circle_line_intersection(
    x: float, y: float, r: float, x0: float, y0: float, x1: float, y1: float, /
) -> tuple[tuple[float, float], ...]:
    """Return coordinates of intersection points of circle and line.

    >>> circle_line_intersection(
    ...     0.6, 0.4, 0.2, 0.0, 0.0, 0.6, 0.4
    ... )  # doctest: +NUMBER
    ((0.7664, 0.5109), (0.4335, 0.2890))
    >>> circle_line_intersection(0.6, 0.4, 0.2, 0.0, 0.0, 0.6, 0.1)
    ()

    """
    dx = x1 - x0
    dy = y1 - y0
    dr = dx * dx + dy * dy
    dd = (x0 - x) * (y1 - y) - (x1 - x) * (y0 - y)
    rdd = r * r * dr - dd * dd  # discriminant
    if rdd < 0 or dr < 1e-16:
        return tuple()  # no intersection
    rdd = math.sqrt(rdd)
    sgn = math.copysign
    return (
        (
            x + (dd * dy + sgn(1, dy) * dx * rdd) / dr,
            y + (-dd * dx + abs(dy) * rdd) / dr,
        ),
        (
            x + (dd * dy - sgn(1, dy) * dx * rdd) / dr,
            y + (-dd * dx - abs(dy) * rdd) / dr,
        ),
    )


def project_phasor_to_line(
    real: ArrayLike,
    imag: ArrayLike,
    real_components: ArrayLike,
    imag_components: ArrayLike,
    /,
    *,
    clip: bool = True,
    axis: int = -1,
) -> tuple[NDArray, NDArray]:
    """Return projected phasor coordinates to the line that joins two phasors.

    By default, the points are clipped to the line segment between components
    and the projection is done into the last axis.

    >>> project_phasor_to_line(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... )  # doctest: +NUMBER
    (array([0.592, 0.508, 0.424]), array([0.344, 0.356, 0.368]))

    """
    real = numpy.copy(real)
    imag = numpy.copy(imag)
    real_components = numpy.asarray(real_components)
    imag_components = numpy.asarray(imag_components)
    if real_components.shape != (2,):
        raise ValueError(f'{real_components.shape=} != (2,)')
    if imag_components.shape != (2,):
        raise ValueError(f'{imag_components.shape=} != (2,)')
    unit_vector, distance_between_components = line_from_components(
        real_components, imag_components
    )
    projected_points = numpy.stack((real, imag), axis=axis) - [
        real_components[0],
        imag_components[0],
    ]
    projection_lengths = numpy.dot(projected_points, unit_vector)
    if clip:
        projection_lengths = numpy.clip(
            projection_lengths, 0, distance_between_components
        )
    projected_points = [
        real_components[0],
        imag_components[0],
    ] + numpy.expand_dims(projection_lengths, axis=axis) * unit_vector
    projected_points_real = projected_points[..., 0]
    projected_points_imag = projected_points[..., 1]
    return projected_points_real, projected_points_imag


def line_from_components(
    real_components: ArrayLike,
    imag_components: ArrayLike,
    /,
) -> tuple[NDArray[Any], float]:
    """Return unit vector and distance between components.

    >>> line_from_components([0.2, 0.9], [0.4, 0.3])  # doctest: +NUMBER
    (array([0.99, -0.14]), 0.71)

    """
    real_components = numpy.asarray(real_components)
    imag_components = numpy.asarray(imag_components)
    if real_components.shape != (2,) or imag_components.shape != (2,):
        raise ValueError('components must have shape (2,)')
    line_vector = numpy.array(
        [
            real_components[1] - real_components[0],
            imag_components[1] - imag_components[0],
        ]
    )
    distance_between_components = numpy.linalg.norm(line_vector)
    if math.isclose(distance_between_components, 0, abs_tol=1e-6):
        raise ValueError('components must have different coordinates')
    unit_vector = line_vector / distance_between_components
    return numpy.asarray(unit_vector), float(distance_between_components)


def mask_cursor(
    real: ArrayLike,
    imag: ArrayLike,
    cursor_real: float,
    cursor_imag: float,
    radius: float,
    /,
) -> NDArray[numpy.bool_]:
    """Return mask for phasors within circular cursor radius.

    >>> mask_cursor([0.6, 0.5, 0.4], [0.4, 0.3, 0.2], 0.5, 0.3, 0.05)
    array([False,  True, False])

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    return numpy.hypot(real - cursor_real, imag - cursor_imag) <= radius


def mask_segment(
    real: ArrayLike,
    imag: ArrayLike,
    start_real: float,
    start_imag: float,
    end_real: float,
    end_imag: float,
    distance_threshold: float,
    /,
) -> NDArray[numpy.bool_]:
    """Return mask for phasors within distance threshold from line segment.

    >>> mask_segment([0.6, 0.5, 0.4], [0.4, 0.3, 0.2], 0.2, 0.4, 0.9, 0.3, 0.1)
    array([ True,  True, False])

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    original_shape = real.shape
    real = real.flatten()
    imag = imag.flatten()
    p_vec = numpy.vstack((real, imag)).T
    v_vec = numpy.array([end_real - start_real, end_imag - start_imag])
    t_values = numpy.dot(
        p_vec - numpy.array([start_real, start_imag]), v_vec
    ) / numpy.dot(v_vec, v_vec)
    t_values = numpy.clip(t_values, 0, 1)
    closest_points = (
        numpy.array([start_real, start_imag])
        + t_values[:, numpy.newaxis] * v_vec
    )
    distances = numpy.linalg.norm(p_vec - closest_points, axis=1)
    mask = distances <= distance_threshold
    return mask.reshape(original_shape)
