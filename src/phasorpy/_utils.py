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
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
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
    indices : ndarray
        Indices used to reorder coordinates.

    Examples
    --------
    >>> sort_coordinates([0, 1, 2, 3], [0, 1, -1, 0])
    (array([2, 3, 1, 0]), array([-1,  0,  1,  0]), array([2, 3, 1, 0]...))

    """
    x, y = numpy.atleast_1d(real, imag)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError(f'invalid {x.shape=} or {y.shape=}')
    if x.size < 4:
        return x, y, numpy.arange(x.size)
    if origin is None:
        origin = x.mean(), y.mean()
    indices = numpy.argsort(numpy.arctan2(y - origin[1], x - origin[0]))
    return x[indices], y[indices], indices


def dilate_coordinates(
    real: ArrayLike,
    imag: ArrayLike,
    offset: float,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return dilated coordinates.

    Parameters
    ----------
    real, imag : array_like
        Coordinates of convex hull, sorted by angle.
    offset : float
        Amount by which to dilate coordinates.

    Returns
    -------
    real, imag : ndarray
        Coordinates dilated by offset.

    Examples
    --------
    >>> dilate_coordinates([2, 3, 1, 0], [-1, 0, 1, 0], 0.05)
    (array([2.022, 3.05, 0.9776, -0.05]), array([-1.045, 0, 1.045, 0]))

    """
    x = numpy.asanyarray(real, dtype=numpy.float64)
    y = numpy.asanyarray(imag, dtype=numpy.float64)
    if x.ndim != 1 or x.shape != y.shape or x.size < 1:
        raise ValueError(f'invalid {x.shape=} or {y.shape=}')
    if x.size > 1:
        dx = numpy.diff(numpy.diff(x, prepend=x[-1], append=x[0]))
        dy = numpy.diff(numpy.diff(y, prepend=y[-1], append=y[0]))
    else:
        # TODO: this assumes coordinate on universal semicircle
        dx = numpy.diff(x, append=0.5)
        dy = numpy.diff(y, append=0.0)
    s = numpy.hypot(dx, dy)
    dx /= s
    dx *= -offset
    dx += x
    dy /= s
    dy *= -offset
    dy += y
    return dx, dy


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
