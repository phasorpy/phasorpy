"""Private auxiliary and convenience functions."""

from __future__ import annotations

__all__: list[str] = [
    'chunk_iter',
    'dilate_coordinates',
    'kwargs_notnone',
    'parse_harmonic',
    'parse_kwargs',
    'parse_signal_axis',
    'phasor_from_polar_scalar',
    'phasor_to_polar_scalar',
    'scale_matrix',
    'sort_coordinates',
    'update_kwargs',
]

import math
import numbers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, Sequence, ArrayLike, Literal, NDArray, Iterator

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


def parse_signal_axis(
    signal: ArrayLike,
    /,
    axis: int | str | None = None,
) -> tuple[int, str]:
    """Return axis over which phasor coordinates are computed.

    The axis parameter is not validated against the signal shape.

    Parameters
    ----------
    signal : array_like
        Image stack.
    axis : int or str, optional
        Axis over which phasor coordinates are computed.
        By default, the 'H' or 'C' axes if `signal` contains such
        dimension names, else the last axis (-1).

    Returns
    -------
    axis : int
        Axis over which phasor coordinates are computed.
    axis_label: str
        Axis label from `signal.dims` if any.

    Raises
    ------
    ValueError
        Axis not found in signal.dims or invalid for signal type.

    Examples
    --------
    >>> parse_signal_axis([])
    (-1, '')
    >>> parse_signal_axis([], 1)
    (1, '')
    >>> class DataArray:
    ...     dims = ('C', 'H', 'Y', 'X')
    ...
    >>> parse_signal_axis(DataArray())
    (1, 'H')
    >>> parse_signal_axis(DataArray(), 'C')
    (0, 'C')
    >>> parse_signal_axis(DataArray(), 1)
    (1, 'H')

    """
    if hasattr(signal, 'dims'):
        assert isinstance(signal.dims, tuple)
        if axis is None:
            for ax in 'HC':
                if ax in signal.dims:
                    return signal.dims.index(ax), ax
            return -1, signal.dims[-1]
        if isinstance(axis, int):
            return axis, signal.dims[axis]
        if axis in signal.dims:
            return signal.dims.index(axis), axis
        raise ValueError(f'{axis=} not found in {signal.dims}')
    if axis is None:
        return -1, ''
    if isinstance(axis, int):
        return axis, ''
    raise ValueError(f'{axis=} not valid for {type(signal)=}')


def parse_harmonic(
    harmonic: int | Sequence[int] | Literal['all'] | str | None,
    harmonic_max: int | None = None,
    /,
) -> tuple[list[int], bool]:
    """Return parsed harmonic parameter.

    This function performs common, but not necessarily all, verifications
    of user-provided `harmonic` parameter.

    Parameters
    ----------
    harmonic : int, sequence of int, 'all', or None
        Harmonic parameter to parse.
    harmonic_max : int, optional
        Maximum value allowed in `hamonic`. Must be one or greater.
        To verify against known number of signal samples,
        pass ``samples // 2``.
        If `harmonic='all'`, a range of harmonics from one to `harmonic_max`
        (included) is returned.

    Returns
    -------
    harmonic : list of int
        Parsed list of harmonics.
    has_harmonic_axis : bool
        False if `harmonic` input parameter is a scalar integer.

    Raises
    ------
    IndexError
        Any element is out of range `[1..harmonic_max]`.
    ValueError
        Elements are not unique.
        Harmonic is empty.
        String input is not 'all'.
        `harmonic_max` is smaller than 1.
    TypeError
        Any element is not an integer.
        `harmonic` is `'all'` and `harmonic_max` is None.

    """
    if harmonic_max is not None and harmonic_max < 1:
        raise ValueError(f'{harmonic_max=} < 1')

    if harmonic is None:
        return [1], False

    if isinstance(harmonic, (int, numbers.Integral)):
        if harmonic < 1 or (
            harmonic_max is not None and harmonic > harmonic_max
        ):
            raise IndexError(f'{harmonic=} out of range [1..{harmonic_max}]')
        return [int(harmonic)], False

    if isinstance(harmonic, str):
        if harmonic == 'all':
            if harmonic_max is None:
                raise TypeError(
                    f'maximum harmonic must be specified for {harmonic=!r}'
                )
            return list(range(1, harmonic_max + 1)), True
        raise ValueError(f'{harmonic=!r} is not a valid harmonic')

    h = numpy.atleast_1d(numpy.asarray(harmonic))
    if h.size == 0:
        raise ValueError(f'{harmonic=} is empty')
    if h.dtype.kind not in 'iu' or h.ndim != 1:
        raise TypeError(f'{harmonic=} element not an integer')
    if numpy.any(h < 1):
        raise IndexError(f'{harmonic=} element < 1')
    if harmonic_max is not None and numpy.any(h > harmonic_max):
        raise IndexError(f'{harmonic=} element > {harmonic_max}]')
    if numpy.unique(h).size != h.size:
        raise ValueError(f'{harmonic=} elements must be unique')
    return h.tolist(), True


def chunk_iter(
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    /,
    dims: Sequence[str] | None = None,
    *,
    pattern: str | None = None,
    squeeze: bool = False,
    use_index: bool = False,
) -> Iterator[tuple[tuple[int | slice, ...], str, bool]]:
    """Yield indices and labels of chunks from ndarray's shape.

    Parameters
    ----------
    shape : tuple of int
        Shape of C-order ndarray to chunk.
    chunk_shape : tuple of int
        Shape of chunks in the most significant dimensions.
    dims : sequence of str, optional
        Labels for each axis in shape if `pattern` is None.
    pattern : str, optional
        String to format chunk indices.
        If None, use ``_[{dims[index]}{chunk_index[index]}]`` for each axis.
    squeeze : bool
        If true, do not include length-1 chunked dimensions in label
        unless dimensions are part of `chunk_shape`.
        Applies only if `pattern` is None.
    use_index : bool
        If true, use indices of chunks in `shape` instead of chunk indices to
        format pattern.

    Yields
    ------
    index : tuple of int or slice
        Indices of chunk in ndarray.
    label : str
        Pattern formatted with chunk indices.
    cropped : bool
        True if chunk exceeds any border of ndarray.
        Indexing ndarray with `index` will yield a slice smaller than
        `chunk_shape`.

    Examples
    --------

    >>> list(chunk_iter((2, 2), (2,), pattern='Y{}'))
    [((0, slice(0, 2, 1)), 'Y0', False), ((1, slice(0, 2, 1)), 'Y1', False)]

    Chunk a four-dimensional image stack into 2x2 sized image tiles:

    >>> stack = numpy.zeros((2, 3, 4, 5))
    >>> for index, label, cropped in chunk_iter(stack.shape, (2, 2)):
    ...     chunk = stack[index]
    ...

    """
    ndim = len(shape)

    sep = '_'
    if dims is None:
        dims = sep * ndim
        sep = ''
    elif ndim != len(dims):
        raise ValueError(f'{len(shape)=} != {len(dims)=}')

    if pattern is not None:
        try:
            pattern.format(*shape)
        except Exception as exc:
            raise ValueError('pattern cannot be formatted') from exc

    # number of high dimensions not included in chaunk_shape
    hdim = ndim - len(chunk_shape)
    if hdim < 0:
        raise ValueError(f'{len(shape)=} < {len(chunk_shape)=}')
    if hdim > 0:
        # prepend length-1 dimensions
        chunk_shape = ((1,) * hdim) + chunk_shape

    chunked_shape = []
    pattern_list = []
    for i, (size, chunk_size, ax) in enumerate(zip(shape, chunk_shape, dims)):
        if size <= 0:
            raise ValueError('shape must contain positive sizes')
        if chunk_size <= 0:
            raise ValueError('chunk_shape must contain positive sizes')
        div, mod = divmod(size, chunk_size)
        chunked_shape.append(div + 1 if mod else div)

        if not squeeze or chunked_shape[-1] > 1:
            if use_index:
                digits = int(math.log10(size)) + 1
            else:
                digits = int(math.log10(chunked_shape[-1])) + 1
            pattern_list.append(f'{sep}{ax}{{{i}:0{digits}d}}')

    if pattern is None:
        pattern = ''.join(pattern_list)

    chunk_index: tuple[int, ...]
    for chunk_index in numpy.ndindex(tuple(chunked_shape)):
        index: tuple[int | slice, ...] = tuple(
            (
                chunk_index[i]
                if i < hdim
                else slice(
                    chunk_index[i] * chunk_shape[i],
                    (chunk_index[i] + 1) * chunk_shape[i],
                    1,
                )
            )
            for i in range(ndim)
        )
        if use_index:
            format_index = tuple(
                chunk_index[i] * chunk_shape[i] for i in range(ndim)
            )
        else:
            format_index = chunk_index
        yield (
            index,
            pattern.format(*format_index),
            any(
                (chunk_index[i] + 1) * chunk_shape[i] > shape[i]
                for i in range(ndim)
            ),
        )
