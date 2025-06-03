"""Private auxiliary and convenience functions."""

from __future__ import annotations

__all__ = [
    'chunk_iter',
    'dilate_coordinates',
    'init_module',
    'kwargs_notnone',
    'parse_harmonic',
    'parse_kwargs',
    'parse_signal_axis',
    'parse_skip_axis',
    'phasor_from_polar_scalar',
    'phasor_to_polar_scalar',
    'scale_matrix',
    'sort_coordinates',
    'squeeze_dims',
    'update_kwargs',
    'xarray_metadata',
]

import math
import numbers
import os
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import (
        Any,
        ArrayLike,
        Literal,
        NDArray,
        Iterator,
        Container,
        PathLike,
    )

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
    axis_label : str
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


def parse_skip_axis(
    skip_axis: int | Sequence[int] | None,
    /,
    ndim: int,
    prepend_axis: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return axes to skip and not to skip.

    This helper function is used to validate and parse `skip_axis`
    parameters.

    Parameters
    ----------
    skip_axis : int or sequence of int, optional
        Axes to skip. If None, no axes are skipped.
    ndim : int
        Dimensionality of array in which to skip axes.
    prepend_axis : bool, optional
        Prepend one dimension and include in `skip_axis`.

    Returns
    -------
    skip_axis : tuple of int
        Ordered, positive values of `skip_axis`.
    other_axis : tuple of int
        Axes indices not included in `skip_axis`.

    Raises
    ------
    IndexError
        If any `skip_axis` value is out of bounds of `ndim`.

    Examples
    --------
    >>> parse_skip_axis((1, -2), 5)
    ((1, 3), (0, 2, 4))

    >>> parse_skip_axis((1, -2), 5, True)
    ((0, 2, 4), (1, 3, 5))

    """
    if ndim < 0:
        raise ValueError(f'invalid {ndim=}')
    if skip_axis is None:
        if prepend_axis:
            return (0,), tuple(range(1, ndim + 1))
        return (), tuple(range(ndim))
    if not isinstance(skip_axis, Sequence):
        skip_axis = (skip_axis,)
    if any(i >= ndim or i < -ndim for i in skip_axis):
        raise IndexError(f'skip_axis={skip_axis} out of range for {ndim=}')
    skip_axis = sorted(int(i % ndim) for i in skip_axis)
    if prepend_axis:
        skip_axis = [0] + [i + 1 for i in skip_axis]
        ndim += 1
    other_axis = tuple(i for i in range(ndim) if i not in skip_axis)
    return tuple(skip_axis), other_axis


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
        Any element is out of range `[1, harmonic_max]`.
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
            raise IndexError(f'{harmonic=} out of range [1, {harmonic_max}]')
        return [int(harmonic)], False

    if isinstance(harmonic, str):
        if harmonic == 'all':
            if harmonic_max is None:
                raise TypeError(
                    f'maximum harmonic must be specified for {harmonic=!r}'
                )
            return list(range(1, harmonic_max + 1)), True
        raise ValueError(f'{harmonic=!r} is not a valid harmonic')

    h = numpy.atleast_1d(harmonic)
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
    return [int(i) for i in harmonic], True


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


def init_module(globs: dict[str, Any], /) -> None:
    """Add names in module to ``__all__`` and set ``__module__`` attributes.

    Parameters
    ----------
    globs : dict
        Module namespace to modify.

    Examples
    --------
    >>> init_module(globals())

    """
    names = globs['__all__']
    module_name = globs['__name__']
    module = sys.modules[module_name]
    for name in dir(module):
        if name.startswith('_') or name in {
            'annotations',
            'init_module',
            'utils',  # TODO: where does this come from?
        }:
            continue
        names.append(name)
        obj = getattr(module, name)
        if hasattr(obj, '__module__'):
            obj.__module__ = module_name
    globs['__all__'] = sorted(set(names))


def xarray_metadata(
    dims: Sequence[str] | None,
    shape: tuple[int, ...],
    /,
    name: str | PathLike[Any] | None = None,
    attrs: dict[str, Any] | None = None,
    **coords: Any,
) -> dict[str, Any]:
    """Return xarray-style dims, coords, and attrs in a dict.

    >>> xarray_metadata('SYX', (3, 2, 1), S=['0', '1', '2'])
    {'dims': ('S', 'Y', 'X'), 'coords': {'S': ['0', '1', '2']}, 'attrs': {}}

    """
    assert dims is not None
    dims = tuple(dims)
    if len(dims) != len(shape):
        raise ValueError(
            f'dims do not match shape {len(dims)} != {len(shape)}'
        )
    coords = {dim: coords[dim] for dim in dims if dim in coords}
    if attrs is None:
        attrs = {}
    metadata = {'dims': dims, 'coords': coords, 'attrs': attrs}
    if name:
        metadata['name'] = os.path.basename(name)
    return metadata


def squeeze_dims(
    shape: Sequence[int],
    dims: Sequence[str],
    /,
    skip: Container[str] = 'XY',
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[bool, ...]]:
    """Return shape and axes with length-1 dimensions removed.

    Remove unused dimensions unless their axes are listed in the `skip`
    parameter.

    Adapted from the tifffile library.

    Parameters
    ----------
    shape : tuple of ints
        Sequence of dimension sizes.
    dims : sequence of str
        Character codes for dimensions in `shape`.
    skip : container of str, optional
        Character codes for dimensions whose length-1 dimensions are
        not removed. The default is 'XY'.

    Returns
    -------
    shape : tuple of ints
        Sequence of dimension sizes with length-1 dimensions removed.
    dims : tuple of str
        Character codes for dimensions in output `shape`.
    squeezed : str
        Dimensions were kept (True) or removed (False).

    Examples
    --------
    >>> squeeze_dims((5, 1, 2, 1, 1), 'TZYXC')
    ((5, 2, 1), ('T', 'Y', 'X'), (True, False, True, True, False))
    >>> squeeze_dims((1,), ('Q',))
    ((1,), ('Q',), (True,))

    """
    if len(shape) != len(dims):
        raise ValueError(f'{len(shape)=} != {len(dims)=}')
    if not dims:
        return tuple(shape), tuple(dims), ()
    squeezed: list[bool] = []
    shape_squeezed: list[int] = []
    dims_squeezed: list[str] = []
    for size, ax in zip(shape, dims):
        if size > 1 or ax in skip:
            squeezed.append(True)
            shape_squeezed.append(size)
            dims_squeezed.append(ax)
        else:
            squeezed.append(False)
    if len(shape_squeezed) == 0:
        squeezed[-1] = True
        shape_squeezed.append(shape[-1])
        dims_squeezed.append(dims[-1])
    return tuple(shape_squeezed), tuple(dims_squeezed), tuple(squeezed)
