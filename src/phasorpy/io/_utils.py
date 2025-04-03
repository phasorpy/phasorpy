"""IO utility functions."""

from __future__ import annotations

__all__ = ['squeeze_dims', 'xarray_metadata']

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._typing import Any, Container, PathLike, Sequence


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
