"""Type annotations.

This module should only be imported when type-checking, for example::

    from __future__ import annotations

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from ._typing import Any, ArrayLike, PathLike

"""

# flake8: noqa: F401
# pylint: disable=unused-import
# autoflake: skip_file

from __future__ import annotations

__all__ = [
    'Any',
    'ArrayLike',
    'Callable',
    'Collection',
    'Container',
    'DTypeLike',
    'DataArray',
    'EllipsisType',
    'IO',
    'ItemsView',
    'Iterable',
    'Iterator',
    'KeysView',
    'Literal',
    'Mapping',
    'NDArray',
    'Optional',
    'PathLike',
    'Pooch',
    'Sequence',
    'TextIO',
    'Union',
    'ValuesView',
    'cast',
    'final',
    'overload',
]

from collections.abc import (
    Callable,
    Collection,
    Container,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from os import PathLike
from types import EllipsisType
from typing import (
    IO,
    Any,
    Literal,
    Optional,
    TextIO,
    Union,
    cast,
    final,
    overload,
)

from numpy.typing import ArrayLike, DTypeLike, NDArray
from pooch import Pooch
from xarray import DataArray
