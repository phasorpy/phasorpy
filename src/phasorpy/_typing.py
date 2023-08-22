"""Type annotations.

This module should only be imported when type-checking, for example::

    from __future__ import annotations

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from ._typing import Any, ArrayLike, PathLike

"""

# flake8: noqa: F401
# pylint: disable=unused-import

from __future__ import annotations

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
from typing import (
    Any,
    BinaryIO,
    Literal,
    Optional,
    TextIO,
    Union,
    cast,
    final,
    overload,
)

from numpy.typing import ArrayLike, DTypeLike, NDArray
