"""Test configuration for phasorpy.io module."""

from __future__ import annotations

__all__ = ['SKIP_FETCH', 'SKIP_PRIVATE', 'TempFileName', 'private_file']

import contextlib
import glob
import os
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

HERE = os.path.dirname(__file__)
TEMP_DIR = os.path.normpath(
    os.environ.get('PHASORPY_TEMP', tempfile.gettempdir())
)
DATA_DIR = os.path.normpath(
    os.environ.get('PHASORPY_DATA', os.path.join(HERE, '..', '..', 'data'))
)
PRIVATE_DIR = os.path.join(DATA_DIR, 'private')

SKIP_PRIVATE = not os.path.exists(PRIVATE_DIR)
SKIP_FETCH = bool(os.environ.get('SKIP_FETCH', ''))


class TempFileName:
    """Temporary file name context manager."""

    name: str
    remove: bool
    pattern: bool

    def __init__(
        self,
        name: str | None = None,
        /,
        *,
        remove: bool = False,
        pattern: bool = False,
    ) -> None:
        self.remove = remove or tempfile.gettempdir() == TEMP_DIR
        if not name:
            with tempfile.NamedTemporaryFile(prefix='test_') as fh:
                self.name = fh.name
        else:
            self.name = os.path.join(TEMP_DIR, f'test_{name}')
        self.pattern = pattern

    def __enter__(self) -> str:
        return self.name

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.remove:
            if self.pattern:
                name, ext = os.path.splitext(self.name)
                for fname in glob.glob(name + '*' + ext):
                    with contextlib.suppress(Exception):
                        os.remove(fname)
            with contextlib.suppress(Exception):
                os.remove(self.name)


def private_file(filename: str, /) -> str:
    """Return path to private test file."""
    return os.path.normpath(os.path.join(PRIVATE_DIR, filename))
