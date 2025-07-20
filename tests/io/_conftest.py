"""Test configuration for io module."""

from __future__ import annotations

__all__ = ['private_file', 'TempFileName', 'SKIP_PRIVATE', 'SKIP_FETCH']

import glob
import os
import tempfile
from typing import Any

HERE = os.path.dirname(__file__)
TEMP_DIR = os.path.normpath(
    os.environ.get('PHASORPY_TEMP', tempfile.gettempdir())
)
DATA_DIR = os.path.normpath(
    os.environ.get('PHASORPY_DATA', os.path.join(HERE, '..', '..', 'data'))
)
PRIVATE_DIR = os.path.join(DATA_DIR, 'private')

SKIP_PRIVATE = not os.path.exists(PRIVATE_DIR)
SKIP_FETCH = os.environ.get('SKIP_FETCH', False)


class TempFileName:
    """Temporary file name context manager."""

    name: str
    remove: bool
    pattern: bool

    def __init__(
        self,
        name: str | None = None,
        remove: bool = False,
        pattern: bool = False,
    ) -> None:
        self.remove = remove or TEMP_DIR == tempfile.gettempdir()
        if not name:
            fh = tempfile.NamedTemporaryFile(prefix='test_')
            self.name = fh.named
            fh.close()
        else:
            self.name = os.path.join(TEMP_DIR, f'test_{name}')
        self.pattern = pattern

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.remove:
            if self.pattern:
                name, ext = os.path.splitext(self.name)
                for fname in glob.glob(name + '*' + ext):
                    try:
                        os.remove(fname)
                    except Exception:
                        pass
            try:
                os.remove(self.name)
            except Exception:
                pass


def private_file(filename: str, /) -> str:
    """Return path to private test file."""
    return os.path.normpath(os.path.join(PRIVATE_DIR, filename))
