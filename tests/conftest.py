"""Pytest configuration."""

import os
from pathlib import Path

import pytest

import phasorpy
from phasorpy.utils import number_threads, versions


def pytest_report_header(config: pytest.Config, start_path: Path) -> str:
    """Return versions of relevant installed packages."""
    return '\n'.join(
        (
            f'versions: {versions(sep=", ")}',
            f'number_threads: {number_threads(0)}',
            f'path: {os.path.dirname(phasorpy.__file__)}',
        )
    )


collect_ignore = ['data']
