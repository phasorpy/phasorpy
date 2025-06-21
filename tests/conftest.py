"""Pytest configuration."""

import os

import phasorpy
from phasorpy.utils import number_threads, versions


def pytest_report_header(config, start_path):
    """Return versions of relevant installed packages."""
    return '\n'.join(
        (
            f'versions: {versions(sep=", ")}',
            f'number_threads: {number_threads(0)}',
            f'path: {os.path.dirname(phasorpy.__file__)}',
        )
    )


collect_ignore = ['data']


# mypy: allow-untyped-defs, allow-untyped-calls
