"""Utility functions.

The ``phasorpy.utils`` module provides auxiliary and convenience functions
that do not naturally fit into other modules.

"""

from __future__ import annotations

__all__ = [
    'logger',
    'number_threads',
    'versions',
]

import logging
import os


def logger() -> logging.Logger:
    """Return ``logging.getLogger('phasorpy')``."""
    return logging.getLogger('phasorpy')


def number_threads(
    num_threads: int | None = None,
    max_threads: int | None = None,
    /,
) -> int:
    """Return number of threads for parallel computations across CPU cores.

    This function is used to parse ``num_threads`` parameters.

    Parameters
    ----------
    num_threads : int, optional
        Number of threads to use for parallel computations on CPU cores.
        If None (default), return 1, disabling multi-threading.
        If greater than zero, return value up to `max_threads` if set.
        If zero, return the value of the ``PHASORPY_NUM_THREADS`` environment
        variable if set, else half the CPU cores up to `max_threads` or 32.
    max_threads : int, optional
        Maximum number of threads to return.

    Returns
    -------
    int
        Number of threads for parallel computations.

    Examples
    --------
    >>> number_threads()
    1
    >>> number_threads(0)  # doctest: +SKIP
    8

    """
    if num_threads is None or num_threads < 0:
        # disable multi-threading by default
        return 1
    if num_threads == 0:
        # return default number of threads
        if max_threads is None:
            max_threads = 32
        else:
            max_threads = max(max_threads, 1)
        if 'PHASORPY_NUM_THREADS' in os.environ:
            return min(
                max_threads, max(1, int(os.environ['PHASORPY_NUM_THREADS']))
            )
        cpu_count: int | None
        if hasattr(os, 'sched_getaffinity'):
            cpu_count = len(os.sched_getaffinity(0))
        else:
            # sched_getaffinity not available on Windows
            cpu_count = os.cpu_count()
        if cpu_count is None:
            return 1
        return min(max_threads, max(1, cpu_count // 2))
    # return num_threads up to max_threads
    if max_threads is None:
        return num_threads
    return min(num_threads, max(max_threads, 1))


def versions(
    *, sep: str = '\n', dash: str = '-', verbose: bool = False
) -> str:
    """Return version information for PhasorPy and its dependencies.

    Parameters
    ----------
    sep : str, optional
        Separator between version items. Defaults to newline.
    dash : str, optional
        Separator between module name and version. Defaults to dash.
    verbose : bool, optional
        Include paths to Python interpreter and modules.

    Returns
    -------
    str
        Formatted string containing version information.
        Format: "<package><dash><version>[<space>(<path>)]<sep>"

    Example
    -------
    >>> print(versions())
    Python-3...
    phasorpy-0...
    numpy-...
    ...

    """
    import importlib.metadata
    import os
    import sys

    if verbose:
        version_strings = [f'Python{dash}{sys.version}  ({sys.executable})']
    else:
        version_strings = [f'Python{dash}{sys.version.split()[0]}']

    for module in (
        'phasorpy',
        'numpy',
        'tifffile',
        'imagecodecs',
        'lfdfiles',
        'sdtfile',
        'ptufile',
        'liffile',
        'matplotlib',
        'scipy',
        'skimage',
        'sklearn',
        'pandas',
        'xarray',
        'click',
        'pooch',
    ):
        try:
            __import__(module)
        except ModuleNotFoundError:
            version_strings.append(f'{module.lower()}{dash}n/a')
            continue
        lib = sys.modules[module]
        try:
            ver = importlib.metadata.version(module)
        except importlib.metadata.PackageNotFoundError:
            ver = getattr(lib, '__version__', 'unknown')
        ver = f'{module.lower()}{dash}{ver}'
        if verbose:
            try:
                path = getattr(lib, '__file__')
            except NameError:
                pass
            else:
                ver += f'  ({os.path.dirname(path)})'
        version_strings.append(ver)
    return sep.join(version_strings)
