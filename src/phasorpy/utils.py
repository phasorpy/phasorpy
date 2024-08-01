"""Utility functions.

The ``phasorpy.utils`` module provides auxiliary and convenience functions
that do not naturally fit into other modules.

"""

from __future__ import annotations

__all__ = ['number_threads']

import os


def number_threads(
    num_threads: int | None = None,
    max_threads: int | None = None,
    /,
) -> int:
    """Return number of threads for parallel computations on CPU cores.

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
        try:
            cpu_count = len(os.sched_getaffinity(0))  # type: ignore
        except AttributeError:
            # sched_getaffinity not available on Windows
            cpu_count = os.cpu_count()
        if cpu_count is None:
            return 1
        return min(max_threads, max(1, cpu_count // 2))
    # return num_threads up to max_threads
    if max_threads is None:
        return num_threads
    return min(num_threads, max(max_threads, 1))
