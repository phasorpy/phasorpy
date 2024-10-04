"""Utility functions.

The ``phasorpy.utils`` module provides auxiliary and convenience functions
that do not naturally fit into other modules.

"""

from __future__ import annotations

__all__ = [
    'anscombe_transformation',
    'anscombe_transformation_inverse',
    'number_threads',
]

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, NDArray, ArrayLike

from ._phasorpy import _anscombe, _anscombe_inverse, _anscombe_inverse_approx


def anscombe_transformation(
    data: ArrayLike,
    /,
    **kwargs: Any,
) -> NDArray[Any]:
    r"""Return Anscombe variance-stabilizing transformation.

    The Anscombe transformation normalizes the standard deviation of noisy,
    Poisson-distributed data.
    It can be used to transform un-normalized phasor coordinates to
    approximate standard Gaussian distributions.

    Parameters
    ----------
    data: array_like
        Noisy Poisson-distributed data to be transformed.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    ndarray
        Anscombe-transformed data with variance of approximately 1.

    Notes
    -----
    The Anscombe transformation according to [1]_:

    .. math::

        z = 2 \cdot \sqrt{x + 3 / 8}

    References
    ----------

    .. [1] Anscombe FJ.
       `The transformation of Poisson, binomial and negative-binomial data
       <https://doi.org/10.2307/2332343>`_.
       *Biometrika*, 35(3-4): 246-254 (1948)

    Examples
    --------

    >>> z = anscombe_transformation(numpy.random.poisson(10, 10000))
    >>> numpy.allclose(numpy.std(z), 1.0, atol=0.1)
    True

    """
    return _anscombe(data, **kwargs)  # type: ignore[no-any-return]


def anscombe_transformation_inverse(
    data: ArrayLike,
    /,
    *,
    approx: bool = False,
    **kwargs: Any,
) -> NDArray[Any]:
    r"""Return inverse Anscombe transformation.

    Parameters
    ----------
    data: array_like
        Anscombe-transformed data.
    approx: bool, default: False
        If true, return approximation of exact unbiased inverse.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    ndarray
        Inverse Anscombe-transformed data.

    Notes
    -----
    The inverse Anscombe transformation according to [1]_:

    .. math::

        x = (z / 2.0)^2 - 3 / 8

    The approximate inverse Anscombe transformation according to [2]_ and [3]_:

    .. math::

        x = 1/4 \cdot z^2{2}
          + 1/4 \cdot \sqrt{3/2} \cdot z^{-1}
          - 11/8 \cdot z^{-2}
          + 5/8 \cdot \sqrt(3/2) \cdot z^{-3}
          - 1/8

    References
    ----------

    .. [2] Makitalo M, and Foi A.
       `A closed-form approximation of the exact unbiased inverse of the
       Anscombe variance-stabilizing transformation
       <https://doi.org/10.1109/TIP.2011.2121085>`_.
       IEEE Trans Image Process, 20(9): 2697-8 (2011).

    .. [3] Makitalo M, and Foi A
       `Optimal inversion of the generalized Anscombe transformation for
       Poisson-Gaussian noise
       <https://doi.org/10.1109/TIP.2012.2202675>`_,
       IEEE Trans Image Process, 22(1): 91-103 (2013)

    Examples
    --------

    >>> x = numpy.random.poisson(10, 100)
    >>> x2 = anscombe_transformation_inverse(anscombe_transformation(x))
    >>> numpy.allclose(x, x2, atol=1e-3)
    True

    """
    if approx:
        return _anscombe_inverse_approx(  # type: ignore[no-any-return]
            data, **kwargs
        )
    return _anscombe_inverse(data, **kwargs)  # type: ignore[no-any-return]


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
