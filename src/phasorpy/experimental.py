"""Experimental functions.

The ``phasorpy.experimental`` module provides functions related to phasor
analysis for evaluation.
The functions may be removed or moved to other modules in future releases.

"""

from __future__ import annotations

__all__ = [
    'anscombe_transform',
    'anscombe_transform_inverse',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, NDArray, ArrayLike

from ._phasorpy import (
    _anscombe,
    _anscombe_inverse,
    _anscombe_inverse_approx,
)


def anscombe_transform(
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
    data : array_like
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

    >>> z = anscombe_transform(numpy.random.poisson(10, 10000))
    >>> numpy.allclose(numpy.std(z), 1.0, atol=0.1)
    True

    """
    return _anscombe(data, **kwargs)  # type: ignore[no-any-return]


def anscombe_transform_inverse(
    data: ArrayLike,
    /,
    *,
    approx: bool = False,
    **kwargs: Any,
) -> NDArray[Any]:
    r"""Return inverse Anscombe transformation.

    Parameters
    ----------
    data : array_like
        Anscombe-transformed data.
    approx : bool, default: False
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

        x = 1/4 \cdot z^2
          + 1/4 \cdot \sqrt{3/2} \cdot z^{-1}
          - 11/8 \cdot z^{-2}
          + 5/8 \cdot \sqrt{3/2} \cdot z^{-3}
          - 1/8

    References
    ----------
    .. [2] Makitalo M, and Foi A.
       `A closed-form approximation of the exact unbiased inverse of the
       Anscombe variance-stabilizing transformation
       <https://doi.org/10.1109/TIP.2011.2121085>`_.
       *IEEE Trans Image Process*, 20(9): 2697-8 (2011)

    .. [3] Makitalo M, and Foi A.
       `Optimal inversion of the generalized Anscombe transformation for
       Poisson-Gaussian noise
       <https://doi.org/10.1109/TIP.2012.2202675>`_,
       *IEEE Trans Image Process*, 22(1): 91-103 (2013)

    Examples
    --------

    >>> x = numpy.random.poisson(10, 100)
    >>> x2 = anscombe_transform_inverse(anscombe_transform(x))
    >>> numpy.allclose(x, x2, atol=1e-3)
    True

    """
    if approx:
        return _anscombe_inverse_approx(  # type: ignore[no-any-return]
            data, **kwargs
        )
    return _anscombe_inverse(data, **kwargs)  # type: ignore[no-any-return]
