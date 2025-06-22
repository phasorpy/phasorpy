"""Experimental functions.

The ``phasorpy.experimental`` module provides functions related to phasor
analysis for evaluation.
The functions may be removed or moved to other modules in future releases.

"""

from __future__ import annotations

__all__ = [
    'anscombe_transform',
    'anscombe_transform_inverse',
    'spectral_vector_denoise',
]

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, NDArray, ArrayLike, DTypeLike, Literal, Sequence

import numpy

from ._phasorpy import (
    _anscombe,
    _anscombe_inverse,
    _anscombe_inverse_approx,
    _phasor_from_signal_vector,
    _signal_denoise_vector,
)
from ._utils import parse_harmonic
from .utils import number_threads


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
          + 5/8 \cdot \sqrt(3/2) \cdot z^{-3}
          - 1/8

    References
    ----------

    .. [2] Makitalo M, and Foi A.
       `A closed-form approximation of the exact unbiased inverse of the
       Anscombe variance-stabilizing transformation
       <https://doi.org/10.1109/TIP.2011.2121085>`_.
       *IEEE Trans Image Process*, 20(9): 2697-8 (2011).

    .. [3] Makitalo M, and Foi A
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


def spectral_vector_denoise(
    signal: ArrayLike,
    /,
    spectral_vector: ArrayLike | None = None,
    *,
    axis: int = -1,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
    sigma: float = 0.05,
    vmin: float | None = None,
    dtype: DTypeLike | None = None,
    num_threads: int | None = None,
) -> NDArray[Any]:
    """Return spectral-vector-denoised signal.

    The spectral vector denoising algorithm is based on a Gaussian weighted
    average calculation, with weights obtained in n-dimensional Chebyshev or
    Fourier space [4]_.

    Parameters
    ----------
    signal : array_like
        Hyperspectral data to be denoised.
        A minimum of three samples are required along `axis`.
        The samples must be uniformly spaced.
    spectral_vector : array_like, optional
        Spectral vector.
        For example, phasor coordinates, PCA projected phasor coordinates,
        or Chebyshev coefficients.
        Must be of same shape as `signal` with `axis` removed and axis
        containing spectral space appended.
        If None (default), phasor coordinates are calculated at specified
        `harmonic`.
    axis : int, optional, default: -1
        Axis over which `spectral_vector` is computed if not provided.
        The default is the last axis (-1).
    harmonic : int, sequence of int, or 'all', optional
        Harmonics to include in calculating `spectral_vector`.
        If `'all'`, include all harmonics for `signal` samples along `axis`.
        Else, harmonics must be at least one and no larger than half the
        number of `signal` samples along `axis`.
        The default is the first harmonic (fundamental frequency).
        A minimum of `harmonic * 2 + 1` samples are required along `axis`
        to calculate correct phasor coordinates at `harmonic`.
    sigma : float, default: 0.05
        Width of Gaussian filter in spectral vector space.
        Weighted averages are calculated using the spectra of signal items
        within an spectral vector Euclidean distance of `3 * sigma` and
        intensity above `vmin`.
    vmin : float, optional
        Signal intensity along `axis` below which not to include in denoising.
    dtype : dtype_like, optional
        Data type of output arrays. Either float32 or float64.
        The default is float64 unless the `signal` is float32.
    num_threads : int, optional
        Number of OpenMP threads to use for parallelization.
        By default, multi-threading is disabled.
        If zero, up to half of logical CPUs are used.
        OpenMP may not be available on all platforms.

    Returns
    -------
    ndarray
        Denoised signal of `dtype`.
        Spectra with integrated intensity below `vmin` are unchanged.

    References
    ----------

    .. [4] Harman RC, Lang RT, Kercher EM, Leven P, and Spring BQ.
       `Denoising multiplexed microscopy images in n-dimensional spectral space
       <https://doi.org/10.1364/BOE.463979>`_.
       *Biomed Opt Express*, 13(8): 4298-4309 (2022)

    Examples
    --------
    Denoise a hyperspectral image with a Gaussian filter width of 0.1 in
    spectral vector space using first and second harmonic:

    >>> signal = numpy.random.randint(0, 255, (8, 16, 16))
    >>> spectral_vector_denoise(signal, axis=0, sigma=0.1, harmonic=[1, 2])
    array([[[...]]])

    """
    num_threads = number_threads(num_threads)

    signal = numpy.asarray(signal)
    if axis == -1 or axis == signal.ndim - 1:
        axis = -1
    else:
        signal = numpy.moveaxis(signal, axis, -1)
    shape = signal.shape
    samples = shape[-1]

    if harmonic is None:
        harmonic = 1
    harmonic, _ = parse_harmonic(harmonic, samples // 2)
    num_harmonics = len(harmonic)

    if vmin is None or vmin < 0.0:
        vmin = 0.0

    sincos = numpy.empty((num_harmonics, samples, 2))
    for i, h in enumerate(harmonic):
        phase = numpy.linspace(
            0,
            h * math.pi * 2.0,
            samples,
            endpoint=False,
            dtype=numpy.float64,
        )
        sincos[i, :, 0] = numpy.cos(phase)
        sincos[i, :, 1] = numpy.sin(phase)

    signal = numpy.ascontiguousarray(signal).reshape(-1, samples)
    size = signal.shape[0]

    if dtype is None:
        if signal.dtype.char == 'f':
            dtype = signal.dtype
        else:
            dtype = numpy.float64
    dtype = numpy.dtype(dtype)
    if dtype.char not in {'d', 'f'}:
        raise ValueError('dtype is not floating point')

    if spectral_vector is None:
        spectral_vector = numpy.zeros((size, num_harmonics * 2), dtype=dtype)
        _phasor_from_signal_vector(
            spectral_vector, signal, sincos, num_threads
        )
    else:
        spectral_vector = numpy.ascontiguousarray(spectral_vector, dtype=dtype)
        if spectral_vector.shape[:-1] != shape[:-1]:
            raise ValueError('signal and spectral_vector shape mismatch')
        spectral_vector = spectral_vector.reshape(
            -1, spectral_vector.shape[-1]
        )

    if dtype == signal.dtype:
        denoised = signal.copy()
    else:
        denoised = numpy.zeros(signal.shape, dtype=dtype)
        denoised[:] = signal
    integrated = numpy.zeros(size, dtype=dtype)
    _signal_denoise_vector(
        denoised, integrated, signal, spectral_vector, sigma, vmin, num_threads
    )

    denoised = denoised.reshape(shape)
    if axis != -1:
        denoised = numpy.moveaxis(denoised, -1, axis)
    return denoised
