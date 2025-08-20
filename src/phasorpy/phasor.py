"""Calculate, convert, and reduce phasor coordinates.

The ``phasorpy.phasor`` module provides functions to:

- calculate phasor coordinates from time-resolved and spectral signals:

  - :py:func:`phasor_from_signal`

- synthesize signals from phasor coordinates:

  - :py:func:`phasor_to_signal`

- convert to and from polar coordinates (phase and modulation):

  - :py:func:`phasor_from_polar`
  - :py:func:`phasor_to_polar`

- transform phasor coordinates:

  - :py:func:`phasor_transform`
  - :py:func:`phasor_multiply`
  - :py:func:`phasor_divide`
  - :py:func:`phasor_normalize`

- reduce dimensionality of arrays of phasor coordinates:

  - :py:func:`phasor_center`
  - :py:func:`phasor_to_principal_plane`

- filter phasor coordinates:

  - :py:func:`phasor_filter_median`
  - :py:func:`phasor_filter_pawflim`
  - :py:func:`phasor_threshold`

- find nearest neighbor phasor coordinates from other phasor coordinates:

  - :py:func:`phasor_nearest_neighbor`

"""

from __future__ import annotations

__all__ = [
    'phasor_center',
    'phasor_divide',
    'phasor_filter_median',
    'phasor_filter_pawflim',
    'phasor_from_polar',
    'phasor_from_signal',
    'phasor_multiply',
    'phasor_nearest_neighbor',
    'phasor_normalize',
    'phasor_threshold',
    'phasor_to_complex',
    'phasor_to_polar',
    'phasor_to_principal_plane',
    'phasor_to_signal',
    'phasor_transform',
]

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import (
        Any,
        NDArray,
        ArrayLike,
        DTypeLike,
        Callable,
        Literal,
    )

import numpy

from ._phasorpy import (
    _median_filter_2d,
    _nearest_neighbor_2d,
    _phasor_divide,
    _phasor_from_polar,
    _phasor_from_signal,
    _phasor_multiply,
    _phasor_threshold_closed,
    _phasor_threshold_mean_closed,
    _phasor_threshold_mean_open,
    _phasor_threshold_nan,
    _phasor_threshold_open,
    _phasor_to_polar,
    _phasor_transform,
    _phasor_transform_const,
)
from ._utils import parse_harmonic, parse_signal_axis, parse_skip_axis
from .utils import number_threads


def phasor_from_signal(
    signal: ArrayLike,
    /,
    *,
    axis: int | str | None = None,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
    sample_phase: ArrayLike | None = None,
    use_fft: bool | None = None,
    rfft: Callable[..., NDArray[Any]] | None = None,
    dtype: DTypeLike = None,
    normalize: bool = True,
    num_threads: int | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    r"""Return phasor coordinates from signal.

    Parameters
    ----------
    signal : array_like
        Frequency-domain, time-domain, or hyperspectral data.
        A minimum of three samples are required along `axis`.
        The samples must be uniformly spaced.
    axis : int or str, optional
        Axis over which to compute phasor coordinates.
        By default, the 'H' or 'C' axes if signal contains such dimension
        names, else the last axis (-1).
    harmonic : int, sequence of int, or 'all', optional
        Harmonics to return.
        If `'all'`, return all harmonics for `signal` samples along `axis`.
        Else, harmonics must be at least one and no larger than half the
        number of `signal` samples along `axis`.
        The default is the first harmonic (fundamental frequency).
        A minimum of `harmonic * 2 + 1` samples are required along `axis`
        to calculate correct phasor coordinates at `harmonic`.
    sample_phase : array_like, optional
        Phase values (in radians) of `signal` samples along `axis`.
        If None (default), samples are assumed to be uniformly spaced along
        one period.
        The array size must equal the number of samples along `axis`.
        Cannot be used with `harmonic!=1` or `use_fft=True`.
    use_fft : bool, optional
        If true, use a real forward Fast Fourier Transform (FFT).
        If false, use a Cython implementation that is optimized (faster and
        resource saving) for calculating few harmonics.
        By default, FFT is only used when all or at least 8 harmonics are
        calculated, or `rfft` is specified.
    rfft : callable, optional
        Drop-in replacement function for ``numpy.fft.rfft``.
        For example, ``scipy.fft.rfft`` or ``mkl_fft._numpy_fft.rfft``.
        Used to calculate the real forward FFT.
    dtype : dtype_like, optional
        Data type of output arrays. Either float32 or float64.
        The default is float64 unless the `signal` is float32.
    normalize : bool, optional
        Return normalized phasor coordinates.
        If true (default), return average of `signal` along `axis` and
        Fourier coefficients divided by sum of `signal` along `axis`.
        Else, return sum of `signal` along `axis` and unscaled Fourier
        coefficients.
        Un-normalized phasor coordinates cannot be used with most of PhasorPy's
        functions but may be required for intermediate processing.
    num_threads : int, optional
        Number of OpenMP threads to use for parallelization when not using FFT.
        By default, multi-threading is disabled.
        If zero, up to half of logical CPUs are used.
        OpenMP may not be available on all platforms.

    Returns
    -------
    mean : ndarray
        Average of `signal` along `axis` (zero harmonic).
    real : ndarray
        Real component of phasor coordinates at `harmonic` along `axis`.
    imag : ndarray
        Imaginary component of phasor coordinates at `harmonic` along `axis`.

    Raises
    ------
    ValueError
        The `signal` has less than three samples along `axis`.
        The `sample_phase` size does not equal the number of samples along
        `axis`.
    IndexError
        `harmonic` is smaller than 1 or greater than half the samples along
        `axis`.
    TypeError
        The `signal`, `dtype`, or `harmonic` types are not supported.

    See Also
    --------
    phasorpy.phasor.phasor_to_signal
    phasorpy.phasor.phasor_normalize
    :ref:`sphx_glr_tutorials_misc_phasorpy_phasor_from_signal.py`

    Notes
    -----
    The normalized phasor coordinates `real` (:math:`G`), `imag` (:math:`S`),
    and average intensity `mean` (:math:`F_{DC}`) are calculated from
    :math:`K \ge 3` samples of the signal :math:`F` at `harmonic` :math:`h`
    according to:

    .. math::

        F_{DC} &= \frac{1}{K} \sum_{k=0}^{K-1} F_{k}

        G &= \frac{1}{K} \sum_{k=0}^{K-1} F_{k}
        \cos{\left (2 \pi h \frac{k}{K} \right )} \cdot \frac{1}{F_{DC}}

        S &= \frac{1}{K} \sum_{k=0}^{K-1} F_{k}
        \sin{\left (2 \pi h \frac{k}{K} \right )} \cdot \frac{1}{F_{DC}}

    If :math:`F_{DC} = 0`, the phasor coordinates are undefined
    (resulting in NaN or infinity).
    Use NaN-aware software to further process the phasor coordinates.

    The phasor coordinates may be zero, for example, in case of only constant
    background in time-resolved signals, or as the result of linear
    combination of non-zero spectral phasors coordinates.

    Examples
    --------
    Calculate phasor coordinates of a phase-shifted sinusoidal waveform:

    >>> sample_phase = numpy.linspace(0, 2 * math.pi, 5, endpoint=False)
    >>> signal = 1.1 * (numpy.cos(sample_phase - 0.785398) * 2 * 0.707107 + 1)
    >>> phasor_from_signal(signal)  # doctest: +NUMBER
    (array(1.1), array(0.5), array(0.5))

    The sinusoidal signal does not have a second harmonic component:

    >>> phasor_from_signal(signal, harmonic=2)  # doctest: +NUMBER
    (array(1.1), array(0.0), array(0.0))

    """
    # TODO: C-order not required by rfft?
    # TODO: preserve array subtypes?

    axis, _ = parse_signal_axis(signal, axis)

    signal = numpy.asarray(signal, order='C')
    if signal.dtype.kind not in 'uif':
        raise TypeError(f'signal must be real valued, not {signal.dtype=}')
    samples = numpy.size(signal, axis)  # this also verifies axis and ndim >= 1
    if samples < 3:
        raise ValueError(f'not enough {samples=} along {axis=}')

    if dtype is None:
        dtype = numpy.float32 if signal.dtype.char == 'f' else numpy.float64
    dtype = numpy.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError(f'{dtype=} not supported')

    harmonic, keepdims = parse_harmonic(harmonic, samples // 2)
    num_harmonics = len(harmonic)

    if sample_phase is not None:
        if use_fft:
            raise ValueError('sample_phase cannot be used with FFT')
        if num_harmonics > 1 or harmonic[0] != 1:
            raise ValueError('sample_phase cannot be used with harmonic != 1')
        sample_phase = numpy.atleast_1d(
            numpy.asarray(sample_phase, dtype=numpy.float64)
        )
        if sample_phase.ndim != 1 or sample_phase.size != samples:
            raise ValueError(f'{sample_phase.shape=} != ({samples},)')

    if use_fft is None:
        use_fft = sample_phase is None and (
            rfft is not None
            or num_harmonics > 7
            or num_harmonics >= samples // 2
        )

    if use_fft:
        if rfft is None:
            rfft = numpy.fft.rfft

        fft: NDArray[Any] = rfft(
            signal, axis=axis, norm='forward' if normalize else 'backward'
        )

        mean = fft.take(0, axis=axis).real
        if not mean.ndim == 0:
            mean = numpy.ascontiguousarray(mean, dtype)
        fft = fft.take(harmonic, axis=axis)
        real = numpy.ascontiguousarray(fft.real, dtype)
        imag = numpy.ascontiguousarray(fft.imag, dtype)
        del fft

        if not keepdims and real.shape[axis] == 1:
            dc = mean
            real = real.squeeze(axis)
            imag = imag.squeeze(axis)
        else:
            # make broadcastable
            dc = numpy.expand_dims(mean, 0)
            real = numpy.moveaxis(real, axis, 0)
            imag = numpy.moveaxis(imag, axis, 0)

        if normalize:
            with numpy.errstate(divide='ignore', invalid='ignore'):
                real /= dc
                imag /= dc
        numpy.negative(imag, out=imag)

        if not keepdims and real.ndim == 0:
            return mean.squeeze(), real.squeeze(), imag.squeeze()

        return mean, real, imag

    num_threads = number_threads(num_threads)

    sincos = numpy.empty((num_harmonics, samples, 2))
    for i, h in enumerate(harmonic):
        if sample_phase is None:
            phase = numpy.linspace(
                0,
                h * math.pi * 2.0,
                samples,
                endpoint=False,
                dtype=numpy.float64,
            )
        else:
            phase = sample_phase
        sincos[i, :, 0] = numpy.cos(phase)
        sincos[i, :, 1] = numpy.sin(phase)

    # reshape to 3D with axis in middle
    axis = axis % signal.ndim
    shape0 = signal.shape[:axis]
    shape1 = signal.shape[axis + 1 :]
    size0 = math.prod(shape0)
    size1 = math.prod(shape1)
    phasor = numpy.empty((num_harmonics * 2 + 1, size0, size1), dtype)
    signal = signal.reshape((size0, samples, size1))

    _phasor_from_signal(phasor, signal, sincos, normalize, num_threads)

    # restore original shape
    shape = shape0 + shape1
    mean = phasor[0].reshape(shape)
    if keepdims:
        shape = (num_harmonics,) + shape
    real = phasor[1 : num_harmonics + 1].reshape(shape)
    imag = phasor[1 + num_harmonics :].reshape(shape)
    if shape:
        return mean, real, imag
    return mean.squeeze(), real.squeeze(), imag.squeeze()


def phasor_to_signal(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    samples: int = 64,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
    axis: int = -1,
    irfft: Callable[..., NDArray[Any]] | None = None,
) -> NDArray[numpy.float64]:
    """Return signal from phasor coordinates using inverse Fourier transform.

    Parameters
    ----------
    mean : array_like
        Average signal intensity (DC).
        If not scalar, shape must match the last dimensions of `real`.
    real : array_like
        Real component of phasor coordinates.
        Multiple harmonics, if any, must be in the first axis.
    imag : array_like
        Imaginary component of phasor coordinates.
        Must be same shape as `real`.
    samples : int, default: 64
        Number of signal samples to return. Must be at least three.
    harmonic : int, sequence of int, or 'all', optional
        Harmonics included in first axis of `real` and `imag`.
        If None, lower harmonics are inferred from the shapes of phasor
        coordinates (most commonly, lower harmonics are present if the number
        of dimensions of `mean` is one less than `real`).
        If `'all'`, the harmonics in the first axis of phasor coordinates are
        the lower harmonics necessary to synthesize `samples`.
        Else, harmonics must be at least one and no larger than half of
        `samples`.
        The phasor coordinates of missing harmonics are zeroed
        if `samples` is greater than twice the number of harmonics.
    axis : int, optional
        Axis at which to return signal samples.
        The default is the last axis (-1).
    irfft : callable, optional
        Drop-in replacement function for ``numpy.fft.irfft``.
        For example, ``scipy.fft.irfft`` or ``mkl_fft._numpy_fft.irfft``.
        Used to calculate the real inverse FFT.

    Returns
    -------
    signal : ndarray
        Reconstructed signal with samples of one period along the last axis.

    See Also
    --------
    phasorpy.phasor.phasor_from_signal

    Notes
    -----
    The reconstructed signal may be undefined if the input phasor coordinates,
    or signal mean contain NaN values.

    Examples
    --------
    Reconstruct exact signal from phasor coordinates at all harmonics:

    >>> sample_phase = numpy.linspace(0, 2 * math.pi, 5, endpoint=False)
    >>> signal = 1.1 * (numpy.cos(sample_phase - 0.785398) * 2 * 0.707107 + 1)
    >>> signal
    array([2.2, 2.486, 0.8566, -0.4365, 0.3938])
    >>> phasor_to_signal(
    ...     *phasor_from_signal(signal, harmonic='all'),
    ...     harmonic='all',
    ...     samples=len(signal)
    ... )  # doctest: +NUMBER
    array([2.2, 2.486, 0.8566, -0.4365, 0.3938])

    Reconstruct a single-frequency waveform from phasor coordinates at
    first harmonic:

    >>> phasor_to_signal(1.1, 0.5, 0.5, samples=5)  # doctest: +NUMBER
    array([2.2, 2.486, 0.8566, -0.4365, 0.3938])

    """
    if samples < 3:
        raise ValueError(f'{samples=} < 3')

    mean = numpy.array(mean, ndmin=0, copy=True)
    real = numpy.array(real, ndmin=0, copy=True)
    imag = numpy.array(imag, ndmin=1, copy=True)

    harmonic_ = harmonic
    harmonic, has_harmonic_axis = parse_harmonic(harmonic, samples // 2)

    if real.ndim == 1 and len(harmonic) > 1 and real.shape[0] == len(harmonic):
        # single axis contains harmonic
        has_harmonic_axis = True
        real = real[..., None]
        imag = imag[..., None]
        keepdims = mean.ndim > 0
    else:
        keepdims = mean.ndim > 0 or real.ndim > 0

    mean = numpy.asarray(numpy.atleast_1d(mean))
    real = numpy.asarray(numpy.atleast_1d(real))

    if real.dtype.kind != 'f' or imag.dtype.kind != 'f':
        raise ValueError(f'{real.dtype=} or {imag.dtype=} not floating point')
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')

    if (
        harmonic_ is None
        and mean.size > 1
        and mean.ndim + 1 == real.ndim
        and mean.shape == real.shape[1:]
    ):
        # infer harmonic from shapes of mean and real
        harmonic = list(range(1, real.shape[0] + 1))
        has_harmonic_axis = True

    if not has_harmonic_axis:
        real = real[None, ...]
        imag = imag[None, ...]

    if len(harmonic) != real.shape[0]:
        raise ValueError(f'{len(harmonic)=} != {real.shape[0]=}')

    real *= mean
    imag *= mean
    numpy.negative(imag, out=imag)

    fft: NDArray[Any] = numpy.zeros(
        (samples // 2 + 1, *real.shape[1:]), dtype=numpy.complex128
    )
    fft.real[[0]] = mean
    fft.real[harmonic] = real[: len(harmonic)]
    fft.imag[harmonic] = imag[: len(harmonic)]

    if irfft is None:
        irfft = numpy.fft.irfft

    signal: NDArray[Any] = irfft(fft, samples, axis=0, norm='forward')

    if not keepdims:
        signal = signal[:, 0]
    elif axis != 0:
        signal = numpy.moveaxis(signal, 0, axis)
    return signal


def phasor_to_complex(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    dtype: DTypeLike = None,
) -> NDArray[numpy.complex64 | numpy.complex128]:
    """Return phasor coordinates as complex numbers.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    dtype : dtype_like, optional
        Data type of output array. Either complex64 or complex128.
        By default, complex64 if `real` and `imag` are float32,
        else complex128.

    Returns
    -------
    complex : ndarray
        Phasor coordinates as complex numbers.

    Examples
    --------
    Convert phasor coordinates to complex number arrays:

    >>> phasor_to_complex([0.4, 0.5], [0.2, 0.3])
    array([0.4+0.2j, 0.5+0.3j])

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    if dtype is None:
        if real.dtype == numpy.float32 and imag.dtype == numpy.float32:
            dtype = numpy.complex64
        else:
            dtype = numpy.complex128
    else:
        dtype = numpy.dtype(dtype)
        if dtype.kind != 'c':
            raise ValueError(f'{dtype=} not a complex type')

    c = numpy.empty(numpy.broadcast(real, imag).shape, dtype)
    c.real = real
    c.imag = imag
    return c


def phasor_multiply(
    real: ArrayLike,
    imag: ArrayLike,
    factor_real: ArrayLike,
    factor_imag: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return complex multiplication of two phasors.

    Complex multiplication can be used, for example, to convolve two signals
    such as exponential decay and instrument response functions.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates to multiply.
    imag : array_like
        Imaginary component of phasor coordinates to multiply.
    factor_real : array_like
        Real component of phasor coordinates to multiply by.
    factor_imag : array_like
        Imaginary component of phasor coordinates to multiply by.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real : ndarray
        Real component of complex multiplication.
    imag : ndarray
        Imaginary component of complex multiplication.

    Notes
    -----
    The phasor coordinates `real` (:math:`G`) and `imag` (:math:`S`)
    are multiplied by phasor coordinates `factor_real` (:math:`g`)
    and `factor_imag` (:math:`s`) according to:

    .. math::

        G' &= G \cdot g - S \cdot s

        S' &= G \cdot s + S \cdot g

    Examples
    --------
    Multiply two sets of phasor coordinates:

    >>> phasor_multiply([0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8])
    (array([-0.16, -0.2]), array([0.22, 0.4]))

    """
    # c = phasor_to_complex(real, imag) * phasor_to_complex(
    #     factor_real, factor_imag
    # )
    # return c.real, c.imag
    return _phasor_multiply(  # type: ignore[no-any-return]
        real, imag, factor_real, factor_imag, **kwargs
    )


def phasor_divide(
    real: ArrayLike,
    imag: ArrayLike,
    divisor_real: ArrayLike,
    divisor_imag: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return complex division of two phasors.

    Complex division can be used, for example, to deconvolve two signals
    such as exponential decay and instrument response functions.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates to divide.
    imag : array_like
        Imaginary component of phasor coordinates to divide.
    divisor_real : array_like
        Real component of phasor coordinates to divide by.
    divisor_imag : array_like
        Imaginary component of phasor coordinates to divide by.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real : ndarray
        Real component of complex division.
    imag : ndarray
        Imaginary component of complex division.

    Notes
    -----
    The phasor coordinates `real` (:math:`G`) and `imag` (:math:`S`)
    are divided by phasor coordinates `divisor_real` (:math:`g`)
    and `divisor_imag` (:math:`s`) according to:

    .. math::

        d &= g \cdot g + s \cdot s

        G' &= (G \cdot g + S \cdot s) / d

        S' &= (G \cdot s - S \cdot g) / d

    Examples
    --------
    Divide two sets of phasor coordinates:

    >>> phasor_divide([-0.16, -0.2], [0.22, 0.4], [0.5, 0.6], [0.7, 0.8])
    (array([0.1, 0.2]), array([0.3, 0.4]))

    """
    # c = phasor_to_complex(real, imag) / phasor_to_complex(
    #     divisor_real, divisor_imag
    # )
    # return c.real, c.imag
    return _phasor_divide(  # type: ignore[no-any-return]
        real, imag, divisor_real, divisor_imag, **kwargs
    )


def phasor_normalize(
    mean_unnormalized: ArrayLike,
    real_unnormalized: ArrayLike,
    imag_unnormalized: ArrayLike,
    /,
    samples: int = 1,
    dtype: DTypeLike = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    r"""Return normalized phasor coordinates.

    Use to normalize the phasor coordinates returned by
    ``phasor_from_signal(..., normalize=False)``.

    Parameters
    ----------
    mean_unnormalized : array_like
        Unnormalized intensity of phasor coordinates.
    real_unnormalized : array_like
        Unnormalized real component of phasor coordinates.
    imag_unnormalized : array_like
        Unnormalized imaginary component of phasor coordinates.
    samples : int, default: 1
        Number of signal samples over which `mean` was integrated.
    dtype : dtype_like, optional
        Data type of output arrays. Either float32 or float64.
        The default is float64 unless the `real` is float32.

    Returns
    -------
    mean : ndarray
        Normalized intensity.
    real : ndarray
        Normalized real component.
    imag : ndarray
        Normalized imaginary component.

    Notes
    -----
    The average intensity `mean` (:math:`F_{DC}`) and normalized phasor
    coordinates `real` (:math:`G`) and `imag` (:math:`S`) are calculated from
    the signal `intensity` (:math:`F`), the  number of `samples` (:math:`K`),
    `real_unnormalized` (:math:`G'`), and `imag_unnormalized` (:math:`S'`)
    according to:

    .. math::

        F_{DC} &= F / K

        G &= G' / F

        S &= S' / F

    If :math:`F = 0`, the normalized phasor coordinates (:math:`G`)
    and (:math:`S`) are undefined (NaN or infinity).

    Examples
    --------
    Normalize phasor coordinates with intensity integrated over 10 samples:

    >>> phasor_normalize([0.0, 0.1], [0.0, 0.3], [0.4, 0.5], samples=10)
    (array([0, 0.01]), array([nan, 3]), array([inf, 5]))

    Normalize multi-harmonic phasor coordinates:

    >>> phasor_normalize(0.1, [0.0, 0.3], [0.4, 0.5], samples=10)
    (array(0.01), array([0, 3]), array([4, 5]))

    """
    if samples < 1:
        raise ValueError(f'{samples=} < 1')

    if (
        dtype is None
        and isinstance(real_unnormalized, numpy.ndarray)
        and real_unnormalized.dtype == numpy.float32
    ):
        real = real_unnormalized.copy()
    else:
        real = numpy.array(real_unnormalized, dtype, copy=True)
    imag = numpy.array(imag_unnormalized, real.dtype, copy=True)
    mean = numpy.array(mean_unnormalized, real.dtype, copy=True)

    with numpy.errstate(divide='ignore', invalid='ignore'):
        numpy.divide(real, mean, out=real)
        numpy.divide(imag, mean, out=imag)
    if samples > 1:
        numpy.divide(mean, samples, out=mean)

    return mean, real, imag


def phasor_transform(
    real: ArrayLike,
    imag: ArrayLike,
    phase: ArrayLike = 0.0,
    modulation: ArrayLike = 1.0,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return rotated and scaled phasor coordinates.

    This function rotates and uniformly scales phasor coordinates around the
    origin.
    It can be used, for example, to calibrate phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates to transform.
    imag : array_like
        Imaginary component of phasor coordinates to transform.
    phase : array_like, optional, default: 0.0
        Rotation angle in radians.
    modulation : array_like, optional, default: 1.0
        Uniform scale factor.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real : ndarray
        Real component of rotated and scaled phasor coordinates.
    imag : ndarray
        Imaginary component of rotated and scaled phasor coordinates.

    Notes
    -----
    The phasor coordinates `real` (:math:`G`) and `imag` (:math:`S`)
    are rotated by `phase` (:math:`\phi`)
    and scaled by `modulation_zero` (:math:`M`)
    around the origin according to:

    .. math::

        g &= M \cdot \cos{\phi}

        s &= M \cdot \sin{\phi}

        G' &= G \cdot g - S \cdot s

        S' &= G \cdot s + S \cdot g

    Examples
    --------
    Use scalar reference coordinates to rotate and scale phasor coordinates:

    >>> phasor_transform(
    ...     [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], 0.1, 0.5
    ... )  # doctest: +NUMBER
    (array([0.0298, 0.0745, 0.119]), array([0.204, 0.259, 0.3135]))

    Use separate reference coordinates for each phasor coordinate:

    >>> phasor_transform(
    ...     [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.2, 0.2, 0.3], [0.5, 0.2, 0.3]
    ... )  # doctest: +NUMBER
    (array([0.00927, 0.0193, 0.0328]), array([0.206, 0.106, 0.1986]))

    """
    if numpy.ndim(phase) == 0 and numpy.ndim(modulation) == 0:
        return _phasor_transform_const(  # type: ignore[no-any-return]
            real,
            imag,
            modulation * numpy.cos(phase),
            modulation * numpy.sin(phase),
        )
    return _phasor_transform(  # type: ignore[no-any-return]
        real, imag, phase, modulation, **kwargs
    )


def phasor_to_polar(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return polar coordinates from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Notes
    -----
    The phasor coordinates `real` (:math:`G`) and `imag` (:math:`S`)
    are converted to polar coordinates `phase` (:math:`\phi`) and
    `modulation` (:math:`M`) according to:

    .. math::

        \phi &= \arctan(S / G)

        M &= \sqrt{G^2 + S^2}

    Returns
    -------
    phase : ndarray
        Angular component of polar coordinates in radians.
    modulation : ndarray
        Radial component of polar coordinates.

    See Also
    --------
    phasorpy.phasor.phasor_from_polar
    :ref:`sphx_glr_tutorials_phasorpy_lifetime_geometry.py`

    Examples
    --------
    Calculate polar coordinates from three phasor coordinates:

    >>> phasor_to_polar([1.0, 0.5, 0.0], [0.0, 0.5, 1.0])  # doctest: +NUMBER
    (array([0, 0.7854, 1.571]), array([1, 0.7071, 1]))

    """
    return _phasor_to_polar(  # type: ignore[no-any-return]
        real, imag, **kwargs
    )


def phasor_from_polar(
    phase: ArrayLike,
    modulation: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return phasor coordinates from polar coordinates.

    Parameters
    ----------
    phase : array_like
        Angular component of polar coordinates in radians.
    modulation : array_like
        Radial component of polar coordinates.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real : ndarray
        Real component of phasor coordinates.
    imag : ndarray
        Imaginary component of phasor coordinates.

    See Also
    --------
    phasorpy.phasor.phasor_to_polar

    Notes
    -----
    The polar coordinates `phase` (:math:`\phi`) and `modulation` (:math:`M`)
    are converted to phasor coordinates `real` (:math:`G`) and
    `imag` (:math:`S`) according to:

    .. math::

        G &= M \cdot \cos{\phi}

        S &= M \cdot \sin{\phi}

    Examples
    --------
    Calculate phasor coordinates from three polar coordinates:

    >>> phasor_from_polar(
    ...     [0.0, math.pi / 4, math.pi / 2], [1.0, math.sqrt(0.5), 1.0]
    ... )  # doctest: +NUMBER
    (array([1, 0.5, 0.0]), array([0, 0.5, 1]))

    """
    return _phasor_from_polar(  # type: ignore[no-any-return]
        phase, modulation, **kwargs
    )


def phasor_to_principal_plane(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    reorient: bool = True,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return multi-harmonic phasor coordinates projected onto principal plane.

    Principal component analysis (PCA) is used to project
    multi-harmonic phasor coordinates onto a plane, along which
    coordinate axes the phasor coordinates have the largest variations.

    The transformed coordinates are not phasor coordinates. However, the
    coordinates can be used in visualization and cursor analysis since
    the transformation is affine (preserving collinearity and ratios
    of distances).

    Parameters
    ----------
    real : array_like
        Real component of multi-harmonic phasor coordinates.
        The first axis is the frequency dimension.
        If less than 2-dimensional, size-1 dimensions are prepended.
    imag : array_like
        Imaginary component of multi-harmonic phasor coordinates.
        Must be of same shape as `real`.
    reorient : bool, optional, default: True
        Reorient coordinates for easier visualization.
        The projected coordinates are rotated and scaled, such that
        the center lies in same quadrant and the projection
        of [1, 0] lies at [1, 0].

    Returns
    -------
    x : ndarray
        X-coordinates of projected phasor coordinates.
        If not `reorient`, this is the coordinate on the first principal axis.
        The shape is ``real.shape[1:]``.
    y : ndarray
        Y-coordinates of projected phasor coordinates.
        If not `reorient`, this is the coordinate on the second principal axis.
    transformation_matrix : ndarray
        Affine transformation matrix used to project phasor coordinates.
        The shape is ``(2, 2 * real.shape[0])``.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_pca.py`

    Notes
    -----

    This implementation does not work with coordinates containing
    undefined NaN values.

    The transformation matrix can be used to project multi-harmonic phasor
    coordinates, where the first axis is the frequency:

    .. code-block:: python

        x, y = numpy.dot(
            numpy.vstack(
                real.reshape(real.shape[0], -1),
                imag.reshape(imag.shape[0], -1),
            ),
            transformation_matrix,
        ).reshape(2, *real.shape[1:])

    An application of PCA to full-harmonic phasor coordinates from MRI signals
    can be found in [1]_.

    References
    ----------
    .. [1] Franssen WMJ, Vergeldt FJ, Bader AN, van Amerongen H, and Terenzi C.
      `Full-harmonics phasor analysis: unravelling multiexponential trends
      in magnetic resonance imaging data
      <https://doi.org/10.1021/acs.jpclett.0c02319>`_.
      *J Phys Chem Lett*, 11(21): 9152-9158 (2020)

    Examples
    --------
    The phasor coordinates of multi-exponential decays may be almost
    indistinguishable at certain frequencies but are separated in the
    projection on the principal plane:

    >>> real = [[0.495, 0.502], [0.354, 0.304]]
    >>> imag = [[0.333, 0.334], [0.301, 0.349]]
    >>> x, y, transformation_matrix = phasor_to_principal_plane(real, imag)
    >>> x, y  # doctest: +SKIP
    (array([0.294, 0.262]), array([0.192, 0.242]))
    >>> transformation_matrix  # doctest: +SKIP
    array([[0.67, 0.33, -0.09, -0.41], [0.52, -0.52, -0.04, 0.44]])

    """
    re, im = numpy.atleast_2d(real, imag)
    if re.shape != im.shape:
        raise ValueError(f'real={re.shape} != imag={im.shape}')

    # reshape to variables in row, observations in column
    frequencies = re.shape[0]
    shape = re.shape[1:]
    re = re.reshape(re.shape[0], -1)
    im = im.reshape(im.shape[0], -1)

    # vector of multi-frequency phasor coordinates
    coordinates = numpy.vstack([re, im])

    # vector of centered coordinates
    center = numpy.nanmean(coordinates, axis=1, keepdims=True)
    coordinates -= center

    # covariance matrix (scatter matrix would also work)
    cov = numpy.cov(coordinates, rowvar=True)

    # calculate eigenvectors
    _, eigvec = numpy.linalg.eigh(cov)

    # projection matrix: two eigenvectors with largest eigenvalues
    transformation_matrix = eigvec.T[-2:][::-1]

    if reorient:
        # for single harmonic, this should restore original coordinates.

        # 1. rotate and scale such that projection of [1, 0] lies at [1, 0]
        x, y = numpy.dot(
            transformation_matrix,
            numpy.vstack(([[1.0]] * frequencies, [[0.0]] * frequencies)),
        )
        x = x.item()
        y = y.item()
        angle = -math.atan2(y, x)
        if angle < 0:
            angle += 2.0 * math.pi
        cos = math.cos(angle)
        sin = math.sin(angle)
        transformation_matrix = numpy.dot(
            [[cos, -sin], [sin, cos]], transformation_matrix
        )
        scale_factor = 1.0 / math.hypot(x, y)
        transformation_matrix = numpy.dot(
            [[scale_factor, 0], [0, scale_factor]], transformation_matrix
        )

        # 2. mirror such that projected center lies in same quadrant
        cs = math.copysign
        x, y = numpy.dot(transformation_matrix, center)
        x = x.item()
        y = y.item()
        transformation_matrix = numpy.dot(
            [
                [-1 if cs(1, x) != cs(1, center[0][0]) else 1, 0],
                [0, -1 if cs(1, y) != cs(1, center[1][0]) else 1],
            ],
            transformation_matrix,
        )

    # project multi-frequency phasor coordinates onto principal plane
    coordinates += center
    coordinates = numpy.dot(transformation_matrix, coordinates)

    return (
        coordinates[0].reshape(shape),  # x coordinates
        coordinates[1].reshape(shape),  # y coordinates
        transformation_matrix,
    )


def phasor_filter_median(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    repeat: int = 1,
    size: int = 3,
    skip_axis: int | Sequence[int] | None = None,
    use_scipy: bool = False,
    num_threads: int | None = None,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return median-filtered phasor coordinates.

    By default, apply a NaN-aware median filter independently to the real
    and imaginary components of phasor coordinates once with a kernel size of 3
    multiplied by the number of dimensions of the input arrays. Return the
    intensity unchanged.

    Parameters
    ----------
    mean : array_like
        Intensity of phasor coordinates.
    real : array_like
        Real component of phasor coordinates to be filtered.
    imag : array_like
        Imaginary component of phasor coordinates to be filtered.
    repeat : int, optional
        Number of times to apply median filter. The default is 1.
    size : int, optional
        Size of median filter kernel. The default is 3.
    skip_axis : int or sequence of int, optional
        Axes in `mean` to exclude from filter.
        By default, all axes except harmonics are included.
    use_scipy : bool, optional
        Use :py:func:`scipy.ndimage.median_filter`.
        This function has undefined behavior if the input arrays contain
        NaN values but is faster when filtering more than 2 dimensions.
        See `issue #87 <https://github.com/phasorpy/phasorpy/issues/87>`_.
    num_threads : int, optional
        Number of OpenMP threads to use for parallelization.
        Applies to filtering in two dimensions when not using scipy.
        By default, multi-threading is disabled.
        If zero, up to half of logical CPUs are used.
        OpenMP may not be available on all platforms.
    **kwargs
        Optional arguments passed to :py:func:`scipy.ndimage.median_filter`.

    Returns
    -------
    mean : ndarray
        Unchanged intensity of phasor coordinates.
    real : ndarray
        Filtered real component of phasor coordinates.
    imag : ndarray
        Filtered imaginary component of phasor coordinates.

    Raises
    ------
    ValueError
        If `repeat` is less than 0.
        If `size` is less than 1.
        The array shapes of `mean`, `real`, and `imag` do not match.

    Examples
    --------
    Apply three times a median filter with a kernel size of three:

    >>> mean, real, imag = phasor_filter_median(
    ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    ...     [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]],
    ...     [[0.3, 0.3, 0.3], [0.6, math.nan, 0.6], [0.4, 0.4, 0.4]],
    ...     size=3,
    ...     repeat=3,
    ... )
    >>> mean
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> real
    array([[0, 0, 0],
           [0.2, 0.2, 0.2],
           [0.2, 0.2, 0.2]])
    >>> imag
    array([[0.3, 0.3, 0.3],
           [0.4, nan, 0.4],
           [0.4, 0.4, 0.4]])

    """
    if repeat < 0:
        raise ValueError(f'{repeat=} < 0')
    if size < 1:
        raise ValueError(f'{size=} < 1')
    if size == 1:
        # no need to filter
        repeat = 0

    mean = numpy.asarray(mean)
    if use_scipy or repeat == 0:  # or using nD numpy filter
        real = numpy.asarray(real)
    elif isinstance(real, numpy.ndarray) and real.dtype == numpy.float32:
        real = real.copy()
    else:
        real = numpy.array(real, numpy.float64, copy=True)
    if use_scipy or repeat == 0:  # or using nD numpy filter
        imag = numpy.asarray(imag)
    elif isinstance(imag, numpy.ndarray) and imag.dtype == numpy.float32:
        imag = imag.copy()
    else:
        imag = numpy.array(imag, numpy.float64, copy=True)

    if mean.shape != real.shape[-mean.ndim if mean.ndim else 1 :]:
        raise ValueError(f'{mean.shape=} != {real.shape=}')
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')

    prepend_axis = mean.ndim + 1 == real.ndim
    _, axes = parse_skip_axis(skip_axis, mean.ndim, prepend_axis)

    # in case mean is also filtered
    # if prepend_axis:
    #     mean = numpy.expand_dims(mean, axis=0)
    # ...
    # if prepend_axis:
    #     mean = numpy.asarray(mean[0])

    if repeat == 0:
        # no need to call filter
        return mean, real, imag

    if use_scipy:
        # use scipy NaN-unaware fallback
        from scipy.ndimage import median_filter

        kwargs.pop('axes', None)

        for _ in range(repeat):
            real = median_filter(real, size=size, axes=axes, **kwargs)
            imag = median_filter(imag, size=size, axes=axes, **kwargs)

        return mean, numpy.asarray(real), numpy.asarray(imag)

    if len(axes) != 2:
        # n-dimensional median filter using numpy
        from numpy.lib.stride_tricks import sliding_window_view

        kernel_shape = tuple(
            size if i in axes else 1 for i in range(real.ndim)
        )
        pad_width = [
            (s // 2, s // 2) if s > 1 else (0, 0) for s in kernel_shape
        ]
        axis = tuple(range(-real.ndim, 0))

        nan_mask = numpy.isnan(real)
        for _ in range(repeat):
            real = numpy.pad(real, pad_width, mode='edge')
            real = sliding_window_view(real, kernel_shape)
            real = numpy.nanmedian(real, axis=axis)
            real = numpy.where(nan_mask, numpy.nan, real)

        nan_mask = numpy.isnan(imag)
        for _ in range(repeat):
            imag = numpy.pad(imag, pad_width, mode='edge')
            imag = sliding_window_view(imag, kernel_shape)
            imag = numpy.nanmedian(imag, axis=axis)
            imag = numpy.where(nan_mask, numpy.nan, imag)

        return mean, real, imag

    # 2-dimensional median filter using optimized Cython implementation
    num_threads = number_threads(num_threads)

    buffer = numpy.empty(
        tuple(real.shape[axis] for axis in axes), dtype=real.dtype
    )

    for index in numpy.ndindex(
        *[real.shape[ax] for ax in range(real.ndim) if ax not in axes]
    ):
        index_list: list[int | slice] = list(index)
        for ax in axes:
            index_list = index_list[:ax] + [slice(None)] + index_list[ax:]
        full_index = tuple(index_list)

        _median_filter_2d(real[full_index], buffer, size, repeat, num_threads)
        _median_filter_2d(imag[full_index], buffer, size, repeat, num_threads)

    return mean, real, imag


def phasor_filter_pawflim(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    sigma: float = 2.0,
    levels: int = 1,
    harmonic: Sequence[int] | None = None,
    skip_axis: int | Sequence[int] | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return pawFLIM wavelet-filtered phasor coordinates.

    This function must only be used with calibrated, unprocessed phasor
    coordinates obtained from FLIM data. The coordinates must not be filtered,
    thresholded, or otherwise pre-processed.

    The pawFLIM wavelet filter is described in [2]_.

    Parameters
    ----------
    mean : array_like
        Intensity of phasor coordinates.
    real : array_like
        Real component of phasor coordinates to be filtered.
        Must have at least two harmonics in the first axis.
    imag : array_like
        Imaginary component of phasor coordinates to be filtered.
        Must have at least two harmonics in the first axis.
    sigma : float, optional
        Significance level to test difference between two phasors.
        Given in terms of the equivalent 1D standard deviations.
        sigma=2 corresponds to ~95% (or 5%) significance.
    levels : int, optional
        Number of levels for wavelet decomposition.
        Controls the maximum averaging area, which has a length of
        :math:`2^level`.
    harmonic : sequence of int or None, optional
        Harmonics included in first axis of `real` and `imag`.
        If None (default), the first axis of `real` and `imag` contains lower
        harmonics starting at and increasing by one.
        All harmonics must have a corresponding half or double harmonic.
    skip_axis : int or sequence of int, optional
        Axes in `mean` to exclude from filter.
        By default, all axes except harmonics are included.

    Returns
    -------
    mean : ndarray
        Unchanged intensity of phasor coordinates.
    real : ndarray
        Filtered real component of phasor coordinates.
    imag : ndarray
        Filtered imaginary component of phasor coordinates.

    Raises
    ------
    ValueError
        If `level` is less than 0.
        The array shapes of `mean`, `real`, and `imag` do not match.
        If `real` and `imag` have no harmonic axis.
        Number of harmonics in `harmonic` is less than 2 or does not match
        the first axis of `real` and `imag`.
        Not all harmonics in `harmonic` have a corresponding half
        or double harmonic.

    References
    ----------
    .. [2] Silberberg M, and Grecco H. `pawFLIM: reducing bias and
      uncertainty to enable lower photon count in FLIM experiments
      <https://doi.org/10.1088/2050-6120/aa72ab>`_.
      *Methods Appl Fluoresc*, 5(2): 024016 (2017)

    Examples
    --------
    Apply a pawFLIM wavelet filter with four significance levels (sigma)
    and three decomposition levels:

    >>> mean, real, imag = phasor_filter_pawflim(
    ...     [[1, 1], [1, 1]],
    ...     [[[0.5, 0.8], [0.5, 0.8]], [[0.2, 0.4], [0.2, 0.4]]],
    ...     [[[0.5, 0.4], [0.5, 0.4]], [[0.4, 0.5], [0.4, 0.5]]],
    ...     sigma=4,
    ...     levels=3,
    ...     harmonic=[1, 2],
    ... )
    >>> mean
    array([[1, 1],
           [1, 1]])
    >>> real
    array([[[0.65, 0.65],
            [0.65, 0.65]],
           [[0.3, 0.3],
            [0.3, 0.3]]])
    >>> imag
    array([[[0.45, 0.45],
            [0.45, 0.45]],
           [[0.45, 0.45],
            [0.45, 0.45]]])

    """
    from pawflim import pawflim  # type: ignore[import-untyped]

    mean = numpy.asarray(mean)
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)

    if levels < 0:
        raise ValueError(f'{levels=} < 0')
    if levels == 0:
        return mean, real, imag

    if mean.shape != real.shape[-mean.ndim if mean.ndim else 1 :]:
        raise ValueError(f'{mean.shape=} != {real.shape=}')
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')

    has_harmonic_axis = mean.ndim + 1 == real.ndim
    if not has_harmonic_axis:
        raise ValueError('no harmonic axis')
    if harmonic is None:
        harmonics, _ = parse_harmonic('all', real.shape[0])
    else:
        harmonics, _ = parse_harmonic(harmonic, None)
    if len(harmonics) < 2:
        raise ValueError(
            'at least two harmonics required, ' f'got {len(harmonics)}'
        )
    if len(harmonics) != real.shape[0]:
        raise ValueError(
            'number of harmonics does not match first axis of real and imag'
        )

    mean = numpy.asarray(numpy.nan_to_num(mean), dtype=float)
    real = numpy.asarray(numpy.nan_to_num(real * mean), dtype=float)
    imag = numpy.asarray(numpy.nan_to_num(imag * mean), dtype=float)

    mean_expanded = numpy.broadcast_to(mean, real.shape).copy()
    original_mean_expanded = mean_expanded.copy()
    real_filtered = real.copy()
    imag_filtered = imag.copy()

    _, axes = parse_skip_axis(skip_axis, mean.ndim, True)

    for index in numpy.ndindex(
        *(
            real.shape[ax]
            for ax in range(real.ndim)
            if ax not in axes and ax != 0
        )
    ):
        index_list: list[int | slice] = list(index)
        for ax in axes:
            index_list = index_list[:ax] + [slice(None)] + index_list[ax:]
        full_index = tuple(index_list)

        processed_harmonics = set()

        for h in harmonics:
            if h in processed_harmonics and (
                h * 4 in harmonics or h * 2 not in harmonics
            ):
                continue
            if h * 2 not in harmonics:
                raise ValueError(
                    f'harmonic {h} does not have a corresponding half '
                    f'or double harmonic in {harmonics}'
                )
            n = harmonics.index(h)
            n2 = harmonics.index(h * 2)

            complex_phasor = numpy.empty(
                (3, *original_mean_expanded[n][full_index].shape),
                dtype=complex,
            )
            complex_phasor[0] = original_mean_expanded[n][full_index]
            complex_phasor[1] = real[n][full_index] + 1j * imag[n][full_index]
            complex_phasor[2] = (
                real[n2][full_index] + 1j * imag[n2][full_index]
            )

            complex_phasor = pawflim(
                complex_phasor, n_sigmas=sigma, levels=levels
            )

            for i, idx in enumerate([n, n2]):
                if harmonics[idx] in processed_harmonics:
                    continue
                mean_expanded[idx][full_index] = complex_phasor[0].real
                real_filtered[idx][full_index] = complex_phasor[i + 1].real
                imag_filtered[idx][full_index] = complex_phasor[i + 1].imag

            processed_harmonics.add(h)
            processed_harmonics.add(h * 2)

    with numpy.errstate(divide='ignore', invalid='ignore'):
        real = numpy.asarray(numpy.divide(real_filtered, mean_expanded))
        imag = numpy.asarray(numpy.divide(imag_filtered, mean_expanded))

    return mean, real, imag


def phasor_threshold(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    mean_min: ArrayLike | None = None,
    mean_max: ArrayLike | None = None,
    *,
    real_min: ArrayLike | None = None,
    real_max: ArrayLike | None = None,
    imag_min: ArrayLike | None = None,
    imag_max: ArrayLike | None = None,
    phase_min: ArrayLike | None = None,
    phase_max: ArrayLike | None = None,
    modulation_min: ArrayLike | None = None,
    modulation_max: ArrayLike | None = None,
    open_interval: bool = False,
    detect_harmonics: bool = True,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return phasor coordinates with values outside interval replaced by NaN.

    Interval thresholds can be set for mean intensity, real and imaginary
    coordinates, and phase and modulation.
    Phasor coordinates smaller than minimum thresholds or larger than maximum
    thresholds are replaced with NaN.
    No threshold is applied by default.
    NaNs in `mean` or any `real` and `imag` harmonic are propagated to
    `mean` and all harmonics in `real` and `imag`.

    Parameters
    ----------
    mean : array_like
        Intensity of phasor coordinates.
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    mean_min : array_like, optional
        Lower threshold for mean intensity.
    mean_max : array_like, optional
        Upper threshold for mean intensity.
    real_min : array_like, optional
        Lower threshold for real coordinates.
    real_max : array_like, optional
        Upper threshold for real coordinates.
    imag_min : array_like, optional
        Lower threshold for imaginary coordinates.
    imag_max : array_like, optional
        Upper threshold for imaginary coordinates.
    phase_min : array_like, optional
        Lower threshold for phase angle.
    phase_max : array_like, optional
        Upper threshold for phase angle.
    modulation_min : array_like, optional
        Lower threshold for modulation.
    modulation_max : array_like, optional
        Upper threshold for modulation.
    open_interval : bool, optional
        If true, the interval is open, and the threshold values are
        not included in the interval.
        If false (default), the interval is closed, and the threshold values
        are included in the interval.
    detect_harmonics : bool, optional
        By default, detect presence of multiple harmonics from array shapes.
        If false, no harmonics are assumed to be present, and the function
        behaves like a numpy universal function.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    mean : ndarray
        Thresholded intensity of phasor coordinates.
    real : ndarray
        Thresholded real component of phasor coordinates.
    imag : ndarray
        Thresholded imaginary component of phasor coordinates.

    Examples
    --------
    Set phasor coordinates to NaN if mean intensity is smaller than 1.1:

    >>> phasor_threshold([1, 2, 3], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], 1.1)
    (array([nan, 2, 3]), array([nan, 0.2, 0.3]), array([nan, 0.5, 0.6]))

    Set phasor coordinates to NaN if real component is smaller than 0.15 or
    larger than 0.25:

    >>> phasor_threshold(
    ...     [1.0, 2.0, 3.0],
    ...     [0.1, 0.2, 0.3],
    ...     [0.4, 0.5, 0.6],
    ...     real_min=0.15,
    ...     real_max=0.25,
    ... )
    (array([nan, 2, nan]), array([nan, 0.2, nan]), array([nan, 0.5, nan]))

    Apply NaNs to other input arrays:

    >>> phasor_threshold(
    ...     [numpy.nan, 2, 3], [0.1, 0.2, 0.3], [0.4, 0.5, numpy.nan]
    ... )
    (array([nan, 2, nan]), array([nan, 0.2, nan]), array([nan, 0.5, nan]))

    """
    threshold_mean_only = None
    if mean_min is None:
        mean_min = numpy.nan
    else:
        threshold_mean_only = True
    if mean_max is None:
        mean_max = numpy.nan
    else:
        threshold_mean_only = True
    if real_min is None:
        real_min = numpy.nan
    else:
        threshold_mean_only = False
    if real_max is None:
        real_max = numpy.nan
    else:
        threshold_mean_only = False
    if imag_min is None:
        imag_min = numpy.nan
    else:
        threshold_mean_only = False
    if imag_max is None:
        imag_max = numpy.nan
    else:
        threshold_mean_only = False
    if phase_min is None:
        phase_min = numpy.nan
    else:
        threshold_mean_only = False
    if phase_max is None:
        phase_max = numpy.nan
    else:
        threshold_mean_only = False
    if modulation_min is None:
        modulation_min = numpy.nan
    else:
        threshold_mean_only = False
    if modulation_max is None:
        modulation_max = numpy.nan
    else:
        threshold_mean_only = False

    if detect_harmonics:
        mean = numpy.asarray(mean)
        real = numpy.asarray(real)
        imag = numpy.asarray(imag)

        shape = numpy.broadcast_shapes(mean.shape, real.shape, imag.shape)
        ndim = len(shape)

        has_harmonic_axis = (
            # detect multi-harmonic in axis 0
            mean.ndim + 1 == ndim
            and real.shape == shape
            and imag.shape == shape
            and mean.shape == shape[-mean.ndim if mean.ndim else 1 :]
        )
    else:
        has_harmonic_axis = False

    if threshold_mean_only is None:
        mean, real, imag = _phasor_threshold_nan(mean, real, imag, **kwargs)

    elif threshold_mean_only:
        mean_func = (
            _phasor_threshold_mean_open
            if open_interval
            else _phasor_threshold_mean_closed
        )
        mean, real, imag = mean_func(
            mean, real, imag, mean_min, mean_max, **kwargs
        )

    else:
        func = (
            _phasor_threshold_open
            if open_interval
            else _phasor_threshold_closed
        )
        mean, real, imag = func(
            mean,
            real,
            imag,
            mean_min,
            mean_max,
            real_min,
            real_max,
            imag_min,
            imag_max,
            phase_min,
            phase_max,
            modulation_min,
            modulation_max,
            **kwargs,
        )

    mean = numpy.asarray(mean)
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    if has_harmonic_axis and mean.ndim > 0:
        # propagate NaN to all dimensions
        mean = numpy.mean(mean, axis=0, keepdims=True)
        mask = numpy.where(numpy.isnan(mean), numpy.nan, 1.0)
        numpy.multiply(real, mask, out=real)
        numpy.multiply(imag, mask, out=imag)
        # remove harmonic dimension created by broadcasting
        mean = numpy.asarray(numpy.asarray(mean)[0])

    return mean, real, imag


def phasor_nearest_neighbor(
    real: ArrayLike,
    imag: ArrayLike,
    neighbor_real: ArrayLike,
    neighbor_imag: ArrayLike,
    /,
    *,
    values: ArrayLike | None = None,
    dtype: DTypeLike | None = None,
    distance_max: float | None = None,
    num_threads: int | None = None,
) -> NDArray[Any]:
    """Return indices or values of nearest neighbors from other coordinates.

    For each phasor coordinate, find the nearest neighbor in another set of
    phasor coordinates and return its flat index. If more than one neighbor
    has the same distance, return the smallest index.

    For phasor coordinates that are NaN, or have a distance to the nearest
    neighbor that is larger than `distance_max`, return an index of -1.

    If `values` are provided, return the values corresponding to the nearest
    neighbor coordinates instead of indices. Return NaN values for indices
    that are -1.

    This function does not support multi-harmonic, multi-channel, or
    multi-frequency phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    neighbor_real : array_like
        Real component of neighbor phasor coordinates.
    neighbor_imag : array_like
        Imaginary component of neighbor phasor coordinates.
    values : array_like, optional
        Array of values corresponding to neighbor coordinates.
        If provided, return the values corresponding to the nearest
        neighbor coordinates.
    distance_max : float, optional
        Maximum Euclidean distance to consider a neighbor valid.
        By default, all neighbors are considered.
    dtype : dtype_like, optional
        Floating point data type used for calculation and output values.
        Either `float32` or `float64`. The default is `float64`.
    num_threads : int, optional
        Number of OpenMP threads to use for parallelization.
        By default, multi-threading is disabled.
        If zero, up to half of logical CPUs are used.
        OpenMP may not be available on all platforms.

    Returns
    -------
    nearest : ndarray
        Flat indices (or the corresponding values if provided) of the nearest
        neighbor coordinates.

    Raises
    ------
    ValueError
        If the shapes of `real`, and `imag` do not match.
        If the shapes of `neighbor_real` and `neighbor_imag` do not match.
        If the shapes of `values` and `neighbor_real` do not match.
        If `distance_max` is less than or equal to zero.

    See Also
    --------
    :ref:`sphx_glr_tutorials_applications_phasorpy_fret_efficiency.py`

    Notes
    -----
    This function uses linear search, which is inefficient for large
    number of coordinates or neighbors.
    ``scipy.spatial.KDTree.query()`` would be more efficient in those cases.
    However, KDTree is known to return non-deterministic results in case of
    multiple neighbors with the same distance.

    Examples
    --------
    >>> phasor_nearest_neighbor(
    ...     [0.1, 0.5, numpy.nan],
    ...     [0.1, 0.5, numpy.nan],
    ...     [0, 0.4],
    ...     [0, 0.4],
    ...     values=[10, 20],
    ... )
    array([10, 20, nan])

    """
    dtype = numpy.dtype(dtype)
    if dtype.char not in {'f', 'd'}:
        raise ValueError(f'{dtype=} is not a floating point type')

    real = numpy.ascontiguousarray(real, dtype=dtype)
    imag = numpy.ascontiguousarray(imag, dtype=dtype)
    neighbor_real = numpy.ascontiguousarray(neighbor_real, dtype=dtype)
    neighbor_imag = numpy.ascontiguousarray(neighbor_imag, dtype=dtype)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if neighbor_real.shape != neighbor_imag.shape:
        raise ValueError(f'{neighbor_real.shape=} != {neighbor_imag.shape=}')

    shape = real.shape
    real = real.ravel()
    imag = imag.ravel()
    neighbor_real = neighbor_real.ravel()
    neighbor_imag = neighbor_imag.ravel()

    indices = numpy.empty(
        real.shape, numpy.min_scalar_type(-neighbor_real.size)
    )

    if distance_max is None:
        distance_max = numpy.inf
    else:
        distance_max = float(distance_max)
        if distance_max <= 0:
            raise ValueError(f'{distance_max=} <= 0')

    num_threads = number_threads(num_threads)

    _nearest_neighbor_2d(
        indices,
        real,
        imag,
        neighbor_real,
        neighbor_imag,
        distance_max,
        num_threads,
    )

    if values is None:
        return numpy.asarray(indices.reshape(shape))

    values = numpy.ascontiguousarray(values, dtype=dtype).ravel()
    if values.shape != neighbor_real.shape:
        raise ValueError(f'{values.shape=} != {neighbor_real.shape=}')

    nearest_values = values[indices]
    nearest_values[indices == -1] = numpy.nan

    return numpy.asarray(nearest_values.reshape(shape))


def phasor_center(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    skip_axis: int | Sequence[int] | None = None,
    method: Literal['mean', 'median'] = 'mean',
    nan_safe: bool = True,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return center of phasor coordinates.

    Parameters
    ----------
    mean : array_like
        Intensity of phasor coordinates.
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    skip_axis : int or sequence of int, optional
        Axes in `mean` to excluded from center calculation.
        By default, all axes except harmonics are included.
    method : str, optional
        Method used for center calculation:

        - ``'mean'``: Arithmetic mean of phasor coordinates.
        - ``'median'``: Spatial median of phasor coordinates.

    nan_safe : bool, optional
        Ensure `method` is applied to same elements of input arrays.
        By default, distribute NaNs among input arrays before applying
        `method`. May be disabled if phasor coordinates were filtered by
        :py:func:`phasor_threshold`.
    **kwargs
        Optional arguments passed to :py:func:`numpy.nanmean` or
        :py:func:`numpy.nanmedian`.

    Returns
    -------
    mean_center : ndarray
        Intensity center coordinates.
    real_center : ndarray
        Real center coordinates.
    imag_center : ndarray
        Imaginary center coordinates.

    Raises
    ------
    ValueError
        If the specified method is not supported.
        If the shapes of `mean`, `real`, and `imag` do not match.

    Examples
    --------
    Compute center coordinates with the default 'mean' method:

    >>> phasor_center(
    ...     [2, 1, 2], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]
    ... )  # doctest: +NUMBER
    (1.67, 0.2, 0.5)

    Compute center coordinates with the 'median' method:

    >>> phasor_center(
    ...     [1, 2, 3], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], method='median'
    ... )
    (2.0, 0.2, 0.5)

    """
    methods = {
        'mean': _mean,
        'median': _median,
    }
    if method not in methods:
        raise ValueError(
            f'Method not supported, supported methods are: '
            f"{', '.join(methods)}"
        )

    mean = numpy.asarray(mean)
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if mean.shape != real.shape[-mean.ndim if mean.ndim else 1 :]:
        raise ValueError(f'{mean.shape=} != {real.shape=}')

    prepend_axis = mean.ndim + 1 == real.ndim
    _, axis = parse_skip_axis(skip_axis, mean.ndim, prepend_axis)
    if prepend_axis:
        mean = numpy.expand_dims(mean, axis=0)

    if nan_safe:
        mean, real, imag = phasor_threshold(mean, real, imag)

    mean, real, imag = methods[method](mean, real, imag, axis=axis, **kwargs)

    if prepend_axis:
        mean = numpy.asarray(mean[0])
    return mean, real, imag


def _mean(
    mean: NDArray[Any],
    real: NDArray[Any],
    imag: NDArray[Any],
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return mean center of phasor coordinates."""
    real = numpy.nanmean(real * mean, **kwargs)
    imag = numpy.nanmean(imag * mean, **kwargs)
    mean = numpy.nanmean(mean, **kwargs)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        real /= mean
        imag /= mean
    return mean, real, imag


def _median(
    mean: NDArray[Any],
    real: NDArray[Any],
    imag: NDArray[Any],
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return spatial median center of phasor coordinates."""
    return (
        numpy.nanmedian(mean, **kwargs),
        numpy.nanmedian(real, **kwargs),
        numpy.nanmedian(imag, **kwargs),
    )
