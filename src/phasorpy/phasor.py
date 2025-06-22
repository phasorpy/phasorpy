"""Calculate, convert, calibrate, and reduce phasor coordinates.

The ``phasorpy.phasor`` module provides functions to:

- calculate phasor coordinates from time-resolved and spectral signals:

  - :py:func:`phasor_from_signal`

- synthesize signals from phasor coordinates or lifetimes:

  - :py:func:`phasor_to_signal`
  - :py:func:`lifetime_to_signal`

- convert between phasor coordinates and single- or multi-component
  fluorescence lifetimes:

  - :py:func:`phasor_from_lifetime`
  - :py:func:`phasor_from_apparent_lifetime`
  - :py:func:`phasor_to_apparent_lifetime`
  - :py:func:`phasor_to_normal_lifetime`

- convert to and from polar coordinates (phase and modulation):

  - :py:func:`phasor_from_polar`
  - :py:func:`phasor_to_polar`
  - :py:func:`polar_from_apparent_lifetime`
  - :py:func:`polar_to_apparent_lifetime`

- transform phasor coordinates:

  - :py:func:`phasor_transform`
  - :py:func:`phasor_multiply`
  - :py:func:`phasor_divide`
  - :py:func:`phasor_normalize`

- calibrate phasor coordinates with reference of known fluorescence
  lifetime:

  - :py:func:`phasor_calibrate`
  - :py:func:`polar_from_reference`
  - :py:func:`polar_from_reference_phasor`

- reduce dimensionality of arrays of phasor coordinates:

  - :py:func:`phasor_center`
  - :py:func:`phasor_to_principal_plane`

- calculate phasor coordinates for FRET donor and acceptor channels:

  - :py:func:`phasor_from_fret_donor`
  - :py:func:`phasor_from_fret_acceptor`

- convert between single component lifetimes and optimal frequency:

  - :py:func:`lifetime_to_frequency`
  - :py:func:`lifetime_from_frequency`

- convert between fractional intensities and pre-exponential amplitudes:

  - :py:func:`lifetime_fraction_from_amplitude`
  - :py:func:`lifetime_fraction_to_amplitude`

- calculate phasor coordinates on universal semicircle at other harmonics:

  - :py:func:`phasor_at_harmonic`

- filter phasor coordinates:

  - :py:func:`phasor_filter_median`
  - :py:func:`phasor_filter_pawflim`
  - :py:func:`phasor_threshold`

- find nearest neighbor phasor coordinates from another set of phasor coordinates:

  - :py:func:`phasor_nearest_neighbor`

"""

from __future__ import annotations

__all__ = [
    'lifetime_fraction_from_amplitude',
    'lifetime_fraction_to_amplitude',
    'lifetime_from_frequency',
    'lifetime_to_frequency',
    'lifetime_to_signal',
    'phasor_at_harmonic',
    'phasor_calibrate',
    'phasor_center',
    'phasor_divide',
    'phasor_filter_median',
    'phasor_filter_pawflim',
    'phasor_from_apparent_lifetime',
    'phasor_from_fret_acceptor',
    'phasor_from_fret_donor',
    'phasor_from_lifetime',
    'phasor_from_polar',
    'phasor_from_signal',
    'phasor_multiply',
    'phasor_nearest_neighbor',
    'phasor_normalize',
    'phasor_semicircle',
    'phasor_semicircle_intersect',
    'phasor_threshold',
    'phasor_to_apparent_lifetime',
    'phasor_to_complex',
    'phasor_to_normal_lifetime',
    'phasor_to_polar',
    'phasor_to_principal_plane',
    'phasor_to_signal',
    'phasor_transform',
    'polar_from_apparent_lifetime',
    'polar_from_reference',
    'polar_from_reference_phasor',
    'polar_to_apparent_lifetime',
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
    _gaussian_signal,
    _intersect_semicircle_line,
    _median_filter_2d,
    _nearest_neighbor_2d,
    _phasor_at_harmonic,
    _phasor_divide,
    _phasor_from_apparent_lifetime,
    _phasor_from_fret_acceptor,
    _phasor_from_fret_donor,
    _phasor_from_lifetime,
    _phasor_from_polar,
    _phasor_from_signal,
    _phasor_from_single_lifetime,
    _phasor_multiply,
    _phasor_threshold_closed,
    _phasor_threshold_mean_closed,
    _phasor_threshold_mean_open,
    _phasor_threshold_nan,
    _phasor_threshold_open,
    _phasor_to_apparent_lifetime,
    _phasor_to_normal_lifetime,
    _phasor_to_polar,
    _phasor_transform,
    _phasor_transform_const,
    _polar_from_apparent_lifetime,
    _polar_from_reference,
    _polar_from_reference_phasor,
    _polar_from_single_lifetime,
    _polar_to_apparent_lifetime,
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
    :ref:`sphx_glr_tutorials_benchmarks_phasorpy_phasor_from_signal.py`

    Notes
    -----
    The normalized phasor coordinates `real` (:math:`G`), `imag` (:math:`S`),
    and average intensity `mean` (:math:`F_{DC}`) are calculated from
    :math:`K\ge3` samples of the signal :math:`F` af `harmonic` :math:`h`
    according to:

    .. math::

        F_{DC} &= \frac{1}{K} \sum_{k=0}^{K-1} F_{k}

        G &= \frac{1}{K} \sum_{k=0}^{K-1} F_{k}
        \cos{\left (2 \pi h \frac{k}{K} \right )} \cdot \frac{1}{F_{DC}}

        S &= \frac{1}{K} \sum_{k=0}^{K-1} F_{k}
        \sin{\left (2 \pi h \frac{k}{K} \right )} \cdot \frac{1}{F_{DC}}

    If :math:`F_{DC} = 0`, the phasor coordinates are undefined
    (:math:`NaN` or :math:`\infty`).
    Use `NaN`-aware software to further process the phasor coordinates.

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
    or signal mean contain `NaN` values.

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


def lifetime_to_signal(
    frequency: float,
    lifetime: ArrayLike,
    fraction: ArrayLike | None = None,
    *,
    mean: ArrayLike | None = None,
    background: ArrayLike | None = None,
    samples: int = 64,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
    zero_phase: float | None = None,
    zero_stdev: float | None = None,
    preexponential: bool = False,
    unit_conversion: float = 1e-3,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    r"""Return synthetic signal from lifetime components.

    Return synthetic signal, instrument response function (IRF), and
    time axis, sampled over one period of the fundamental frequency.
    The signal is convoluted with the IRF, which is approximated by a
    normal distribution.

    Parameters
    ----------
    frequency : float
        Fundamental laser pulse or modulation frequency in MHz.
    lifetime : array_like
        Lifetime components in ns.
    fraction : array_like, optional
        Fractional intensities or pre-exponential amplitudes of the lifetime
        components. Fractions are normalized to sum to 1.
        Must be specified if `lifetime` is not a scalar.
    mean : array_like, optional, default: 1.0
        Average signal intensity (DC). Must be scalar for now.
    background : array_like, optional, default: 0.0
        Background signal intensity. Must be smaller than `mean`.
    samples : int, default: 64
        Number of signal samples to return. Must be at least 16.
    harmonic : int, sequence of int, or 'all', optional, default: 'all'
        Harmonics used to synthesize signal.
        If `'all'`, all harmonics are used.
        Else, harmonics must be at least one and no larger than half of
        `samples`.
        Use `'all'` to synthesize an exponential time-domain decay signal,
        or `1` to synthesize a homodyne signal.
    zero_phase : float, optional
        Position of instrument response function in radians.
        Must be in range [0, pi]. The default is the 8th sample.
    zero_stdev : float, optional
        Standard deviation of instrument response function in radians.
        Must be at least 1.5 samples and no more than one tenth of samples
        to allow for sufficient sampling of the function.
        The default is 1.5 samples. Increase `samples` to narrow the IRF.
    preexponential : bool, optional, default: False
        If true, `fraction` values are pre-exponential amplitudes,
        else fractional intensities.
    unit_conversion : float, optional, default: 1e-3
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.

    Returns
    -------
    signal : ndarray
        Signal generated from lifetimes at frequency, convoluted with
        instrument response function.
    zero : ndarray
        Instrument response function.
    time : ndarray
        Time for each sample in signal in units of `lifetime`.

    See Also
    --------
    phasorpy.phasor.phasor_from_lifetime
    phasorpy.phasor.phasor_to_signal
    :ref:`sphx_glr_tutorials_api_phasorpy_lifetime_to_signal.py`

    Notes
    -----
    This implementation is based on an inverse digital Fourier transform (DFT).
    Because DFT cannot be used on signals with discontinuities
    (for example, an exponential decay starting at zero) without producing
    strong artifacts (ripples), the signal is convoluted with a continuous
    instrument response function (IRF). The minimum width of the IRF is
    limited due to sampling requirements.

    Examples
    --------
    Synthesize a multi-exponential time-domain decay signal for two
    lifetime components of 4.2 and 0.9 ns at 40 MHz:

    >>> signal, zero, times = lifetime_to_signal(
    ...     40, [4.2, 0.9], fraction=[0.8, 0.2], samples=16
    ... )
    >>> signal  # doctest: +NUMBER
    array([0.2846, 0.1961, 0.1354, ..., 0.8874, 0.6029, 0.4135])

    Synthesize a homodyne frequency-domain waveform signal for
    a single lifetime:

    >>> signal, zero, times = lifetime_to_signal(
    ...     40.0, 4.2, samples=16, harmonic=1
    ... )
    >>> signal  # doctest: +NUMBER
    array([0.2047, -0.05602, -0.156, ..., 1.471, 1.031, 0.5865])

    """
    if harmonic is None:
        harmonic = 'all'
    all_hamonics = harmonic == 'all'
    harmonic, _ = parse_harmonic(harmonic, samples // 2)

    if samples < 16:
        raise ValueError(f'{samples=} < 16')

    if background is None:
        background = 0.0
    background = numpy.asarray(background)

    if mean is None:
        mean = 1.0
    mean = numpy.asarray(mean)
    mean -= background
    if numpy.any(mean < 0.0):
        raise ValueError('mean - background must not be less than zero')

    scale = samples / (2.0 * math.pi)
    if zero_phase is None:
        zero_phase = 8.0 / scale
    phase = zero_phase * scale  # in sample units
    if zero_stdev is None:
        zero_stdev = 1.5 / scale
    stdev = zero_stdev * scale  # in sample units

    if zero_phase < 0 or zero_phase > 2.0 * math.pi:
        raise ValueError(f'{zero_phase=} out of range [0, 2 pi]')
    if stdev < 1.5:
        raise ValueError(
            f'{zero_stdev=} < {1.5 / scale} cannot be sampled sufficiently'
        )
    if stdev >= samples / 10:
        raise ValueError(f'{zero_stdev=} > pi / 5 not supported')

    frequencies = numpy.atleast_1d(frequency)
    if frequencies.size > 1 or frequencies[0] <= 0.0:
        raise ValueError('frequency must be scalar and positive')
    frequencies = numpy.linspace(
        frequency, samples // 2 * frequency, samples // 2
    )
    frequencies = frequencies[[h - 1 for h in harmonic]]

    real, imag = phasor_from_lifetime(
        frequencies,
        lifetime,
        fraction,
        preexponential=preexponential,
        unit_conversion=unit_conversion,
    )
    real, imag = numpy.atleast_1d(real, imag)

    zero = numpy.zeros(samples, dtype=numpy.float64)
    _gaussian_signal(zero, phase, stdev)
    zero_mean, zero_real, zero_imag = phasor_from_signal(
        zero, harmonic=harmonic
    )
    if real.ndim > 1:
        # make broadcastable with real and imag
        zero_real = zero_real[:, None]
        zero_imag = zero_imag[:, None]
    if not all_hamonics:
        zero = phasor_to_signal(
            zero_mean, zero_real, zero_imag, samples=samples, harmonic=harmonic
        )

    phasor_multiply(real, imag, zero_real, zero_imag, out=(real, imag))

    if len(harmonic) == 1:
        harmonic = harmonic[0]
    signal = phasor_to_signal(
        mean, real, imag, samples=samples, harmonic=harmonic
    )
    signal += numpy.asarray(background)

    time = numpy.linspace(0, 1.0 / (unit_conversion * frequency), samples)

    return signal.squeeze(), zero.squeeze(), time


def phasor_semicircle(
    samples: int = 101, /
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    r"""Return equally spaced phasor coordinates on universal semicircle.

    Parameters
    ----------
    samples : int, optional, default: 101
        Number of coordinates to return.

    Returns
    -------
    real : ndarray
        Real component of phasor coordinates on universal semicircle.
    imag : ndarray
        Imaginary component of phasor coordinates on universal semicircle.

    Raises
    ------
    ValueError
        The number of `samples` is smaller than 1.

    Notes
    -----
    If more than one sample, the first and last phasor coordinates returned
    are ``(0, 0)`` and ``(1, 0)``.
    The center coordinate, if any, is ``(0.5, 0.5)``.

    The universal semicircle is composed of the phasor coordinates of
    single lifetime components, where the relation of polar coordinates
    (phase :math:`\phi` and modulation :math:`M`) is:

    .. math::

        M = \cos{\phi}

    Examples
    --------
    Calculate three phasor coordinates on universal semicircle:

    >>> phasor_semicircle(3)  # doctest: +NUMBER
    (array([0, 0.5, 1]), array([0.0, 0.5, 0]))

    """
    if samples < 1:
        raise ValueError(f'{samples=} < 1')
    arange = numpy.linspace(math.pi, 0.0, samples)
    real = numpy.cos(arange)
    real += 1.0
    real *= 0.5
    imag = numpy.sin(arange)
    imag *= 0.5
    return real, imag


def phasor_semicircle_intersect(
    real0: ArrayLike,
    imag0: ArrayLike,
    real1: ArrayLike,
    imag1: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return intersection of line through phasors with universal semicircle.

    Return the phasor coordinates of two intersections of the universal
    semicircle with the line between two phasor coordinates.
    Return NaN if the line does not intersect the semicircle.

    Parameters
    ----------
    real0 : array_like
        Real component of first set of phasor coordinates.
    imag0 : array_like
        Imaginary component of first set of phasor coordinates.
    real1 : array_like
        Real component of second set of phasor coordinates.
    imag1 : array_like
        Imaginary component of second set of phasor coordinates.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real0 : ndarray
        Real component of first intersect of phasors with semicircle.
    imag0 : ndarray
        Imaginary component of first intersect of phasors with semicircle.
    real1 : ndarray
        Real component of second intersect of phasors with semicircle.
    imag1 : ndarray
        Imaginary component of second intersect of phasors with semicircle.

    Examples
    --------
    Calculate two intersects of a line through two phasor coordinates
    with the universal semicircle:

    >>> phasor_semicircle_intersect(0.2, 0.25, 0.6, 0.25)  # doctest: +NUMBER
    (0.066, 0.25, 0.933, 0.25)

    The line between two phasor coordinates may not intersect the semicircle
    at two points:

    >>> phasor_semicircle_intersect(0.2, 0.0, 0.6, 0.25)  # doctest: +NUMBER
    (nan, nan, 0.817, 0.386)

    """
    return _intersect_semicircle_line(  # type: ignore[no-any-return]
        real0, imag0, real1, imag1, **kwargs
    )


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
    and (:math:`S`) are undefined (:math:`NaN` or :math:`\infty`).

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


def phasor_calibrate(
    real: ArrayLike,
    imag: ArrayLike,
    reference_mean: ArrayLike,
    reference_real: ArrayLike,
    reference_imag: ArrayLike,
    /,
    frequency: ArrayLike,
    lifetime: ArrayLike,
    *,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
    skip_axis: int | Sequence[int] | None = None,
    fraction: ArrayLike | None = None,
    preexponential: bool = False,
    unit_conversion: float = 1e-3,
    method: Literal['mean', 'median'] = 'mean',
    nan_safe: bool = True,
    reverse: bool = False,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return calibrated/referenced phasor coordinates.

    Calibration of phasor coordinates from time-resolved measurements is
    necessary to account for the instrument response function (IRF) and delays
    in the electronics.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates to be calibrated.
    imag : array_like
        Imaginary component of phasor coordinates to be calibrated.
    reference_mean : array_like or None
        Intensity of phasor coordinates from reference of known lifetime.
        Used to re-normalize averaged phasor coordinates.
    reference_real : array_like
        Real component of phasor coordinates from reference of known lifetime.
        Must be measured with the same instrument setting as the phasor
        coordinates to be calibrated. Dimensions must be the same as `real`.
    reference_imag : array_like
        Imaginary component of phasor coordinates from reference of known
        lifetime.
        Must be measured with the same instrument setting as the phasor
        coordinates to be calibrated.
    frequency : array_like
        Fundamental laser pulse or modulation frequency in MHz.
    lifetime : array_like
        Lifetime components in ns. Must be scalar or one-dimensional.
    harmonic : int, sequence of int, or 'all', default: 1
        Harmonics included in `real` and `imag`.
        If an integer, the harmonics at which `real` and `imag` were acquired
        or calculated.
        If a sequence, the harmonics included in the first axis of `real` and
        `imag`.
        If `'all'`, the first axis of `real` and `imag` contains lower
        harmonics.
        The default is the first harmonic (fundamental frequency).
    skip_axis : int or sequence of int, optional
        Axes in `reference_mean` to exclude from reference center calculation.
        By default, all axes except harmonics are included.
    fraction : array_like, optional
        Fractional intensities or pre-exponential amplitudes of the lifetime
        components. Fractions are normalized to sum to 1.
        Must be same size as `lifetime`.
    preexponential : bool, optional
        If true, `fraction` values are pre-exponential amplitudes,
        else fractional intensities (default).
    unit_conversion : float, optional
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    method : str, optional
        Method used for calculating center of reference phasor coordinates:

        - ``'mean'``: Arithmetic mean.
        - ``'median'``: Spatial median.

    nan_safe : bool, optional
        Ensure `method` is applied to same elements of reference arrays.
        By default, distribute NaNs among reference arrays before applying
        `method`.
    reverse : bool, optional
        Reverse calibration.

    Returns
    -------
    real : ndarray
        Calibrated real component of phasor coordinates.
    imag : ndarray
        Calibrated imaginary component of phasor coordinates.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag`, or `reference_real` and
        `reference_imag` do not match.
        Number of harmonics or frequencies does not match the first axis
        of `real` and `imag`.

    See Also
    --------
    phasorpy.phasor.phasor_transform
    phasorpy.phasor.polar_from_reference_phasor
    phasorpy.phasor.phasor_center
    phasorpy.phasor.phasor_from_lifetime

    Notes
    -----
    This function is a convenience wrapper for the following operations:

    .. code-block:: python

        phasor_transform(
            real,
            imag,
            *polar_from_reference_phasor(
                *phasor_center(
                    reference_mean,
                    reference_real,
                    reference_imag,
                    skip_axis,
                    method,
                    nan_safe,
                )[1:],
                *phasor_from_lifetime(
                    frequency,
                    lifetime,
                    fraction,
                    preexponential,
                    unit_conversion,
                ),
            ),
        )

    Calibration can be reversed such that

    .. code-block:: python

        real, imag == phasor_calibrate(
            *phasor_calibrate(real, imag, *args, **kwargs),
            *args,
            reverse=True,
            **kwargs
        )

    Examples
    --------
    >>> phasor_calibrate(
    ...     [0.1, 0.2, 0.3],
    ...     [0.4, 0.5, 0.6],
    ...     [1.0, 1.0, 1.0],
    ...     [0.2, 0.3, 0.4],
    ...     [0.5, 0.6, 0.7],
    ...     frequency=80,
    ...     lifetime=4,
    ... )  # doctest: +NUMBER
    (array([0.0658, 0.132, 0.198]), array([0.2657, 0.332, 0.399]))

    Undo the previous calibration:

    >>> phasor_calibrate(
    ...     [0.0658, 0.132, 0.198],
    ...     [0.2657, 0.332, 0.399],
    ...     [1.0, 1.0, 1.0],
    ...     [0.2, 0.3, 0.4],
    ...     [0.5, 0.6, 0.7],
    ...     frequency=80,
    ...     lifetime=4,
    ...     reverse=True,
    ... )  # doctest: +NUMBER
    (array([0.1, 0.2, 0.3]), array([0.4, 0.5, 0.6]))

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    reference_mean = numpy.asarray(reference_mean)
    reference_real = numpy.asarray(reference_real)
    reference_imag = numpy.asarray(reference_imag)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if reference_real.shape != reference_imag.shape:
        raise ValueError(f'{reference_real.shape=} != {reference_imag.shape=}')

    has_harmonic_axis = reference_mean.ndim + 1 == reference_real.ndim
    harmonic, _ = parse_harmonic(
        harmonic,
        (
            reference_real.shape[0]
            if has_harmonic_axis
            and isinstance(harmonic, str)
            and harmonic == 'all'
            else None
        ),
    )

    frequency = numpy.asarray(frequency)
    frequency = frequency * harmonic

    if has_harmonic_axis:
        if real.ndim == 0:
            raise ValueError(
                f'{real.shape=} != {len(frequency)} frequencies or harmonics'
            )
        if real.shape[0] != len(frequency):
            raise ValueError(
                f'{real.shape[0]=} != {len(frequency)} '
                'frequencies or harmonics'
            )
        if reference_real.shape[0] != len(frequency):
            raise ValueError(
                f'{reference_real.shape[0]=} != {len(frequency)} '
                'frequencies or harmonics'
            )
        if reference_mean.shape != reference_real.shape[1:]:
            raise ValueError(
                f'{reference_mean.shape=} != {reference_real.shape[1:]=}'
            )
    elif reference_mean.shape != reference_real.shape:
        raise ValueError(f'{reference_mean.shape=} != {reference_real.shape=}')
    elif len(harmonic) > 1:
        raise ValueError(
            f'{reference_mean.shape=} does not have harmonic axis'
        )

    _, measured_re, measured_im = phasor_center(
        reference_mean,
        reference_real,
        reference_imag,
        skip_axis=skip_axis,
        method=method,
        nan_safe=nan_safe,
    )

    known_re, known_im = phasor_from_lifetime(
        frequency,
        lifetime,
        fraction,
        preexponential=preexponential,
        unit_conversion=unit_conversion,
    )

    skip_axis, axis = parse_skip_axis(
        skip_axis, real.ndim - int(has_harmonic_axis), has_harmonic_axis
    )

    if has_harmonic_axis and any(skip_axis):
        known_re = numpy.expand_dims(
            known_re, tuple(range(1, measured_re.ndim))
        )
        known_re = numpy.broadcast_to(
            known_re, (len(frequency), *measured_re.shape[1:])
        )
        known_im = numpy.expand_dims(
            known_im, tuple(range(1, measured_im.ndim))
        )
        known_im = numpy.broadcast_to(
            known_im, (len(frequency), *measured_im.shape[1:])
        )

    phi_zero, mod_zero = polar_from_reference_phasor(
        measured_re, measured_im, known_re, known_im
    )

    if numpy.ndim(phi_zero) > 0:
        if reverse:
            numpy.negative(phi_zero, out=phi_zero)
            numpy.reciprocal(mod_zero, out=mod_zero)
        if axis is not None:
            phi_zero = numpy.expand_dims(phi_zero, axis=axis)
            mod_zero = numpy.expand_dims(mod_zero, axis=axis)
    elif reverse:
        phi_zero = -phi_zero
        mod_zero = 1.0 / mod_zero

    return phasor_transform(real, imag, phi_zero, mod_zero)


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


def polar_from_reference_phasor(
    measured_real: ArrayLike,
    measured_imag: ArrayLike,
    known_real: ArrayLike,
    known_imag: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return polar coordinates for calibration from reference phasor.

    Return rotation angle and scale factor for calibrating phasor coordinates
    from measured and known phasor coordinates of a reference, for example,
    a sample of known lifetime.

    Parameters
    ----------
    measured_real : array_like
        Real component of measured phasor coordinates.
    measured_imag : array_like
        Imaginary component of measured phasor coordinates.
    known_real : array_like
        Real component of reference phasor coordinates.
    known_imag : array_like
        Imaginary component of reference phasor coordinates.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    phase_zero : ndarray
        Angular component of polar coordinates for calibration in radians.
    modulation_zero : ndarray
        Radial component of polar coordinates for calibration.

    See Also
    --------
    phasorpy.phasor.polar_from_reference

    Notes
    -----
    This function performs the following operations:

    .. code-block:: python

        polar_from_reference(
            *phasor_to_polar(measured_real, measured_imag),
            *phasor_to_polar(known_real, known_imag),
        )

    Examples
    --------
    >>> polar_from_reference_phasor(0.5, 0.0, 1.0, 0.0)
    (0.0, 2.0)

    """
    return _polar_from_reference_phasor(  # type: ignore[no-any-return]
        measured_real, measured_imag, known_real, known_imag, **kwargs
    )


def polar_from_reference(
    measured_phase: ArrayLike,
    measured_modulation: ArrayLike,
    known_phase: ArrayLike,
    known_modulation: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return polar coordinates for calibration from reference coordinates.

    Return rotation angle and scale factor for calibrating phasor coordinates
    from measured and known polar coordinates of a reference, for example,
    a sample of known lifetime.

    Parameters
    ----------
    measured_phase : array_like
        Angular component of measured polar coordinates in radians.
    measured_modulation : array_like
        Radial component of measured polar coordinates.
    known_phase : array_like
        Angular component of reference polar coordinates in radians.
    known_modulation : array_like
        Radial component of reference polar coordinates.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    phase_zero : ndarray
        Angular component of polar coordinates for calibration in radians.
    modulation_zero : ndarray
        Radial component of polar coordinates for calibration.

    See Also
    --------
    phasorpy.phasor.polar_from_reference_phasor

    Examples
    --------
    >>> polar_from_reference(0.2, 0.4, 0.4, 1.3)
    (0.2, 3.25)

    """
    return _polar_from_reference(  # type: ignore[no-any-return]
        measured_phase,
        measured_modulation,
        known_phase,
        known_modulation,
        **kwargs,
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


def phasor_to_apparent_lifetime(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return apparent single lifetimes from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    unit_conversion : float, optional
        Product of `frequency` and returned `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    phase_lifetime : ndarray
        Apparent single lifetime from angular component of phasor coordinates.
    modulation_lifetime : ndarray
        Apparent single lifetime from radial component of phasor coordinates.

    See Also
    --------
    phasorpy.phasor.phasor_from_apparent_lifetime
    :ref:`sphx_glr_tutorials_phasorpy_lifetime_geometry.py`

    Notes
    -----
    The phasor coordinates `real` (:math:`G`) and `imag` (:math:`S`)
    are converted to apparent single lifetimes
    `phase_lifetime` (:math:`\tau_{\phi}`) and
    `modulation_lifetime` (:math:`\tau_{M}`) at frequency :math:`f`
    according to:

    .. math::

        \omega &= 2 \pi f

        \tau_{\phi} &= \omega^{-1} \cdot S / G

        \tau_{M} &= \omega^{-1} \cdot \sqrt{1 / (S^2 + G^2) - 1}

    Examples
    --------
    The apparent single lifetimes from phase and modulation are equal
    only if the phasor coordinates lie on the universal semicircle:

    >>> phasor_to_apparent_lifetime(
    ...     0.5, [0.5, 0.45], frequency=80
    ... )  # doctest: +NUMBER
    (array([1.989, 1.79]), array([1.989, 2.188]))

    Apparent single lifetimes of phasor coordinates outside the universal
    semicircle are undefined:

    >>> phasor_to_apparent_lifetime(-0.1, 1.1, 80)  # doctest: +NUMBER
    (-21.8, 0.0)

    Apparent single lifetimes at the universal semicircle endpoints are
    infinite and zero:

    >>> phasor_to_apparent_lifetime([0, 1], [0, 0], 80)  # doctest: +NUMBER
    (array([inf, 0]), array([inf, 0]))

    """
    omega = numpy.array(frequency, dtype=numpy.float64)  # makes copy
    omega *= math.pi * 2.0 * unit_conversion
    return _phasor_to_apparent_lifetime(  # type: ignore[no-any-return]
        real, imag, omega, **kwargs
    )


def phasor_from_apparent_lifetime(
    phase_lifetime: ArrayLike,
    modulation_lifetime: ArrayLike | None,
    /,
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return phasor coordinates from apparent single lifetimes.

    Parameters
    ----------
    phase_lifetime : ndarray
        Apparent single lifetime from phase.
    modulation_lifetime : ndarray, optional
        Apparent single lifetime from modulation.
        If None, `modulation_lifetime` is same as `phase_lifetime`.
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    unit_conversion : float, optional
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
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
    phasorpy.phasor.phasor_to_apparent_lifetime

    Notes
    -----
    The apparent single lifetimes `phase_lifetime` (:math:`\tau_{\phi}`)
    and `modulation_lifetime` (:math:`\tau_{M}`) are converted to phasor
    coordinates `real` (:math:`G`) and `imag` (:math:`S`) at
    frequency :math:`f` according to:

    .. math::

        \omega &= 2 \pi f

        \phi & = \arctan(\omega \tau_{\phi})

        M &= 1 / \sqrt{1 + (\omega \tau_{M})^2}

        G &= M \cdot \cos{\phi}

        S &= M \cdot \sin{\phi}

    Examples
    --------
    If the apparent single lifetimes from phase and modulation are equal,
    the phasor coordinates lie on the universal semicircle, else inside:

    >>> phasor_from_apparent_lifetime(
    ...     1.9894, [1.9894, 2.4113], frequency=80.0
    ... )  # doctest: +NUMBER
    (array([0.5, 0.45]), array([0.5, 0.45]))

    Zero and infinite apparent single lifetimes define the endpoints of the
    universal semicircle:

    >>> phasor_from_apparent_lifetime(
    ...     [0.0, 1e9], [0.0, 1e9], frequency=80
    ... )  # doctest: +NUMBER
    (array([1, 0.0]), array([0, 0.0]))

    """
    omega = numpy.array(frequency, dtype=numpy.float64)  # makes copy
    omega *= math.pi * 2.0 * unit_conversion
    if modulation_lifetime is None:
        return _phasor_from_single_lifetime(  # type: ignore[no-any-return]
            phase_lifetime, omega, **kwargs
        )
    return _phasor_from_apparent_lifetime(  # type: ignore[no-any-return]
        phase_lifetime, modulation_lifetime, omega, **kwargs
    )


def phasor_to_normal_lifetime(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> NDArray[Any]:
    r"""Return normal lifetimes from phasor coordinates.

    The normal lifetime of phasor coordinates represents the single lifetime
    equivalent corresponding to the perpendicular projection of the coordinates
    onto the universal semicircle, as defined in [3]_.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    unit_conversion : float, optional
        Product of `frequency` and returned `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    normal_lifetime : ndarray
        Normal lifetime from of phasor coordinates.

    See Also
    --------
    :ref:`sphx_glr_tutorials_phasorpy_lifetime_geometry.py`

    Notes
    -----
    The phasor coordinates `real` (:math:`G`) and `imag` (:math:`S`)
    are converted to normal lifetimes `normal_lifetime` (:math:`\tau_{N}`)
    at frequency :math:`f` according to:

    .. math::

        \omega &= 2 \pi f

        G_{N} &= 0.5 \cdot (1 + \cos{\arctan{\frac{S}{G - 0.5}}})

        \tau_{N} &= \sqrt{\frac{1 - G_{N}}{\omega^{2} \cdot G_{N}}}

    References
    ----------

    .. [3] Silberberg M, and Grecco H. `pawFLIM: reducing bias and
      uncertainty to enable lower photon count in FLIM experiments
      <https://doi.org/10.1088/2050-6120/aa72ab>`_.
      *Methods Appl Fluoresc*, 5(2): 024016 (2017)

    Examples
    --------
    The normal lifetimes of phasor coordinates with a real component of 0.5
    are independent of the imaginary component:

    >>> phasor_to_normal_lifetime(
    ...     0.5, [0.5, 0.45], frequency=80
    ... )  # doctest: +NUMBER
    array([1.989, 1.989])

    """
    omega = numpy.array(frequency, dtype=numpy.float64)  # makes copy
    omega *= math.pi * 2.0 * unit_conversion
    return _phasor_to_normal_lifetime(  # type: ignore[no-any-return]
        real, imag, omega, **kwargs
    )


def lifetime_to_frequency(
    lifetime: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
) -> NDArray[numpy.float64]:
    r"""Return optimal frequency for resolving single component lifetime.

    Parameters
    ----------
    lifetime : array_like
        Single component lifetime.
    unit_conversion : float, optional, default: 1e-3
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.

    Returns
    -------
    frequency : ndarray
        Optimal laser pulse or modulation frequency for resolving `lifetime`.

    Notes
    -----
    The optimal frequency :math:`f` to resolve a single component lifetime
    :math:`\tau` is
    (:ref:`Redford & Clegg 2005 <redford-clegg-2005>`. Eq. B.6):

    .. math::

        \omega &= 2 \pi f

        \omega^2 &= \frac{1 + \sqrt{3}}{2 \tau^2}

    Examples
    --------
    Measurements of a lifetime near 4 ns should be made at 47 MHz,
    near 1 ns at 186 MHz:

    >>> lifetime_to_frequency([4.0, 1.0])  # doctest: +NUMBER
    array([46.5, 186])

    """
    t = numpy.reciprocal(lifetime, dtype=numpy.float64)
    t *= 0.18601566519848653 / unit_conversion
    return t


def lifetime_from_frequency(
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
) -> NDArray[numpy.float64]:
    r"""Return single component lifetime best resolved at frequency.

    Parameters
    ----------
    frequency : array_like
        Laser pulse or modulation frequency.
    unit_conversion : float, optional, default: 1e-3
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.

    Returns
    -------
    lifetime : ndarray
        Single component lifetime best resolved at `frequency`.

    Notes
    -----
    The lifetime :math:`\tau` that is best resolved at frequency :math:`f` is
    (:ref:`Redford & Clegg 2005 <redford-clegg-2005>`. Eq. B.6):

    .. math::

        \omega &= 2 \pi f

        \tau^2 &=  \frac{1 + \sqrt{3}}{2 \omega^2}

    Examples
    --------
    Measurements at frequencies of 47 and 186 MHz are best for measuring
    lifetimes near 4 and 1 ns respectively:

    >>> lifetime_from_frequency([46.5, 186])  # doctest: +NUMBER
    array([4, 1])

    """
    t = numpy.reciprocal(frequency, dtype=numpy.float64)
    t *= 0.18601566519848653 / unit_conversion
    return t


def lifetime_fraction_to_amplitude(
    lifetime: ArrayLike, fraction: ArrayLike, *, axis: int = -1
) -> NDArray[numpy.float64]:
    r"""Return pre-exponential amplitude from fractional intensity.

    Parameters
    ----------
    lifetime : array_like
        Lifetime components.
    fraction : array_like
        Fractional intensities of lifetime components.
        Fractions are normalized to sum to 1.
    axis : int, optional
        Axis over which to compute pre-exponential amplitudes.
        The default is the last axis (-1).

    Returns
    -------
    amplitude : ndarray
        Pre-exponential amplitudes.
        The product of `amplitude` and `lifetime` sums to 1 along `axis`.

    See Also
    --------
    phasorpy.phasor.lifetime_fraction_from_amplitude

    Notes
    -----
    The pre-exponential amplitude :math:`a` of component :math:`j` with
    lifetime :math:`\tau` and fractional intensity :math:`\alpha` is:

    .. math::

        a_{j} = \frac{\alpha_{j}}{\tau_{j} \cdot \sum_{j} \alpha_{j}}

    Examples
    --------
    >>> lifetime_fraction_to_amplitude(
    ...     [4.0, 1.0], [1.6, 0.4]
    ... )  # doctest: +NUMBER
    array([0.2, 0.2])

    """
    t = numpy.array(fraction, dtype=numpy.float64)  # makes copy
    t /= numpy.sum(t, axis=axis, keepdims=True)
    numpy.true_divide(t, lifetime, out=t)
    return t


def lifetime_fraction_from_amplitude(
    lifetime: ArrayLike, amplitude: ArrayLike, *, axis: int = -1
) -> NDArray[numpy.float64]:
    r"""Return fractional intensity from pre-exponential amplitude.

    Parameters
    ----------
    lifetime : array_like
        Lifetime of components.
    amplitude : array_like
        Pre-exponential amplitudes of lifetime components.
    axis : int, optional
        Axis over which to compute fractional intensities.
        The default is the last axis (-1).

    Returns
    -------
    fraction : ndarray
        Fractional intensities, normalized to sum to 1 along `axis`.

    See Also
    --------
    phasorpy.phasor.lifetime_fraction_to_amplitude

    Notes
    -----
    The fractional intensity :math:`\alpha` of component :math:`j` with
    lifetime :math:`\tau` and pre-exponential amplitude :math:`a` is:

    .. math::

        \alpha_{j} = \frac{a_{j} \tau_{j}}{\sum_{j} a_{j} \tau_{j}}

    Examples
    --------
    >>> lifetime_fraction_from_amplitude(
    ...     [4.0, 1.0], [1.0, 1.0]
    ... )  # doctest: +NUMBER
    array([0.8, 0.2])

    """
    t: NDArray[numpy.float64]
    t = numpy.multiply(amplitude, lifetime, dtype=numpy.float64)
    t /= numpy.sum(t, axis=axis, keepdims=True)
    return t


def phasor_at_harmonic(
    real: ArrayLike,
    harmonic: ArrayLike,
    other_harmonic: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    r"""Return phasor coordinates on universal semicircle at other harmonics.

    Return phasor coordinates at any harmonic, given the real component of
    phasor coordinates of a single exponential lifetime at a certain harmonic.
    The input and output phasor coordinates lie on the universal semicircle.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates of single exponential lifetime
        at `harmonic`.
    harmonic : array_like
        Harmonic of `real` coordinate. Must be integer >= 1.
    other_harmonic : array_like
        Harmonic for which to return phasor coordinates. Must be integer >= 1.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real_other : ndarray
        Real component of phasor coordinates at `other_harmonic`.
    imag_other : ndarray
        Imaginary component of phasor coordinates at `other_harmonic`.

    Notes
    -----
    The phasor coordinates
    :math:`g_{n}` (`real_other`) and :math:`s_{n}` (`imag_other`)
    of a single exponential lifetime at harmonic :math:`n` (`other_harmonic`)
    is calculated from the real part of the phasor coordinates
    :math:`g_{m}` (`real`) at harmonic :math:`m` (`harmonic`) according to
    (:ref:`Torrado, Malacrida, & Ranjit. 2022 <torrado-2022>`. Eq. 25):

    .. math::

        g_{n} &= \frac{m^2 \cdot g_{m}}{n^2 + (m^2-n^2) \cdot g_{m}}

        s_{n} &= \sqrt{G_{n} - g_{n}^2}

    This function is equivalent to the following operations:

    .. code-block:: python

        phasor_from_lifetime(
            frequency=other_harmonic,
            lifetime=phasor_to_apparent_lifetime(
                real, sqrt(real - real * real), frequency=harmonic
            )[0],
        )

    Examples
    --------
    The phasor coordinates at higher harmonics are approaching the origin:

    >>> phasor_at_harmonic(0.5, 1, [1, 2, 4, 8])  # doctest: +NUMBER
    (array([0.5, 0.2, 0.05882, 0.01538]), array([0.5, 0.4, 0.2353, 0.1231]))

    """
    harmonic = numpy.asarray(harmonic, dtype=numpy.int32)
    if numpy.any(harmonic < 1):
        raise ValueError('invalid harmonic')

    other_harmonic = numpy.asarray(other_harmonic, dtype=numpy.int32)
    if numpy.any(other_harmonic < 1):
        raise ValueError('invalid other_harmonic')

    return _phasor_at_harmonic(  # type: ignore[no-any-return]
        real, harmonic, other_harmonic, **kwargs
    )


def phasor_from_lifetime(
    frequency: ArrayLike,
    lifetime: ArrayLike,
    fraction: ArrayLike | None = None,
    *,
    preexponential: bool = False,
    unit_conversion: float = 1e-3,
    keepdims: bool = False,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    r"""Return phasor coordinates from lifetime components.

    Calculate phasor coordinates as a function of frequency, single or
    multiple lifetime components, and the pre-exponential amplitudes
    or fractional intensities of the components.

    Parameters
    ----------
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
        A scalar or one-dimensional sequence.
    lifetime : array_like
        Lifetime components in ns. See notes below for allowed dimensions.
    fraction : array_like, optional
        Fractional intensities or pre-exponential amplitudes of the lifetime
        components. Fractions are normalized to sum to 1.
        See notes below for allowed dimensions.
    preexponential : bool, optional, default: False
        If true, `fraction` values are pre-exponential amplitudes,
        else fractional intensities.
    unit_conversion : float, optional, default: 1e-3
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    keepdims : bool, optional, default: False
        If true, length-one dimensions are left in phasor coordinates.

    Returns
    -------
    real : ndarray
        Real component of phasor coordinates.
    imag : ndarray
        Imaginary component of phasor coordinates.

        See notes below for dimensions of the returned arrays.

    Raises
    ------
    ValueError
        Input arrays exceed their allowed dimensionality or do not match.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_phasor_from_lifetime.py`
    :ref:`sphx_glr_tutorials_phasorpy_lifetime_geometry.py`

    Notes
    -----
    The phasor coordinates :math:`G` (`real`) and :math:`S` (`imag`) for
    many lifetime components :math:`j` with lifetimes :math:`\tau` and
    pre-exponential amplitudes :math:`\alpha` at frequency :math:`f` are:

    .. math::

        \omega &= 2 \pi f

        g_{j} &= \alpha_{j} / (1 + (\omega \tau_{j})^2)

        G &= \sum_{j} g_{j}

        S &= \sum_{j} \omega \tau_{j} g_{j}

    The relation between pre-exponential amplitudes :math:`a` and
    fractional intensities :math:`\alpha` is:

    .. math::
        F_{DC} &= \sum_{j} a_{j} \tau_{j}

        \alpha_{j} &= a_{j} \tau_{j} / F_{DC}

    The following combinations of `lifetime` and `fraction` parameters are
    supported:

    - `lifetime` is scalar or one-dimensional, holding single component
      lifetimes. `fraction` is None.
      Return arrays of shape `(frequency.size, lifetime.size)`.

    - `lifetime` is two-dimensional, `fraction` is one-dimensional.
      The last dimensions match in size, holding lifetime components and
      their fractions.
      Return arrays of shape `(frequency.size, lifetime.shape[1])`.

    - `lifetime` is one-dimensional, `fraction` is two-dimensional.
      The last dimensions must match in size, holding lifetime components and
      their fractions.
      Return arrays of shape `(frequency.size, fraction.shape[1])`.

    - `lifetime` and `fraction` are up to two-dimensional of same shape.
      The last dimensions hold lifetime components and their fractions.
      Return arrays of shape `(frequency.size, lifetime.shape[0])`.

    Length-one dimensions are removed from returned arrays
    if `keepdims` is false (default).

    Examples
    --------
    Phasor coordinates of a single lifetime component (in ns) at a
    frequency of 80 MHz:

    >>> phasor_from_lifetime(80.0, 1.9894368)  # doctest: +NUMBER
    (0.5, 0.5)

    Phasor coordinates of two lifetime components with equal fractional
    intensities:

    >>> phasor_from_lifetime(
    ...     80.0, [3.9788735, 0.9947183], [0.5, 0.5]
    ... )  # doctest: +NUMBER
    (0.5, 0.4)

    Phasor coordinates of two lifetime components with equal pre-exponential
    amplitudes:

    >>> phasor_from_lifetime(
    ...     80.0, [3.9788735, 0.9947183], [0.5, 0.5], preexponential=True
    ... )  # doctest: +NUMBER
    (0.32, 0.4)

    Phasor coordinates of many single-component lifetimes (fractions omitted):

    >>> phasor_from_lifetime(
    ...     80.0, [3.9788735, 1.9894368, 0.9947183]
    ... )  # doctest: +NUMBER
    (array([0.2, 0.5, 0.8]), array([0.4, 0.5, 0.4]))

    Phasor coordinates of two lifetime components with varying fractions:

    >>> phasor_from_lifetime(
    ...     80.0, [3.9788735, 0.9947183], [[1, 0], [0.5, 0.5], [0, 1]]
    ... )  # doctest: +NUMBER
    (array([0.2, 0.5, 0.8]), array([0.4, 0.4, 0.4]))

    Phasor coordinates of multiple two-component lifetimes with constant
    fractions, keeping dimensions:

    >>> phasor_from_lifetime(
    ...     80.0, [[3.9788735, 0.9947183], [1.9894368, 1.9894368]], [0.5, 0.5]
    ... )  # doctest: +NUMBER
    (array([0.5, 0.5]), array([0.4, 0.5]))

    Phasor coordinates of multiple two-component lifetimes with specific
    fractions at multiple frequencies. Frequencies are in Hz, lifetimes in ns:

    >>> phasor_from_lifetime(
    ...     [40e6, 80e6],
    ...     [[1e-9, 0.9947183e-9], [3.9788735e-9, 0.9947183e-9]],
    ...     [[0, 1], [0.5, 0.5]],
    ...     unit_conversion=1.0,
    ... )  # doctest: +NUMBER
    (array([[0.941, 0.721], [0.8, 0.5]]), array([[0.235, 0.368], [0.4, 0.4]]))

    """
    if unit_conversion < 1e-16:
        raise ValueError(f'{unit_conversion=} < 1e-16')
    frequency = numpy.atleast_1d(numpy.asarray(frequency, dtype=numpy.float64))
    if frequency.ndim != 1:
        raise ValueError('frequency is not one-dimensional array')
    lifetime = numpy.atleast_1d(numpy.asarray(lifetime, dtype=numpy.float64))
    if lifetime.ndim > 2:
        raise ValueError('lifetime must be one- or two-dimensional array')

    if fraction is None:
        # single-component lifetimes
        if lifetime.ndim > 1:
            raise ValueError(
                'lifetime must be one-dimensional array if fraction is None'
            )
        lifetime = lifetime.reshape(-1, 1)  # move components to last axis
        fraction = numpy.ones_like(lifetime)  # not really used
    else:
        fraction = numpy.atleast_1d(
            numpy.asarray(fraction, dtype=numpy.float64)
        )
        if fraction.ndim > 2:
            raise ValueError('fraction must be one- or two-dimensional array')

    if lifetime.ndim == 1 and fraction.ndim == 1:
        # one multi-component lifetime
        if lifetime.shape != fraction.shape:
            raise ValueError(
                f'{lifetime.shape=} does not match {fraction.shape=}'
            )
        lifetime = lifetime.reshape(1, -1)
        fraction = fraction.reshape(1, -1)
        nvar = 1
    elif lifetime.ndim == 2 and fraction.ndim == 2:
        # multiple, multi-component lifetimes
        if lifetime.shape[1] != fraction.shape[1]:
            raise ValueError(f'{lifetime.shape[1]=} != {fraction.shape[1]=}')
        nvar = lifetime.shape[0]
    elif lifetime.ndim == 2 and fraction.ndim == 1:
        # variable components, same fractions
        fraction = fraction.reshape(1, -1)
        nvar = lifetime.shape[0]
    elif lifetime.ndim == 1 and fraction.ndim == 2:
        # same components, varying fractions
        lifetime = lifetime.reshape(1, -1)
        nvar = fraction.shape[0]
    else:
        # unreachable code
        raise RuntimeError(f'{lifetime.shape=}, {fraction.shape=}')

    phasor = numpy.empty((2, frequency.size, nvar), dtype=numpy.float64)

    _phasor_from_lifetime(
        phasor, frequency, lifetime, fraction, unit_conversion, preexponential
    )

    if not keepdims:
        phasor = phasor.squeeze()
    return phasor[0], phasor[1]


def polar_to_apparent_lifetime(
    phase: ArrayLike,
    modulation: ArrayLike,
    /,
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return apparent single lifetimes from polar coordinates.

    Parameters
    ----------
    phase : array_like
        Angular component of polar coordinates.
    imag : array_like
        Radial component of polar coordinates.
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    unit_conversion : float, optional
        Product of `frequency` and returned `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    phase_lifetime : ndarray
        Apparent single lifetime from `phase`.
    modulation_lifetime : ndarray
        Apparent single lifetime from `modulation`.

    See Also
    --------
    phasorpy.phasor.polar_from_apparent_lifetime

    Notes
    -----
    The polar coordinates `phase` (:math:`\phi`) and `modulation` (:math:`M`)
    are converted to apparent single lifetimes
    `phase_lifetime` (:math:`\tau_{\phi}`) and
    `modulation_lifetime` (:math:`\tau_{M}`) at frequency :math:`f`
    according to:

    .. math::

        \omega &= 2 \pi f

        \tau_{\phi} &= \omega^{-1} \cdot \tan{\phi}

        \tau_{M} &= \omega^{-1} \cdot \sqrt{1 / M^2 - 1}

    Examples
    --------
    The apparent single lifetimes from phase and modulation are equal
    only if the polar coordinates lie on the universal semicircle:

    >>> polar_to_apparent_lifetime(
    ...     math.pi / 4, numpy.hypot([0.5, 0.45], [0.5, 0.45]), frequency=80
    ... )  # doctest: +NUMBER
    (array([1.989, 1.989]), array([1.989, 2.411]))

    """
    omega = numpy.array(frequency, dtype=numpy.float64)  # makes copy
    omega *= math.pi * 2.0 * unit_conversion
    return _polar_to_apparent_lifetime(  # type: ignore[no-any-return]
        phase, modulation, omega, **kwargs
    )


def polar_from_apparent_lifetime(
    phase_lifetime: ArrayLike,
    modulation_lifetime: ArrayLike | None,
    /,
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return polar coordinates from apparent single lifetimes.

    Parameters
    ----------
    phase_lifetime : ndarray
        Apparent single lifetime from phase.
    modulation_lifetime : ndarray, optional
        Apparent single lifetime from modulation.
        If None, `modulation_lifetime` is same as `phase_lifetime`.
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    unit_conversion : float, optional
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    phase : ndarray
        Angular component of polar coordinates.
    modulation : ndarray
        Radial component of polar coordinates.

    See Also
    --------
    phasorpy.phasor.polar_to_apparent_lifetime

    Notes
    -----
    The apparent single lifetimes `phase_lifetime` (:math:`\tau_{\phi}`)
    and `modulation_lifetime` (:math:`\tau_{M}`) are converted to polar
    coordinates `phase` (:math:`\phi`) and `modulation` (:math:`M`) at
    frequency :math:`f` according to:

    .. math::

        \omega &= 2 \pi f

        \phi & = \arctan(\omega \tau_{\phi})

        M &= 1 / \sqrt{1 + (\omega \tau_{M})^2}

    Examples
    --------
    If the apparent single lifetimes from phase and modulation are equal,
    the polar coordinates lie on the universal semicircle, else inside:

    >>> polar_from_apparent_lifetime(
    ...     1.9894, [1.9894, 2.4113], frequency=80.0
    ... )  # doctest: +NUMBER
    (array([0.7854, 0.7854]), array([0.7071, 0.6364]))

    """
    omega = numpy.array(frequency, dtype=numpy.float64)  # makes copy
    omega *= math.pi * 2.0 * unit_conversion
    if modulation_lifetime is None:
        return _polar_from_single_lifetime(  # type: ignore[no-any-return]
            phase_lifetime, omega, **kwargs
        )
    return _polar_from_apparent_lifetime(  # type: ignore[no-any-return]
        phase_lifetime, modulation_lifetime, omega, **kwargs
    )


def phasor_from_fret_donor(
    frequency: ArrayLike,
    donor_lifetime: ArrayLike,
    *,
    fret_efficiency: ArrayLike = 0.0,
    donor_fretting: ArrayLike = 1.0,
    donor_background: ArrayLike = 0.0,
    background_real: ArrayLike = 0.0,
    background_imag: ArrayLike = 0.0,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return phasor coordinates of FRET donor channel.

    Calculate phasor coordinates of a FRET (Förster Resonance Energy Transfer)
    donor channel as a function of frequency, donor lifetime, FRET efficiency,
    fraction of donors undergoing FRET, and background fluorescence.

    The phasor coordinates of the donor channel contain fractions of:

    - donor not undergoing energy transfer
    - donor quenched by energy transfer
    - background fluorescence

    Parameters
    ----------
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    donor_lifetime : array_like
        Lifetime of donor without FRET in ns.
    fret_efficiency : array_like, optional, default 0
        FRET efficiency in range [0, 1].
    donor_fretting : array_like, optional, default 1
        Fraction of donors participating in FRET. Range [0, 1].
    donor_background : array_like, optional, default 0
        Weight of background fluorescence in donor channel
        relative to fluorescence of donor without FRET.
        A weight of 1 means the fluorescence of background and donor
        without FRET are equal.
    background_real : array_like, optional, default 0
        Real component of background fluorescence phasor coordinate
        at `frequency`.
    background_imag : array_like, optional, default 0
        Imaginary component of background fluorescence phasor coordinate
        at `frequency`.
    unit_conversion : float, optional
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real : ndarray
        Real component of donor channel phasor coordinates.
    imag : ndarray
        Imaginary component of donor channel phasor coordinates.

    See Also
    --------
    phasorpy.phasor.phasor_from_fret_acceptor
    :ref:`sphx_glr_tutorials_api_phasorpy_fret.py`
    :ref:`sphx_glr_tutorials_applications_phasorpy_fret_efficiency.py`

    Examples
    --------
    Compute the phasor coordinates of a FRET donor channel at three
    FRET efficiencies:

    >>> phasor_from_fret_donor(
    ...     frequency=80,
    ...     donor_lifetime=4.2,
    ...     fret_efficiency=[0.0, 0.3, 1.0],
    ...     donor_fretting=0.9,
    ...     donor_background=0.1,
    ...     background_real=0.11,
    ...     background_imag=0.12,
    ... )  # doctest: +NUMBER
    (array([0.1766, 0.2737, 0.1466]), array([0.3626, 0.4134, 0.2534]))

    """
    omega = numpy.array(frequency, dtype=numpy.float64)  # makes copy
    omega *= math.pi * 2.0 * unit_conversion
    return _phasor_from_fret_donor(  # type: ignore[no-any-return]
        omega,
        donor_lifetime,
        fret_efficiency,
        donor_fretting,
        donor_background,
        background_real,
        background_imag,
        **kwargs,
    )


def phasor_from_fret_acceptor(
    frequency: ArrayLike,
    donor_lifetime: ArrayLike,
    acceptor_lifetime: ArrayLike,
    *,
    fret_efficiency: ArrayLike = 0.0,
    donor_fretting: ArrayLike = 1.0,
    donor_bleedthrough: ArrayLike = 0.0,
    acceptor_bleedthrough: ArrayLike = 0.0,
    acceptor_background: ArrayLike = 0.0,
    background_real: ArrayLike = 0.0,
    background_imag: ArrayLike = 0.0,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return phasor coordinates of FRET acceptor channel.

    Calculate phasor coordinates of a FRET (Förster Resonance Energy Transfer)
    acceptor channel as a function of frequency, donor and acceptor lifetimes,
    FRET efficiency, fraction of donors undergoing FRET, fraction of directly
    excited acceptors, fraction of donor fluorescence in acceptor channel,
    and background fluorescence.

    The phasor coordinates of the acceptor channel contain fractions of:

    - acceptor sensitized by energy transfer
    - directly excited acceptor
    - donor bleedthrough
    - background fluorescence

    Parameters
    ----------
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    donor_lifetime : array_like
        Lifetime of donor without FRET in ns.
    acceptor_lifetime : array_like
        Lifetime of acceptor in ns.
    fret_efficiency : array_like, optional, default 0
        FRET efficiency in range [0, 1].
    donor_fretting : array_like, optional, default 1
        Fraction of donors participating in FRET. Range [0, 1].
    donor_bleedthrough : array_like, optional, default 0
        Weight of donor fluorescence in acceptor channel
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence from donor and fully sensitized
        acceptor are equal.
        The background in the donor channel does not bleed through.
    acceptor_bleedthrough : array_like, optional, default 0
        Weight of fluorescence from directly excited acceptor
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence from directly excited acceptor
        and fully sensitized acceptor are equal.
    acceptor_background : array_like, optional, default 0
        Weight of background fluorescence in acceptor channel
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence of background and fully
        sensitized acceptor are equal.
    background_real : array_like, optional, default 0
        Real component of background fluorescence phasor coordinate
        at `frequency`.
    background_imag : array_like, optional, default 0
        Imaginary component of background fluorescence phasor coordinate
        at `frequency`.
    unit_conversion : float, optional
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real : ndarray
        Real component of acceptor channel phasor coordinates.
    imag : ndarray
        Imaginary component of acceptor channel phasor coordinates.

    See Also
    --------
    phasorpy.phasor.phasor_from_fret_donor
    :ref:`sphx_glr_tutorials_api_phasorpy_fret.py`

    Examples
    --------
    Compute the phasor coordinates of a FRET acceptor channel at three
    FRET efficiencies:

    >>> phasor_from_fret_acceptor(
    ...     frequency=80,
    ...     donor_lifetime=4.2,
    ...     acceptor_lifetime=3.0,
    ...     fret_efficiency=[0.0, 0.3, 1.0],
    ...     donor_fretting=0.9,
    ...     donor_bleedthrough=0.1,
    ...     acceptor_bleedthrough=0.1,
    ...     acceptor_background=0.1,
    ...     background_real=0.11,
    ...     background_imag=0.12,
    ... )  # doctest: +NUMBER
    (array([0.1996, 0.05772, 0.2867]), array([0.3225, 0.3103, 0.4292]))

    """
    omega = numpy.array(frequency, dtype=numpy.float64)  # makes copy
    omega *= math.pi * 2.0 * unit_conversion
    return _phasor_from_fret_acceptor(  # type: ignore[no-any-return]
        omega,
        donor_lifetime,
        acceptor_lifetime,
        fret_efficiency,
        donor_fretting,
        donor_bleedthrough,
        acceptor_bleedthrough,
        acceptor_background,
        background_real,
        background_imag,
        **kwargs,
    )


def phasor_to_principal_plane(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    reorient: bool = True,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return multi-harmonic phasor coordinates projected onto principal plane.

    Principal Component Analysis (PCA) is used to project
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
    undefined `NaN` values.

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
    coordinates = numpy.vstack((re, im))

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
        `NaN` values but is faster when filtering more than 2 dimensions.
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
    """Return phasor coordinates with values out of interval replaced by NaN.

    Interval thresholds can be set for mean intensity, real and imaginary
    coordinates, and phase and modulation.
    Phasor coordinates smaller than minimum thresholds or larger than maximum
    thresholds are replaced NaN.
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
    """Return indices or values of nearest neighbor from other coordinates.

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
