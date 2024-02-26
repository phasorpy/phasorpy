"""Calculate, convert, calibrate, and reduce phasor coordinates.

The ``phasorpy.phasor`` module provides functions to:

- calculate phasor coordinates from time-resolved and spectral signals:

  - :py:func:`phasor_from_signal`
  - :py:func:`phasor_from_signal_fft`

- calculate phasor coordinates from single- or multi-component fluorescence
  lifetimes:

  - :py:func:`phasor_from_lifetime`
  - :py:func:`phasor_from_apparent_lifetime` (not implemented yet)
  - :py:func:`phasor_to_apparent_lifetime` (not implemented yet)

- convert between phasor and polar (phase and modulation) coordinates:

  - :py:func:`phasor_from_polar`
  - :py:func:`phasor_to_polar`

- calibrate phasor coordinates with reference of known fluorescence
  lifetime:

  - :py:func:`phasor_calibrate`
  - :py:func:`phasor_transform`
  - :py:func:`polar_from_reference`
  - :py:func:`polar_from_reference_phasor`

- reduce arrays of phasor coordinates to single coordinates:

  - :py:func:`phasor_center`

"""

from __future__ import annotations

__all__ = [
    'phasor_calibrate',
    'phasor_center',
    # 'phasor_from_apparent_lifetime',
    'phasor_from_lifetime',
    'phasor_from_polar',
    'phasor_from_signal',
    'phasor_from_signal_fft',
    'phasor_semicircle',
    # 'phasor_to_apparent_lifetime',
    'phasor_to_polar',
    'phasor_transform',
    'polar_from_reference',
    'polar_from_reference_phasor',
]

import math
import os
import warnings
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import (
        Any,
        NDArray,
        ArrayLike,
        DTypeLike,
        Callable,
        Literal,
        Sequence,
    )

import numpy

from ._phasor import _phasor_from_lifetime, _phasor_from_signal
from .utils import number_threads


def phasor_from_signal(
    signal: ArrayLike,
    /,
    *,
    axis: int = -1,
    harmonic: int | Sequence[int] | None = None,
    sample_phase: ArrayLike | None = None,
    dtype: DTypeLike = None,
    num_threads: int | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return phasor coordinates from signal.

    Parameters
    ----------
    signal : array_like
        Frequency-domain, time-domain, or hyperspectral data.
        A minimum of three samples are required along `axis`.
        The samples must be uniformly spaced.
    axis : int, optional
        Axis over which to compute phasor coordinates.
        The default is the last axis (-1).
    harmonic : int or sequence of int, optional
        Harmonics to return. Must be >= 1.
        The default is the first harmonic (fundamental frequency).
    sample_phase : array_like, optional
        Phase values (in radians) of `signal` samples along `axis`.
        If None (default), samples are assumed to be uniformly spaced along
        one period.
        The array size must equal the number of samples along `axis`.
        Cannot be used with `harmonic`.
    dtype : dtype_like, optional
        Data type of output arrays. Either float32 or float64 (default).
    num_threads : int, optional
        Number of OpenMP threads to use for parallelization.
        By default, multi-threading is disabled.
        If zero, up to half of logical CPUs are used.
        OpenMP may not be available on all platforms.

    Returns
    -------
    mean : ndarray
        Average of signal along axis (zero harmonic).
    real : ndarray
        Real component of phasor coordinates at `harmonic` along `axis`.
    imag : ndarray
        Imaginary component of phasor coordinates at `harmonic` along `axis`.

    Notes
    -----
    Compared to the :py:func:`phasor_from_signal_fft` reference implementation,
    this function does not use FFT, uses less memory, should be faster for
    few harmonics, supports out-of-order samples, and returns zeros instead
    of nan, inf, or excessively large values.

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

    Examples
    --------
    Calculate phasor coordinates of a phase-shifted sinusoidal waveform:

    >>> sample_phase = numpy.linspace(0, 2 * math.pi, 5, endpoint=False)[::-1]
    >>> signal = 1.1 * (
    ...     numpy.cos(sample_phase - 0.78539816) * 2 *  0.70710678 + 1
    ... )
    >>> phasor_from_signal(
    ...    signal, sample_phase=sample_phase
    ... )  # doctest: +NUMBER
    (1.1, 0.5, 0.5)

    The sinusoidal signal does not have a second harmonic component:

    >>> phasor_from_signal(signal, harmonic=2)  # doctest: +NUMBER
    (1.1, 0.0, 0.0)

    """
    signal = numpy.array(signal, order='C', ndmin=1, copy=False)
    samples = signal.shape[axis]  # this also verifies axis

    if sample_phase is not None:
        if harmonic is not None:
            raise ValueError('sample_phase cannot be used with harmonic')
        harmonics = [1]  # value not used
        sample_phase = numpy.array(
            sample_phase, dtype=numpy.float64, copy=False, ndmin=1
        )
        if sample_phase.ndim != 1 or sample_phase.size != samples:
            raise ValueError(f'{sample_phase.shape=} != ({samples},)')

    max_harmonic = samples // 2 + 1
    if harmonic is None:
        harmonics = [1]
    elif isinstance(harmonic, int):
        harmonics = [harmonic]
    else:
        a = numpy.array(harmonic, ndmin=1)
        if a.dtype.kind not in 'iu' or a.ndim != 1:
            raise TypeError(f'invalid {harmonic=} type')
        harmonics = a.tolist()
        del a
    num_harmonics = len(harmonics)

    num_threads = number_threads(num_threads)

    # pure numpy implementation for reference:
    # shape = [1] * signal.ndim
    # shape[axis] = sample_phase.size
    # sample_phase = sample_phase.reshape(shape)  # make broadcastable
    # mean = numpy.mean(signal, axis=axis)
    # real = numpy.mean(signal * numpy.cos(sample_phase), axis=axis)
    # real /= mean
    # imag = numpy.mean(signal * numpy.sin(sample_phase), axis=axis)
    # imag /= mean
    # return mean, real, imag

    sincos = numpy.empty((num_harmonics, samples, 2))
    for i, h in enumerate(harmonics):
        if h < 1 or h >= max_harmonic:
            raise IndexError(
                f'harmonic={h} out of range 1..{max_harmonic - 1}'
            )
        if sample_phase is None:
            phase = numpy.linspace(
                0,
                h * math.pi * 2,
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

    _phasor_from_signal(phasor, signal, sincos, num_threads)

    # restore original shape
    shape = shape0 + shape1
    mean = phasor[0].reshape(shape)
    if num_harmonics > 1:
        shape = (num_harmonics,) + shape
    real = phasor[1 : num_harmonics + 1].reshape(shape)
    imag = phasor[1 + num_harmonics :].reshape(shape)
    if shape:
        return mean, real, imag
    return mean.item(), real.item(), imag.item()


def phasor_from_signal_fft(
    signal: ArrayLike,
    /,
    *,
    axis: int = -1,
    harmonic: int | Sequence[int] | None = None,
    fft_func: Callable[..., NDArray[Any]] | None = None,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return phasor coordinates from signal using fast Fourier transform.

    Parameters
    ----------
    signal : array_like
        Frequency-domain, time-domain, or hyperspectral data.
        A minimum of three uniformly spaced samples are required along `axis`.
    axis : int, optional
        Axis over which to compute phasor coordinates.
        The default is the last axis (-1).
    harmonic : int or sequence of int, optional
        Harmonics to return. Must be >= 1.
        The default is the first harmonic (fundamental frequency).
    fft_func : callable, optional
        A drop-in replacement function for ``numpy.fft.fft``.

    Returns
    -------
    mean : ndarray
        Average of signal along axis (zero harmonic).
    real : ndarray
        Real component of phasor coordinates at `harmonic` along `axis`.
    imag : ndarray
        Imaginary component of phasor coordinates at `harmonic` along `axis`.

    Raises
    ------
    ValueError
        The `signal` has less than three samples along `axis`.
    IndexError
        `harmonic` is smaller than 1 or greater than half the samples along
        `axis`.

    Examples
    --------
    Calculate phasor coordinates of a phase-shifted sinusoidal signal:

    >>> sample_phase = numpy.linspace(0, 2 * math.pi, 5, endpoint=False)
    >>> signal = 1.1 * (
    ...     numpy.cos(sample_phase - 0.78539816) * 2 *  0.70710678 + 1
    ... )
    >>> phasor_from_signal_fft(signal, harmonic=[1, 2])  # doctest: +NUMBER
    (1.1, array([0.5, 0.0]), array([0.5, -0]))

    """
    signal = numpy.array(signal, copy=False, ndmin=1)
    samples = numpy.size(signal, axis)
    if samples < 3:
        raise ValueError(f'not enough {samples=} along {axis=}')

    max_harmonic = samples // 2
    if harmonic is None:
        harmonic = 1
    elif isinstance(harmonic, int):
        if harmonic < 1 or harmonic > max_harmonic:
            raise IndexError(
                f'harmonic={harmonic} out of range 1..{max_harmonic}'
            )
    else:
        a = numpy.array(harmonic)
        if a.dtype.kind not in 'iu' or a.ndim != 1:
            raise TypeError(f'invalid {harmonic=} type')
        if numpy.any(a < 1) or numpy.any(a > max_harmonic):
            raise IndexError(f'{harmonic=} out of range 1..{max_harmonic}')
        harmonic = a.tolist()
        del a

    if fft_func is None:
        fft_func = numpy.fft.fft

    fft: NDArray = fft_func(signal, axis=axis, norm='forward')

    mean = fft.take(0, axis=axis).real.copy()
    fft = fft.take(harmonic, axis=axis)  # type: ignore
    if mean.ndim == fft.ndim:
        dc = mean
    elif fft.shape[axis] == 1:
        dc = mean
        fft = fft.squeeze(axis)
    else:
        dc = numpy.expand_dims(mean, 0)
        fft = numpy.moveaxis(fft, axis, 0)
    real = fft.real.copy()
    imag = fft.imag.copy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        real /= dc
        imag /= dc
    imag *= -1
    return mean, real, imag


def phasor_semicircle(
    samples: int = 33, /
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Return equally spaced phasor coordinates on universal semicircle.

    Parameters
    ----------
    samples : int, optional
        Number of coordinates to return. The default is 33.

    Returns
    -------
    real : ndarray
        Real component of phasor coordinates.
    imag : ndarray
        Imaginary component of phasor coordinates.

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
    (phase :math:`φ` and modulation :math:`M`) is :math:`M=cos(φ)`.

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

def phasor_calibrate(real, imag, reference_real, reference_imag, frequency, lifetime, fraction, **kwargs):
    """
    Return calibrated/referenced phasor coordinates.

    Calibration of phasor coordinates in fluorescence lifetime analysis is
    necessary to account for the instrument response function (IRF) and delays
    in the electronics.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates to be calibrated.
    imag : array_like
        Imaginary component of phasor coordinates to be calibrated.
    reference_real : array_like
        Real component of phasor coordinates from reference of known lifetime.
    reference_imag : array_like
        Imaginary component of phasor coordinates from reference of known 
        lifetime.
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
        A scalar or one-dimensional sequence.
    lifetime : array_like
        Lifetime components in ns. See notes below for allowed dimensions.
    fraction : array_like, optional
        Fractional intensities or pre-exponential amplitudes of the lifetime
        components. Fractions are normalized to sum to 1.
        See notes below for allowed dimensions.
    **kwargs
        Additional keyword arguments passed to :py:func:`phasor_center`
        and :py:func:`phasor_from_lifetime`,

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

    Examples
    --------
    Use scalar reference coordinates to calibrate phasor coordinates:

    >>> phasor_calibrate([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 0.5, 0.2)
    (array([-0.208, -0.1284, -0.04876]), array([0.798, 1.069, 1.341]))

    Use separate reference coordinates for each phasor coordinate:

    >>> phasor_calibrate(
    ...     [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.5, 0.2, 0.3], [1.5, 2.0, 0.3]
    ... )
    (array([-1.56, 1.934, 0.3279]), array([5.985, 10.6, 1.986]))

    """
    re = numpy.asarray(real)
    im = numpy.asarray(imag)
    if re.shape != im.shape:
        raise ValueError(f'{re.shape=} != {im.shape=}')
    ref_re = numpy.asarray(reference_real)
    ref_im = numpy.asarray(reference_imag)
    if ref_re.shape != ref_im.shape:
        raise ValueError(f'{ref_re.shape=} != {ref_im.shape=}')
    kwargs_center = {key: value for key, value in kwargs.items() if key in inspect.signature(phasor_center).parameters}
    measured_re, measured_im = phasor_center(reference_real, reference_imag, **kwargs_center)
    kwargs_lifetime = {key: value for key, value in kwargs.items() if key in inspect.signature(phasor_from_lifetime).parameters}
    known_re, known_im = phasor_from_lifetime(frequency, lifetime, fraction, **kwargs_lifetime)
    phi_shift, mod_ratio = polar_from_reference_phasor(measured_re, measured_im, known_re, known_im)
    return phasor_transform(re, im, phi_shift, mod_ratio)

def phasor_transform(
    real: ArrayLike,
    imag: ArrayLike,
    phase_shift: ArrayLike = 0.0,
    modulation_ratio: ArrayLike = 1.0,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return transformed/rotated phasor coordinates.

    This function is used for calibration purposes, but can also be used to
    transform and/or rotate phasor coordinates. The transform is applied in
    the polar system and returned as phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates to transform/rotate.
    imag : array_like
        Imaginary component of phasor coordinates to transform/rotate.
    phase_shift : array_like, optional
        Angular component of polar coordinates for transformation/rotatio in radians.
        Defaults to 0.0.
    modulation_ratio : array_like, optional
        Radial component of polar coordinates for transformation/rotatio.
        Defaults to 1.0.

    Returns
    -------
    real : ndarray
        Transformed/rotated real component of phasor coordinates.
    imag : ndarray
        Transformed/rotated imaginary component of phasor coordinates.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag`, or `phase_shift` and `modulation_ratio`
        do not match.

    Examples
    --------
    Use scalar reference coordinates to transform/rotate phasor coordinates:

    >>> phasor_trasnform([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 0.5, 0.2)
    (array([-0.208, -0.1284, -0.04876]), array([0.798, 1.069, 1.341]))

    Use separate reference coordinates for each phasor coordinate:

    >>> phasor_trasnform(
    ...     [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.5, 0.2, 0.3], [1.5, 2.0, 0.3]
    ... )
    (array([-1.56, 1.934, 0.3279]), array([5.985, 10.6, 1.986]))

    """
    phi_shift = numpy.asarray(phase_shift)
    mod_ratio = numpy.asarray(modulation_ratio)
    if phi_shift.shape != mod_ratio.shape:
        raise ValueError(f'{phi_shift.shape=} != {mod_ratio.shape=}')
    re = numpy.array(real, copy=True, dtype=float)
    im = numpy.array(imag, copy=True, dtype=float)
    if re.shape != im.shape:
        raise ValueError(f'{re.shape=} != {im.shape=}')
    cos = numpy.cos(phi_shift)
    cos *= mod_ratio
    sin = numpy.sin(phi_shift)
    sin *= mod_ratio
    re_calibrated = re * cos
    re_calibrated -= im * sin
    im_calibrated = re
    im_calibrated *= sin
    im *= cos
    im_calibrated += im
    return re_calibrated, im_calibrated


def polar_from_reference_phasor(
    measured_real: ArrayLike,
    measured_imag: ArrayLike,
    known_real: ArrayLike,
    known_imag: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return polar coordinates for calibration from reference phasor.

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

    Returns
    -------
    phase_shift : ndarray
        Angular component of polar coordinates for calibration in radians.
    modulation_ratio : ndarray
        Radial component of polar coordinates for calibration.

    Raises
    ------
    ValueError
        The array shapes of `measured_real` and `measured_imag`, or
        `known_real` and `known_imag` do not match.

    Examples
    --------
    >>> polar_from_reference_phasor(0.5, 0.0, 1.0, 0.0)
    (0.0, 2.0)

    """
    measured_real = numpy.asarray(measured_real)
    measured_imag = numpy.asarray(measured_imag)
    if measured_real.shape != measured_imag.shape:
        raise ValueError(f'{measured_real.shape=} != {measured_imag.shape=}')
    known_real = numpy.asarray(known_real)
    known_imag = numpy.asarray(known_imag)
    if known_real.shape != known_imag.shape:
        raise ValueError(f'{known_real.shape=} != {known_imag.shape=}')
    measured_phi, measured_mod = phasor_to_polar(measured_real, measured_imag)
    known_phi, known_mod = phasor_to_polar(known_real, known_imag)
    phase_shift = known_phi - measured_phi
    modulation_ratio = known_mod / measured_mod
    phase_shift = phase_shift.item() if numpy.isscalar(modulation_ratio) else phase_shift
    return phase_shift, modulation_ratio


def polar_from_reference(
    measured_phase: ArrayLike,
    measured_modulation: ArrayLike,
    known_phase: ArrayLike,
    known_modulation: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return polar coordinates for calibration from reference coordinates.

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

    Returns
    -------
    phase_shift : ndarray
        Angular component of polar coordinates for calibration in radians.
    modulation_ratio : ndarray
        Radial component of polar coordinates for calibration.

    Raises
    ------
    ValueError
        The array shapes of `measured_phase` and `measured_modulation`, or
        `known_phase` and `known_modulation` do not match.

    Examples
    --------
    >>> polar_from_reference(0.4, 1.3, 0.2, 0.4)
    (0.2, 3.25)

    """
    measured_phase = numpy.asarray(measured_phase)
    measured_modulation = numpy.asarray(measured_modulation)
    if measured_phase.shape != measured_modulation.shape:
        raise ValueError(
            f'{measured_phase.shape=} != {measured_modulation.shape=}'
        )
    known_phase = numpy.asarray(known_phase)
    known_modulation = numpy.asarray(known_modulation)
    if known_phase.shape != known_modulation.shape:
        raise ValueError(f'{known_phase.shape=} != {known_modulation.shape=}')
    phase_shift = measured_phase - known_phase
    modulation_ratio = measured_modulation / known_modulation
    return phase_shift, modulation_ratio


def phasor_to_polar(
    real: ArrayLike,
    imag: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return polar coordinates from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.

    Returns
    -------
    phase : ndarray
        Angular component of polar coordinates in radians.
    modulation : ndarray
        Radial component of polar coordinates.

    Raises
    ------
    ValueError
        The shapes of the `real` and `imag` do not match.

    Examples
    --------
    Calculate polar coordinates from three phasor coordinates:

    >>> phasor_to_polar([1.0, 0.5, 0.0], [0.0, 0.5, 1.0])  # doctest: +NUMBER
    (array([0, 0.7854, 1.571]), array([1, 0.7071, 1]))

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    phase = numpy.arctan2(imag, real)
    modulation = numpy.hypot(real, imag)
    return phase, modulation


def phasor_from_polar(
    phase: ArrayLike,
    modulation: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return phasor coordinates from polar coordinates.

    Parameters
    ----------
    phase : array_like
        Angular component of polar coordinates in radians.
    modulation : array_like
        Radial component of polar coordinates.

    Returns
    -------
    real : ndarray
        Real component of phasor coordinates.
    imag : ndarray
        Imaginary component of phasor coordinates.

    Raises
    ------
    ValueError
        The shapes of `phase` and `modulation` do not match.

    Examples
    --------
    Calculate phasor coordinates from three polar coordinates:

    >>> phasor_from_polar(
    ...     [0.0, math.pi / 4, math.pi / 2], [1.0, math.sqrt(0.5), 1.0]
    ... )  # doctest: +NUMBER
    (array([1, 0.5, 0.0]), array([0, 0.5, 1]))

    """
    phase = numpy.asarray(phase)
    modulation = numpy.asarray(modulation)
    if phase.shape != modulation.shape:
        raise ValueError(f'{phase.shape=} != {modulation.shape=}')
    real = numpy.cos(phase)
    real *= modulation
    imag = numpy.sin(phase)
    imag *= modulation
    return real, imag


def phasor_from_lifetime(
    frequency: ArrayLike,
    lifetime: ArrayLike,
    fraction: ArrayLike | None = None,
    *,
    preexponential: bool = False,
    unit_conversion: float = 1e-3,
    squeeze: bool = True,
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
    preexponential : bool, optional
        If true, `fraction` values are pre-exponential amplitudes,
        else fractional intensities (default).
    unit_conversion : float
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    squeeze : bool, optional
        If true (default), length-one dimensions are removed from phasor
        coordinates.

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

    Notes
    -----
    The phasor coordinates :math:`G` (`real`) and :math:`S` (`imag`) for
    many lifetime components :math:`j` with lifetimes :math:`𝜏` and
    pre-exponential amplitudes :math:`α` at radial frequency :math:`ω` are:

    .. math::
        g_{j} &= a_{j} / (1 + (ω𝜏_{j})^2)

        G &= \sum_{j} g_{j}

        S &= \sum_{j} ω𝜏_{j}g_{j}

    The relation between pre-exponential amplitudes :math:`α` and
    fractional intensities :math:`a` is:

    .. math::
        F_{DC} &= \sum_{j} a_{j}𝜏_{j}

        α_{j} &= a_{j}𝜏_{j} / F_{DC}

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
      The last dimensions holding lifetime components and their fractions.
      Return arrays of shape `(frequency.size, lifetime.shape[0])`.

    Length-one dimensions are removed from returned arrays if `squeeze` is
    true.

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
    ...     80.0, [3.9788735, 1.9894368, 0.9947183],
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
    ...    unit_conversion=1.0
    ... )  # doctest: +NUMBER
    (array([[0.941, 0.721], [0.8, 0.5]]), array([[0.235, 0.368], [0.4, 0.4]]))

    """
    if unit_conversion < 1e-16:
        raise ValueError(f'{unit_conversion=} < 1e-16')
    frequency = numpy.array(frequency, dtype=numpy.float64, ndmin=1)
    if frequency.ndim != 1:
        raise ValueError('frequency is not one-dimensional array')
    lifetime = numpy.array(lifetime, dtype=numpy.float64, ndmin=1)
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
        fraction = numpy.array(fraction, dtype=numpy.float64, ndmin=1)
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

    if squeeze:
        phasor = phasor.squeeze()
    return phasor[0], phasor[1]


def phasor_center(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    skip_axes: tuple[int, ...] | None = None,
    method: Literal['mean', 'median'] = 'mean',
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return center of phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    skip_axes : tuple of int, optional
        Axes to be excluded during center calculation. If None, all
        axes are considered.
    method : str, optional
        Method used for center calculation:

        - ``'mean'``: Arithmetic mean of phasor coordinates.
        - ``'median'``: Spatial median of phasor coordinates.

    Returns
    -------
    real_center : ndarray
        Real center coordinates calculated based on the specified method.
    imag_center : ndarray
        Imaginary center coordinates calculated based on the specified method.

    Raises
    ------
    ValueError
        If the specified method is not supported.
        If the shapes of the 'real' and 'imag' do not match.

    Examples
    --------
    Compute center coordinates with the 'mean' method:

    >>> phasor_center([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], method='mean')
    (2.0, 5.0)

    Compute center coordinates with the 'median' method:

    >>> phasor_center([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], method='median')
    (2.0, 5.0)

    """
    supported_methods = ['mean', 'median']
    if method not in supported_methods:
        raise ValueError(
            f"Method not supported, supported methods are: "
            f"{', '.join(supported_methods)}"
        )
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    return {
        'mean': _mean,
        'median': _median,
    }[
        method
    ](real, imag, skip_axes)


def _mean(
    real: NDArray[Any],
    imag: NDArray[Any],
    skip_axes: tuple[int, ...] | None = None,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the mean center of phasor coordinates.

    Parameters
    ----------
    real : numpy.ndarray
        Real components of phasor coordinates.
    imag : numpy.ndarray
        Imaginary components of phasor coordinates.
    skip_axes : tuple of int, optional
        Axes to be excluded during center calculation. If None, all
        axes are considered.

    Returns
    -------
    real_center : ndarray
        Mean real center coordinates.
    imag_center : ndarray
        Mean imaginary center coordinates.

    Examples
    --------
    >>> _mean([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    (2.0, 5.0)

    """
    if skip_axes is None:
        return numpy.mean(real), numpy.mean(imag)
    else:
        included_axes = tuple(set(range(real.ndim)) - set(skip_axes))
        return numpy.mean(real, axis=included_axes), numpy.mean(
            imag, axis=included_axes
        )


def _median(
    real: NDArray[Any],
    imag: NDArray[Any],
    skip_axes: tuple[int, ...] | None = None,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the spatial median center of phasor coordinates.

    Parameters
    ----------
    real : numpy.ndarray
        Real components of the phasor coordinates.
    imag : numpy.ndarray
        Imaginary components of the phasor coordinates.
    skip_axes : tuple of int, optional
        Axes to be excluded during center calculation. If None, all
        axes are considered.

    Returns
    -------
    real_center : ndarray
        Spatial median center for real coordinates.
    imag_center : ndarray
        Spatial median center for imaginary coordinates.

    Examples
    --------
    >>> _median([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    (2.0, 5.0)

    """
    if skip_axes is None:
        return numpy.median(real), numpy.median(imag)
    else:
        included_axes = tuple(set(range(real.ndim)) - set(skip_axes))
        return numpy.median(real, axis=included_axes), numpy.median(
            imag, axis=included_axes
        )
