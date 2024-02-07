"""Calculate, convert, calibrate, and reduce phasor coordinates.

The ``phasorpy.phasor`` module provides functions to:

- calculate phasor coordinates from time-resolved and spectral signals:

  - :py:func:`phasor_from_signal`

- calculate phasor coordinates from single- or multi-component fluorescence
  lifetimes:

  - :py:func:`phasor_from_lifetime`
  - :py:func:`phasor_from_apparent_lifetime` (not implemented yet)
  - :py:func:`phasor_to_apparent_lifetime` (not implemented yet)

- convert between phasor and polar (phase and modulation) coordinates:

  - :py:func:`phasor_from_polar` (not implemented yet)
  - :py:func:`phasor_to_polar`

- calibrate phasor coordinates with reference of known fluorescence
  lifetime:

  - :py:func:`phasor_calibrate`
  - :py:func:`polar_from_reference`
  - :py:func:`polar_from_reference_phasor`

- reduce arrays of phasor coordinates to single coordinates:

  - :py:func:`phasor_center`

"""

from __future__ import annotations

__all__ = [
    "phasor_calibrate",
    "polar_from_reference_phasor",
    "polar_from_reference",
    "phasor_to_polar",
    "phasor_from_lifetime",
    "phasor_center",
    "phasor_from_signal",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, NDArray, ArrayLike, Literal

import math

import numpy
import warnings


def phasor_calibrate(
    real: ArrayLike,
    imag: ArrayLike,
    phase0: ArrayLike = 0.0,
    modulation0: ArrayLike = 1.0,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return calibrated/referenced phasor coordinates.

    Calibration of phasor coordinates in fluorescence lifetime analysis is
    necessary to account for the instrument response function (IRF) and delays
    in the electronics.

    This function can also be used to transform/rotate any phasor coordinate.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    phase0 : array_like, optional
        Angular component of polar coordinates for calibration. Defaults
        to 0.0.
    modulation0 : array_like, optional
        Radial component of polar coordinates for calibration. Defaults to 1.0.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag`, or `phase0` and `modulation0`
        do not match.

    Returns
    -------
    real: ndarray
        Calibrated real component of phasor coordinates.
    imag: ndarray
        Calibrated imaginary component of phasor coordinates.

    Examples
    --------
    Use scalar reference coordinates to calibrate phasor coordinates:

    >>> phasor_calibrate([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 0.5, 0.2)
    (array([-0.208, -0.1284, -0.04876]), array([0.798, 1.069, 1.341]))

    Use separate reference coordinates for each phasor coordinate:

    >>> phasor_calibrate(
    ... [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.5, 0.2, 0.3], [1.5, 2.0, 0.3]
    ... )
    (array([-1.56, 1.934, 0.3279]), array([5.985, 10.6, 1.986]))

    """
    phi0 = numpy.asarray(phase0)
    mod0 = numpy.asarray(modulation0)
    if phi0.shape != mod0.shape:
        raise ValueError(f'{phi0.shape=} != {mod0.shape=}')
    re = numpy.array(real, copy=True, dtype=float)
    im = numpy.array(imag, copy=True, dtype=float)
    if re.shape != im.shape:
        raise ValueError(f'{re.shape=} != {im.shape=}')
    cos = numpy.cos(phi0)
    cos *= mod0
    sin = numpy.sin(phi0)
    sin *= mod0
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

    Parameters
    ----------
    measured_real: array_like
        Real component of measured phasor coordinates.
    measured_imag: array_like
        Imaginary component of the measured phasor coordinates.
    known_real: array_like
        Real component of the reference phasor coordinates.
    known_imag: array_like
        Imaginary component of the reference phasor coordinates.

    Returns
    -------
    phase0: ndarray
        Angular component of polar coordinates for calibration.
    modulation0: ndarray
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
    phase0 = known_phi - measured_phi
    modulation0 = known_mod / measured_mod
    phase0 = phase0.item() if numpy.isscalar(modulation0) else phase0
    return phase0, modulation0


def polar_from_reference(
    measured_phase: ArrayLike,
    measured_modulation: ArrayLike,
    known_phase: ArrayLike,
    known_modulation: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return components for calibration from reference polar coordinates.

    Parameters
    ----------
    measured_phase: array_like
        Angular component of measured polar coordinates (in radians).
    measured_modulation: array_like
        Radial component of the measured polar coordinates.
    known_phase: array_like
        Angular component of the reference polar coordinates (in radians).
    known_modulation: array_like
        Radial component of the reference polar coordinates.

    Returns
    -------
    phase0: ndarray
        Angular component of polar coordinates for calibration.

    modulation0: ndarray
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
    phase0 = measured_phase - known_phase
    modulation0 = measured_modulation / known_modulation
    return phase0, modulation0


def phasor_to_polar(
    real: ArrayLike,
    imag: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return polar coordinates from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of the phasor coordinates.
    imag : array_like
        Imaginary component of the phasor coordinates.

    Returns
    -------
    phase: ndarray
        Phase values calculated from the phasor coordinates.
    modulation: ndarray
        Modulation values calculated from the phasor coordinates.

    Raises
    ------
    ValueError
        If the shapes of the 'real' and 'imag' do not match.

    Examples
    --------
    >>> phasor_to_polar(
    ...     [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    ... )
    (array([1.326, 1.19, 1.107]), array([4.123, 5.385, 6.708]))

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    phase = numpy.arctan2(imag, real)
    modulation = numpy.hypot(real, imag)
    return phase, modulation


def phasor_from_lifetime(
    frequency: float,
    lifetime: ArrayLike,
    /,
    fraction: ArrayLike | None = None,
    *,
    is_preexp: bool = False,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Return phasor coordinates from multi-exponential lifetime components.

    Parameters
    ----------
    frequency: float
        Laser pulse or modulation frequency in MHz.
    lifetime: array_like
        Lifetime components in ns. May be a scalar for single-exponential
        or a sequence for multi-exponential components.
    fraction: array_like, optional
        Fractional intensities or pre-exponential amplitudes of the lifetime
        components. Must be of same shape as `lifetime`.
        If None (default), fractions are 1 for all lifetime components.
        Fractions are normalized internally.
    is_preexp: bool, optional
        If true, `fraction` values are pre-exponential amplitudes,
        else fractional intensities (default).

    Returns
    -------
    real: ndarray
        Real component of phasor coordinates.
    imag: ndarray
        Imaginary components of phasor coordinates.

    Raises
    ------
    ValueError
        Input arrays do not match or are more than one dimensional.

    Examples
    --------
    Phasor coordinates of a single lifetime component at 80 MHz:

    >>> phasor_from_lifetime(80.0, 1.9894368)  # doctest: +NUMBER
    (0.5, 0.5)

    Phasor coordinates of two lifetime components with equal fractional
    intensities:

    >>> phasor_from_lifetime(
    ...     80.0, [3.9788735, 0.9947183], [0.5, 0.5]
    ... )  # doctest: +NUMBER
    (0.5, 0.4)

    Phasor coordinates of a double-exponential decay with equal
    pre-exponential amplitudes:

    >>> phasor_from_lifetime(
    ...     80.0, [3.9788735, 0.9947183], [0.5, 0.5], is_preexp=True
    ... )  # doctest: +NUMBER
    (0.32, 0.4)

    """
    tau = numpy.array(lifetime, dtype=numpy.float64, copy=True)
    if tau.ndim > 1:
        raise ValueError('lifetime must be scalar or one-dimensional array')
    if fraction is None:
        frac = numpy.ones_like(tau)
    else:
        frac = numpy.array(fraction, dtype=numpy.float64, copy=True)
        if tau.shape != frac.shape:
            raise ValueError(f'shape mismatch {tau.shape} != {frac.shape}')
    if is_preexp:
        # preexponential amplitudes to fractional intensities
        frac *= tau
    frac /= numpy.sum(frac)  # TODO: check for zero
    tau *= frequency * 2e-3 * math.pi  # omega_tau
    tmp = numpy.square(tau)
    tmp += 1.0
    tmp **= -1
    tmp *= frac
    re = numpy.sum(tmp)
    tmp *= tau
    im = numpy.sum(tmp)
    return re, im


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
        Real component of the phasor coordinates.
    imag : array_like
        Imaginary component of the phasor coordinates.
    skip_axes : tuple of int, optional
        Axes to be excluded during center calculation. If None, all
        axes are considered.
    method : str, optional
        Method used for center calculation:
            - 'mean': Arithmetic mean of the coordinates.
            - 'median': Spatial median of the coordinates.

    Returns
    -------
    real_center: ndarray
        Real center coordinates calculated based on the specified method.
    imag_center: ndarray
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
        Real components of the phasor coordinates.
    imag : numpy.ndarray
        Imaginary components of the phasor coordinates.
    skip_axes : tuple of int, optional
        Axes to be excluded during center calculation. If None, all
        axes are considered.

    Returns
    -------
    real_center: ndarray
        Mean real center coordinates.
    imag_center: ndarray
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
    real_center: ndarray
        Spatial median center for real coordinates.
    imag_center: ndarray
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


def phasor_from_signal(
    signal: NDArray[Any], fft, /
    , *, harmonic: int = 1, axis: int=0, norm: str=None
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Return the phasor transform using the
    fast fourier transform algorithm.

    Parameters
    ----------
    signal : array_like
        description: Input array.
            dims = ZXY. Where z is along axis 0.
    fft: fast fouerier function from other dependecies. 
        ex: numpy.fft.fft.
    harmonic : int, optional
        description:, by default 1, correspond to the nth
            harmonic.
    axis: int, optional. Axis where to compute the fft. Default 0.
    norm: string, optional.

    Returns
    -------
        dc: nd array or tuple.
            Contains the sum over the given axis.
        real: nd array or tuple.
            Contains the real part of the fft transform.
        imag: nd array or tuple.
            Contains the imaginary part of the fft transform.

    Example
    -------
    >>> _phasor_from_signal(numpy.random.rand(32, 64, 64), numpy.fft.fft)

    """
    if harmonic < 1: 
        raise ValueError("harmonic must be greater than 1")
    else:
        if norm: 
            ft = fft(signal, axis=axis, norm=norm)
        else:
            ft = fft(signal, axis=axis)
        dc = ft[0].real
        real = ft[harmonic].real
        imag = ft[harmonic].imag
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            re /= dc
            im /= -dc
    return dc, real, imag
