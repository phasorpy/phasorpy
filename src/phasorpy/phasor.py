"""Calculate, convert, calibrate, and reduce phasor coordinates.

The ``phasorpy.phasor`` module provides functions to:

- calculate phasor coordinates from time-resolved and spectral signals
- calculate phasor coordinates from single- or multi-component
    fluorescence lifetimes
- convert between phasor and polar (phase and modulation) coordinates,
    and apparent single lifetimes
- calibrate phasor coordinates with reference signal of known
    fluorescence lifetime
- reduce arrays of phasor coordinates to single coordinates

"""

from __future__ import annotations

__all__ = [
    "phasor_calibrate",
    "polar_from_reference_phasor",
    "polar_from_reference",
    "phasor_to_polar",
    "phasor_from_lifetime",
    "phasor_center",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, NDArray, ArrayLike, Literal

import math

import numpy


def phasor_calibrate(
    real: ArrayLike,
    imag: ArrayLike,
    phase0: ArrayLike = 0.0,
    modulation0: ArrayLike = 1.0,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return calibrated/referenced phasor coordinates.

    Parameters
    ----------
    real : array_like
        Scalar or array containing the real components of phasor coordinates.
    imag : array_like
        Scalar or array containing the imaginary components of phasor
        coordinates.
    phase0 : array_like, optional
        Angular component of polar coordinates for calibration. Defaults
        to 0.0.
    modulation0 : array_like, optional
        Radial component of polar coordinates for calibration. Defaults to 1.0.

    Returns
    -------
    real, imag:
        Calibrated real and imaginary components of phasor coordinates.

    Examples
    --------
    >>> real = numpy.array([1.0, 2.0, 3.0])
    >>> imag = numpy.array([4.0, 5.0, 6.0])
    >>> phase0 = 0.5
    >>> modulation0 = 2.0
    >>> real_calibrated, imag_calibrated = phasor_calibrate(
    ...     real, imag,
    ...     phase0,
    ...     modulation0
    ... )
    >>> real_calibrated
    array(...)
    >>> imag_calibrated
    array(...)
    >>> phase0 = numpy.array([0.5, 0.2, 0.3])
    >>> modulation0 = numpy.array([1.5, 2.0, 0.3])
    >>> real_calibrated, imag_calibrated = phasor_calibrate(
    ...     real, imag,
    ...     phase0,
    ...     modulation0
    ... )
    >>> real_calibrated
    array(...)
    >>> imag_calibrated
    array(...)
    """
    phi0 = numpy.asarray(phase0)
    mod0 = numpy.asarray(modulation0)
    re = numpy.asarray(real)
    im = numpy.asarray(imag)
    if numpy.all(phi0 == 0) and numpy.all(mod0 == 1):
        return re, im
    re_calibrated = (re * numpy.cos(phi0) - im * numpy.sin(phi0)) * mod0
    im_calibrated = (re * numpy.sin(phi0) + im * numpy.cos(phi0)) * mod0
    return re_calibrated, im_calibrated


def polar_from_reference_phasor(
    measured_real: ArrayLike,
    measured_imag: ArrayLike,
    known_real: ArrayLike,
    known_imag: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return  components for calibration from reference phasor.

    Parameters
    ----------
    measured_real: array_like
        Scalar or array containing the real component of measured
        phasor coordinates.
    measured_imag: array_like
        Scalar or array containing the imaginary component of the measured
        phasor coordinates.
    known_real: array_like
        Scalar or array containing the real component of the reference
        phasor coordinates.
    known_imag: array_like
        Scalar or array containing the imaginary component of the reference
        phasor coordinates.

    Returns
    -------
    phase0, modulation0: ndarray
        Angular and radial components of polar coordinates for calibration.

    Examples
    --------
    >>> polar_from_reference_phasor(0.5, 0.0, 1.0, 0.0)
    (0.0, 2.0)
    """
    measured_real = numpy.asarray(measured_real)
    measured_imag = numpy.asarray(measured_imag)
    known_real = numpy.asarray(known_real)
    known_imag = numpy.asarray(known_imag)
    measured_phi, measured_mod = phasor_to_polar(measured_real, measured_imag)
    known_phi, known_mod = phasor_to_polar(known_real, known_imag)
    phase0 = numpy.where(
        measured_real < 0,
        known_phi - numpy.pi - measured_phi,
        known_phi - measured_phi,
    )
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
        Scalar or array containing the angular component of measured polar
        coordinates (in radians).
    measured_modulation: array_like
        Scalar or array containing the radial component of the measured
        polar coordinates.
    known_phase: array_like
        Scalar or array containing the angular component of the reference
        polar coordinates (in radians).
    known_modulation: array_like
        Scalar or array containing the radial component of the reference
        polar coordinates.

    Returns
    -------
    phase0, modulation0:
        Angular and radial components of polar coordinates for calibration.

    Examples
    --------
    >>> polar_from_reference(0.4, 1.3, 0.2, 0.4)
    (0.2, 3.25)
    """
    measured_phase = numpy.asarray(measured_phase)
    measured_modulation = numpy.asarray(measured_modulation)
    known_phase = numpy.asarray(known_phase)
    known_modulation = numpy.asarray(known_modulation)
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
        Scalar or array containing the real components of the
        phasor coordinates.
    imag : array_like
        Scalar or array containing the imaginary components of the
        phasor coordinates.

    Returns
    -------
    phase, modulation:
        Phase and modulation values calculated from the phasor coordinates.

    Examples
    --------
    >>> real_data = numpy.array([1.0, 2.0, 3.0])
    >>> imag_data = numpy.array([4.0, 5.0, 6.0])
    >>> phase, modulation = phasor_to_polar(
    ...     real_data, imag_data
    ... )
    >>> phase
    array(...)
    >>> modulation
    array(...)
    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
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
    real, imag: ndarray
        Real and imaginary components of phasor coordinates.

    Raises
    ------
    ValueError
        Input arrays do not match or are more than one dimensional.

    Examples
    --------
    >>> phasor_from_lifetime(80.0, 1.9894368)  # 1000/(2*pi*80) = 1.9894368
    (0..., 0...)
    >>> phasor_from_lifetime(80.0, [3.9788735, 0.9947183], [0.5, 0.4])
    (0..., 0...)
    >>> phasor_from_lifetime(
    ...     80.0, [3.9788735, 0.9947183], [0.5, 0.5], is_preexp=True
    ... )
    (0..., 0...)

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
    method: Literal['mean', 'spatial_median', 'geometric_median'] = 'mean',
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return center of phasor coordinates.

    Parameters
    ----------
    real : array_like
        Scalar or array containing the real components of the
        phasor coordinates.
    imag : array_like
        Scalar or array containing the imaginary components of the
        phasor coordinates.
    skip_axes : tuple[int, ...] or None, optional
        Axes to be excluded during center calculation. If None, all
        axes are considered.
    method : str, optional
        Method used for center calculation:
            - 'mean': Arithmetic mean of the coordinates.
            - 'spatial_median': Spatial median of the coordinates.
            - 'geometric_median': Geometric median of the coordinates.

    Returns
    -------
    real_center, imag_center:
        Real and imaginary center coordinates calculated based on
        the specified method.

    Examples
    --------
    >>> real = numpy.array([1.0, 2.0, 3.0])
    >>> imag = numpy.array([4.0, 5.0, 6.0])
    >>> phasor_center(real, imag, method='mean')
    (2.0, 5.0)
    >>> phasor_center(real, imag, method='spatial_median')
    (2.0, 5.0)
    >>> phasor_center(real, imag, method='geometric_median')
    (2.0, 5.0)
    """
    supported_methods = ['mean', 'spatial_median', 'geometric_median']
    assert method in supported_methods, (
        f"Method not supported, supported methods are: "
        f"{', '.join(supported_methods)}"
    )
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    return {
        'mean': _mean,
        'spatial_median': _spatial_median,
        'geometric_median': _geometric_median,
    }[method](real, imag)


def _mean(
    real: NDArray[Any], imag: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the mean center of phasor coordinates.

    Parameters
    ----------
    real : numpy.ndarray
        Array containing the real components of the phasor coordinates.
    imag : numpy.ndarray
        Array containing the imaginary components of the phasor coordinates.

    Returns
    -------
    real_center, imag_center:
        Mean real and imaginary center coordinates.

    Examples
    --------
    >>> real_data = numpy.array([1.0, 2.0, 3.0])
    >>> imag_data = numpy.array([4.0, 5.0, 6.0])
    >>> _mean(real_data, imag_data)
    (2.0, 5.0)
    """
    return numpy.mean(real), numpy.mean(imag)


def _spatial_median(
    real: NDArray[Any], imag: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the spatial median center of phasor coordinates.

    Parameters
    ----------
    real : numpy.ndarray
        Array containing the real components of the phasor coordinates.
    imag : numpy.ndarray
        Array containing the imaginary components of the phasor coordinates.

    Returns
    -------
    real_center, imag_center:
        Spatial median center for real and imaginary coordinates.

    Examples
    --------
    >>> real_data = numpy.array([1.0, 2.0, 3.0])
    >>> imag_data = numpy.array([4.0, 5.0, 6.0])
    >>> _spatial_median(real_data, imag_data)
    (2.0, 5.0)
    """
    points = numpy.column_stack((real.flatten(), imag.flatten()))
    medians = numpy.median(points, axis=0)
    return medians[0], medians[1]


def _geometric_median(
    real: NDArray[Any], imag: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return the geometric median center of phasor coordinates.

    Parameters
    ----------
    real : numpy.ndarray
        Array containing the real components of the phasor coordinates.
    imag : numpy.ndarray
        Array containing the imaginary components of the phasor coordinates.

    Returns
    -------
    real_center, imag_center:
        Geometric median center for real and imaginary coordinates.

    Examples
    --------
    >>> real_data = numpy.array([1.0, 2.0, 3.0])
    >>> imag_data = numpy.array([4.0, 5.0, 6.0])
    >>> _geometric_median(real_data, imag_data)
    (2.0, 5.0)
    """
    points = numpy.column_stack((real.flatten(), imag.flatten()))
    x = numpy.median(points, axis=0)
    for _ in range(100):
        distances = numpy.linalg.norm(points - x, axis=1)
        weights = 1 / (distances + 1e-6)
        x_new = numpy.sum(
            weights[:, numpy.newaxis] * points, axis=0
        ) / numpy.sum(weights)
        if numpy.linalg.norm(x_new - x) < 1e-6:
            break
        x = x_new
    return x[0], x[1]
