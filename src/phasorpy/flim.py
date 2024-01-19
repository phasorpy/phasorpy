"""Manipulate Fluorescence Lifetime Microscopy (FLIM) data for phasor analysis.

The ``phasorpy.flim`` module provides functions for the calibration of time-
resolved fluorescence lifetime imaging (FLIM) data using phasor techniques.
The module is designed to facilitate the processing and interpretation of FLIM
data, including the correction of phase and modulation parameters and the
application of phasor analysis.

Lifetime values for known fluorophores can be obtained from the ISS website:
https://iss.com/resources#lifetime-data-of-selected-fluorophores.

"""

from __future__ import annotations

__all__ = [
    "calibration_parameters",
    "calibrate_phasor",
    "cartesian_center_of_mass",
    "polar_center_of_mass",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import ArrayLike, Literal

import numpy


def calibration_parameters(
    reference_lifetime: float,
    frequency: float,
    /,
    *,
    reference_real: ArrayLike | None = None,
    reference_imag: ArrayLike | None = None,
    reference_phase: ArrayLike | None = None,
    reference_modulation: ArrayLike | None = None,
    center_function: Literal[
        "mean", "spatial_median", "geometric_median"
    ] = "mean",
) -> tuple[float, float]:
    """Return phase and modulation correction parameters based on the observed
    location of a reference fluorophore in phasor coordinates and the known
    lifetime of fluorophores.

    Parameters
    ----------
    reference_lifetime : float
        Lifetime in nanoseconds of the fluorophore used as a reference.
    frequency : float
        Repetition frequency in MHz.
    reference_real : array_like,
        Real component of the phasor coordinates of the reference fluorophore.
    reference_imag : array_like,
        Imaginary component of the phasor coordinates of the reference
        fluorophore.
    reference_phase : array_like,
        Angular phase values for reference fluorophore.
    reference_modulation : array_like,
        Modulation values reference fluorophore.

    center_function : str
        The method used to calculate the center of mass or centroid of the
        phasor coordinates. Supported values are 'mean' (default),
        'spatial_median', and 'geometric_median'.

    Returns
    -------
    tuple[float, float]
        phase_correction : Factor to perform the phase calibration.
        modulation_correction : Factor to perform the modulation calibration.

    Examples
    --------
    >>> real = numpy.array([1,2,3])
    >>> imag = numpy.array([4,5,6])
    >>> calibration_parameters(4, 80, reference_real=real, reference_imag=imag)
    (..., ...)

    >>> calibration_parameters(
    ...     2.5, 80, reference_real=real, reference_imag=imag,
    ...     center_function='geometric_median'
    ... )
    (..., ...)
    """
    omega_tau = (
        2 * numpy.pi * float(frequency) / 1000 * float(reference_lifetime)
    )
    theoretical_real = 1 / (1 + numpy.power(omega_tau, 2))
    theoretical_imaginary = theoretical_real * omega_tau
    phase_theoretical = numpy.arctan(theoretical_imaginary / theoretical_real)
    modulation_theoretical = numpy.sqrt(
        numpy.power(theoretical_real, 2)
        + numpy.power(theoretical_imaginary, 2)
    )
    if reference_real is not None or reference_imag is not None:
        assert (
            reference_real is not None and reference_imag is not None
        ), "Both 'reference_real' and 'reference_imag' must be provided"
        reference_real = numpy.asarray(reference_real)
        reference_imag = numpy.asarray(reference_imag)
        real_center, imag_center = cartesian_center_of_mass(
            reference_real,
            reference_imag,
            method=center_function,
        )
        phase_observed = numpy.arctan(imag_center / real_center)
        modulation_observed = numpy.sqrt(
            numpy.power(real_center, 2) + numpy.power(imag_center, 2)
        )
        phase_correction = (
            phase_theoretical - numpy.pi - phase_observed
            if real_center < 0
            else phase_theoretical - phase_observed
        )
    elif reference_phase is not None or reference_modulation is not None:
        assert (
            reference_phase is not None and reference_modulation is not None
        ), "Both 'reference_phase' and 'reference_modulation' must be provided"
        reference_phase = numpy.asarray(reference_phase)
        reference_modulation = numpy.asarray(reference_modulation)
        phase_observed, modulation_observed = polar_center_of_mass(
            reference_phase,
            reference_modulation,
            method=center_function,
        )
        phase_correction = phase_theoretical - phase_observed
        if phase_correction < -numpy.pi:
            phase_correction += 2 * numpy.pi
        elif phase_correction > numpy.pi:
            phase_correction -= 2 * numpy.pi

    modulation_correction = modulation_theoretical / modulation_observed
    return phase_correction, modulation_correction


def calibrate_phasor(
    phase_correction: float,
    modulation_correction: float,
    /,
    *,
    real: ArrayLike | None = None,
    imag: ArrayLike | None = None,
    phase: ArrayLike | None = None,
    modulation: ArrayLike | None = None,
) -> tuple[ArrayLike, ArrayLike]:
    """Calibrate phasor coordinates by applying phase and modulation
    correction based on the provided calibration parameters.

    Parameters
    ----------
    phase_correction : float
        Phase correction parameter. Can be calculated with
        `calibration_parameters` function.
    modulation_correction : float
        Modulation correction parameter. Can be calculated with
        `calibration_parameters` function.
    real : array_like
        Real component of the phasor to be calibrated.
    imag : array_like
        Imaginary component of the phasor to be calibrated.
    phase : array_like
        Angular phase values of the phasor to be calibrated.
    modulation : array_like
        Modulation values of the phasor to be calibrated.

    Returns
    -------
    Tuple[array_like, array_like]
        real_calibrated: Array or float containing the calibrated real
            component of the phasor.
        imag_calibrated: Array or float containing the calibrated
            imaginary component of the phasor.

    Examples
    --------
    >>> real = numpy.array([1,2,3])
    >>> imag = numpy.array([4,5,6])
    >>> calibration_params = (0, 1)
    >>> calibrate_phasor(*calibration_params, real=real, imag=imag)
    (array([1., 2., 3.]), array([4., 5., 6.]))
    """
    phase_correction = float(phase_correction)
    modulation_correction = float(modulation_correction)
    if real is not None or imag is not None:
        assert (
            real is not None and imag is not None
        ), "Both 'real' and 'imag' must be provided'"
        real = numpy.asarray(real)
        imag = numpy.asarray(imag)

        phase_correction_matrix = numpy.array(
            (
                (numpy.cos(phase_correction), -numpy.sin(phase_correction)),
                (numpy.sin(phase_correction), numpy.cos(phase_correction)),
            )
        )
        real_calibrated, imag_calibrated = (
            phase_correction_matrix.dot(
                numpy.vstack([real.flatten(), imag.flatten()])
            )
            * modulation_correction
        ).reshape((2, *real.shape))
        return real_calibrated, imag_calibrated
    elif phase is not None or modulation is not None:
        assert (
            phase is not None and modulation is not None
        ), "Both 'phase' and 'modulation' must be provided"
        phase = numpy.asarray(phase)
        modulation = numpy.asarray(modulation)
        phase_calibrated = phase + phase_correction
        modulation_calibrated = modulation * modulation_correction
        return phase_calibrated, modulation_calibrated
    else:
        raise ValueError('Incomplete data input')


def cartesian_center_of_mass(
    x_coords: ArrayLike,
    y_coords: ArrayLike,
    /,
    *,
    method: Literal["mean", "spatial_median", "geometric_median"] = "mean",
) -> tuple[float, float]:
    """Return the center of mass (centroid) of a set of 2D coordinates.

    Parameters
    ----------
    x_coords : array_like
        Values for the x-coordinates of the points.
    y_coords : array_like
        Values for the y-coordinates of the points.
    method : str
        The method used to calculate the center of mass. Supported values are
        'mean' (default), 'spatial_median', and 'geometric_median'.

    Returns
    -------
    tuple
        A tuple representing the (x, y) coordinates of the center of mass.

    Raises
    ------
    ValueError
        If an invalid method is provided.

    Examples
    --------
    >>> x = numpy.array([1,2,3])
    >>> y = numpy.array([4,5,6])
    >>> cartesian_center_of_mass(x, y)
    (...,...)

    >>> cartesian_center_of_mass(x, y, method='spatial_median')
    (...,...)

    >>> cartesian_center_of_mass(x, y, method='geometric_median')
    (...,...)
    """
    x_coords = numpy.asarray(x_coords)
    y_coords = numpy.asarray(y_coords)
    if method == "mean":
        return numpy.mean(x_coords), numpy.mean(y_coords)
    elif method == "spatial_median":
        points = numpy.column_stack((x_coords.flatten(), y_coords.flatten()))
        medians = numpy.median(points, axis=0)
        return medians[0], medians[1]
    elif method == "geometric_median":
        points = numpy.column_stack((x_coords.flatten(), y_coords.flatten()))
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
    else:
        raise ValueError(
            "Invalid method. Supported methods: "
            "'mean', 'spatial_median', 'geometric_median'"
        )


def polar_center_of_mass(
    phase: ArrayLike,
    modulation: ArrayLike,
    /,
    *,
    method: Literal["mean", "spatial_median", "geometric_median"] = "mean",
) -> tuple[float, float]:
    """Return the center of mass (centroid) of a set of points in polar
    coordinates.

    Parameters
    ----------
    phase : array_like
        Values for the phase coordinates (in radians) of the points.
    modulation : array_like
        Values for the modulation coordinates of the points.
    method : str
        The method used to calculate the center of mass. Supported values are
        'mean' (default), 'spatial_median', and 'geometric_median'.

    Returns
    -------
    tuple
        A tuple representing the (phase, modulation) coordinates of the center
        of mass.

    Raises
    ------
    ValueError
        If an invalid method is provided.

    Examples
    --------
    >>> x = numpy.array([1,2,3])
    >>> y = numpy.array([4,5,6])
    >>> polar_center_of_mass(x, y)
    (...,...)

    >>> polar_center_of_mass(x, y, method='spatial_median')
    (...,...)

    >>> polar_center_of_mass(x, y, method='geometric_median')
    (...,...)
    """
    phase = numpy.asarray(phase)
    modulation = numpy.asarray(modulation)
    if method == "mean":
        return numpy.mean(phase), numpy.mean(modulation)
    elif method == "spatial_median":
        return numpy.median(phase), numpy.median(modulation)
    elif method == "geometric_median":
        modulation_median = numpy.median(modulation)
        for _ in range(100):
            weights = 1 / (modulation + 1e-6)
            phase_new = numpy.arctan2(
                numpy.sum(weights * numpy.sin(phase)),
                numpy.sum(weights * numpy.cos(phase)),
            )
            if numpy.abs(phase_new - numpy.median(phase)) < 1e-6:
                break
            phase = numpy.asarray(phase_new)
        return phase_new, modulation_median
    else:
        raise ValueError(
            "Invalid method. Supported methods: "
            "'mean', 'spatial_median', 'geometric_median'"
        )
