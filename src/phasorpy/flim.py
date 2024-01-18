"""Manipulate Fluorescence Lifetime Microscopy (FLIM) data for phasor analysis.

The ``phasorpy.flim`` module provides functions for the calibration of time-
resolved fluorescence lifetime imaging (FLIM) data using phasor techniques. The
module is designed to facilitate the processing and interpretation of FLIM data,
including the correction of phase and modulation parameters and the application
of phasor analysis.

Lifetime values for known fluorophores can be obtained from the ISS website:
https://iss.com/resources#lifetime-data-of-selected-fluorophores.

"""

from __future__ import annotations

__all__ = [
    'calibration_parameters',
    'calibrate_phasor',
    'center_of_mass',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray, Sequence, Literal

import numpy


def calibration_parameters(
    reference_real_phasor: ArrayLike | Sequence | float,
    reference_imaginary_phasor: ArrayLike | Sequence | float,
    reference_tau: float,
    laser_frequency: float,
    center_function: Literal[
        'mean', 'spatial_median', 'geometric_median'
    ] = 'mean',
) -> tuple[float, float]:
    """Get phase and modulation correction parameters based on the observed
    location of a reference fluorophore in phasor coordinates and the known
    lifetime of fluorophores.

    Parameters
    ----------
    reference_real_phasor : ArrayLike | Sequence | float,
        Real part of the phasor coordinates of the reference fluorophore.
    reference_imaginary_phasor : ArrayLike | Sequence | float,
        Imaginary part of the phasor coordinates of the reference fluorophore.
    laser_frequency : float
        Laser repetition frequency in MHz.
    reference_tau : float
        Lifetime in nanoseconds of the fluorophore used as a reference.
    center_function : Literal[
            'mean',
            'spatial_median',
            'geometric_median'
        ], optional
        Function to calculate the center of mass or centroid of the phasor
        coordinates, by default 'mean'.

    Returns
    -------
    tuple[float, float]
        phi_correction : Factor to perform the phase calibration.
        modulation_correction : Factor to perform the modulation calibration.

    Examples
    --------
    >>> real_phasor = numpy.array([1,2,3])
    >>> imaginary_phasor = numpy.array([4,5,6])
    >>> calibration_parameters(real_phasor, imaginary_phasor, 2.5, 80)
    (..., ...)

    >>> calibration_parameters(
    ...     real_phasor, imaginary_phasor, 2.5, 80,
    ...     center_function='geometric_median'
    ... )
    (..., ...)
    """
    reference_real_phasor = numpy.asarray(reference_real_phasor)
    reference_imaginary_phasor = numpy.asarray(reference_imaginary_phasor)
    omega_tau = (
        2 * numpy.pi * float(laser_frequency) / 1000 * float(reference_tau)
    )
    theoretical_real = 1 / (1 + numpy.power(omega_tau, 2))
    theoretical_imaginary = theoretical_real * omega_tau
    phi_theoretical = numpy.arctan(theoretical_imaginary / theoretical_real)
    modulation_theoretical = numpy.sqrt(
        numpy.power(theoretical_real, 2)
        + numpy.power(theoretical_imaginary, 2)
    )
    real_center, imaginary_center = center_of_mass(
        reference_real_phasor,
        reference_imaginary_phasor,
        method=center_function,
    )
    phi_observed = numpy.arctan(imaginary_center / real_center)
    modulation_observed = numpy.sqrt(
        numpy.power(real_center, 2) + numpy.power(imaginary_center, 2)
    )
    phi_correction = (
        phi_theoretical - numpy.pi - phi_observed
        if real_center < 0
        else phi_theoretical - phi_observed
    )
    modulation_correction = modulation_theoretical / modulation_observed
    return phi_correction, modulation_correction


def calibrate_phasor(
    real_phasor: ArrayLike | Sequence | float,
    imaginary_phasor: ArrayLike | Sequence | float,
    calibration_params: tuple[float, float],
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Calibrate fluorescence lifetime imaging (FLIM) data.

    This function applies phase and modulation correction to FLIM data
    based on the provided calibration parameters.

    Parameters
    ----------
    real_phasor : ArrayLike | Sequence | float
        Real component of the phasor to be calibrated.
    imaginary_phasor : ArrayLike | Sequence | float
        Imaginary component of the phasor to be calibrated.
    calibration_parameters : tuple[float,float]
        Pre-calculated phase and modulation correction parameters. Can be
        calculated with `calibration_parameters` function.

    Returns
    -------
    Tuple[NDArray[Any], NDArray[Any]]
        real_phasor_calibrated: Array containing the calibrated real
            component of the phasor.
        imaginary_phasor_calibrated: Array containing the calibrated
            imaginary component of the phasor.

    Examples
    --------
    >>> real_phasor = numpy.array([1,2,3])
    >>> imaginary_phasor = numpy.array([4,5,6])
    >>> calibration_params = (0, 1)
    >>> calibrate_phasor(real_phasor, imaginary_phasor, calibration_params)
    (array([1., 2., 3.]), array([4., 5., 6.]))
    """
    real_phasor = numpy.asarray(real_phasor)
    imaginary_phasor = numpy.asarray(imaginary_phasor)
    phi_correction = calibration_params[0]
    modulation_correction = calibration_params[1]
    phase_correction_matrix = numpy.array(
        (
            (numpy.cos(phi_correction), -numpy.sin(phi_correction)),
            (numpy.sin(phi_correction), numpy.cos(phi_correction)),
        )
    )
    real_phasor_calibrated, imaginary_phasor_calibrated = (
        phase_correction_matrix.dot(
            numpy.vstack([real_phasor.flatten(), imaginary_phasor.flatten()])
        )
        * modulation_correction
    ).reshape((2, *real_phasor.shape))
    if isinstance(real_phasor, list):
        real_phasor_calibrated = real_phasor_calibrated.tolist()
    if isinstance(imaginary_phasor, list):
        imaginary_phasor_calibrated = imaginary_phasor_calibrated.tolist()
    return real_phasor_calibrated, imaginary_phasor_calibrated


def center_of_mass(
    x_coords: ArrayLike | Sequence | float,
    y_coords: ArrayLike | Sequence | float,
    method: Literal['mean', 'spatial_median', 'geometric_median'] = 'mean',
) -> tuple[float, float]:
    """Calculate the center of mass (centroid) of a set of 2D coordinates.

    Parameters
    ----------
    x_coords : ArrayLike | Sequence | float
        Array containing x-coordinates of the points.
    y_coords : ArrayLike | Sequence | float
        Array containing y-coordinates of the points.
    method : Literal[
            'mean',
            'spatial_median',
            'geometric_median'
        ], optional
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
    >>> center_of_mass(x, y)
    (...,...)

    >>> center_of_mass(x, y, method='spatial_median')
    (...,...)

    >>> center_of_mass(x, y, method='geometric_median')
    (...,...)
    """
    x_coords = numpy.asarray(x_coords)
    y_coords = numpy.asarray(y_coords)
    if method == 'mean':
        return numpy.mean(x_coords), numpy.mean(y_coords)
    elif method == 'spatial_median':
        points = numpy.column_stack((x_coords.flatten(), y_coords.flatten()))
        medians = numpy.median(points, axis=0)
        return medians[0], medians[1]
    elif method == 'geometric_median':
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
