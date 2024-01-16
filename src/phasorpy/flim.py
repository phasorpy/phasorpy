"""Manipulate Fluorescence Lifetime Microscopy (FLIM) data for phasor analysis.

The ``phasorpy.flim`` module provides functions and classes for the calibration and analysis
of time-resolved fluorescence lifetime imaging (FLIM) data using phasor techniques. The module
is designed to facilitate the processing and interpretation of FLIM data, including the correction
of phase and modulation parameters and the application of phasor analysis.

Classes
-------
FlimData
    Represents fluorescence lifetime imaging (FLIM) data with calibration capabilities.
    - Attributes:
        - `laser_frequency`
        - `original_data`
        - `average_phasor`
        - `real_phasor_coordinates`
        - `imaginary_phasor_coordinates`
        - `axis`
        - `phi_correction`
        - `modulation_correction`
        - `calibration_status`: 

    - Methods:
        - `calibrate_flim`
        
Functions
---------
calibrate_multiple_flim
    Calibrates a list of time-resolved FLIM data using reference with known lifetime, or with phase and modulation correction parameters.

Constants
---------
TAU_REFERENCES: dict
    Lifetime data of selected fluorophores to use as references for calibration.
    Data is obtained from the ISS website: https://iss.com/resources#lifetime-data-of-selected-fluorophores.

"""

from __future__ import annotations

__all__ = [
    'FlimData',
    'TAU_REFERENCES',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, NDArray, Callable, Dict, Sequence

import numpy
from phasorpy.phasor import phasor

from numpy.testing import assert_array_equal


class FlimData:
    """
    Represents fluorescence lifetime imaging (FLIM) data with calibration capabilities.

    Parameters
    ----------
    laser_frequency : float
        The frequency of the laser used for data acquisition.
    original_data : numpy.ndarray
        The original FLIM data.
    average_phasor : numpy.ndarray
        The average image along the specified axis
    real_phasor_coordinates : numpy.ndarray
        The real component of the fft transformed FLIM data
    imaginary_phasor_coordinates : numpy.ndarray
        The imaginary component of the fft transformed FLIM data
    axis : int, optional
        The axis along which phasor calculations are performed (default is 0).
    phi_correction : float, optional
        The initial correction applied to the phase angle (default is 0).
    modulation_correction : float, optional
        The initial correction applied to the modulation amplitude (default is 1).
    calibration_status : bool, optional
        The initial calibration status (default is False).

    Methods
    -------
    calibrate_flim(recalibrate=False, reference_data=None, reference_name=None, reference_tau=None,
                   center_function=None, **kwargs)
        Calibrates the FlimData object based on reference data and correction parameters.

    Attributes
    ----------
    laser_frequency : float
        The frequency of the laser used for data acquisition.
    original_data : numpy.ndarray
        The original FLIM data.
    average_phasor : numpy.ndarray
        The average image along the specified axis
    real_phasor_coordinates : numpy.ndarray
        The real component of the fft transformed FLIM data
    imaginary_phasor_coordinates : numpy.ndarray
        The imaginary component of the fft transformed FLIM data
    phi_correction : float
        The correction applied to the phase angle.
    modulation_correction : float
        The correction applied to the modulation amplitude.
    calibration_status : bool
        The calibration status indicating whether the data is calibrated.
    """

    def __init__(
        self,
        laser_frequency: float,
        original_data: NDArray[Any] | None = None,
        average_phasor: NDArray[Any] | None = None,
        real_phasor_coordinates: NDArray[Any] | None = None,
        imaginary_phasor_coordinates: NDArray[Any] | None = None,
        axis: int = 0,
        phi_correction: float = 0,
        modulation_correction: float = 1,
        calibration_status: bool = False,
    ) -> None:
        """
        Initialize a FlimData object.

        Parameters
        ----------
        laser_frequency : float
            The frequency of the laser used for data acquisition in MHz.
        original_data : numpy.ndarray
            The original fluorescence lifetime imaging (FLIM) data.
        average_phasor : numpy.ndarray , optional
            The average image along the specified axis
        real_phasor_coordinates : numpy.ndarray , optional
            The real component of the fft transformed FLIM data
        imaginary_phasor_coordinates : numpy.ndarray , optional
            The imaginary component of the fft transformed FLIM data
        axis : int, optional
            The axis along which phasor calculations are performed (default is 0).
        phi_correction : float, optional
            The initial correction applied to the phase angle (default is 0).
        modulation_correction : float, optional
            The initial correction applied to the modulation amplitude (default is 1).
        calibration_status : bool, optional
            The initial calibration status. If True, the data is considered calibrated;
            if False, the data is uncalibrated (default is False).

        Raises
        ------
        ValueError
            If the original data is not provided and either the average phasor or the real and imaginary coordinates are also not provided. 

        Examples
        --------
        >>> flim_data = FlimData(80, original_data = original_data)
        >>> flim_data.original_data
        array(...)
        >>> flim_data.laser_frequency
        80.0
        >>> flim_data.average_phasor
        array(...)
        >>> flim_data.phi_correction
        0.0
        >>> flim_data.calibration_status
        False
        
        """
        self.laser_frequency = float(laser_frequency)
        self.original_data = original_data
        if original_data is not None:
            (
                average_phasor,
                real_phasor_coordinates,
                imaginary_phasor_coordinates,
            ) = phasor(
                original_data
            )  # TODO: add axis
        elif (
            average_phasor is None
            or real_phasor_coordinates is None
            or imaginary_phasor_coordinates is None
        ):
            raise ValueError(
                'Either oringinal flim data or average image and phasor coordinates must be provided'
            )
        self.average_phasor = average_phasor
        self.real_phasor_coordinates = real_phasor_coordinates
        self.imaginary_phasor_coordinates = imaginary_phasor_coordinates

        self.phi_correction = float(phi_correction)
        self.modulation_correction = float(modulation_correction)

        self.calibration_status = calibration_status
    
    def __eq__(self, other):
        if isinstance(other, FlimData):
            assert_array_equal(self.original_data, other.original_data)
            assert self.laser_frequency == other.laser_frequency
            assert self.phi_correction == other.phi_correction
            assert self.modulation_correction == other.modulation_correction
            assert_array_equal(self.average_phasor, other.average_phasor)
            assert_array_equal(self.real_phasor_coordinates, other.real_phasor_coordinates)
            assert_array_equal(self.imaginary_phasor_coordinates, other.imaginary_phasor_coordinates)
            assert self.calibration_status == other.calibration_status
            return True
        return False

    def calibrate_flim(
        self,
        recalibrate: bool = False,
        reference_data: NDArray[Any] | FlimData | None = None,
        reference_name: str | None = None,
        reference_tau: float | None = None,
        center_function: Callable[..., tuple[float, float]] | str = 'mean',
        **kwargs: Any,
    ) -> None:
        """
        Calibrate the FlimData object based on reference data and correction parameters.

        Parameters
        ----------
        recalibrate : bool, optional
            If True, it forces recalibration even if the data is already calibrated, by default False.
        reference_data : numpy.ndarray or None, optional
            Reference fluorescence lifetime imaging (FLIM) data for calibration. Can be either the data as array or as a FlimData object, by default None.
        reference_name : str or None, optional
            Name of the reference for calibration. If provided, 'reference_tau' is ignored, by default None.
        reference_tau : float or None, optional
            Time constant of the reference for calibration. Ignored if 'reference_name' is provided, by default None.
        center_function : callable or None, optional
            Function to calculate the center of mass. Should take 'g_observed', 's_observed', and optional keyword arguments,
            and return two values: real_center and imaginary_center, by default 'mean'.
        **kwargs : dict
            Additional keyword arguments passed to the 'center_function', if provided.

        Raises
        ------
        ValueError
            If the data is already calibrated and 'recalibrate' is set to False,
            or if neither 'reference_name' nor 'reference_tau' is provided when 'reference_data' is given.
        KeyError
            If reference name is not in the selected fluorophores speified in REFERENCES_TAU

        Notes
        -----
        When called with reference data, the method adjusts the calibration parameters, such as phase and modulation corrections,
        based on the provided reference data and correction functions. The calibration process updates
        the 'phi_correction', 'modulation_correction', and 'calibration_status' attributes of the FlimData object.
        
        Examples
        --------
        >>> # Example 1: Recalibrating with reference data
        >>> flim_data.calibrate(reference_data=reference_data, reference_name='Rhodamine 110')
        >>> flim_data.calibration_status
        True

        >>> # Example 2: Calibrating with phi and modulation correction
        >>> flim_data.phi_correction = 1.5
        >>> flim_data.modulation_correction = -0.5
        >>> flim_data.calibrate()

        >>> # Example 3: Calibrating with a custom center function
        >>> flim_data.calibrate(center_function=my_custom_center_function, additional_arg=42)
   
        """
        if self.calibration_status and not recalibrate:
            raise ValueError(
                'Data is already calibrated. To force recalibration, set recalibrate=True.'
            )

        if reference_data is not None:
            if reference_name is not None:
                try:
                    reference_tau = TAU_REFERENCES[reference_name]  # type: ignore
                except KeyError:
                    raise KeyError(
                        f'{reference_name} not in reference list, search the correct name or enter the reference_tau instead'
                    )
            elif reference_tau is None:
                raise ValueError(
                    'Either reference name or reference tau must be provided'
                )
            omega_tau = (
                2
                * numpy.pi
                * float(self.laser_frequency)
                / 1000
                * float(reference_tau)
            )   
            g_reference = 1 / (1 + numpy.power(omega_tau, 2))
            s_reference = g_reference * omega_tau
            phi_reference = numpy.arctan(s_reference / g_reference)
            modulation_reference = numpy.sqrt(
                numpy.power(g_reference, 2) + numpy.power(s_reference, 2)
            )
            _, g_observed, s_observed = phasor(
                reference_data.original_data if isinstance(reference_data, FlimData) else reference_data
            )  # TODO add axis optional
            if center_function != 'mean':
                try:
                    real_center, imaginary_center = center_function(
                        g_observed, s_observed, **kwargs
                    )
                except:
                    raise ValueError(
                        'Function to calculate center of mass should recieve arguments real and imaginary coordinates, and the optional keyword arguments, and return two values: real_center and imaginary_center.'
                    )
            else:
                real_center, imaginary_center = numpy.mean(
                    g_observed
                ), numpy.mean(s_observed)
            phi_observed = numpy.arctan(imaginary_center / real_center)
            modulation_observed = numpy.sqrt(
                numpy.power(real_center, 2)
                + numpy.power(imaginary_center, 2)
            )

            self.phi_correction = (
                phi_reference - numpy.pi - phi_observed
                if real_center < 0
                else phi_reference - phi_observed
            )
            self.modulation_correction = (
                modulation_reference / modulation_observed
            )

        phase_correction_matrix = numpy.array(
            (
                (
                    numpy.cos(self.phi_correction),
                    -numpy.sin(self.phi_correction),
                ),
                (
                    numpy.sin(self.phi_correction),
                    numpy.cos(self.phi_correction),
                ),
            )
        )

        self.real_phasor_coordinates, self.imaginary_phasor_coordinates = (
            phase_correction_matrix.dot(
                numpy.vstack(
                    [
                        self.real_phasor_coordinates.flatten(),
                        self.imaginary_phasor_coordinates.flatten(),
                    ]
                )
            )
            * self.modulation_correction
        ).reshape((2, *self.real_phasor_coordinates.shape))

        self.calibration_status = True



def calibrate_multiple_flim(
    image_list: Sequence[FlimData],
    recalibrate: bool = False,
    reference_data: NDArray[Any] | None = None,
    reference_name: str | None = None,
    reference_tau: float | None = None,
    phi_correction: float | None = None,
    modulation_correction: float | None = None,
    center_function: Callable[..., tuple[float, float]] | str = 'mean',
    **kwargs: Any,
) -> None:
    """Calibrates a sequence of time-resolved FLIM data using the same reference with known lifetime, or with phase and modulation correction parameters.

    Parameters
    ----------
    image_sequence : list
        Sequence with images of time-resolved FLIM data as FlimData objects.
    recalibrate : bool, optional
        If True, it forces recalibration even if the data is already calibrated, by default False.
    reference_data : numpy.ndarray or None, optional
        Reference fluorescence lifetime imaging (FLIM) data for calibration, by default None.
    reference_name : str or None, optional
        Name of the reference for calibration. If provided, 'reference_tau' is ignored, by default None.
    reference_tau : float or None, optional
        Time constant of the reference for calibration. Ignored if 'reference_name' is provided, by default None.
    center_function : callable or None, optional
        Function to calculate the center of mass. Should take 'g_observed', 's_observed', and optional keyword arguments,
        and return two values: real_center and imaginary_center, by default 'mean'.
    **kwargs : dict
        Additional keyword arguments passed to the 'center_function', if provided.

    Raises
    ------
    ValueError
        If 'reference_data' is given without providing 'reference_name' or 'reference_tau', or if neither 'reference_data' or 'phi_correction' and 'modulation_correction' are provided.
    
    Examples
    --------
    >>> image_list = [image1, image2]
    >>> calibrate_multiple_flim(image_list, reference_data=reference_data, reference_tau=4)
    >>> image1.calibration_status
    True
    >>> image2.calibration_status
    True
    
    """
    for image in image_list:
        if phi_correction is not None and modulation_correction is not None:
            image.phi_correction = phi_correction
            image.modulation_correction = modulation_correction
            image.calibrate_flim(
                recalibrate=recalibrate,
                center_function=center_function,
                **kwargs,
            )
        elif reference_data is not None and (
            reference_name is not None or reference_tau is not None
        ):
            image.calibrate_flim(
                recalibrate=recalibrate,
                reference_data=reference_data,
                reference_name=reference_name,
                reference_tau=reference_tau,
                center_function=center_function,
                **kwargs,
            )
        else:
            raise ValueError(
                'Provide either reference data and reference name or tau, or provide phase and modulation correction values.'
            )


TAU_REFERENCES: Dict[str, float] = {
    'ATTO 565': 3.4,  # Water
    'ATTO 655': 3.6,  # Water
    'Acridine Orange': 2.0,  # PB pH 7.8
    'Alexa Fluor 488': 4.1,  # PB pH 7.4
    'Alexa Fluor 546': 4.0,  # PB pH 7.4
    'Alexa Fluor 633': 3.2,  # Water
    'Alexa Fluor 647': 1.0,  # Water
    'Alexa Fluor 680': 1.2,  # PB pH 7.5
    'BODIPY FL': 5.7,  # Methanol
    'BODIPY TR-X': 5.4,  # Methanol
    'Coumarin 6': 2.5,  # Ethanol
    'CY3B': 2.8,  # PBS
    'CY3': 0.3,  # PBS
    'CY3.5': 0.5,  # PBS
    'CY5': 1.0,  # PBS
    'CY5.5': 1.0,  # PBS
    'DAPI': 0.16,  # TRIS/EDTA
    'DAPI + ssDNA': 1.88,  # TRIS/EDTA
    'DAPI + dsDNA': 2.20,  # TRIS/EDTA
    'Ethidium Bromide - no DNA': 1.6,  # TRIS/EDTA
    'Ethidium Bromide + ssDNA': 25.1,  # TRIS/EDTA
    'Ethidium Bromide + dsDNA': 28.3,  # TRIS/EDTA
    'FITC': 4.1,  # PB pH 7.8
    'Fluorescein': 4.0,  # PB pH 7.5
    'GFP': 3.2,  # Buffer pH 8
    'Hoechst 33258 - no DNA': 0.2,  # TRIS/EDTA
    'Hoechst 33258 + ssDNA': 1.22,  # TRIS/EDTA
    'Hoechst 33258 + dsDNA': 1.94,  # TRIS/EDTA
    'Hoechst 33342 - no DNA': 0.35,  # TRIS/EDTA
    'Hoechst 33342 + ssDNA': 1.05,  # TRIS/EDTA
    'Hoechst 33342 + dsDNA': 2.21,  # TRIS/EDTA
    'HPTS': 5.4,  # PB pH 7.8
    'Indocyanine Green': 0.52,  # Water
    'Lucifer Yellow': 5.7,  # Water
    'Oregon Green 488': 4.1,  # Buffer pH 9
    'Oregon Green 500': 2.18,  # Buffer pH 2
    'Prodan': 1.41,  # Water
    'Rhodamine 101': 4.32,  # Water
    'Rhodamine 110': 4.0,  # Water
    'Rhodamine 6G': 4.08,  # Water
    'Rhodamine B': 1.68,  # Water
    'Ru(bpy)3[PF6]2': 600,  # Water
    'Ru(bpy)2(dcpby)[PF6]2': 375,  # Buffer pH 7
    'SeTau-380-NHS': 32.5,  # Water
    'SeTau-404-NHS': 9.3,  # Water
    'SeTau-405-NHS': 9.3,  # Water
    'SeTau-425-NHS': 26.2,  # Water
    'Texas Red': 4.2,  # Water
    'TOTO-1': 2.2,  # Water
    'YOYO-1 no DNA': 2.1,  # TRIS/EDTA
    'YOYO-1 + ssDNA': 1.67,  # TRIS/EDTA
    'YOYO-1 + dsDNA': 2.3,  # TRIS/EDTA
}
"""Lifetime data of selected fluorophores to use as reference for calibration.

Data is obtained from the ISS website: https://iss.com/resources#lifetime-data-of-selected-fluorophores.

Lifetime is in nanoseconds (ns). Solvent where the lifetime was measured is stated as a comment.
"""
