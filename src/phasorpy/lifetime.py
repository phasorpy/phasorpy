"""


The `phasorpy.lifetime` module provides essential functions for the analysis of fluorescence lifetime imaging (FLIM) data. 
Here's an overview of the key functions and their roles in the context of the phasorpy project:

phasor_coordinates
Purpose: Converts phase and modulation data into phasor coordinates in the frequency domain.

phasemodule_values
Purpose: Converts phasor coordinates back into phase and module.

lifetime_computation_phasor
Purpose: Calculates lifetime values from phasor coordinates and laser repetition frequency, yielding τm and τφ.

lifetime_computation_array
Purpose: Calculates lifetime values for each pixel from phasor coordinates and laser repetition frequency, to map τm and τφ.

phasor_lifetime
Purpose: Computes lifetime values and standard deviations from phasor coordinates and laser repetition frequency with error propagation.

refplot
Purpose: Generates reference phasor coordinates based on sample lifetimes and laser repetition frequency.

fractional_intensities
Purpose: Calculates intensity fractions between two reference points and a target phasor.

apparent_lifetime
Purpose: Calculate apparent lifetime and relative fractions.

Please note that FRET-related functionalities are currently under development and will be added by experts in the future. The `phasorpy.lifetime` module forms a valuable foundation for FLIM data analysis in your project, with these functions serving as key components for your analysis and research needs.

"""
from __future__ import annotations
__all__ = [
    'phasor_coordinates',
    'phasemodule_values',
    'lifetime_computation_phasor',
    'lifetime_computation_array',
    'phasor_lifetime',
    'refplot',
    'fractional_intensities',
    'apparent_lifetime'
    ]

from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, PathLike, Sequence


def phasor_coordinates(
    phi: ArrayLike,
    mod: ArrayLike,
    /,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Convert phase and module information into phasor coordinates in the frequency domain.
    Parameters:
    ----------
    phi: array_like
        Phase angle values in radians for each pixel
    mod: array_like
        Module values for each pixel
    Returns:
    -------
    g: array_like
        Real part of the phasor coordinates.
    s: array_like
        Imaginary part of the phasor coordinates.
    """
    g = mod * numpy.cos(phi)
    s = mod * numpy.sin(phi)
    return g, s


def phasemodule_values(
    g: ArrayLike,
    s: ArrayLike,
    /,
) -> tuple[ArrayLike, ArrayLike]:
    """
    # Convert phasor coordinates into phase and module
    # Parameters:
    # - g: array_like
    #   Real part of the phasor coordinates.
    # - s: array_like
    #   Imaginary part of the phasor coordinates.
    # Returns:
    # - phi: array_like
    #   Phase angle in radians.
    # - mod: array_like 
    #   Module.
    """
    g = numpy.asarray(g)  # Convert to NumPy array if not already
    s = numpy.asarray(s)  # Convert to NumPy array if not already
    phi = numpy.arctan2(s, g)
    mod = (g**2 + s**2) ** 0.5

    return phi, mod


def lifetime_computation_phasor(
    g: float,
    s: float,
    laser_rep_frequency: float,
    /,
) -> tuple[float, float]:
    """
    # Calculate lifetime values from phasor coordinates and FLIM frequency.
    # Parameters:
    # - g: float
    #   Real part of the phasor coordinates.
    # - s: float
    #   Imaginary part of the phasor coordinates.
    # - laser_rep_frequency: int
    #   laser repetition frequency.
    # Returns:
    # - tau_m: float
    #   Lifetime modulation.
    # - tau_phi: float
    #   Lifetime phase.
    """
    omega = laser_rep_frequency * numpy.pi * 2
    tau_m = (
        numpy.sqrt((1 - (g**2 + s**2)) / (omega**2 * (g**2 + s**2)))
    ) * 10e2
    tau_phi = 1 / omega * s / g * 10e2
    return tau_m, tau_phi


def lifetime_computation_array(
    g: ArrayLike,
    s: ArrayLike,
    laser_rep_frequency: int,
    /,
) -> tuple[ArrayLike, ArrayLike]:
    """
    # Calculate lifetime values from phasor coordinates and FLIM frequency.
    # Parameters:
    # - g: array_like
    #   Real part of the phasor coordinates.
    # - s: array_like
    #   Imaginary part of the phasor coordinates.
    # - laser_rep_frequency: int
    #   laser repetition frequency.
    # Returns:
    # - tau_m: array_like
    #   Lifetime modulation.
    # - tau_phi: array_like
    #   Lifetime phase.
    """
    g = numpy.asarray(g)  # Convert to NumPy array if not already
    s = numpy.asarray(s)  # Convert to NumPy array if not already
    omega = laser_rep_frequency * numpy.pi * 2
    tau_m = numpy.zeros(numpy.size(g))
    tau_phi = numpy.zeros(numpy.size(g))
    tau_m = (
        numpy.sqrt((1 - (g**2 + s**2)) / (omega**2 * (g**2 + s**2)))
    ) * 10e2
    tau_phi = 1 / omega * s / g * 10e2
    tau_m[numpy.isnan(tau_m) > 0] = 0
    tau_phi[numpy.isnan(tau_phi) > 0] = 0
    return tau_m, tau_phi


# def phasor_lifetime(
#    g: Union[float,ArrayLike], s: Union[float,ArrayLike], laser_rep_frequency: int, /,
# ) -> tuple[Union[list[float],ArrayLike],Union[list[float],ArrayLike], Union[list[float],ArrayLike], Union[list[float],ArrayLike]]:


def phasor_lifetime(
    g: ArrayLike,
    s: ArrayLike,
    laser_rep_frequency: int,
    /,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    # Compute lifetime values and standard deviations from phasor coordinates and FLIM frequency.
    # Parameters:
    # - g: array_like
    #   Real part of the phasor coordinates.
    # - s: array_like
    #   Imaginary part of the phasor coordinates.
    # - laser_rep_frequency: int
    #   laser repetition frequency.
    # Returns:
    # - lt: list[float]
    #   List containing lifetime values, [lt_m, lt_phi].
    # - lt_std: list[float]
    #   List containing standard deviations for lifetime values, [lt_m_std, lt_phi_std].
    # - phasor_position: list[float]
    #   List with the average phasor coordinates, [g_av, s_av].
    # - phasor_std: list[float]
    #   List with standard deviations for phasor coordinates, [g_std, s_std].
    """
    g = numpy.asarray(g)  # Convert to NumPy array if not already
    s = numpy.asarray(s)  # Convert to NumPy array if not already
    omega = laser_rep_frequency * 2 * numpy.pi

    g_av = numpy.average(g)
    s_av = numpy.average(s)
    phasor_position = [g_av, s_av]
    g_var = numpy.average((g - g_av) ** 2)
    s_var = numpy.average((s - s_av) ** 2)
    g_std = numpy.sqrt(g_var)
    s_std = numpy.sqrt(s_var)
    phasor_position = [g_av, s_av]
    phasor_std = [g_std, s_std]
    lt_m, lt_phi = lifetime_computation_phasor(
        g_av, s_av, laser_rep_frequency
    )
    lt_phi_std = numpy.sqrt((g_std / g_av) ** 2 + (s_std / s_av) ** 2)
    lt_m_std = 2 * g_std / numpy.sqrt(g_av**2) + 2 * s_std / numpy.sqrt(
        s_av**2
    )
    lt = [lt_m, lt_phi]
    lt_std = [lt_m_std, lt_phi_std]
    if not lt_m:
        print(
            'Due to the phasor position it was not possible to determine a lifetime finite value'
        )
    return lt, lt_std, phasor_position, phasor_std


def refplot(
    names: list[str],
    lifetimes: list[float],
    laser_rep_frequency: int,
    /,
) -> tuple[list[float], list[float]]:
    """
    # Generates reference phasor coordinates based on sample lifetimes and FLIM frequency.
    # Parameters:
    # - names: 
    #   List of sample names.
    # - lifetimes: List of lifetime values corresponding to the samples.
    # - laser_rep_frequency: laser repetition frequency.
    # Returns:
    # - g_ref: Real part of the reference phasor coordinates.
    # - s_ref: Imaginary part of the reference phasor coordinates.
    """
    g_ref = []
    s_ref = []
    omega = laser_rep_frequency * numpy.pi * 10e-3
    for tau in lifetimes:
        M = 1 / numpy.sqrt(1 + (omega * tau) ** 2)
        phi = numpy.arctan(omega * tau)
        g_ref.append(M * numpy.cos(phi))
        s_ref.append(M * numpy.sin(phi))
    return g_ref, s_ref


def fractional_intensities(
    RefA: tuple[float, float],
    RefB: tuple[float, float],
    target: tuple[float, float],
) -> list[float]:
    """
    # Calculate intensity fractions between two reference points and a target point.
    # Parameters:
    # - RefA: Coordinates of the first reference point.
    # - RefB: Coordinates of the second reference point.
    # - target: Coordinates of the target phasor.
    # - labels: A dictionary for labeling categories.
    # Returns:
    # - ratio a list of intensity fractions.
    """

    Tot_distance = ((RefA[0] - RefB[0]) ** 2 + (RefA[1] - RefB[1]) ** 2) ** 0.5
    distanceA = (
        (RefA[0] - target[0]) ** 2 + (RefA[1] - target[1]) ** 2
    ) ** 0.5
    distanceB = (
        (RefB[0] - target[0]) ** 2 + (RefB[1] - target[1]) ** 2
    ) ** 0.5

    fractionA = distanceB / (distanceA + distanceB)
    fractionB = distanceA / (distanceA + distanceB)

    ratios = [fractionA, fractionB]
    return ratios


def apparent_lifetime(
    g: float,
    s: float,
    laser_rep_frequency: int,
    /,
    ref_tau: float,
) -> tuple[list[float], float, float]:
    """
    Calculate apparent lifetime and relative fractions.

    Parameters:
    - g,s: Real and imaginary part of phasor coordinates. (could be implemented to input list also)
    - laser_rep_frequency: Laser repetition frequency.
    - ref_tau: Reference lifetime of one of the two putable species.

    Returns:
    - lifetime2: apparent lifetimes calculated for different phasor coordinates.
    - frac_species1: species 1 fractions.
    - frac_species2: species 2 fractions.
    """
    r = 0.5
    omega = laser_rep_frequency * numpy.pi * 10 ** (-3)
    M = 1 / numpy.sqrt(1 + (omega * ref_tau) ** 2)
    phi = numpy.arctan(ref_tau * omega)
    xref = M * numpy.cos(phi)
    yref = M * numpy.sin(phi)
    # alpha is the angle of the segment connecting the center of the circle to the nad_free phasor
    # this angle is found via trigonometry
    a = xref - r
    c = numpy.sqrt((r - xref) ** 2 + (yref) ** 2)
    alpha = numpy.arctan(yref / c)
    x1 = xref - g
    y1 = s - yref
    frac1 = ((x1) ** 2 + (y1) ** 2) ** 0.5
    beta = numpy.arctan(y1 / x1)
    # gamma is the angle of the chord of a circle theorem chord=2*r*sin(gamma)
    # in this case r is 0.5
    gamma = numpy.pi / 2 - beta - alpha
    tot_dist = numpy.sin(gamma)
    # tot_dist is also the length of the hypotenuse of the triangle connecting (e.g.) free and bound nadh
    # hence one can compute the nadh fractions and the position of the intercept
    frac_species2 = frac1 / tot_dist
    frac_species1 = 1 - frac1 / tot_dist
    x_int = xref - tot_dist * numpy.cos(beta)
    y_int = yref + tot_dist * numpy.sin(beta)
    intersection_points = [x_int, y_int]
    (
        lifetime_int_point,
        _,
        _,
        _,
    ) = phasor_lifetime(x_int, y_int, laser_rep_frequency)
    lifetime2 = lifetime_int_point

    return lifetime2, frac_species1, frac_species2
