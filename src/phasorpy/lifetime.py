"""
The `phasorpy.lifetime` module provides functions to:
- Convert phase and modulation data into phasor coordinates and vice-versa in the frequency domain
- Manage phase and modulation lifetime computations
- Manage apparent single lifetimes
- Compute fractional intensities in mixtures of two components

Future implementations to be considered:
- Spatial filtering: linear (gaussian) and non-linear (median)
- Making some functions flexible enough to manage both arrays and single values
- Compute intensity fractions for mixture of three species

Please note: FRET- and time-domain related functionalities are not yet implemented.

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
    'apparent_lifetime',
]

import warnings
from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, PathLike, Sequence


def phasor_coordinates(
    phi: ArrayLike, mod: ArrayLike, /,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Convert phase and module information into phasor coordinates in the frequency domain.
    Parameters:
    ----------
    phi : array_like
        Phase angle values of type ``float32` for each pixel.
    mod : array_like
        Module values of type ``float32` for each pixel.
    Returns:
    -------
    g : array_like
        Real part of the phasor coordinates of type ``float32`.
    s : array_like
        Imaginary part of the phasor coordinates of type ``float32`.

    Examples
    --------
    >>> phi = numpy.random.rand(256, 256)
    >>> mod = numpy.random.rand(256, 256)
    >>> g,s=phasor_coordinates(phi,mod)
    >>> g.dtype
    dtype('float64')
    >>> s.dtype
    dtype('float64')
    >>> s.shape
    (256, 256)
    >>> g.shape
    (256, 256)
    """
    phi = numpy.radians(phi)
    g = mod * numpy.cos(phi)
    s = mod * numpy.sin(phi)
    return g, s


def phasemodule_values(
    g: ArrayLike, s: ArrayLike, /,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Convert phasor coordinates into phase and module
    Parameters:
    ----------
    g: array_like
      Real part of the phasor coordinates of type ``float32`.
    s: array_like
      Imaginary part of the phasor coordinates of type ``float32`.
    Returns:
    -------
    phi: array_like
      Phase angle values in radians of type ``float32`.
    mod: array_like
      Module values Module values of type ``float32`.
    Examples
    --------
    >>> g = numpy.random.rand(256, 256)
    >>> s = numpy.random.rand(256, 256)
    >>> g.shape
    (256, 256)
    >>> g.dtype
    dtype('float64')
    >>> phi, mod = phasemodule_values(g, s)
    >>> phi.shape
    (256, 256)
    >>> phi.dtype
    dtype('float64')
    """
    g = numpy.asarray(g)  # Convert to NumPy array if not already
    s = numpy.asarray(s)  # Convert to NumPy array if not already
    phi = numpy.arctan2(s, g)
    mod = numpy.hypot(g, s)
    return phi, mod


def lifetime_computation_phasor(
    g: float, s: float, laser_rep_frequency: float, /,
) -> tuple[float, float]:
    """
    Calculate lifetime values from phasor coordinates and laser repetition frequency.
    Parameters:
    ----------
    g: float
      Real part value of the phasor coordinates.
    s: float
      Imaginary part value of the phasor coordinates.
    laser_rep_frequency: float
      laser repetition frequency in MHz.
    Returns:
    -------
    tau_m: float
      Lifetime modulation value.
    tau_phi: float
      Lifetime phase value.
    Examples
    --------
    >>> g=0.195
    >>> s=0.400
    >>> laser_rep_frequency=80
    >>> tau_m, tau_phi=lifetime_computation_phasor(g, s, laser_rep_frequency)
    >>> tau_phi, tau_m
    (4.080895976715265, 4.00359878497335)
    """
    omega = laser_rep_frequency * numpy.pi * 2
    tau_m = (
        numpy.sqrt((1 - (g * g + s * s)) / (omega * omega * (g * g + s * s)))
    ) * 10e2
    tau_phi = 1 / omega * s / g * 10e2
    return tau_m, tau_phi


def lifetime_computation_array(
    g: ArrayLike, s: ArrayLike, laser_rep_frequency: float, /,
) -> tuple[ArrayLike, ArrayLike]:
    """
     Calculate lifetime values from phasor coordinates and laser repetition frequency.
    Parameters:
    ----------
    g: array_like
      Real part of the phasor coordinates of type ``float32`.
    s: array_like
      Imaginary part of the phasor coordinates of type ``float32`.
    laser_rep_frequency: float
      laser repetition frequency in MHz.
    Returns:
    -------
    tau_m: array_like
      Lifetime modulation values for each pixel of type ``float32`.
    tau_phi: array_like
      Lifetime phase values for each pixel of type ``float32.
    >>> phi = numpy.random.rand(256, 256)
    >>> mod = numpy.random.rand(256, 256)
    >>> laser_rep_frequency = 80.0
    >>> g,s=phasor_coordinates(phi,mod)
    >>> tau_m, tau_phi = lifetime_computation_array(g, s, laser_rep_frequency)
    >>> tau_phi.shape
    (256, 256)
    """
    # Filter out RuntimeWarnings for divide by zero and invalid value
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="divide by zero encountered in divide",
    )
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in sqrt",
    )
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in divide",
    )
    g = numpy.asarray(g)  # Convert to NumPy array if not already
    s = numpy.asarray(s)  # Convert to NumPy array if not already
    omega = laser_rep_frequency * numpy.pi * 2
    tau_m = numpy.zeros(numpy.size(g))
    tau_phi = numpy.zeros(numpy.size(g))
    tau_m = (
        numpy.sqrt((1 - (g * g + s * s)) / (omega * omega * (g * g + s * s)))
    ) * 10e2
    tau_phi = 1 / omega * s / g * 10e2
    tau_m[numpy.isnan(tau_m) > 0] = 0
    tau_phi[numpy.isnan(tau_phi) > 0] = 0
    return tau_m, tau_phi


def phasor_lifetime(
    g: ArrayLike, s: ArrayLike, laser_rep_frequency: float, /,
) -> tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]:
    """
    Compute lifetime values and standard deviations from phasor coordinates and laser repetition frequency.
    Parameters:
    ----------
    g: array_like
      Real part of the phasor coordinates of type ``float32`.
    s: array_like
      Imaginary part of the phasor coordinates of type ``float32`.
    laser_rep_frequency: float
      laser repetition frequency in MHz.
    Returns:
    -------
    lt: Sequence[float]
      Sequence containing lifetime values, [lt_m, lt_phi].
    lt_std: Sequence[float]
      Sequence containing standard deviations for lifetime values, [lt_m_std, lt_phi_std].
    phasor_position: Sequence[float]
      Sequence with the average phasor coordinates, [g_av, s_av].
    phasor_std: Sequence[float]
      Sequence with standard deviations for phasor coordinates, [g_std, s_std].
    Examples
    --------
    >>> g = numpy.random.rand(256, 256)
    >>> s = numpy.random.rand(256, 256)
    >>> laser_rep_frequency = 80
    >>> lt, lt_std, phasor_position, phasor_std=phasor_lifetime(g,s,laser_rep_frequency)
    >>> numpy.shape(lt)
    (2,)
    >>> numpy.shape(lt_std)
    (2,)
    >>> numpy.shape(phasor_position)
    (2,)
    """
    g = numpy.asarray(g)  # Convert to NumPy array if not already
    s = numpy.asarray(s)  # Convert to NumPy array if not already
    omega = laser_rep_frequency * 2 * numpy.pi

    g_av = numpy.average(g)
    s_av = numpy.average(s)
    phasor_position = [g_av, s_av]
    g_var = numpy.average((g - g_av) * (g - g_av))
    s_var = numpy.average((s - s_av) * (s - s_av))
    g_std = numpy.sqrt(g_var)
    s_std = numpy.sqrt(s_var)
    phasor_position = [g_av, s_av]
    phasor_std = [g_std, s_std]
    lt_m, lt_phi = lifetime_computation_phasor(g_av, s_av, laser_rep_frequency)
    lt_phi_std = numpy.hypot(g_std / g_av, s_std / s_av)
    lt_m_std = 2 * g_std / numpy.sqrt(g_av * g_av) + 2 * s_std / numpy.sqrt(
        s_av * s_av
    )
    lt = [lt_m, lt_phi]
    lt_std = [lt_m_std, lt_phi_std]
    if not lt_m:
        print(
            'Due to the phasor position it was not possible to determine a lifetime finite value'
        )
    return lt, lt_std, phasor_position, phasor_std


def refplot(
    names: Sequence[str],
    lifetimes: Sequence[float],
    laser_rep_frequency: float,
    /,
) -> tuple[Sequence[float], Sequence[float]]:
    """
    Generates reference phasor coordinates based on sample lifetimes and laser repetition frequency.
    Parameters:
    ----------
    names: sequence of str
       List of sample names.
    lifetimes: sequence of floating-point numbers.
      List of lifetime values corresponding to the samples (nanoseconds).
    laser_rep_frequency: float
      laser repetition frequency in MHz.
    Returns:
    -------
    g_ref: sequence of float
      Real part of the reference phasor coordinates.
    s_ref: sequence of float
      Imaginary part of the reference phasor coordinates.
    Examples
    --------
    >>> names=['free','bound']
    >>> lifetimes = [0.37, 3.4]
    >>> laser_rep_frequency = 80
    >>> g_ref,s_ref=refplot(names, lifetimes, laser_rep_frequency)
    >>> g_ref,s_ref
    ([0.5362691540599069, 0.013509993393265021], [0.4986828134834594, 0.11544467710457233])
    """
    g_ref = []
    s_ref = []
    omega = laser_rep_frequency * numpy.pi * 10e-3
    for tau in lifetimes:
        M = 1 / numpy.sqrt(1 + (omega * tau) * (omega * tau))
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
    Calculate intensity fractions between two reference points and a target point.
    Parameters:
    ----------
    RefA:
      Coordinates of the first reference point.
    RefB:
      Coordinates of the second reference point.
    target:
      Coordinates of the target phasor.
    labels:
      A dictionary for labeling categories.
    Returns:
    -------
    ratios: Sequence of floating numbers
      the sequence of intensity fractions.
    Examples
    --------
    >>> RefA=[0.99,0.09]
    >>> RefB=[0.31,0.46]
    >>> target=[0.51,0.35]
    >>> ratios=fractional_intensities(RefA,RefB,target)
    >>> ratios
    [0.2948457429517342, 0.7051542570482657]
    """
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
    g: float, s: float, laser_rep_frequency: float, /, ref_tau: float,
) -> tuple[Sequence[float], float, float]:
    """
    Calculate apparent lifetime and relative fractions exploiting circle chord theorem.

    Parameters:
    ----------
    g: float
      Real part value of the phasor coordinates.
    s: float
      Imaginary part value of the phasor coordinates.
    laser_rep_frequency:
      Laser repetition frequency.
    ref_tau:
      Reference lifetime of one of the two putable species.
    Returns:
    -------
    lifetime2: Sequence of float
      Phase and module lifetime calculated for second putative species.
    frac_species1: float
      species 1 fraction.
    frac_species2: float
      species 2 fraction.
    >>> g=0.6
    >>> s=0.4
    >>> laser_rep_frequency = 80
    >>> ref_tau=0.37
    >>> lifetime2, frac_species1,frac_species2=apparent_lifetime(g,s,laser_rep_frequency,0.37)
    >>> lifetime2, frac_species1,frac_species2
    ([2.09816056498088, 2.11118113734811], 0.24660520747448866, 0.7533947925255113)
    """
    r = 0.5
    omega = laser_rep_frequency * numpy.pi * 10 ** (-3)
    M = 1 / numpy.sqrt(1 + (omega * ref_tau) * (omega * ref_tau))
    phi = numpy.arctan(ref_tau * omega)
    xref = M * numpy.cos(phi)
    yref = M * numpy.sin(phi)
    # alpha is the angle of the segment connecting the center of the circle to the nad_free phasor
    # this angle is found via trigonometry
    a = xref - r
    c = numpy.sqrt((r - xref) * (r - xref) + (yref) * (yref))
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
    (lifetime_int_point, _, _, _,) = phasor_lifetime(
        x_int, y_int, laser_rep_frequency
    )
    lifetime2 = lifetime_int_point

    return lifetime2, frac_species1, frac_species2
