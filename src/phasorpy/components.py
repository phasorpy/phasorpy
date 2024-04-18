"""Component analysis of phasor coordinates.

The ``phasorpy.components`` module provides functions to:


"""

from __future__ import annotations

__all__ = [
    'fractional_intensities_from_phasor'
    'fractional_intensities_from_phasor_old'
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import numpy
import math

from ._utils import project_phasor_to_line
from .phasor import polar_from_reference, phasor_to_polar


def fractional_intensities_from_phasor(
    real: ArrayLike, imag: ArrayLike, real_components: ArrayLike, imaginary_components: ArrayLike, /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return fractions of two components from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    real_components: array_like
        Real coordinates of the components.
    imaginary_components: array_like
        Imaginary coordinates of the components.

    Returns
    -------
    fractions : ndarray
        Fractions for all components.

    Raises
    ------
    ValueError
        The `signal` has less than three samples along `axis`.

    Examples
    --------
    >>> fractional_intensities_from_phasor(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.4], [0.9, 0.3]
    ... ) # doctest: +NUMBER
    (...)

    """
    projected_real, projected_imag = project_phasor_to_line(
        real, imag, real_components, imaginary_components
    )
    components_coordinates = numpy.array([real_components, imaginary_components])
    projected_real = numpy.atleast_1d(projected_real)
    projected_imag = numpy.atleast_1d(projected_imag)
    fractions = []
    for element in zip(projected_real,projected_imag):
        row_fractions = numpy.linalg.solve(components_coordinates, element)
        fractions.append(row_fractions)
    fractions = numpy.stack(fractions, axis=-1)
    return fractions
    



def fractional_intensities_from_phasor_old(
    real: ArrayLike, imag: ArrayLike, real_components: ArrayLike, imaginary_components: ArrayLike, /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return fractions of two components from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    real_components: array_like
        Real coordinates of the components.
    imaginary_components: array_like
        Imaginary coordinates of the components.

    Returns
    -------
    fractions : ndarray
        Fractions for all components.

    Raises
    ------
    ValueError
        The `signal` has less than three samples along `axis`.

    Examples
    --------
    >>> fractional_intensities_from_phasor(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.4], [0.9, 0.3]
    ... ) # doctest: +NUMBER
    (...)

    """
    projected_real, projected_imag = project_phasor_to_line(
        real, imag, real_components, imaginary_components
    )
    first_component_phasor = numpy.array([real_components[0], imaginary_components[0]])
    second_component_phasor = numpy.array([real_components[1], imaginary_components[1]])
    total_distance_between_components = math.sqrt(
        (second_component_phasor[0] - first_component_phasor[0]) ** 2
        + (second_component_phasor[1] - first_component_phasor[1]) ** 2
    )
    distances_to_first_component = numpy.sqrt(
        (numpy.array(projected_real) - first_component_phasor[0]) ** 2
        + (numpy.array(projected_imag) - first_component_phasor[1]) ** 2
    )
    fraction_of_second_component = (
        distances_to_first_component / total_distance_between_components
    )
    fraction_of_second_component = numpy.clip(fraction_of_second_component, 0 , 1)
    return 1 - fraction_of_second_component, fraction_of_second_component

def fractional_intensities_polar(
    real: ArrayLike, imag: ArrayLike, real_components: ArrayLike, imaginary_components: ArrayLike, /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return fractions of two components from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    real_components: array_like
        Real coordinates of the components.
    imaginary_components: array_like
        Imaginary coordinates of the components.

    Returns
    -------
    fractions : ndarray
        Fractions for all components.

    Raises
    ------
    ValueError
        The `signal` has less than three samples along `axis`.

    Examples
    --------
    >>> fractional_intensities_from_phasor(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.4], [0.9, 0.3]
    ... ) # doctest: +NUMBER
    (...)

    """
    # projected_real, projected_imag = project_phasor_to_line(
    #     real, imag, real_components, imaginary_components
    # )
    phase, mod = phasor_to_polar(real, imag)
    first_component = phasor_to_polar(real_components[0], imaginary_components[0])
    second_component = phasor_to_polar(real_components[1], imaginary_components[1])
    # phi0_first_component, mod0_first_component = polar_from_reference(phase, mod, *first_component)
    # phi0_second_component, mod0_second_component = polar_from_reference(phase, mod, *second_component)
    # Reshape single values to arrays with one column
    # first_component = numpy.array([first_component[0]]).reshape(-1, 1)
    # second_component = numpy.array([second_component[0]]).reshape(-1, 1)
    phase = numpy.array([phase]).reshape(-1, 1)
    mod = numpy.array([mod]).reshape(-1, 1)
    # Create coefficient matrix
    # A = numpy.concatenate((first_component, second_component), axis=1)
    # Augment coefficient matrix with the equation weight1 + weight2 = 1
    A = numpy.vstack([first_component, second_component, [1, 1]])

    # Create phase vector
    b = numpy.concatenate((phase, mod, numpy.array([[1]])))

    # Solve the augmented system of equations
    fractions = numpy.linalg.solve(A, b)
    return fractions