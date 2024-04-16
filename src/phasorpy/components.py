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
    # fraction1 = []
    # fraction2 = []
    # projected_real = numpy.atleast_1d(projected_real)
    # projected_imag = numpy.atleast_1d(projected_imag)
    # for element in zip(projected_real,projected_imag):
    #     fractions = numpy.linalg.solve(components_coordinates, element)
    #     fraction1.append(fractions[0])
    #     fraction2.append(fractions[1])
    # fraction1 = numpy.asarray(fraction1)
    # fraction2 = numpy.asarray(fraction2)
    # fraction1 = numpy.clip(fraction1, 0, 1)
    # fraction2 = numpy.clip(fraction2, 0, 1)
    # return fraction1.reshape(projected_real.shape), fraction2.reshape(projected_real.shape)
    projected_real = numpy.atleast_1d(projected_real)
    projected_imag = numpy.atleast_1d(projected_imag)
    i = 0
    fractions = []
    for element in zip(projected_real,projected_imag):
        row_fractions = numpy.linalg.solve(components_coordinates, element)
        fractions.append(row_fractions)
        i += 1
    print('i=',i)
    fractions = numpy.stack(fractions, axis=2)
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