"""Component analysis of phasor coordinates.

The ``phasorpy.components`` module provides functions to:


"""

from __future__ import annotations

__all__ = [
    'component_fractions_from_phasor'
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import numpy
import math

from ._utils import project_phasor_to_line


def linear_fractions_from_phasor(
    real: ArrayLike, imag: ArrayLike, first_component_phasor: ArrayLike, second_component_phasor: ArrayLike, /,
) -> tuple[NDArray[Any], NDArray[Any]]:
   """Return fractions of two components from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    first_component_phasor: array_like
        Coordinates from the first component
    second_component_phasor: array_like
        Coordinates from the second component

    Returns
    -------
    fractions_of_first_component : ndarray
        Fractions of the first component.
    fractions_of_second_component : ndarray
        Fractions of the second component

    Raises
    ------
    ValueError
        The `signal` has less than three samples along `axis`.

    Examples
    --------
    >>> component_fractions_from_phasor(...)  # doctest: +NUMBER
    (...)

    """
    projected_real, projected_imag = project_phasor_to_line(
        real, imag, first_component_phasor, second_component_phasor
    )
    total_distance_between_components = math.sqrt(
        (second_component_phasor[0] - first_component_phasor[0]) ** 2
        + (second_component_phasor[1] - first_component_phasor[1]) ** 2
    )
    distances_to_first_component = numpy.sqrt(
        (numpy.array(projected_real) - first_component_phasor[0]) ** 2
        + (numpy.array(projected_imag) - first_component_phasor[1]) ** 2
    )
    fraction_of_first_component = (
        distances_to_first_component / total_distance_between_components
    )
    return fraction_of_first_component, 1 - fraction_of_first_component


