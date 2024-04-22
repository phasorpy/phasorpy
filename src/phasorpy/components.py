"""Component analysis of phasor coordinates.

The ``phasorpy.components`` module provides functions to:


"""

from __future__ import annotations

__all__ = ['two_fractions_from_phasor']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import math

import numpy

from ._utils import project_phasor_to_line


def two_fractions_from_phasor(
    real: ArrayLike,
    imag: ArrayLike,
    real_components: ArrayLike,
    imag_components: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return fractions of two components from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    real_components: array_like
        Real coordinates of the pair of components.
    imag_components: array_like
        Imaginary coordinates of the pair of components.

    Returns
    -------
    fractions : ndarray
        Fractions for all components.

    Raises
    ------
    ValueError
        If the real and/or imaginary coordinates of the known components are not of size 2.
        If the 
        
    Examples
    --------
    >>> two_fractions_from_phasor(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.4], [0.9, 0.3]
    ... ) # doctest: +NUMBER
    (...)

    """
    real_components = numpy.asarray(real_components)
    imag_components = numpy.asarray(imag_components)
    if real_components.size < 2:
        raise ValueError(f'{real_components.size=} must have at least two coordinates')
    if imag_components.size < 2:
        raise ValueError(f'{imag_components.size=} must have at least two coordinates')
    if real_components.all == imag_components.all:
        raise ValueError('components must have different coordinates')
    projected_real, projected_imag = project_phasor_to_line(
        real, imag, real_components, imag_components
    )
    first_component_phasor = numpy.array(
        [real_components[0], imag_components[0]]
    )
    second_component_phasor = numpy.array(
        [real_components[1], imag_components[1]]
    )
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
    fraction_of_second_component = numpy.clip(
        fraction_of_second_component, 0, 1
    )
    return 1 - fraction_of_second_component, fraction_of_second_component
