"""Component analysis of phasor coordinates.

The ``phasorpy.components`` module provides functions to:

- calculate fractions of two components of known location by projecting to
  line between components:

  - :py:func:`two_fractions_from_phasor`

- calculate phasor coordinates of second component if only one is
  known (not implemented)

- calculate fractions of three or four known components by using higher
  harmonic information (not implemented)

- calculate fractions of two or three components of known location by
  resolving graphically with histogram (not implemented)

- blindly resolve fractions of n components by using harmonic
  information (not implemented)

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
        Real coordinates of the first and second components.
    imag_components: array_like
        Imaginary coordinates of the first and second components.

    Returns
    -------
    fraction_of_first_component : ndarray
        Fractions of the first component.
    fraction_of_second_component : ndarray
        Fractions of the second component.

    Notes
    -----
    For the moment, calculation of fraction of components from different
    channels or frequencies is not supported. Only one pair of components can
    be analyzed and will be broadcasted to all channels/frequencies.

    Raises
    ------
    ValueError
        If the real and/or imaginary coordinates of the known components are
        not of size 2.
        If the two components have the same coordinates.

    Examples
    --------
    >>> two_fractions_from_phasor(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... ) # doctest: +NUMBER
    (array([0.44, 0.56, 0.68]), array([0.56, 0.44, 0.32]))

    """
    real_components = numpy.asarray(real_components)
    imag_components = numpy.asarray(imag_components)
    if real_components.shape != (2,):
        raise ValueError(f'{real_components.shape=} != (2,)')
    if imag_components.shape != (2,):
        raise ValueError(f'{imag_components.shape=} != (2,)')
    first_component_phasor = numpy.array(
        [real_components[0], imag_components[0]]
    )
    second_component_phasor = numpy.array(
        [real_components[1], imag_components[1]]
    )
    total_distance_between_components = math.hypot(
        (second_component_phasor[0] - first_component_phasor[0]),
        (second_component_phasor[1] - first_component_phasor[1]),
    )
    if math.isclose(total_distance_between_components, 0, abs_tol=1e-6):
        raise ValueError('components must have different coordinates')
    projected_real, projected_imag = project_phasor_to_line(
        real, imag, real_components, imag_components
    )
    distances_to_first_component = numpy.hypot(
        (numpy.array(projected_real) - first_component_phasor[0]),
        (numpy.array(projected_imag) - first_component_phasor[1]),
    )
    fraction_of_second_component = (
        distances_to_first_component / total_distance_between_components
    )
    return 1 - fraction_of_second_component, fraction_of_second_component
