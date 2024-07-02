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
  resolving graphically with histogram:

  - :py:func:`graphical_component_analysis`

- blindly resolve fractions of n components by using harmonic
  information (not implemented)

"""

from __future__ import annotations

__all__ = [
    'two_fractions_from_phasor',
    'graphical_component_analysis',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import numpy

from ._utils import (
    line_from_components,
    mask_cursor,
    mask_segment,
    project_phasor_to_line,
)


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
    fractions : tuple of ndarray
        A tuple containing arrays with the fractions of a specific component.
        The order of the arrays in the tuple corresponds to the order of the
        components used in the calculation.

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

    Examples
    --------
    >>> two_fractions_from_phasor(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... )  # doctest: +NUMBER
    (array([0.44, 0.56, 0.68]), array([0.56, 0.44, 0.32]))

    """
    real_components = numpy.asarray(real_components)
    imag_components = numpy.asarray(imag_components)
    if real_components.shape != (2,):
        raise ValueError(f'{real_components.shape=} != (2,)')
    if imag_components.shape != (2,):
        raise ValueError(f'{imag_components.shape=} != (2,)')
    _, distance_between_components = line_from_components(
        real_components, imag_components
    )
    projected_real, projected_imag = project_phasor_to_line(
        real, imag, real_components, imag_components
    )
    distances_to_first_component = numpy.hypot(
        numpy.asarray(projected_real) - real_components[0],
        numpy.asarray(projected_imag) - imag_components[0],
    )
    second_component_fractions = (
        distances_to_first_component / distance_between_components
    )
    first_component_fractions = 1 - second_component_fractions
    return first_component_fractions, second_component_fractions


def graphical_component_analysis(
    real: ArrayLike,
    imag: ArrayLike,
    real_components: ArrayLike,
    imag_components: ArrayLike,
    /,
    *,
    cursor_diameter: float = 0.1,
    number_of_steps: int = 100,
) -> tuple[tuple[NDArray[Any], ...], NDArray[Any]]:
    """Return fractions of two or three components from phasor coordinates by
    solving graphically.

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
    cursor_diameter: float, optional
        Diameter of the cursor in phasor coordinates.
    number_of_steps: int, optional
        Number of steps to move the cursor along the line between components.

    Returns
    -------
    fractions : tuple of ndarray
        A tuple of arrays containing phasor counts along each line segment
        connecting the components, ordered as follows: 1-2, 1-3, 2-3 (for 3
        components) or simply 1-2 (for 2 components).

    Notes
    -----
    For the moment, calculation of fraction of components from different
    channels or frequencies is not supported. Only one set of components can
    be analyzed and will be broadcasted to all channels/frequencies.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag`, or `real_components` and
        `imag_components` do not match.
        Number of components is less than 2 or greater than 3.

    Examples
    --------
    Count the number of phasors and fractions between two components:

    >>> graphical_component_analysis(
    ...     [0.6, 0.3], [0.35, 0.38], [0.2, 0.9], [0.4, 0.3], number_of_steps=5
    ... )  # doctest: +NUMBER
    ((array([0, 0, 1, 0, 1, 0]),), array([0, 0.2, 0.4, 0.6, 0.8, 1]))

    Count the number of phasors and fractions between the combinations
    of three components:

    >>> graphical_component_analysis(
    ...     [0.4, 0.5],
    ...     [0.2, 0.3],
    ...     [0.0, 0.2, 0.9],
    ...     [0.0, 0.4, 0.3],
    ...     number_of_steps=5,
    ... )  # doctest: +NUMBER +NORMALIZE_WHITESPACE
    ((array([0, 1, 1, 1, 1, 0]),
    array([0, 1, 0, 0, 0, 0]),
    array([0, 1, 2, 0, 0, 0])),
    array([0, 0.2, 0.4, 0.6, 0.8, 1]))

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    real_components = numpy.asarray(real_components)
    imag_components = numpy.asarray(imag_components)
    if (
        real.shape != imag.shape
        or real_components.shape != imag_components.shape
    ):
        raise ValueError("Input array shapes must match")
    num_components = len(real_components)
    if num_components not in (2, 3):
        raise ValueError("Number of components must be 2 or 3")
    counts = []
    fractions = numpy.asarray(numpy.linspace(0, 1, number_of_steps + 1))
    for i, (real_a, imag_a) in enumerate(
        zip(real_components, imag_components)
    ):
        for j in range(i + 1, num_components):
            real_b, imag_b = real_components[j], imag_components[j]
            unit_vector, distance = line_from_components(
                [real_b, real_a], [imag_b, imag_a]
            )
            cursor_real, cursor_imag = real_b, imag_b
            step_size = distance / number_of_steps
            component_counts = []
            for _ in range(number_of_steps + 1):
                if num_components == 2:
                    mask = mask_cursor(
                        real, imag, cursor_real, cursor_imag, cursor_diameter
                    )
                elif num_components == 3:
                    real_c, imag_c = (
                        real_components[3 - i - j],
                        imag_components[3 - i - j],
                    )
                    mask = mask_segment(
                        real,
                        imag,
                        cursor_real,
                        cursor_imag,
                        real_c,
                        imag_c,
                        cursor_diameter / 2,
                    )
                fraction_counts = numpy.sum(mask)
                component_counts.append(fraction_counts)
                cursor_real += step_size * unit_vector[0]
                cursor_imag += step_size * unit_vector[1]
            counts.append(numpy.asarray(component_counts))
    return (tuple(counts), fractions)
