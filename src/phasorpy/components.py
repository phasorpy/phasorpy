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

__all__ = [
    'two_fractions_from_phasor',
    'graphical_component_analysis'
    ]

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
    ... )  # doctest: +NUMBER
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
        numpy.asarray(projected_real) - first_component_phasor[0],
        numpy.asarray(projected_imag) - first_component_phasor[1],
    )
    fraction_of_second_component = (
        distances_to_first_component / total_distance_between_components
    )
    return 1 - fraction_of_second_component, fraction_of_second_component

def graphical_component_analysis(
    real: ArrayLike,
    imag: ArrayLike,
    real_components: ArrayLike,
    imag_components: ArrayLike,
    /,
    *,
    cursor_diameter: float = 0.1,
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
    cursor_diameter: float, optional
        Diameter of the cursor in phasor coordinates.

    Returns
    -------
    fractions : ndarray
        Fractions of the components.

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
    >>> graphical_component_analysis(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... )  # doctest: +NUMBER
    (array([0.44, 0.56, 0.68]), array([0.56, 0.44, 0.32]))

    """
    real = numpy.atleast_1d(real)
    imag = numpy.atleast_1d(imag)
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    real_components = numpy.atleast_1d(real_components)
    imag_components = numpy.atleast_1d(imag_components)
    if real_components.shape != imag_components.shape:
        raise ValueError(f'{real_components.shape=} != {imag_components.shape=}')
    fractions = []
    for component in range(len(real_components)):
        component_fractions_avg = []
        other_components = [id for id in range(len(real_components)) if id != component]
        if len(other_components) < 2:
            fractions_first_component = _graphical_first_fraction(real, imag, real_components, imag_components, cursor_diameter=cursor_diameter)
            return fractions_first_component, 1 - fractions_first_component
        components_combinations = [(b, c) for id, b in enumerate(other_components) for c in other_components[id + 1:]]
        for component_b, component_c in components_combinations:
            line_vector = numpy.array([real_components[component_c] - real_components[component_b], imag_components[component_c] - imag_components[component_b]])
            total_distance_between_components = numpy.linalg.norm(line_vector)
            if math.isclose(total_distance_between_components, 0, abs_tol=1e-6):
                raise ValueError('components must have different coordinates')
            unit_vector = line_vector / total_distance_between_components
            number_of_steps = math.ceil(total_distance_between_components / cursor_diameter)
            cursor_x, cursor_y = real_components[component_b], imag_components[component_b]
            for step in range(0, number_of_steps + 1):
                component_fractions = _graphical_first_fraction(real, imag, [real_components[component], cursor_x], [imag_components[component], cursor_y], cursor_diameter=cursor_diameter)
                component_fractions_avg.append(component_fractions)
                cursor_x, cursor_y = _move_cursor_along_line(cursor_x, cursor_y, cursor_diameter, unit_vector)
        fractions.append(numpy.nanmean(numpy.array(component_fractions_avg), axis = 0))
    fractions = fractions / numpy.sum(fractions, axis=0) 
    return tuple(fractions)

def _graphical_first_fraction(
    real: ArrayLike,
    imag: ArrayLike,
    real_components: ArrayLike,
    imag_components: ArrayLike,
    /,
    *,
    cursor_diameter: float = 0.1,
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
    cursor_diameter: float, optional
        Diameter of the cursor in phasor coordinates.

    Returns
    -------


    Notes
    -----

    Raises
    ------

    Examples
    --------

    

    """
    real = numpy.atleast_1d(real)
    imag = numpy.atleast_1d(imag)
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    real_components = numpy.atleast_1d(real_components)
    imag_components = numpy.atleast_1d(imag_components)
    if real_components.shape != imag_components.shape:
        raise ValueError(f'{real_components.shape=} != {imag_components.shape=}')
    fractions = numpy.full_like(real, numpy.nan)
    line_vector = numpy.array([real_components[1] - real_components[0], imag_components[1] - imag_components[0]])
    total_distance_between_components = numpy.linalg.norm(line_vector)
    if math.isclose(total_distance_between_components, 0, abs_tol=1e-6):
        raise ValueError('components must have different coordinates')
    unit_vector = line_vector / total_distance_between_components
    number_of_steps = math.ceil(total_distance_between_components / cursor_diameter)
    cursor_x, cursor_y = real_components[0], imag_components[0]
    for step in range(0, number_of_steps):
        mask = _mask_cursor(real, imag, cursor_x, cursor_y, cursor_diameter)
        fraction =  (number_of_steps - step) / number_of_steps
        fractions[mask] = fraction
        cursor_x, cursor_y = _move_cursor_along_line(cursor_x, cursor_y, cursor_diameter, unit_vector)
    return fractions

def _move_cursor_along_line(cursor_x, cursor_y, cursor_diameter, unit_vector) -> tuple[float, float]:
    displacement = cursor_diameter * unit_vector
    cursor_x += displacement[0]
    cursor_y += displacement[1]
    return cursor_x, cursor_y

def _mask_cursor(
    real: NDArray[Any],
    imag: NDArray[Any],
    cursor_x: float,
    cursor_y: float,
    cursor_diameter: float,
    /,
) -> NDArray[Any]:
    """Return array with cursor masked.

    Parameters
    ----------
    real : ndarray
        Real component of phasor coordinates.
    imag : ndarray
        Imaginary component of phasor coordinates.
    cursor_x : float
        x-coordinate of the cursor.
    cursor_y : float
        y-coordinate of the cursor.
    cursor_diameter : float
        Diameter of the cursor in phasor coordinates.

    Returns
    -------
    masked_array : ndarray
        Masked array.

    """
    cursor_radius = cursor_diameter / 2
    distances = numpy.hypot(real - cursor_x, imag - cursor_y)
    return numpy.where(distances <= cursor_radius, True, False)  

