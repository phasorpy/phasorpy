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

  - :py:func:`graphical_two_fractions`
  - :py:func:`graphical_component_analysis`

- blindly resolve fractions of n components by using harmonic
  information (not implemented)

"""

from __future__ import annotations

__all__ = [
    'two_fractions_from_phasor',
    'graphical_two_fractions',
    'graphical_component_analysis',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import numpy

import matplotlib.pyplot as plt
from phasorpy.plot import PhasorPlot
from matplotlib.patches import Circle

from ._utils import (
    line_from_components,
    mask_cursor,
    move_cursor_along_line,
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
    number_of_steps: int, optional
        Number of steps to move the cursor along the line between components.

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
        raise ValueError(
            f'{real_components.shape=} != {imag_components.shape=}'
        )
    fractions = []
    for component in range(len(real_components)):
        component_fractions_avg = []
        other_components = [
            id for id in range(len(real_components)) if id != component
        ]
        if len(other_components) < 2:
            fractions_first_component, fractions_second_component = (
                graphical_two_fractions(
                    real,
                    imag,
                    real_components,
                    imag_components,
                    cursor_diameter=cursor_diameter,
                    number_of_steps=number_of_steps,
                )
            )
            return fractions_first_component, fractions_second_component
        components_combinations = [
            (b, c)
            for id, b in enumerate(other_components)
            for c in other_components[id + 1 :]
        ]
        for component_b, component_c in components_combinations:
            unit_vector, distance_between_components = line_from_components(
                [
                    real_components[component_b],
                    real_components[component_c],
                ],
                [
                    imag_components[component_b],
                    imag_components[component_c],
                ],
            )
            # number_of_steps = math.ceil(
            #     distance_between_components / cursor_diameter
            # )
            cursor_real, cursor_imag = (
                real_components[component_b],
                imag_components[component_b],
            )
            fig, ax = plt.subplots()
            plot = PhasorPlot(frequency = 80, ax=ax)
            plot.plot(real_components, imag_components, linestyle = '-')
            plot.plot(real, imag)
            print('COMPONENT: ', component)
            for step in range(0, number_of_steps + 1):
                component_fractions, _ = graphical_two_fractions(
                    real,
                    imag,
                    [real_components[component], cursor_real],
                    [imag_components[component], cursor_imag],
                    cursor_diameter=cursor_diameter,
                    number_of_steps=number_of_steps,
                    ax = ax,
                )
                circle = Circle((cursor_real, cursor_imag), cursor_diameter/2, fill=False, edgecolor='green')  # Create a circle patch
                ax.add_patch(circle)
                component_fractions_avg.append(component_fractions)
                cursor_real, cursor_imag = move_cursor_along_line(
                    cursor_real,
                    cursor_imag,
                    distance_between_components / number_of_steps,
                    unit_vector,
                )
            plot.show()
        fractions.append(
            numpy.nanmean(numpy.array(component_fractions_avg), axis=0)
        )
    fractions = fractions / numpy.sum(fractions, axis=0)
    return tuple(fractions)


def graphical_two_fractions(
    real: ArrayLike,
    imag: ArrayLike,
    real_components: ArrayLike,
    imag_components: ArrayLike,
    /,
    *,
    cursor_diameter: float = 0.1,
    number_of_steps: int = 100,
    ax=None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return fractions of the first component by the graphical solution.

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
        A tuple containing arrays with the fractions of a specific component.
        The order of the arrays in the tuple corresponds to the order of the
        components used in the calculation.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag`, or `real_components` and
        `imag_components` do not match.

    Examples
    --------
    >>> graphical_two_fractions(
    ...     [0.7, 0.55, 0.4],
    ...     [0.35, 0.37, 0.39],
    ...     [0.2, 0.9],
    ...     [0.4, 0.3],
    ...     cursor_diameter=0.05,
    ... )  # doctest: +NUMBER
    (array([0.29, 0.50, 0.70]), array([0.71, 0.51, 0.30]))

    """
    real = numpy.atleast_1d(real)
    imag = numpy.atleast_1d(imag)
    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    real_components = numpy.atleast_1d(real_components)
    imag_components = numpy.atleast_1d(imag_components)
    if real_components.shape != imag_components.shape:
        raise ValueError(
            f'{real_components.shape=} != {imag_components.shape=}'
        )
    fractions = numpy.full_like(real, numpy.nan)
    unit_vector, distance_between_components = line_from_components(
        real_components, imag_components
    )
    # number_of_steps = math.ceil(distance_between_components / cursor_diameter)
    cursor_real, cursor_imag = real_components[0], imag_components[0]
    for step in range(0, number_of_steps + 1):
        circle = Circle((cursor_real, cursor_imag), cursor_diameter/2, fill=False, edgecolor='red')  # Create a circle patch
        ax.add_patch(circle)
        mask = mask_cursor(
            real, imag, cursor_real, cursor_imag, cursor_diameter
        )
        # fraction = (number_of_steps + 1 - step) / (number_of_steps + 1)
        fraction = numpy.hypot(cursor_real - real_components[1], cursor_imag - imag_components[1]) / distance_between_components
        fractions[mask] = fraction
        cursor_real, cursor_imag = move_cursor_along_line(
            cursor_real,
            cursor_imag,
            distance_between_components / number_of_steps,
            unit_vector,
        )
    return (fractions, 1 - fractions)
