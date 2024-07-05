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
    components_real: ArrayLike,
    components_imag: ArrayLike,
    /,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return fractions of two components from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    components_real: array_like
        Real coordinates of the first and second components.
    components_imag: array_like
        Imaginary coordinates of the first and second components.

    Returns
    -------
    fractions : tuple of ndarray
        A tuple containing arrays with the fractions of a specific component.
        The order of the arrays in the tuple corresponds to the order of the
        components used in the calculation.

    Notes
    -----
    For now, calculation of fraction of components from different
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
    components_real = numpy.asarray(components_real)
    components_imag = numpy.asarray(components_imag)
    if components_real.shape != (2,):
        raise ValueError(f'{components_real.shape=} != (2,)')
    if components_imag.shape != (2,):
        raise ValueError(f'{components_imag.shape=} != (2,)')
    _, distance_between_components = line_from_components(
        components_real, components_imag
    )
    projected_real, projected_imag = project_phasor_to_line(
        real, imag, components_real, components_imag
    )
    distances_to_first_component = numpy.hypot(
        numpy.asarray(projected_real) - components_real[0],
        numpy.asarray(projected_imag) - components_imag[0],
    )
    second_component_fractions = (
        distances_to_first_component / distance_between_components
    )
    first_component_fractions = 1 - second_component_fractions
    return first_component_fractions, second_component_fractions


def graphical_component_analysis(
    real: ArrayLike,
    imag: ArrayLike,
    components_real: ArrayLike,
    components_imag: ArrayLike,
    /,
    *,
    radius: float = 0.05,
    steps: int = 100,
) -> tuple[tuple[NDArray[Any], ...], NDArray[Any]]:
    """Return fractions of two or three components from phasor coordinates.

    The graphical method is based on moving circular cursors along the line
    between pairs of components, and quantifying the phasors for each fraction.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    components_real: array_like
        Real coordinates for two or three components.
    components_imag: array_like
        Imaginary coordinates for two or three components.
    radius: float, optional
        Diameter of the cursor in phasor coordinates.
    steps: int, optional
        Number of steps to move the cursor along the line between components.

    Returns
    -------
    counts : tuple of ndarray
        Counts along each line segment connecting the components, ordered
        0-1, 0-2, 1-2 (for 3 components) or simply 0-1 (for 2 components).
    fractions : ndarray
        Fractions for the combinations of each pair of components, from 0 to 1.

    Notes
    -----
    For now, calculation of fraction of components from different
    channels or frequencies is not supported. Only one set of components can
    be analyzed and will be broadcasted to all channels/frequencies.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag`, or `components_real` and
        `components_imag` do not match.
        Number of components is less than 2 or greater than 3.

    See Also
    --------
    :ref:`sphx_glr_tutorials_phasorpy_components.py`

    Examples
    --------
    Count the number of phasors and fractions between two components:

    >>> graphical_component_analysis(
    ...     [0.6, 0.3], [0.35, 0.38], [0.2, 0.9], [0.4, 0.3], steps=5
    ... )  # doctest: +NUMBER
    ((array([0, 0, 1, 0, 1, 0]),), array([0, 0.2, 0.4, 0.6, 0.8, 1]))

    Count the number of phasors and fractions between the combinations
    of three components:

    >>> graphical_component_analysis(
    ...     [0.4, 0.5],
    ...     [0.2, 0.3],
    ...     [0.0, 0.2, 0.9],
    ...     [0.0, 0.4, 0.3],
    ...     steps=5,
    ... )  # doctest: +NUMBER +NORMALIZE_WHITESPACE
    ((array([0, 1, 1, 1, 1, 0]),
    array([0, 1, 0, 0, 0, 0]),
    array([0, 1, 2, 0, 0, 0])),
    array([0, 0.2, 0.4, 0.6, 0.8, 1]))

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    components_real = numpy.asarray(components_real)
    components_imag = numpy.asarray(components_imag)
    if (
        real.shape != imag.shape
        or components_real.shape != components_imag.shape
    ):
        raise ValueError("Input array shapes must match")
    if components_real.ndim != 1:
        raise ValueError(
            'Components arrays are not one-dimensional: '
            f'{components_real.ndim} dimensions found'
        )
    num_components = len(components_real)
    if num_components not in {2, 3}:
        raise ValueError("Number of components must be 2 or 3")
    counts = []
    fractions = numpy.linspace(0, 1, steps + 1)
    for i, (real_a, imag_a) in enumerate(
        zip(components_real, components_imag)
    ):
        for j in range(i + 1, num_components):
            real_b, imag_b = components_real[j], components_imag[j]
            unit_vector, distance = line_from_components(
                [real_b, real_a], [imag_b, imag_a]
            )
            cursor_real, cursor_imag = real_b, imag_b
            step_size = distance / steps
            component_counts = []
            for _ in range(steps + 1):
                if num_components == 2:
                    mask = mask_cursor(
                        real, imag, cursor_real, cursor_imag, radius
                    )
                elif num_components == 3:
                    real_c, imag_c = (
                        components_real[3 - i - j],
                        components_imag[3 - i - j],
                    )
                    mask = mask_segment(
                        real,
                        imag,
                        cursor_real,
                        cursor_imag,
                        real_c,
                        imag_c,
                        radius,
                    )
                fraction_counts = numpy.sum(mask)
                component_counts.append(fraction_counts)
                cursor_real += step_size * unit_vector[0]
                cursor_imag += step_size * unit_vector[1]
            counts.append(numpy.asarray(component_counts))
    return (tuple(counts), fractions)
