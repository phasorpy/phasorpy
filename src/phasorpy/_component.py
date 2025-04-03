"""Analyze components in phasor coordinates."""

from __future__ import annotations

__all__ = [
    'phasor_component_fraction',
    'phasor_component_graphical_analysis',
]

import numbers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import numpy

from ._phasorpy import (
    _fraction_on_segment,
    _is_inside_circle,
    _is_inside_stadium,
    _segment_direction_and_length,
)


def phasor_component_fraction(
    real: ArrayLike,
    imag: ArrayLike,
    component_real: ArrayLike,
    component_imag: ArrayLike,
    /,
) -> NDArray[Any]:
    """Return fraction of first of two components from phasor coordinates.

    Return the relative distance (normalized by the distance between the two
    components) to the second component for each phasor coordinate projected
    onto the line between two components.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    component_real : array_like, shape (2,)
        Real coordinates of first and second components.
    component_imag : array_like, shape (2,)
        Imaginary coordinates of first and second components.

    Returns
    -------
    fraction : ndarray
        Fractions of first component.

    Raises
    ------
    ValueError
        If the real or imaginary coordinates of the known components are
        not of size 2.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_components.py`

    Notes
    -----
    The fraction of the second component is ``1.0 - fraction``.

    For now, calculation of fraction of components from different
    channels or frequencies is not supported. Only one pair of components can
    be analyzed and will be broadcast to all channels/frequencies.

    Examples
    --------
    >>> phasor_component_fraction(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... )  # doctest: +NUMBER
    array([0.44, 0.56, 0.68])

    """
    component_real = numpy.asarray(component_real)
    component_imag = numpy.asarray(component_imag)
    if component_real.shape != (2,):
        raise ValueError(f'{component_real.shape=} != (2,)')
    if component_imag.shape != (2,):
        raise ValueError(f'{component_imag.shape=} != (2,)')
    if (
        component_real[0] == component_real[1]
        and component_imag[0] == component_imag[1]
    ):
        raise ValueError('components must have different coordinates')

    return _fraction_on_segment(  # type: ignore[no-any-return]
        real,
        imag,
        component_real[0],
        component_imag[0],
        component_real[1],
        component_imag[1],
    )


def phasor_component_graphical_analysis(
    real: ArrayLike,
    imag: ArrayLike,
    component_real: ArrayLike,
    component_imag: ArrayLike,
    /,
    *,
    radius: float = 0.05,
    fractions: ArrayLike | None = None,
) -> tuple[NDArray[Any], ...]:
    r"""Return fractions of two or three components from phasor coordinates.

    The graphical method is based on moving circular cursors along the line
    between pairs of components and quantifying the phasors for each
    fraction.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    component_real : array_like, shape (2,) or (3,)
        Real coordinates for two or three components.
    component_imag : array_like, shape (2,) or (3,)
        Imaginary coordinates for two or three components.
    radius : float, optional, default: 0.05
        Radius of cursor.
    fractions : array_like or int, optional
        Number of equidistant fractions, or 1D array of fraction values.
        Fraction values must be in range [0.0, 1.0].
        If an integer, ``numpy.linspace(0.0, 1.0, fractions)`` fraction values
        are used.
        If None (default), the number of fractions is determined from the
        longest distance between any pair of components and the radius of
        the cursor (see Notes below).

    Returns
    -------
    counts : tuple of ndarray
        Counts along each line segment connecting components.
        Ordered 0-1 (2 components) or 0-1, 0-2, 1-2 (3 components).

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag`, or `component_real` and
        `component_imag` do not match.
        The number of components is not 2 or 3.
        Fraction values are out of range [0.0, 1.0].

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_components.py`

    Notes
    -----
    For now, calculation of fraction of components from different
    channels or frequencies is not supported. Only one set of components can
    be analyzed and will be broadcast to all channels/frequencies.

    The graphical method was first introduced in [2]_.

    If no `fractions` are provided, the number of fractions (:math:`N`) used
    is determined from the longest distance between any pair of components
    (:math:`D`) and the radius of the cursor (:math:`R`):

    .. math::

        N = \frac{2 \cdot D}{R} + 1

    The fractions can be retrieved by:

    .. code-block:: python

        fractions = numpy.linspace(0.0, 1.0, len(counts[0]))

    References
    ----------

    .. [2] Ranjit S, Datta R, Dvornikov A, and Gratton E.
      `Multicomponent analysis of phasor plot in a single pixel to
      calculate changes of metabolic trajectory in biological systems
      <https://doi.org/10.1021/acs.jpca.9b07880>`_.
      *J Phys Chem A*, 123(45): 9865-9873 (2019)

    Examples
    --------
    Count the number of phasors between two components:

    >>> phasor_component_graphical_analysis(
    ...     [0.6, 0.3], [0.35, 0.38], [0.2, 0.9], [0.4, 0.3], fractions=6
    ... )  # doctest: +NUMBER
    (array([0, 0, 1, 0, 1, 0]),)

    Count the number of phasors between the combinations of three components:

    >>> phasor_component_graphical_analysis(
    ...     [0.4, 0.5],
    ...     [0.2, 0.3],
    ...     [0.0, 0.2, 0.9],
    ...     [0.0, 0.4, 0.3],
    ...     fractions=6,
    ... )  # doctest: +NUMBER +NORMALIZE_WHITESPACE
    (array([0, 1, 1, 1, 1, 0]),
     array([0, 1, 0, 0, 0, 0]),
     array([0, 1, 2, 0, 0, 0]))

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    component_real = numpy.asarray(component_real)
    component_imag = numpy.asarray(component_imag)
    if (
        real.shape != imag.shape
        or component_real.shape != component_imag.shape
    ):
        raise ValueError('input array shapes must match')
    if component_real.ndim != 1:
        raise ValueError(
            'component arrays are not one-dimensional: '
            f'{component_real.ndim} dimensions found'
        )
    num_components = len(component_real)
    if num_components not in {2, 3}:
        raise ValueError('number of components must be 2 or 3')

    if fractions is None:
        longest_distance = 0
        for i in range(num_components):
            a_real = component_real[i]
            a_imag = component_imag[i]
            for j in range(i + 1, num_components):
                b_real = component_real[j]
                b_imag = component_imag[j]
                _, _, length = _segment_direction_and_length(
                    a_real, a_imag, b_real, b_imag
                )
                longest_distance = max(longest_distance, length)
        fractions = numpy.linspace(
            0.0, 1.0, int(round(longest_distance / (radius / 2) + 1))
        )
    elif isinstance(fractions, (int, numbers.Integral)):
        fractions = numpy.linspace(0.0, 1.0, fractions)
    else:
        fractions = numpy.asarray(fractions)
        if fractions.ndim != 1:
            raise ValueError('fractions is not a one-dimensional array')

    counts = []
    for i in range(num_components):
        a_real = component_real[i]
        a_imag = component_imag[i]
        for j in range(i + 1, num_components):
            b_real = component_real[j]
            b_imag = component_imag[j]
            ab_real = a_real - b_real
            ab_imag = a_imag - b_imag

            component_counts = []
            for f in fractions:
                if f < 0.0 or f > 1.0:
                    raise ValueError(f'fraction {f} out of bounds [0.0, 1.0]')
                if num_components == 2:
                    mask = _is_inside_circle(
                        real,
                        imag,
                        b_real + f * ab_real,  # cursor_real
                        b_imag + f * ab_imag,  # cursor_imag
                        radius,
                    )
                else:
                    # num_components == 3
                    mask = _is_inside_stadium(
                        real,
                        imag,
                        b_real + f * ab_real,  # cursor_real
                        b_imag + f * ab_imag,  # cursor_imag
                        component_real[3 - i - j],  # c_real
                        component_imag[3 - i - j],  # c_imag
                        radius,
                    )
                fraction_counts = numpy.sum(mask)
                component_counts.append(fraction_counts)

            counts.append(numpy.asarray(component_counts))

    return tuple(counts)
