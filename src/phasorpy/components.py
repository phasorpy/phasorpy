"""Component analysis of phasor coordinates.

The ``phasorpy.components`` module provides functions to:

- calculate fractions of two known components by projecting onto the
  line between the components:

  - :py:func:`two_fractions_from_phasor`

- calculate phasor coordinates of second component if only one is
  known (not implemented)

- calculate fractions of three or four known components by using higher
  harmonic information (not implemented)

- calculate fractions of two or three known components by resolving
  graphically with histogram:

  - :py:func:`graphical_component_analysis`

- phasor-based resolve fractions of `n` components by using harmonic
  information

  - :py:func: `n_fractions_from_phasor`

"""

from __future__ import annotations

__all__ = [
    'two_fractions_from_phasor',
    'graphical_component_analysis',
    'n_fractions_from_phasor',
]

import numbers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import numpy
import scipy

from ._phasorpy import (
    _fraction_on_segment,
    _is_inside_circle,
    _is_inside_stadium,
    _segment_direction_and_length,
)


def two_fractions_from_phasor(
    real: ArrayLike,
    imag: ArrayLike,
    components_real: ArrayLike,
    components_imag: ArrayLike,
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
    components_real: array_like, shape (2,)
        Real coordinates of the first and second components.
    components_imag: array_like, shape (2,)
        Imaginary coordinates of the first and second components.

    Returns
    -------
    fraction : ndarray
        Fractions of first component.

    Raises
    ------
    ValueError
        If the real and/or imaginary coordinates of the known components are
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
    >>> two_fractions_from_phasor(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... )  # doctest: +NUMBER
    array([0.44, 0.56, 0.68])

    """
    components_real = numpy.asarray(components_real)
    components_imag = numpy.asarray(components_imag)
    if components_real.shape != (2,):
        raise ValueError(f'{components_real.shape=} != (2,)')
    if components_imag.shape != (2,):
        raise ValueError(f'{components_imag.shape=} != (2,)')
    if (
        components_real[0] == components_real[1]
        and components_imag[0] == components_imag[1]
    ):
        raise ValueError('components must have different coordinates')

    return _fraction_on_segment(  # type: ignore[no-any-return]
        real,
        imag,
        components_real[0],
        components_imag[0],
        components_real[1],
        components_imag[1],
    )


def graphical_component_analysis(
    real: ArrayLike,
    imag: ArrayLike,
    components_real: ArrayLike,
    components_imag: ArrayLike,
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
    components_real: array_like, shape (2,) or (3,)
        Real coordinates for two or three components.
    components_imag: array_like, shape (2,) or (3,)
        Imaginary coordinates for two or three components.
    radius: float, optional, default: 0.05
        Radius of the cursor in phasor coordinates.
    fractions: array_like or int, optional
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
        Counts along each line segment connecting the components, ordered
        0-1 (2 components) or 0-1, 0-2, 1-2 (3 components).

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag`, or `components_real` and
        `components_imag` do not match.
        The number of components is not 2 or 3.
        Fraction values are not in range [0.0, 1.0].

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_components.py`

    Notes
    -----
    For now, calculation of fraction of components from different
    channels or frequencies is not supported. Only one set of components can
    be analyzed and will be broadcast to all channels/frequencies.

    The graphical method was first introduced in [1]_.

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

    .. [1] Ranjit S, Datta R, Dvornikov A, and Gratton E.
      `Multicomponent analysis of phasor plot in a single pixel to
      calculate changes of metabolic trajectory in biological systems
      <https://doi.org/10.1021/acs.jpca.9b07880>`_.
      *J Phys Chem A*, 123(45): 9865-9873 (2019)

    Examples
    --------
    Count the number of phasors between two components:

    >>> graphical_component_analysis(
    ...     [0.6, 0.3], [0.35, 0.38], [0.2, 0.9], [0.4, 0.3], fractions=6
    ... )  # doctest: +NUMBER
    (array([0, 0, 1, 0, 1, 0]),)

    Count the number of phasors between the combinations of three components:

    >>> graphical_component_analysis(
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
    components_real = numpy.asarray(components_real)
    components_imag = numpy.asarray(components_imag)
    if (
        real.shape != imag.shape
        or components_real.shape != components_imag.shape
    ):
        raise ValueError('input array shapes must match')
    if components_real.ndim != 1:
        raise ValueError(
            'component arrays are not one-dimensional: '
            f'{components_real.ndim} dimensions found'
        )
    num_components = len(components_real)
    if num_components not in {2, 3}:
        raise ValueError('number of components must be 2 or 3')

    if fractions is None:
        longest_distance = 0
        for i in range(num_components):
            a_real = components_real[i]
            a_imag = components_imag[i]
            for j in range(i + 1, num_components):
                b_real = components_real[j]
                b_imag = components_imag[j]
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
        a_real = components_real[i]
        a_imag = components_imag[i]
        for j in range(i + 1, num_components):
            b_real = components_real[j]
            b_imag = components_imag[j]
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
                        components_real[3 - i - j],  # c_real
                        components_imag[3 - i - j],  # c_imag
                        radius,
                    )
                fraction_counts = numpy.sum(mask)
                component_counts.append(fraction_counts)

            counts.append(numpy.asarray(component_counts))

    return tuple(counts)


def n_fractions_from_phasor(
    real: ArrayLike,
    imag: ArrayLike,
    coeff_matrix: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], ...]:
    """
    Returns the fractions of each component in each pixel.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Real component of phasor coordinates.
    coeff_matrix : array_like
        Pure components coefficients to compute unmixing.
        Matrix is like ``[real1, imag1, ..., realN, imagN, [1...1]]``
        with real and imaginary component of phasor coordinates for the
        pure components.
    **kwargs : optional
        Additional arguments passed to ``scipy.linalg.lstsq()``.

    Returns
    -------
    fractions : ndarray
        fractions : tuple of ndarray
        Fractions of pure components.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag` do not match.
        The coefficient matrix is empty.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_components.py`

    References
    ----------
    [2] Vallmitjana A, Lepanto P, Irigoin F, Malacrida L.
    `Phasor-based multi-harmonic unmixing for in-vivohyperspectral imaging
    <https://doi.org/10.1088/2050-6120/ac9ae9>`_.
    *Methods Appl Fluoresc.*, (2022)

    Example
    -------
    >>> n_fractions_from_phasor(
    ...     [0.5, 0.3], [0.2, 0.7], [[0.1, 0.3], [0.2, 0.8], [1.0, 1.0]]
    ... )  # doctest: +NUMBER
    (array(0.8162), array(0.1958))

    """

    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    coeff_matrix = numpy.asarray(coeff_matrix)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')

    if coeff_matrix.size == 0:
        raise ValueError('the coefficient matrix is empty.')

    real = numpy.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
    imag = numpy.nan_to_num(imag, nan=0.0, posinf=0.0, neginf=0.0)

    if real.ndim == 1:
        vec_b = numpy.hstack([real[:1], imag[:1], 1])
        return tuple(
            map(numpy.asarray, scipy.linalg.lstsq(coeff_matrix, vec_b)[0])
        )

    else:
        if real.ndim == 2:
            n, m = real.shape
            nh = 1  # number of harmonics
            real = real.reshape(nh, n, m)
            imag = imag.reshape(nh, n, m)
        elif real.ndim == 3:
            nh, n, m = real.shape
        else:
            raise ValueError(
                f'invalid shape {real.shape}, expected (nh, n, m) or (n, m)'
            )

        real_reshaped = real.reshape(nh, -1)
        imag_reshaped = imag.reshape(nh, -1)

        ones_array = numpy.ones((1, n * m))

        vec_b = numpy.concatenate(
            [real_reshaped, imag_reshaped, ones_array], axis=0
        )

        fractions = scipy.linalg.lstsq(coeff_matrix, vec_b, **kwargs)[0]

        return tuple(map(numpy.asarray, fractions))
