"""Component analysis of phasor coordinates.

The ``phasorpy.components`` module provides functions to:

- calculate fractions of two known components by projecting onto the
  line between the components (:py:func:`phasor_component_fraction`)

- calculate phasor coordinates of second component if only one is
  known (not implemented)

- calculate fractions of multiple known components by using higher
  harmonic information (:py:func:`phasor_component_fit`)

- calculate fractions of two or three known components by resolving
  graphically with histogram (:py:func:`phasor_component_graphical`)

- blindly resolve fractions of multiple components by using harmonic
  information (:py:func:`phasor_component_blind`, not implemented)

"""

from __future__ import annotations

__all__ = [
    # phasor_component_blind,
    'phasor_component_fit',
    'phasor_component_fraction',
    'phasor_component_graphical',
]

import numbers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import numpy

from ._phasorpy import (
    _blend_and,
    _fraction_on_segment,
    _is_inside_circle,
    _is_inside_stadium,
    _segment_direction_and_length,
)
from .phasor import phasor_threshold


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


def phasor_component_graphical(
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

    >>> phasor_component_graphical(
    ...     [0.6, 0.3], [0.35, 0.38], [0.2, 0.9], [0.4, 0.3], fractions=6
    ... )  # doctest: +NUMBER
    (array([0, 0, 1, 0, 1, 0], dtype=uint64),)

    Count the number of phasors between the combinations of three components:

    >>> phasor_component_graphical(
    ...     [0.4, 0.5],
    ...     [0.2, 0.3],
    ...     [0.0, 0.2, 0.9],
    ...     [0.0, 0.4, 0.3],
    ...     fractions=6,
    ... )  # doctest: +NUMBER +NORMALIZE_WHITESPACE
    (array([0, 1, 1, 1, 1, 0], dtype=uint64),
     array([0, 1, 0, 0, 0, 0], dtype=uint64),
     array([0, 1, 2, 0, 0, 0], dtype=uint64))

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
                fraction_counts = numpy.sum(mask, dtype=numpy.uint64)
                component_counts.append(fraction_counts)

            counts.append(numpy.asarray(component_counts))

    return tuple(counts)


def phasor_component_fit(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    component_real: ArrayLike,
    component_imag: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], ...]:
    """Return fractions of multiple components from phasor coordinates.

    Component fractions are obtained from the least-squares solution of a
    linear matrix equation that relates phasor coordinates from one or
    multiple harmonics to component fractions according to [2]_.

    Up to ``2 * number harmonics + 1`` components can be fit to multi-harmonic
    phasor coordinates, that is up to three components for single harmonic
    phasor coordinates.

    Parameters
    ----------
    mean : array_like
        Intensity of phasor coordinates.
    real : array_like
        Real component of phasor coordinates.
        Harmonics, if any, must be in the first dimension.
    imag : array_like
        Imaginary component of phasor coordinates.
        Harmonics, if any, must be in the first dimension.
    component_real : array_like
        Real coordinates of components.
        Must be one or two-dimensional with harmonics in the first dimension.
    component_imag : array_like
        Imaginary coordinates of components.
        Must be one or two-dimensional with harmonics in the first dimension.
    **kwargs : optional
        Additional arguments passed to :py:func:`scipy.linalg.lstsq()`.

    Returns
    -------
    fractions : tuple of ndarray
        Component fractions, one array per component.
        Fractions may not exactly add up to 1.0.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag` do not match.
        The array shapes of `component_real` and `component_imag` do not match.
        The number of harmonics in the components does not
        match the ones in the phasor coordinates.
        The system is underdetermined; the component matrix having more
        columns than rows.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_components.py`
    :ref:`sphx_glr_tutorials_applications_phasorpy_component_fit.py`

    Notes
    -----
    For now, calculation of fractions of components from different channels
    or frequencies is not supported. Only one set of components can be
    analyzed and is broadcast to all channels/frequencies.

    The method builds a linear matrix equation,
    :math:`A\\mathbf{x} = \\mathbf{b}`, where :math:`A` consists of the
    phasor coordinates of individual components, :math:`\\mathbf{x}` are
    the unknown fractions, and :math:`\\mathbf{b}` represents the measured
    phasor coordinates in the mixture. The least-squares solution of this
    linear matrix equation yields the fractions.

    References
    ----------
    .. [2] Vallmitjana A, Lepanto P, Irigoin F, and Malacrida L.
      `Phasor-based multi-harmonic unmixing for in-vivo hyperspectral
      imaging <https://doi.org/10.1088/2050-6120/ac9ae9>`_.
      *Methods Appl Fluoresc*, 11(1): 014001 (2022)

    Example
    -------
    >>> phasor_component_fit(
    ...     [1, 1, 1], [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... )  # doctest: +NUMBER
    (array([0.4644, 0.5356, 0.6068]), array([0.5559, 0.4441, 0.3322]))

    """
    from scipy.linalg import lstsq

    mean = numpy.atleast_1d(mean)
    real = numpy.atleast_1d(real)
    imag = numpy.atleast_1d(imag)
    component_real = numpy.atleast_1d(component_real)
    component_imag = numpy.atleast_1d(component_imag)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if mean.shape != real.shape[-mean.ndim :]:
        raise ValueError(f'{mean.shape=} does not match {real.shape=}')

    if component_real.shape != component_imag.shape:
        raise ValueError(f'{component_real.shape=} != {component_imag.shape=}')
    if numpy.isnan(component_real).any() or numpy.isnan(component_imag).any():
        raise ValueError(
            'component phasor coordinates must not contain NaN values'
        )
    if numpy.isinf(component_real).any() or numpy.isinf(component_imag).any():
        raise ValueError(
            'component phasor coordinates must not contain infinite values'
        )

    if component_real.ndim == 1:
        component_real = component_real.reshape(1, -1)
        component_imag = component_imag.reshape(1, -1)
    elif component_real.ndim > 2:
        raise ValueError(f'{component_real.ndim=} > 2')

    num_harmonics, num_components = component_real.shape

    # create component matrix for least squares solving:
    # [real coordinates of components (for each harmonic)] +
    # [imaginary coordinates of components (for each harmonic)] +
    # [ones for intensity constraint]
    component_matrix = numpy.ones((2 * num_harmonics + 1, num_components))
    component_matrix[:num_harmonics] = component_real
    component_matrix[num_harmonics : 2 * num_harmonics] = component_imag

    if component_matrix.shape[0] < component_matrix.shape[1]:
        raise ValueError(
            'the system is undetermined '
            f'({num_components=} > {num_harmonics * 2 + 1=})'
        )

    has_harmonic_axis = mean.ndim + 1 == real.ndim
    if not has_harmonic_axis:
        real = numpy.expand_dims(real, axis=0)
        imag = numpy.expand_dims(imag, axis=0)
    elif real.shape[0] != num_harmonics:
        raise ValueError(f'{real.shape[0]=} != {component_real.shape[0]=}')

    # TODO: replace Inf with NaN values?
    mean, real, imag = phasor_threshold(mean, real, imag)

    # replace NaN values with 0.0 for least squares solving
    real = numpy.nan_to_num(real, nan=0.0, copy=False)
    imag = numpy.nan_to_num(imag, nan=0.0, copy=False)

    # create coordinates matrix for least squares solving:
    # [real coordinates (for each harmonic)] +
    # [imaginary coordinates (for each harmonic)] +
    # [ones for intensity constraint]
    coords = numpy.ones((2 * num_harmonics + 1,) + real.shape[1:])
    coords[:num_harmonics] = real
    coords[num_harmonics : 2 * num_harmonics] = imag

    fractions = lstsq(
        component_matrix, coords.reshape(coords.shape[0], -1), **kwargs
    )[0]

    # reshape to match input dimensions
    fractions = fractions.reshape((num_components,) + coords.shape[1:])

    # TODO: normalize fractions to sum up to 1.0?
    # fractions /= numpy.sum(fractions, axis=0, keepdims=True)

    # restore NaN values in fractions from mean
    _blend_and(mean, fractions, out=fractions)

    return tuple(fractions)
