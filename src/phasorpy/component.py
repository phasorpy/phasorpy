"""Analyze components in phasor coordinates.

The ``phasorpy.component`` module provides functions to:

- calculate fractions of two known components by projecting onto the
  line between the components (:py:func:`phasor_component_fraction`)

- calculate phasor coordinates of a second component if only one is
  known (not implemented yet)

- calculate fractions of multiple known components by using higher-harmonic
  information (:py:func:`phasor_component_fit`)

- calculate fractions of two or three known components by resolving
  graphically with a histogram (:py:func:`phasor_component_graphical`)

- calculate mean value coordinates of phasors with respect to three
  or more components (:py:func:`phasor_component_mvc`)

- blindly resolve fractions of multiple components by using harmonic
  information (:py:func:`phasor_component_blind`, not implemented yet)

- calculate phasor coordinates from fractional intensities of
  components (:py:func:`phasor_from_component`)

- calculate absolute concentrations of two components from phasor
  coordinates and calibration (:py:func:`phasor_component_concentration`)

"""

from __future__ import annotations

__all__ = [
    # phasor_component_blind,
    'phasor_component_concentration',
    'phasor_component_fit',
    'phasor_component_fraction',
    'phasor_component_graphical',
    'phasor_component_mvc',
    'phasor_from_component',
]

import numbers
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, DTypeLike, NDArray

import numpy

from ._phasorpy import (
    _blend_and,
    _fraction_on_segment,
    _intersect_line_line,
    _is_inside_circle,
    _is_inside_stadium,
    _mean_value_coordinates,
    _segment_direction_and_length,
)
from ._utils import sort_coordinates
from .filter import phasor_threshold
from .utils import number_threads


def phasor_from_component(
    component_real: ArrayLike,
    component_imag: ArrayLike,
    fraction: ArrayLike,
    /,
    axis: int = 0,
    dtype: DTypeLike | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return phasor coordinates from fractional intensities of components.

    Return the dot products of the fractional intensities with the real and
    imaginary phasor coordinates of the components.

    Multidimensional component arrays are not supported yet.

    Parameters
    ----------
    component_real : array_like, shape (n,)
        Real coordinates of components.
        At least two components are required.
    component_imag : array_like, shape (n,)
        Imaginary coordinates of components.
    fraction : array_like
        Fractional intensities of components.
        Fractions are normalized to sum to 1 along `axis`.
    axis : int, optional, default: 0
        Axis of components in `fraction`.
    dtype : dtype_like, optional
        Floating point data type used for calculation and output values.
        Either `float32` or `float64`. The default is `float64`.

    Returns
    -------
    real : ndarray
        Real component of phasor coordinates.
    imag : ndarray
        Imaginary component of phasor coordinates.

    Raises
    ------
    ValueError
        If `dtype` is not a floating-point type.
        If the array shapes of `component_real` and `component_imag` do not
        match.
        If the `fraction` array has less than two components along `axis`.
        If the component coordinates contain NaN or infinite values.

    See Also
    --------
    phasorpy.phasor.phasor_combine

    Examples
    --------
    Calculate phasor coordinates from two components and their fractional
    intensities:

    >>> phasor_from_component(
    ...     [0.6, 0.4], [0.3, 0.2], [[1.0, 0.2, 0.9], [0.0, 0.8, 0.1]]
    ... )
    (array([0.6, 0.44, 0.58]), array([0.3, 0.22, 0.29]))

    """
    dtype = numpy.dtype(dtype)
    if dtype.char not in {'f', 'd'}:
        msg = f'{dtype=} is not a floating-point type'
        raise ValueError(msg)

    fraction = numpy.asarray(fraction, dtype=dtype, copy=True)
    if fraction.ndim < 1:
        msg = f'{fraction.ndim=} < 1'
        raise ValueError(msg)
    if fraction.shape[axis] < 2:
        msg = f'{fraction.shape[axis]=} < 2'
        raise ValueError(msg)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        fraction /= fraction.sum(axis=axis, keepdims=True)

    component_real = numpy.asarray(component_real, dtype=dtype)
    component_imag = numpy.asarray(component_imag, dtype=dtype)
    if component_real.shape != component_imag.shape:
        msg = f'{component_real.shape=} != {component_imag.shape=}'
        raise ValueError(msg)
    if component_real.ndim != 1:
        msg = f'{component_real.ndim=} != 1'
        raise ValueError(msg)
    if component_real.size != fraction.shape[axis]:
        msg = f'{component_real.size=} != {fraction.shape[axis]=}'
        raise ValueError(msg)
    if numpy.isnan(component_real).any() or numpy.isnan(component_imag).any():
        msg = 'component coordinates must not contain NaN values'
        raise ValueError(msg)
    if numpy.isinf(component_real).any() or numpy.isinf(component_imag).any():
        msg = 'component coordinates must not contain infinite values'
        raise ValueError(msg)

    fraction = numpy.moveaxis(fraction, axis, -1)
    real = numpy.dot(fraction, component_real)
    imag = numpy.dot(fraction, component_imag)
    return real, imag


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
        If the two components have identical coordinates.
        If the component coordinates contain NaN or infinite values.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_component.py`

    Notes
    -----
    The fraction of the second component is ``1.0 - fraction``.

    Calculation of fractions of components from different channels
    or frequencies is not supported yet. Only one pair of components
    can be analyzed and is broadcast to all channels/frequencies.

    Examples
    --------
    >>> phasor_component_fraction(
    ...     [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... )
    array([0.44, 0.56, 0.68])

    """
    component_real = numpy.asarray(component_real)
    component_imag = numpy.asarray(component_imag)
    if component_real.shape != (2,):
        msg = f'{component_real.shape=} != (2,)'
        raise ValueError(msg)
    if component_imag.shape != (2,):
        msg = f'{component_imag.shape=} != (2,)'
        raise ValueError(msg)
    if (
        component_real[0] == component_real[1]
        and component_imag[0] == component_imag[1]
    ):
        msg = 'components must have different coordinates'
        raise ValueError(msg)
    if numpy.isnan(component_real).any() or numpy.isnan(component_imag).any():
        msg = 'component coordinates must not contain NaN values'
        raise ValueError(msg)
    if numpy.isinf(component_real).any() or numpy.isinf(component_imag).any():
        msg = 'component coordinates must not contain infinite values'
        raise ValueError(msg)

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
) -> NDArray[Any]:
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
        Fraction values must be in the range [0.0, 1.0].
        If an integer, ``numpy.linspace(0.0, 1.0, fractions)`` fraction values
        are used.
        By default, the number of fractions is determined from the
        longest distance between any pair of components and the radius of
        the cursor (see Notes below).

    Returns
    -------
    counts : ndarray
        Counts along each line segment connecting components.
        Ordered 0-1 (2 components) or 0-1, 0-2, 1-2 (3 components).
        Shaped ``(number fractions,)`` (2 components) or
        ``(3, number fractions)`` (3 components).

    Raises
    ------
    ValueError
        If the array shapes of `real` and `imag`, or `component_real` and
        `component_imag` do not match.
        If the number of components is not 2 or 3.
        If the `radius` is not positive.
        If the component coordinates contain NaN or infinite values.
        If `fractions` values are out of range [0, 1].

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_component.py`

    Notes
    -----
    Calculation of fractions of components from different channels
    or frequencies is not supported yet. Only one set of components
    can be analyzed and is broadcast to all channels/frequencies.

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
    ... )
    array([0, 0, 1, 0, 1, 0], dtype=uint8)

    Count the number of phasors between the combinations of three components:

    >>> phasor_component_graphical(
    ...     [0.4, 0.5],
    ...     [0.2, 0.3],
    ...     [0.0, 0.2, 0.9],
    ...     [0.0, 0.4, 0.3],
    ...     fractions=6,
    ... )
    array([[0, 1, 1, 1, 1, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 1, 2, 0, 0, 0]], dtype=uint8)

    """
    if radius <= 0:
        msg = f'{radius=} <= 0'
        raise ValueError(msg)

    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    component_real = numpy.asarray(component_real)
    component_imag = numpy.asarray(component_imag)
    if (
        real.shape != imag.shape
        or component_real.shape != component_imag.shape
    ):
        msg = 'input array shapes must match'
        raise ValueError(msg)
    if component_real.ndim != 1:
        msg = f'{component_real.ndim=} != 1'
        raise ValueError(msg)
    num_components = len(component_real)
    if num_components not in {2, 3}:
        msg = 'number of components must be 2 or 3'
        raise ValueError(msg)
    if numpy.isnan(component_real).any() or numpy.isnan(component_imag).any():
        msg = 'component coordinates must not contain NaN values'
        raise ValueError(msg)
    if numpy.isinf(component_real).any() or numpy.isinf(component_imag).any():
        msg = 'component coordinates must not contain infinite values'
        raise ValueError(msg)

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
            0.0, 1.0, round(longest_distance / (radius / 2) + 1)
        )
    elif isinstance(fractions, (int, numbers.Integral)):
        if fractions < 0:
            msg = f'{fractions=} < 0'
            raise ValueError(msg)
        fractions = numpy.linspace(0.0, 1.0, fractions)
    else:
        fractions = numpy.asarray(fractions)
        if fractions.ndim != 1:
            msg = 'fractions is not a one-dimensional array'
            raise ValueError(msg)
        if fractions.size > 0 and (
            fractions.min() < 0.0 or fractions.max() > 1.0
        ):
            msg = 'fraction values must be in range [0, 1]'
            raise ValueError(msg)

    dtype = numpy.min_scalar_type(real.size)
    counts = numpy.empty(
        (1 if num_components == 2 else 3, fractions.size), dtype=dtype
    )

    c = 0
    for i in range(num_components):
        a_real = component_real[i]
        a_imag = component_imag[i]
        for j in range(i + 1, num_components):
            b_real = component_real[j]
            b_imag = component_imag[j]
            ab_real = a_real - b_real
            ab_imag = a_imag - b_imag

            for k, f in enumerate(fractions):
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
                counts[c, k] = numpy.sum(mask, dtype=dtype)
            c += 1

    return counts[0] if num_components == 2 else counts


def phasor_component_fit(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    component_real: ArrayLike,
    component_imag: ArrayLike,
    /,
    **kwargs: Any,
) -> NDArray[Any]:
    r"""Return fractions of multiple components from phasor coordinates.

    Component fractions are obtained from the least-squares solution of a
    linear matrix equation that relates phasor coordinates from one or
    multiple harmonics to component fractions according to [2]_.

    Up to ``2 * number harmonics + 1`` components can be fitted to
    multi-harmonic phasor coordinates, that is up to three components
    for single harmonic phasor coordinates.

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
    **kwargs
        Optional arguments passed to :py:func:`scipy.linalg.lstsq`.

    Returns
    -------
    fractions : ndarray
        Component fractions.
        Fractions may not exactly add up to 1.0.

    Raises
    ------
    ValueError
        If the array shapes of `real` and `imag` do not match.
        If the array shapes of `component_real` and `component_imag` do not
        match.
        If the number of harmonics in the components does not match the
        ones in the phasor coordinates.
        If the system is underdetermined; the component matrix having more
        columns than rows.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_component.py`
    :ref:`sphx_glr_tutorials_applications_phasorpy_component_fit.py`

    Notes
    -----
    Calculation of fractions of components from different channels
    or frequencies is not supported yet. Only one set of components
    can be analyzed and is broadcast to all channels/frequencies.

    The method builds a linear matrix equation,
    :math:`A\mathbf{x} = \mathbf{b}`, where :math:`A` consists of the
    phasor coordinates of individual components, :math:`\mathbf{x}` are
    the unknown fractions, and :math:`\mathbf{b}` represents the measured
    phasor coordinates in the mixture. The least-squares solution of this
    linear matrix equation yields the fractions.

    References
    ----------
    .. [2] Vallmitjana A, Lepanto P, Irigoin F, and Malacrida L.
       `Phasor-based multi-harmonic unmixing for in-vivo hyperspectral
       imaging <https://doi.org/10.1088/2050-6120/ac9ae9>`_.
       *Methods Appl Fluoresc*, 11(1): 014001 (2022)

    Examples
    --------
    >>> phasor_component_fit(
    ...     [1, 1, 1], [0.6, 0.5, 0.4], [0.4, 0.3, 0.2], [0.2, 0.9], [0.4, 0.3]
    ... )
    array([[0.4644, 0.5356, 0.6068],
           [0.5559, 0.4441, 0.3322]])

    """
    from scipy.linalg import lstsq

    mean = numpy.atleast_1d(mean)
    real = numpy.atleast_1d(real)
    imag = numpy.atleast_1d(imag)
    component_real = numpy.atleast_1d(component_real)
    component_imag = numpy.atleast_1d(component_imag)

    if real.shape != imag.shape:
        msg = f'{real.shape=} != {imag.shape=}'
        raise ValueError(msg)
    if mean.shape != real.shape[-mean.ndim :]:
        msg = f'{mean.shape=} does not match {real.shape=}'
        raise ValueError(msg)

    if component_real.shape != component_imag.shape:
        msg = f'{component_real.shape=} != {component_imag.shape=}'
        raise ValueError(msg)
    if numpy.isnan(component_real).any() or numpy.isnan(component_imag).any():
        msg = 'component phasor coordinates must not contain NaN values'
        raise ValueError(msg)
    if numpy.isinf(component_real).any() or numpy.isinf(component_imag).any():
        msg = 'component phasor coordinates must not contain infinite values'
        raise ValueError(msg)

    if component_real.ndim == 1:
        component_real = component_real.reshape((1, -1))
        component_imag = component_imag.reshape((1, -1))
    elif component_real.ndim > 2:
        msg = f'{component_real.ndim=} > 2'
        raise ValueError(msg)

    num_harmonics, num_components = component_real.shape

    # create component matrix for least squares solving:
    # [real coordinates of components (for each harmonic)] +
    # [imaginary coordinates of components (for each harmonic)] +
    # [ones for intensity constraint]
    component_matrix = numpy.ones((2 * num_harmonics + 1, num_components))
    component_matrix[:num_harmonics] = component_real
    component_matrix[num_harmonics : 2 * num_harmonics] = component_imag

    if component_matrix.shape[0] < component_matrix.shape[1]:
        msg = (
            'the system is underdetermined '
            f'({num_components=} > {num_harmonics * 2 + 1=})'
        )
        raise ValueError(msg)

    has_harmonic_axis = mean.ndim + 1 == real.ndim
    if not has_harmonic_axis:
        real = numpy.expand_dims(real, axis=0)
        imag = numpy.expand_dims(imag, axis=0)
    elif real.shape[0] != num_harmonics:
        msg = f'{real.shape[0]=} != {component_real.shape[0]=}'
        raise ValueError(msg)

    # TODO: replace Inf with NaN values?
    mean, real, imag = phasor_threshold(mean, real, imag)

    # replace NaN values with 0.0 for least squares solving
    real = numpy.nan_to_num(real, nan=0.0, copy=False)
    imag = numpy.nan_to_num(imag, nan=0.0, copy=False)

    real = numpy.asarray(real)  # for mypy
    imag = numpy.asarray(imag)

    # create coordinates matrix for least squares solving:
    # [real coordinates (for each harmonic)] +
    # [imaginary coordinates (for each harmonic)] +
    # [ones for intensity constraint]
    coords = numpy.ones((2 * num_harmonics + 1, *real.shape[1:]))
    coords[:num_harmonics] = real
    coords[num_harmonics : 2 * num_harmonics] = imag

    fractions = lstsq(
        component_matrix,
        coords.reshape((coords.shape[0], -1)),
        **kwargs,
    )[0]

    # reshape to match input dimensions
    fractions = fractions.reshape((num_components, *coords.shape[1:]))

    # TODO: normalize fractions to sum up to 1.0?
    # fractions /= numpy.sum(fractions, axis=0, keepdims=True)

    # restore NaN values in fractions from mean
    _blend_and(mean, fractions, out=fractions)

    return numpy.asarray(fractions)


@overload
def phasor_component_concentration(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    component_real: ArrayLike,
    component_imag: ArrayLike,
    /,
    reference_mean: float,
    reference_real: float,
    reference_imag: float,
    reference_concentration: float,
    *,
    brightness_ratio: None = ...,
) -> NDArray[Any]: ...


@overload
def phasor_component_concentration(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    component_real: ArrayLike,
    component_imag: ArrayLike,
    /,
    reference_mean: float,
    reference_real: float,
    reference_imag: float,
    reference_concentration: float,
    *,
    brightness_ratio: float,
) -> tuple[NDArray[Any], NDArray[Any]]: ...


def phasor_component_concentration(
    mean: ArrayLike,
    real: ArrayLike,
    imag: ArrayLike,
    component_real: ArrayLike,
    component_imag: ArrayLike,
    /,
    reference_mean: float,
    reference_real: float,
    reference_imag: float,
    reference_concentration: float,
    *,
    brightness_ratio: float | None = None,
) -> NDArray[Any] | tuple[NDArray[Any], NDArray[Any]]:
    r"""Return concentrations of two components from phasor coordinates.

    Calculate the absolute concentration of the first component and,
    optionally, the second component of a two-component system from
    phasor coordinates, using an intensity calibration based on a solution
    of known concentration according to [4]_.

    The algorithm uses geometric line-line intersections in phasor space
    to determine fractional contributions and scale them to absolute
    concentrations.

    Parameters
    ----------
    mean : array_like
        Mean intensity of phasor coordinates.
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    component_real : array_like, shape (2,)
        Real coordinates of the two components.
        The first component is the calibrated component, whose pure solution
        is used as the reference;
    component_imag : array_like, shape (2,)
        Imaginary coordinates of the two components.
    reference_mean : float
        Mean fluorescence intensity of calibration solution.
        The calibration solution must contain only the first component at a
        known concentration.
    reference_real : float
        Real coordinate of calibration solution phasor.
    reference_imag : float
        Imaginary coordinate of calibration solution phasor.
    reference_concentration : float
        Known concentration of calibration solution.
        Same units as the returned concentrations.
    brightness_ratio : float, optional
        Ratio of molecular brightness of second to first component.
        If provided, the second-component concentration is also returned.

    Returns
    -------
    conc_first : ndarray
        Absolute concentration of first component.
    conc_second : ndarray
        Absolute concentration of second component.
        Only returned when `brightness_ratio` is provided.

    Raises
    ------
    ValueError
        If `component_real` values are equal (degenerate component line).
        If `component_real` or `component_imag` do not have shape ``(2,)``.
        If `reference_mean` is zero (cannot normalize).
        If `reference_concentration` is not positive.
        If `brightness_ratio` is not positive.

    See Also
    --------
    phasorpy.component.phasor_component_fraction
    :ref:`sphx_glr_tutorials_applications_phasorpy_nadh_concentration.py`

    Notes
    -----
    The algorithm is based on the ``concandfrac`` procedure from SimFCS and
    the method described in [4]_.
    The implementation has been validated against the published cellular NADH
    concentration, but no independent theoretical or cross-instrument
    validation has been performed.
    Users applying the method to other analytes or instrument platforms
    are encouraged to verify results against known standards.

    The calibration solution must contain only component 0 (no component 1)
    at a precisely known concentration (`reference_concentration`).
    It must be acquired under identical instrument settings as the sample
    (same laser power, detector gain, acquisition time, and objective),
    because the algorithm relates pixel mean intensities directly to the
    calibration mean intensity. The mean intensity of the calibration solution
    should be comparable to the typical pixel mean intensities of the sample.

    The calibration factor :math:`k` relates the component-0 phasor
    position to an absolute concentration:

    .. math::

        k = c_\text{ref} \cdot \frac{g_0 - g_\text{cal}}
            {g_\text{cal}}

    where :math:`g_0` is the real coordinate of the component-0 phasor,
    :math:`g_\text{cal}` is the intersection of the line from :math:`g_0`
    through the origin with the line connecting the component-1 phasor
    scaled by :math:`m=0.5` and the calibration phasor scaled by
    :math:`m=0.5`, and :math:`c_\text{ref}` is the reference concentration.

    For each pixel, a normalized intensity is computed:

    .. math::

        m = \frac{I}{I_\text{ref} + I}

    where :math:`I` is the pixel mean intensity and
    :math:`I_\text{ref} = 2 \cdot \text{reference_mean}` is twice the
    fluorescence-only mean intensity of the calibration solution.

    The component-0 concentration is then:

    .. math::

        c_0 = \left|
            \frac{g_\text{pix} \cdot k}{g_0 - g_\text{pix}} \right|

    where :math:`g_\text{pix}` is the real coordinate of the intersection of
    the line from :math:`g_0` through the origin with the line from the
    intensity-scaled component-1 phasor :math:`g_1 \cdot m` through
    the measured phasor.
    Note: The absolute value in the :math:`c_0` formula is required for phasor
    geometries where :math:`g_\text{pix}` falls outside the interval
    :math:`[0, g_0]` (as is typical for NADH), but has not been validated
    for other analytes or geometries.

    When `brightness_ratio` :math:`\varepsilon = \varepsilon_1 / \varepsilon_0`
    is provided, the component-0 fraction :math:`f_0` at each pixel is
    determined by the intersection :math:`g_\text{frac}` of the line from
    the origin through the measured phasor with the component line:

    .. math::

        f_0 = \frac{g_1 - g_\text{frac}}
            {g_1 - g_0}

    The component-1 concentration and total concentration are then:

    .. math::

        c_1 &= c_0 \cdot
            \frac{1 - f_0}{f_0 \cdot \varepsilon}

        c_\text{total} &= c_0 + c_1

    References
    ----------
    .. [4] Ma N, Digman M A, Malacrida L, and Gratton E.
       `Measurements of absolute concentrations of NADH in cells using
       the phasor FLIM method
       <https://doi.org/10.1364/BOE.7.002441>`_.
       *Biomed Opt Express*, 7(7): 2441-2452 (2016)

    Examples
    --------
    Verify the calibration self-consistency:
    when ``mean = 2 * reference_mean`` (intensity modulation ``m = 0.5``)
    and the pixel phasor equals the reference phasor scaled by 0.5, the
    result equals ``reference_concentration``:

    >>> phasor_component_concentration(
    ...     1000.0, 0.4, 0.05, [0.6, 0.2], [0.1, 0.4], 500.0, 0.8, 0.1, 100.0
    ... )
    array(100)

    """
    component_real = numpy.asarray(component_real, dtype=float)
    component_imag = numpy.asarray(component_imag, dtype=float)
    if component_real.shape != (2,):
        msg = f'{component_real.shape=} != (2,)'
        raise ValueError(msg)
    if component_imag.shape != (2,):
        msg = f'{component_imag.shape=} != (2,)'
        raise ValueError(msg)
    c0_real = float(component_real[0])
    c0_imag = float(component_imag[0])
    c1_real = float(component_real[1])
    c1_imag = float(component_imag[1])
    reference_real = float(reference_real)
    reference_imag = float(reference_imag)
    reference_mean = float(reference_mean)
    reference_concentration = float(reference_concentration)

    if c0_real == c1_real:
        msg = 'component_real values must differ'
        raise ValueError(msg)
    if reference_mean == 0.0:
        msg = 'reference_mean must not be zero'
        raise ValueError(msg)
    if reference_concentration <= 0.0:
        msg = f'{reference_concentration=} is not positive'
        raise ValueError(msg)
    if brightness_ratio is not None and brightness_ratio <= 0.0:
        msg = f'{brightness_ratio=} is not positive'
        raise ValueError(msg)

    # line(c0 -> origin) X line(c1/2 -> cal/2)
    # the raw reference phasor is scaled by 0.5 (m=0.5 condition)
    g_cal, _ = _intersect_line_line(
        c0_real,
        c0_imag,
        0.0,
        0.0,
        c1_real * 0.5,
        c1_imag * 0.5,
        reference_real * 0.5,
        reference_imag * 0.5,
    )
    # calibration factor
    k = reference_concentration * (c0_real - g_cal) / g_cal

    # TODO: preserve float32
    mean = numpy.asarray(mean, dtype=numpy.float64)
    real = numpy.asarray(real, dtype=numpy.float64)
    imag = numpy.asarray(imag, dtype=numpy.float64)

    # normalized intensity
    m = mean / ((2.0 * reference_mean) + mean)

    # scale second-component phasor by normalized intensity
    c1_real_scaled = c1_real * m
    c1_imag_scaled = c1_imag * m

    # line(c0 -> origin) X line(scaled_c1 -> data)
    g_pix, _ = _intersect_line_line(
        c0_real,
        c0_imag,
        0.0,
        0.0,
        c1_real_scaled,
        c1_imag_scaled,
        real,
        imag,
    )

    # absolute concentration of first component
    # abs() matches the SimFCS implementation and is required when g_pix
    # falls outside [0, c0_real]
    with numpy.errstate(divide='ignore', invalid='ignore'):
        conc_first = numpy.asarray(numpy.abs(g_pix * k / (c0_real - g_pix)))

    if brightness_ratio is None:
        return conc_first

    # first-component fraction from line(origin -> data) X line(c0 -> c1)
    # this matches the SimFCS implementation; alternatively, the closest point
    # on the component line to the pixel phasor could be used
    g_frac, _ = _intersect_line_line(
        c0_real,
        c0_imag,
        c1_real,
        c1_imag,
        0.0,
        0.0,
        real,
        imag,
    )
    with numpy.errstate(divide='ignore', invalid='ignore'):
        frac_first = numpy.asarray(
            1.0 - (g_frac - c0_real) / (c1_real - c0_real)
        )
        frac_second = 1.0 - frac_first
        conc_second = numpy.asarray(
            conc_first * frac_second / (frac_first * brightness_ratio)
        )

    return conc_first, conc_second


def phasor_component_mvc(
    real: ArrayLike,
    imag: ArrayLike,
    component_real: ArrayLike,
    component_imag: ArrayLike,
    /,
    *,
    dtype: DTypeLike | None = None,
    num_threads: int | None = None,
) -> NDArray[Any]:
    """Return mean value coordinates of phasor coordinates from components.

    The mean value coordinates of phasor coordinates with respect to three or
    more components spanning an arbitrary simple polygon are computed using
    the stable method described in [3]_.
    For three components, mean value coordinates are equivalent to
    barycentric coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    component_real : array_like
        Real coordinates of at least three components.
    component_imag : array_like
        Imaginary coordinates of at least three components.
    dtype : dtype_like, optional
        Floating point data type used for calculation and output values.
        Either `float32` or `float64`. The default is `float64`.
    num_threads : int, optional
        Number of OpenMP threads to use for parallelization.
        By default, multithreading is disabled.
        If zero, up to half of logical CPUs are used.
        OpenMP may not be available on all platforms.

    Returns
    -------
    fractions : ndarray
        Mean value coordinates for each phasor coordinate.

    Raises
    ------
    ValueError
        If the array shapes of `real` and `imag` do not match.
        If the array shapes of `component_real` and `component_imag` do not
        match.

    Notes
    -----
    Calculation of fractions of components from different channels
    or frequencies is not supported yet. Only one set of components
    can be analyzed and is broadcast to all channels/frequencies.

    For three components, this function returns the same result as
    :py:func:`phasor_component_fit`. For more than three components,
    the system is underdetermined and the mean value coordinates represent
    one of multiple solutions. However, the special properties of the mean
    value coordinates make them particularly useful for interpolating and
    visualizing multi-component data.

    References
    ----------
    .. [3] Fuda C and Hormann K.
       `A new stable method to compute mean value coordinates
       <https://doi.org/10.1016/j.cagd.2024.102310>`_.
       *Computer Aided Geometric Design*, 111: 102310 (2024)

    Examples
    --------
    Calculate the barycentric coordinates of a phasor coordinate
    in a triangle defined by three components:

    >>> phasor_component_mvc(0.6, 0.3, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    array([0.3, 0.6, 0.1])

    The barycentric coordinates of phasor coordinates outside the polygon
    defined by the components may be outside the range [0.0, 1.0]:

    >>> phasor_component_mvc(0.6, 0.6, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    array([0.6, 0.6, -0.2])

    """
    num_threads = number_threads(num_threads)

    dtype = numpy.dtype(dtype)
    if dtype.char not in {'f', 'd'}:
        msg = f'{dtype=} is not a floating-point type'
        raise ValueError(msg)

    real = numpy.ascontiguousarray(real, dtype=dtype)
    imag = numpy.ascontiguousarray(imag, dtype=dtype)
    component_real = numpy.ascontiguousarray(component_real, dtype=dtype)
    component_imag = numpy.ascontiguousarray(component_imag, dtype=dtype)

    if real.shape != imag.shape:
        msg = f'{real.shape=} != {imag.shape=}'
        raise ValueError(msg)
    if component_real.shape != component_imag.shape:
        msg = f'{component_real.shape=} != {component_imag.shape=}'
        raise ValueError(msg)
    if component_real.ndim != 1 or component_real.size < 3:
        msg = 'number of components must be three or more'
        raise ValueError(msg)
    if numpy.isnan(component_real).any() or numpy.isnan(component_imag).any():
        msg = 'component coordinates must not contain NaN values'
        raise ValueError(msg)
    if numpy.isinf(component_real).any() or numpy.isinf(component_imag).any():
        msg = 'component coordinates must not contain infinite values'
        raise ValueError(msg)

    # TODO: sorting not strictly required for three components?
    component_real, component_imag, indices = sort_coordinates(
        component_real, component_imag
    )

    shape = real.shape
    real = real.reshape(-1)
    imag = imag.reshape(-1)
    fraction = numpy.zeros((component_real.size, real.size), dtype=dtype)

    _mean_value_coordinates(
        fraction,
        indices,
        real,
        imag,
        component_real,
        component_imag,
        num_threads,
    )

    return numpy.asarray(fraction.reshape((-1, *shape)).squeeze())
