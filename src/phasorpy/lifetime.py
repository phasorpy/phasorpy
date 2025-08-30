"""Calculate, convert, and calibrate phasor coordinates of lifetimes.

The ``phasorpy.lifetime`` module provides functions to:

- synthesize time and frequency domain signals from fluorescence lifetimes:

  - :py:func:`lifetime_to_signal`

- convert between phasor coordinates and single- or multi-component
  fluorescence lifetimes:

  - :py:func:`phasor_from_lifetime`
  - :py:func:`phasor_from_apparent_lifetime`
  - :py:func:`phasor_to_apparent_lifetime`
  - :py:func:`phasor_to_normal_lifetime`
  - :py:func:`phasor_to_lifetime_search`

- convert to and from polar coordinates (phase and modulation):

  - :py:func:`polar_from_apparent_lifetime`
  - :py:func:`polar_to_apparent_lifetime`

- calibrate phasor coordinates with a reference of known fluorescence
  lifetime:

  - :py:func:`phasor_calibrate`
  - :py:func:`polar_from_reference`
  - :py:func:`polar_from_reference_phasor`

- calculate phasor coordinates for FRET donor and acceptor channels:

  - :py:func:`phasor_from_fret_donor`
  - :py:func:`phasor_from_fret_acceptor`

- convert between single component lifetimes and optimal frequency:

  - :py:func:`lifetime_to_frequency`
  - :py:func:`lifetime_from_frequency`

- convert between fractional intensities and pre-exponential amplitudes:

  - :py:func:`lifetime_fraction_from_amplitude`
  - :py:func:`lifetime_fraction_to_amplitude`

- calculate phasor coordinates on the universal semicircle:

  - :py:func:`phasor_semicircle`
  - :py:func:`phasor_semicircle_intersect`
  - :py:func:`phasor_at_harmonic`

"""

from __future__ import annotations

__all__ = [
    'lifetime_fraction_from_amplitude',
    'lifetime_fraction_to_amplitude',
    'lifetime_from_frequency',
    'lifetime_to_frequency',
    'lifetime_to_signal',
    'phasor_at_harmonic',
    'phasor_calibrate',
    'phasor_from_apparent_lifetime',
    'phasor_from_fret_acceptor',
    'phasor_from_fret_donor',
    'phasor_from_lifetime',
    'phasor_semicircle',
    'phasor_semicircle_intersect',
    'phasor_to_apparent_lifetime',
    'phasor_to_lifetime_search',
    'phasor_to_normal_lifetime',
    'polar_from_apparent_lifetime',
    'polar_from_reference',
    'polar_from_reference_phasor',
    'polar_to_apparent_lifetime',
]

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, NDArray, ArrayLike, DTypeLike, Literal

import numpy

from ._phasorpy import (
    _gaussian_signal,
    _intersect_semicircle_line,
    _lifetime_search_2,
    _phasor_at_harmonic,
    _phasor_from_apparent_lifetime,
    _phasor_from_fret_acceptor,
    _phasor_from_fret_donor,
    _phasor_from_lifetime,
    _phasor_from_single_lifetime,
    _phasor_to_apparent_lifetime,
    _phasor_to_normal_lifetime,
    _polar_from_apparent_lifetime,
    _polar_from_reference,
    _polar_from_reference_phasor,
    _polar_from_single_lifetime,
    _polar_to_apparent_lifetime,
)
from ._utils import parse_harmonic, parse_skip_axis
from .phasor import (
    phasor_center,
    phasor_from_signal,
    phasor_multiply,
    phasor_to_signal,
    phasor_transform,
)
from .utils import number_threads


def phasor_from_lifetime(
    frequency: ArrayLike,
    lifetime: ArrayLike,
    fraction: ArrayLike | None = None,
    *,
    preexponential: bool = False,
    unit_conversion: float = 1e-3,
    keepdims: bool = False,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    r"""Return phasor coordinates from lifetime components.

    Calculate phasor coordinates as a function of frequency, single or
    multiple lifetime components, and the pre-exponential amplitudes
    or fractional intensities of the components.

    Parameters
    ----------
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
        A scalar or one-dimensional sequence.
    lifetime : array_like
        Lifetime components in ns. See notes below for allowed dimensions.
    fraction : array_like, optional
        Fractional intensities or pre-exponential amplitudes of the lifetime
        components. Fractions are normalized to sum to 1.
        See notes below for allowed dimensions.
    preexponential : bool, optional, default: False
        If true, `fraction` values are pre-exponential amplitudes,
        else fractional intensities.
    unit_conversion : float, optional, default: 1e-3
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    keepdims : bool, optional, default: False
        If true, length-one dimensions are left in phasor coordinates.

    Returns
    -------
    real : ndarray
        Real component of phasor coordinates.
    imag : ndarray
        Imaginary component of phasor coordinates.

        See notes below for dimensions of the returned arrays.

    Raises
    ------
    ValueError
        Input arrays exceed their allowed dimensionality or do not match.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_phasor_from_lifetime.py`
    :ref:`sphx_glr_tutorials_phasorpy_lifetime_geometry.py`

    Notes
    -----
    The phasor coordinates :math:`G` (`real`) and :math:`S` (`imag`) for
    many lifetime components :math:`j` with lifetimes :math:`\tau` and
    pre-exponential amplitudes :math:`\alpha` at frequency :math:`f` are:

    .. math::

        \omega &= 2 \pi f

        g_{j} &= \alpha_{j} / (1 + (\omega \tau_{j})^2)

        G &= \sum_{j} g_{j}

        S &= \sum_{j} \omega \tau_{j} g_{j}

    The relation between pre-exponential amplitudes :math:`a` and
    fractional intensities :math:`\alpha` is:

    .. math::
        F_{DC} &= \sum_{j} a_{j} \tau_{j}

        \alpha_{j} &= a_{j} \tau_{j} / F_{DC}

    The following combinations of `lifetime` and `fraction` parameters are
    supported:

    - `lifetime` is scalar or one-dimensional, holding single component
      lifetimes. `fraction` is None.
      Return arrays of shape `(frequency.size, lifetime.size)`.

    - `lifetime` is two-dimensional, `fraction` is one-dimensional.
      The last dimensions match in size, holding lifetime components and
      their fractions.
      Return arrays of shape `(frequency.size, lifetime.shape[1])`.

    - `lifetime` is one-dimensional, `fraction` is two-dimensional.
      The last dimensions must match in size, holding lifetime components and
      their fractions.
      Return arrays of shape `(frequency.size, fraction.shape[1])`.

    - `lifetime` and `fraction` are up to two-dimensional of same shape.
      The last dimensions hold lifetime components and their fractions.
      Return arrays of shape `(frequency.size, lifetime.shape[0])`.

    Length-one dimensions are removed from returned arrays
    if `keepdims` is false (default).

    Examples
    --------
    Phasor coordinates of a single lifetime component (in ns) at a
    frequency of 80 MHz:

    >>> phasor_from_lifetime(80.0, 1.9894368)  # doctest: +NUMBER
    (0.5, 0.5)

    Phasor coordinates of two lifetime components with equal fractional
    intensities:

    >>> phasor_from_lifetime(
    ...     80.0, [3.9788735, 0.9947183], [0.5, 0.5]
    ... )  # doctest: +NUMBER
    (0.5, 0.4)

    Phasor coordinates of two lifetime components with equal pre-exponential
    amplitudes:

    >>> phasor_from_lifetime(
    ...     80.0, [3.9788735, 0.9947183], [0.5, 0.5], preexponential=True
    ... )  # doctest: +NUMBER
    (0.32, 0.4)

    Phasor coordinates of many single-component lifetimes (fractions omitted):

    >>> phasor_from_lifetime(
    ...     80.0, [3.9788735, 1.9894368, 0.9947183]
    ... )  # doctest: +NUMBER
    (array([0.2, 0.5, 0.8]), array([0.4, 0.5, 0.4]))

    Phasor coordinates of two lifetime components with varying fractions:

    >>> phasor_from_lifetime(
    ...     80.0, [3.9788735, 0.9947183], [[1, 0], [0.5, 0.5], [0, 1]]
    ... )  # doctest: +NUMBER
    (array([0.2, 0.5, 0.8]), array([0.4, 0.4, 0.4]))

    Phasor coordinates of multiple two-component lifetimes with constant
    fractions, keeping dimensions:

    >>> phasor_from_lifetime(
    ...     80.0, [[3.9788735, 0.9947183], [1.9894368, 1.9894368]], [0.5, 0.5]
    ... )  # doctest: +NUMBER
    (array([0.5, 0.5]), array([0.4, 0.5]))

    Phasor coordinates of multiple two-component lifetimes with specific
    fractions at multiple frequencies. Frequencies are in Hz, lifetimes in ns:

    >>> phasor_from_lifetime(
    ...     [40e6, 80e6],
    ...     [[1e-9, 0.9947183e-9], [3.9788735e-9, 0.9947183e-9]],
    ...     [[0, 1], [0.5, 0.5]],
    ...     unit_conversion=1.0,
    ... )  # doctest: +NUMBER
    (array([[0.941, 0.721], [0.8, 0.5]]), array([[0.235, 0.368], [0.4, 0.4]]))

    """
    if unit_conversion < 1e-16:
        raise ValueError(f'{unit_conversion=} < 1e-16')
    frequency = numpy.array(
        frequency, dtype=numpy.float64, ndmin=1, order='C', copy=None
    )
    if frequency.ndim != 1:
        raise ValueError('frequency is not one-dimensional array')
    lifetime = numpy.array(
        lifetime, dtype=numpy.float64, ndmin=1, order='C', copy=None
    )
    if lifetime.ndim > 2:
        raise ValueError('lifetime must be one- or two-dimensional array')

    if fraction is None:
        # single-component lifetimes
        if lifetime.ndim > 1:
            raise ValueError(
                'lifetime must be one-dimensional array if fraction is None'
            )
        lifetime = lifetime.reshape(-1, 1)  # move components to last axis
        fraction = numpy.ones_like(lifetime)  # not really used
    else:
        fraction = numpy.array(
            fraction, dtype=numpy.float64, ndmin=1, order='C', copy=None
        )
        if fraction.ndim > 2:
            raise ValueError('fraction must be one- or two-dimensional array')

    if lifetime.ndim == 1 and fraction.ndim == 1:
        # one multi-component lifetime
        if lifetime.shape != fraction.shape:
            raise ValueError(
                f'{lifetime.shape=} does not match {fraction.shape=}'
            )
        lifetime = lifetime.reshape(1, -1)
        fraction = fraction.reshape(1, -1)
        nvar = 1
    elif lifetime.ndim == 2 and fraction.ndim == 2:
        # multiple, multi-component lifetimes
        if lifetime.shape[1] != fraction.shape[1]:
            raise ValueError(f'{lifetime.shape[1]=} != {fraction.shape[1]=}')
        nvar = lifetime.shape[0]
    elif lifetime.ndim == 2 and fraction.ndim == 1:
        # variable components, same fractions
        fraction = fraction.reshape(1, -1)
        nvar = lifetime.shape[0]
    elif lifetime.ndim == 1 and fraction.ndim == 2:
        # same components, varying fractions
        lifetime = lifetime.reshape(1, -1)
        nvar = fraction.shape[0]
    else:
        # unreachable code
        raise RuntimeError(f'{lifetime.shape=}, {fraction.shape=}')

    phasor = numpy.empty((2, frequency.size, nvar), dtype=numpy.float64)

    _phasor_from_lifetime(
        phasor, frequency, lifetime, fraction, unit_conversion, preexponential
    )

    if not keepdims:
        phasor = phasor.squeeze()
    return phasor[0], phasor[1]


def lifetime_to_signal(
    frequency: float,
    lifetime: ArrayLike,
    fraction: ArrayLike | None = None,
    *,
    mean: ArrayLike | None = None,
    background: ArrayLike | None = None,
    samples: int = 64,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
    zero_phase: float | None = None,
    zero_stdev: float | None = None,
    preexponential: bool = False,
    unit_conversion: float = 1e-3,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    r"""Return synthetic signal from lifetime components.

    Return synthetic signal, instrument response function (IRF), and
    time axis, sampled over one period of the fundamental frequency.
    The signal is convolved with the IRF, which is approximated by a
    normal distribution.

    Parameters
    ----------
    frequency : float
        Fundamental laser pulse or modulation frequency in MHz.
    lifetime : array_like
        Lifetime components in ns.
    fraction : array_like, optional
        Fractional intensities or pre-exponential amplitudes of the lifetime
        components. Fractions are normalized to sum to 1.
        Must be specified if `lifetime` is not a scalar.
    mean : array_like, optional, default: 1.0
        Average signal intensity (DC). Must be scalar for now.
    background : array_like, optional, default: 0.0
        Background signal intensity. Must be smaller than `mean`.
    samples : int, default: 64
        Number of signal samples to return. Must be at least 16.
    harmonic : int, sequence of int, or 'all', optional, default: 'all'
        Harmonics used to synthesize signal.
        If `'all'`, all harmonics are used.
        Else, harmonics must be at least one and no larger than half of
        `samples`.
        Use `'all'` to synthesize an exponential time-domain decay signal,
        or `1` to synthesize a homodyne signal.
    zero_phase : float, optional
        Position of instrument response function in radians.
        Must be in range [0, pi]. The default is the 8th sample.
    zero_stdev : float, optional
        Standard deviation of instrument response function in radians.
        Must be at least 1.5 samples and no more than one tenth of samples
        to allow for sufficient sampling of the function.
        The default is 1.5 samples. Increase `samples` to narrow the IRF.
    preexponential : bool, optional, default: False
        If true, `fraction` values are pre-exponential amplitudes,
        else fractional intensities.
    unit_conversion : float, optional, default: 1e-3
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.

    Returns
    -------
    signal : ndarray
        Signal generated from lifetimes at frequency, convolved with
        instrument response function.
    zero : ndarray
        Instrument response function.
    time : ndarray
        Time for each sample in signal in units of `lifetime`.

    See Also
    --------
    phasorpy.lifetime.phasor_from_lifetime
    phasorpy.phasor.phasor_to_signal
    :ref:`sphx_glr_tutorials_api_phasorpy_lifetime_to_signal.py`

    Notes
    -----
    This implementation is based on an inverse discrete Fourier transform
    (DFT). Because DFT cannot be used on signals with discontinuities
    (for example, an exponential decay starting at zero) without producing
    strong artifacts (ripples), the signal is convolved with a continuous
    instrument response function (IRF). The minimum width of the IRF is
    limited due to sampling requirements.

    Examples
    --------
    Synthesize a multi-exponential time-domain decay signal for two
    lifetime components of 4.2 and 0.9 ns at 40 MHz:

    >>> signal, zero, times = lifetime_to_signal(
    ...     40, [4.2, 0.9], fraction=[0.8, 0.2], samples=16
    ... )
    >>> signal  # doctest: +NUMBER
    array([0.2846, 0.1961, 0.1354, ..., 0.8874, 0.6029, 0.4135])

    Synthesize a homodyne frequency-domain waveform signal for
    a single lifetime:

    >>> signal, zero, times = lifetime_to_signal(
    ...     40.0, 4.2, samples=16, harmonic=1
    ... )
    >>> signal  # doctest: +NUMBER
    array([0.2047, -0.05602, -0.156, ..., 1.471, 1.031, 0.5865])

    """
    if harmonic is None:
        harmonic = 'all'
    all_harmonics = harmonic == 'all'
    harmonic, _ = parse_harmonic(harmonic, samples // 2)

    if samples < 16:
        raise ValueError(f'{samples=} < 16')

    if background is None:
        background = 0.0
    background = numpy.asarray(background)

    if mean is None:
        mean = 1.0
    mean = numpy.asarray(mean)
    mean -= background
    if numpy.any(mean < 0.0):
        raise ValueError('mean - background must not be less than zero')

    scale = samples / (2.0 * math.pi)
    if zero_phase is None:
        zero_phase = 8.0 / scale
    phase = zero_phase * scale  # in sample units
    if zero_stdev is None:
        zero_stdev = 1.5 / scale
    stdev = zero_stdev * scale  # in sample units

    if zero_phase < 0 or zero_phase > 2.0 * math.pi:
        raise ValueError(f'{zero_phase=} out of range [0, 2 pi]')
    if stdev < 1.5:
        raise ValueError(
            f'{zero_stdev=} < {1.5 / scale} cannot be sampled sufficiently'
        )
    if stdev >= samples / 10:
        raise ValueError(f'{zero_stdev=} > pi / 5 not supported')

    frequencies = numpy.atleast_1d(frequency)
    if frequencies.size > 1 or frequencies[0] <= 0.0:
        raise ValueError('frequency must be scalar and positive')
    frequencies = numpy.linspace(
        frequency, samples // 2 * frequency, samples // 2
    )
    frequencies = frequencies[[h - 1 for h in harmonic]]

    real, imag = phasor_from_lifetime(
        frequencies,
        lifetime,
        fraction,
        preexponential=preexponential,
        unit_conversion=unit_conversion,
    )
    real, imag = numpy.atleast_1d(real, imag)

    zero = numpy.zeros(samples, dtype=numpy.float64)
    _gaussian_signal(zero, phase, stdev)
    zero_mean, zero_real, zero_imag = phasor_from_signal(
        zero, harmonic=harmonic
    )
    if real.ndim > 1:
        # make broadcastable with real and imag
        zero_real = zero_real[:, None]
        zero_imag = zero_imag[:, None]
    if not all_harmonics:
        zero = phasor_to_signal(
            zero_mean, zero_real, zero_imag, samples=samples, harmonic=harmonic
        )

    phasor_multiply(real, imag, zero_real, zero_imag, out=(real, imag))

    if len(harmonic) == 1:
        harmonic = harmonic[0]
    signal = phasor_to_signal(
        mean, real, imag, samples=samples, harmonic=harmonic
    )
    signal += numpy.asarray(background)

    time = numpy.linspace(0, 1.0 / (unit_conversion * frequency), samples)

    return signal.squeeze(), zero.squeeze(), time


def phasor_calibrate(
    real: ArrayLike,
    imag: ArrayLike,
    reference_mean: ArrayLike,
    reference_real: ArrayLike,
    reference_imag: ArrayLike,
    /,
    frequency: ArrayLike,
    lifetime: ArrayLike,
    *,
    harmonic: int | Sequence[int] | Literal['all'] | str | None = None,
    skip_axis: int | Sequence[int] | None = None,
    fraction: ArrayLike | None = None,
    preexponential: bool = False,
    unit_conversion: float = 1e-3,
    method: Literal['mean', 'median'] = 'mean',
    nan_safe: bool = True,
    reverse: bool = False,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return calibrated/referenced phasor coordinates.

    Calibration of phasor coordinates from time-resolved measurements is
    necessary to account for the instrument response function (IRF) and delays
    in the electronics.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates to be calibrated.
    imag : array_like
        Imaginary component of phasor coordinates to be calibrated.
    reference_mean : array_like or None
        Intensity of phasor coordinates from reference of known lifetime.
        Used to re-normalize averaged phasor coordinates.
    reference_real : array_like
        Real component of phasor coordinates from reference of known lifetime.
        Must be measured with the same instrument setting as the phasor
        coordinates to be calibrated. Dimensions must be the same as `real`.
    reference_imag : array_like
        Imaginary component of phasor coordinates from reference of known
        lifetime.
        Must be measured with the same instrument setting as the phasor
        coordinates to be calibrated.
    frequency : array_like
        Fundamental laser pulse or modulation frequency in MHz.
    lifetime : array_like
        Lifetime components in ns. Must be scalar or one-dimensional.
    harmonic : int, sequence of int, or 'all', default: 1
        Harmonics included in `real` and `imag`.
        If an integer, the harmonics at which `real` and `imag` were acquired
        or calculated.
        If a sequence, the harmonics included in the first axis of `real` and
        `imag`.
        If `'all'`, the first axis of `real` and `imag` contains lower
        harmonics.
        The default is the first harmonic (fundamental frequency).
    skip_axis : int or sequence of int, optional
        Axes in `reference_mean` to exclude from reference center calculation.
        By default, all axes except harmonics are included.
    fraction : array_like, optional
        Fractional intensities or pre-exponential amplitudes of the lifetime
        components. Fractions are normalized to sum to 1.
        Must be same size as `lifetime`.
    preexponential : bool, optional
        If true, `fraction` values are pre-exponential amplitudes,
        else fractional intensities (default).
    unit_conversion : float, optional
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    method : str, optional
        Method used for calculating center of reference phasor coordinates:

        - ``'mean'``: Arithmetic mean.
        - ``'median'``: Spatial median.

    nan_safe : bool, optional
        Ensure `method` is applied to same elements of reference arrays.
        By default, distribute NaNs among reference arrays before applying
        `method`.
    reverse : bool, optional
        Reverse calibration.

    Returns
    -------
    real : ndarray
        Calibrated real component of phasor coordinates.
    imag : ndarray
        Calibrated imaginary component of phasor coordinates.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag`, or `reference_real` and
        `reference_imag` do not match.
        Number of harmonics or frequencies does not match the first axis
        of `real` and `imag`.

    See Also
    --------
    phasorpy.phasor.phasor_transform
    phasorpy.phasor.phasor_center
    phasorpy.lifetime.polar_from_reference_phasor
    phasorpy.lifetime.phasor_from_lifetime

    Notes
    -----
    This function is a convenience wrapper for the following operations:

    .. code-block:: python

        phasor_transform(
            real,
            imag,
            *polar_from_reference_phasor(
                *phasor_center(
                    reference_mean,
                    reference_real,
                    reference_imag,
                    skip_axis,
                    method,
                    nan_safe,
                )[1:],
                *phasor_from_lifetime(
                    frequency,
                    lifetime,
                    fraction,
                    preexponential,
                    unit_conversion,
                ),
            ),
        )

    Calibration can be reversed such that

    .. code-block:: python

        real, imag == phasor_calibrate(
            *phasor_calibrate(real, imag, *args, **kwargs),
            *args,
            reverse=True,
            **kwargs
        )

    Examples
    --------
    >>> phasor_calibrate(
    ...     [0.1, 0.2, 0.3],
    ...     [0.4, 0.5, 0.6],
    ...     [1.0, 1.0, 1.0],
    ...     [0.2, 0.3, 0.4],
    ...     [0.5, 0.6, 0.7],
    ...     frequency=80,
    ...     lifetime=4,
    ... )  # doctest: +NUMBER
    (array([0.0658, 0.132, 0.198]), array([0.2657, 0.332, 0.399]))

    Undo the previous calibration:

    >>> phasor_calibrate(
    ...     [0.0658, 0.132, 0.198],
    ...     [0.2657, 0.332, 0.399],
    ...     [1.0, 1.0, 1.0],
    ...     [0.2, 0.3, 0.4],
    ...     [0.5, 0.6, 0.7],
    ...     frequency=80,
    ...     lifetime=4,
    ...     reverse=True,
    ... )  # doctest: +NUMBER
    (array([0.1, 0.2, 0.3]), array([0.4, 0.5, 0.6]))

    """
    real = numpy.asarray(real)
    imag = numpy.asarray(imag)
    reference_mean = numpy.asarray(reference_mean)
    reference_real = numpy.asarray(reference_real)
    reference_imag = numpy.asarray(reference_imag)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if reference_real.shape != reference_imag.shape:
        raise ValueError(f'{reference_real.shape=} != {reference_imag.shape=}')

    has_harmonic_axis = reference_mean.ndim + 1 == reference_real.ndim
    harmonic, _ = parse_harmonic(
        harmonic,
        (
            reference_real.shape[0]
            if has_harmonic_axis
            and isinstance(harmonic, str)
            and harmonic == 'all'
            else None
        ),
    )

    frequency = numpy.asarray(frequency)
    frequency = frequency * harmonic

    if has_harmonic_axis:
        if real.ndim == 0:
            raise ValueError(
                f'{real.shape=} != {len(frequency)} frequencies or harmonics'
            )
        if real.shape[0] != len(frequency):
            raise ValueError(
                f'{real.shape[0]=} != {len(frequency)} '
                'frequencies or harmonics'
            )
        if reference_real.shape[0] != len(frequency):
            raise ValueError(
                f'{reference_real.shape[0]=} != {len(frequency)} '
                'frequencies or harmonics'
            )
        if reference_mean.shape != reference_real.shape[1:]:
            raise ValueError(
                f'{reference_mean.shape=} != {reference_real.shape[1:]=}'
            )
    elif reference_mean.shape != reference_real.shape:
        raise ValueError(f'{reference_mean.shape=} != {reference_real.shape=}')
    elif len(harmonic) > 1:
        raise ValueError(
            f'{reference_mean.shape=} does not have harmonic axis'
        )

    _, measured_re, measured_im = phasor_center(
        reference_mean,
        reference_real,
        reference_imag,
        skip_axis=skip_axis,
        method=method,
        nan_safe=nan_safe,
    )

    known_re, known_im = phasor_from_lifetime(
        frequency,
        lifetime,
        fraction,
        preexponential=preexponential,
        unit_conversion=unit_conversion,
    )

    skip_axis, axis = parse_skip_axis(
        skip_axis, real.ndim - int(has_harmonic_axis), has_harmonic_axis
    )

    if has_harmonic_axis and any(skip_axis):
        known_re = numpy.expand_dims(
            known_re, tuple(range(1, measured_re.ndim))
        )
        known_re = numpy.broadcast_to(
            known_re, (len(frequency), *measured_re.shape[1:])
        )
        known_im = numpy.expand_dims(
            known_im, tuple(range(1, measured_im.ndim))
        )
        known_im = numpy.broadcast_to(
            known_im, (len(frequency), *measured_im.shape[1:])
        )

    phi_zero, mod_zero = polar_from_reference_phasor(
        measured_re, measured_im, known_re, known_im
    )

    if numpy.ndim(phi_zero) > 0:
        if reverse:
            numpy.negative(phi_zero, out=phi_zero)
            numpy.reciprocal(mod_zero, out=mod_zero)
        if axis is not None:
            phi_zero = numpy.expand_dims(phi_zero, axis=axis)
            mod_zero = numpy.expand_dims(mod_zero, axis=axis)
    elif reverse:
        phi_zero = -phi_zero
        mod_zero = 1.0 / mod_zero

    return phasor_transform(real, imag, phi_zero, mod_zero)


def polar_from_reference_phasor(
    measured_real: ArrayLike,
    measured_imag: ArrayLike,
    known_real: ArrayLike,
    known_imag: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return polar coordinates for calibration from reference phasor.

    Return rotation angle and scale factor for calibrating phasor coordinates
    from measured and known phasor coordinates of a reference, for example,
    a sample of known lifetime.

    Parameters
    ----------
    measured_real : array_like
        Real component of measured phasor coordinates.
    measured_imag : array_like
        Imaginary component of measured phasor coordinates.
    known_real : array_like
        Real component of reference phasor coordinates.
    known_imag : array_like
        Imaginary component of reference phasor coordinates.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    phase_zero : ndarray
        Angular component of polar coordinates for calibration in radians.
    modulation_zero : ndarray
        Radial component of polar coordinates for calibration.

    See Also
    --------
    phasorpy.lifetime.polar_from_reference

    Notes
    -----
    This function performs the following operations:

    .. code-block:: python

        polar_from_reference(
            *phasor_to_polar(measured_real, measured_imag),
            *phasor_to_polar(known_real, known_imag),
        )

    Examples
    --------
    >>> polar_from_reference_phasor(0.5, 0.0, 1.0, 0.0)
    (0.0, 2.0)

    """
    return _polar_from_reference_phasor(  # type: ignore[no-any-return]
        measured_real, measured_imag, known_real, known_imag, **kwargs
    )


def polar_from_reference(
    measured_phase: ArrayLike,
    measured_modulation: ArrayLike,
    known_phase: ArrayLike,
    known_modulation: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return polar coordinates for calibration from reference coordinates.

    Return rotation angle and scale factor for calibrating phasor coordinates
    from measured and known polar coordinates of a reference, for example,
    a sample of known lifetime.

    Parameters
    ----------
    measured_phase : array_like
        Angular component of measured polar coordinates in radians.
    measured_modulation : array_like
        Radial component of measured polar coordinates.
    known_phase : array_like
        Angular component of reference polar coordinates in radians.
    known_modulation : array_like
        Radial component of reference polar coordinates.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    phase_zero : ndarray
        Angular component of polar coordinates for calibration in radians.
    modulation_zero : ndarray
        Radial component of polar coordinates for calibration.

    See Also
    --------
    phasorpy.lifetime.polar_from_reference_phasor

    Examples
    --------
    >>> polar_from_reference(0.2, 0.4, 0.4, 1.3)
    (0.2, 3.25)

    """
    return _polar_from_reference(  # type: ignore[no-any-return]
        measured_phase,
        measured_modulation,
        known_phase,
        known_modulation,
        **kwargs,
    )


def phasor_to_lifetime_search(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    frequency: float,
    *,
    lifetime_range: tuple[float, float, float] | None = None,
    unit_conversion: float = 1e-3,
    dtype: DTypeLike = None,
    num_threads: int | None = None,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return two lifetime components from multi-harmonic phasor coordinates.

    Return estimated lifetimes and fractional intensities of two
    single-exponential components from a set of multi-harmonic
    phasor coordinates using the graphical approach described
    in [1]_.

    Return NaN for coordinates outside the universal semicircle.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
        Must contain at least two linearly increasing harmonics
        in the first dimension.
    imag : array_like
        Imaginary component of phasor coordinates.
        Must have same shape as `real`.
    frequency : float
        Laser pulse or modulation frequency in MHz.
    lifetime_range : tuple of float, optional
        Start, stop, and step of lifetime range in ns to search for components.
        Defines the search range for the first lifetime component.
        The default is ``(0.0, 20.0, 0.1)``.
    unit_conversion : float, optional, default: 1e-3
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    dtype : dtype_like, optional
        Floating point data type used for calculation and output values.
        Either `float32` or `float64`. The default is `float64`.
    num_threads : int, optional
        Number of OpenMP threads to use for parallelization.
        By default, multi-threading is disabled.
        If zero, up to half of logical CPUs are used.
        OpenMP may not be available on all platforms.

    Returns
    -------
    lifetime : ndarray
        Lifetime components, shaped ``(2, *real.shape[1:])``.
    fraction : ndarray
        Fractional intensities of resolved lifetime components.

    Raises
    ------
    ValueError
        The shapes of real and imaginary coordinates do not match.
        The number of harmonics is less than the number of components.
        The lifetime range is invalid.

    References
    ----------
    .. [1] Vallmitjana A, Torrado B, Dvornikov A, Ranjit S, and Gratton E.
      `Blind resolution of lifetime components in individual pixels of
      fluorescence lifetime images using the phasor approach
      <https://doi.org/10.1021/acs.jpcb.0c06946>`_.
      *J Phys Chem B*, 124(45): 10126-10137 (2020)

    Notes
    -----
    This function currently supports only two lifetime components.

    Examples
    --------
    Resolve two lifetime components from the phasor coordinates of a mixture
    of 4.2 and 0.9 ns lifetimes with 70/30% fractions at 80 and 160 MHz:

    >>> phasor_to_lifetime_search(
    ...     [0.3773104, 0.20213886], [0.3834715, 0.30623315], frequency=80.0
    ... )
    (array([0.9, 4.2]), array([0.3, 0.7]))

    """
    num_components = 2

    if lifetime_range is None:
        lifetime_range = (0.0, 20.0, 0.1)
    elif (
        lifetime_range[0] < 0.0
        or lifetime_range[1] <= lifetime_range[0]
        or lifetime_range[2] <= 0.0
        or lifetime_range[2] >= lifetime_range[1] - lifetime_range[0]
    ):
        raise ValueError(f'invalid {lifetime_range=}')

    num_threads = number_threads(num_threads)

    dtype = numpy.dtype(dtype)
    if dtype.char not in {'f', 'd'}:
        raise ValueError(f'{dtype=} is not a floating point type')

    real = numpy.ascontiguousarray(real, dtype=dtype)
    imag = numpy.ascontiguousarray(imag, dtype=dtype)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if real.shape[0] < num_components:
        raise ValueError(f'{real.shape[0]=} < {num_components=}')

    shape = real.shape[1:]
    real = real[:num_components].reshape((num_components, -1))
    imag = imag[:num_components].reshape((num_components, -1))
    size = real.shape[-1]

    lifetime = numpy.zeros((num_components, size), dtype=dtype)
    fraction = numpy.zeros((num_components, size), dtype=dtype)

    candidate = phasor_from_lifetime(
        frequency,
        numpy.arange(*lifetime_range, dtype=dtype),
        unit_conversion=unit_conversion,
    )[0]

    omega = frequency * math.pi * 2.0 * unit_conversion
    omega *= omega

    _lifetime_search_2(
        lifetime, fraction, real, imag, candidate, omega, num_threads
    )

    lifetime = lifetime.reshape(num_components, *shape)
    fraction = fraction.reshape(num_components, *shape)

    return lifetime, fraction


def phasor_to_apparent_lifetime(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return apparent single lifetimes from phasor coordinates.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    unit_conversion : float, optional
        Product of `frequency` and returned `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    phase_lifetime : ndarray
        Apparent single lifetime from angular component of phasor coordinates.
    modulation_lifetime : ndarray
        Apparent single lifetime from radial component of phasor coordinates.

    See Also
    --------
    phasorpy.lifetime.phasor_from_apparent_lifetime
    :ref:`sphx_glr_tutorials_phasorpy_lifetime_geometry.py`

    Notes
    -----
    The phasor coordinates `real` (:math:`G`) and `imag` (:math:`S`)
    are converted to apparent single lifetimes
    `phase_lifetime` (:math:`\tau_{\phi}`) and
    `modulation_lifetime` (:math:`\tau_{M}`) at frequency :math:`f`
    according to:

    .. math::

        \omega &= 2 \pi f

        \tau_{\phi} &= \omega^{-1} \cdot S / G

        \tau_{M} &= \omega^{-1} \cdot \sqrt{1 / (S^2 + G^2) - 1}

    Examples
    --------
    The apparent single lifetimes from phase and modulation are equal
    only if the phasor coordinates lie on the universal semicircle:

    >>> phasor_to_apparent_lifetime(
    ...     0.5, [0.5, 0.45], frequency=80
    ... )  # doctest: +NUMBER
    (array([1.989, 1.79]), array([1.989, 2.188]))

    Apparent single lifetimes of phasor coordinates outside the universal
    semicircle are undefined:

    >>> phasor_to_apparent_lifetime(-0.1, 1.1, 80)  # doctest: +NUMBER
    (-21.8, 0.0)

    Apparent single lifetimes at the universal semicircle endpoints are
    infinite and zero:

    >>> phasor_to_apparent_lifetime([0, 1], [0, 0], 80)  # doctest: +NUMBER
    (array([inf, 0]), array([inf, 0]))

    """
    omega = numpy.asarray(frequency, dtype=numpy.float64, copy=True)
    omega *= math.pi * 2.0 * unit_conversion
    return _phasor_to_apparent_lifetime(  # type: ignore[no-any-return]
        real, imag, omega, **kwargs
    )


def phasor_from_apparent_lifetime(
    phase_lifetime: ArrayLike,
    modulation_lifetime: ArrayLike | None,
    /,
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return phasor coordinates from apparent single lifetimes.

    Parameters
    ----------
    phase_lifetime : ndarray
        Apparent single lifetime from phase.
    modulation_lifetime : ndarray, optional
        Apparent single lifetime from modulation.
        If None, `modulation_lifetime` is same as `phase_lifetime`.
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    unit_conversion : float, optional
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real : ndarray
        Real component of phasor coordinates.
    imag : ndarray
        Imaginary component of phasor coordinates.

    See Also
    --------
    phasorpy.lifetime.phasor_to_apparent_lifetime

    Notes
    -----
    The apparent single lifetimes `phase_lifetime` (:math:`\tau_{\phi}`)
    and `modulation_lifetime` (:math:`\tau_{M}`) are converted to phasor
    coordinates `real` (:math:`G`) and `imag` (:math:`S`) at
    frequency :math:`f` according to:

    .. math::

        \omega &= 2 \pi f

        \phi & = \arctan(\omega \tau_{\phi})

        M &= 1 / \sqrt{1 + (\omega \tau_{M})^2}

        G &= M \cdot \cos{\phi}

        S &= M \cdot \sin{\phi}

    Examples
    --------
    If the apparent single lifetimes from phase and modulation are equal,
    the phasor coordinates lie on the universal semicircle, else inside:

    >>> phasor_from_apparent_lifetime(
    ...     1.9894, [1.9894, 2.4113], frequency=80.0
    ... )  # doctest: +NUMBER
    (array([0.5, 0.45]), array([0.5, 0.45]))

    Zero and infinite apparent single lifetimes define the endpoints of the
    universal semicircle:

    >>> phasor_from_apparent_lifetime(
    ...     [0.0, 1e9], [0.0, 1e9], frequency=80
    ... )  # doctest: +NUMBER
    (array([1, 0.0]), array([0, 0.0]))

    """
    omega = numpy.asarray(frequency, dtype=numpy.float64, copy=True)
    omega *= math.pi * 2.0 * unit_conversion
    if modulation_lifetime is None:
        return _phasor_from_single_lifetime(  # type: ignore[no-any-return]
            phase_lifetime, omega, **kwargs
        )
    return _phasor_from_apparent_lifetime(  # type: ignore[no-any-return]
        phase_lifetime, modulation_lifetime, omega, **kwargs
    )


def phasor_to_normal_lifetime(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> NDArray[Any]:
    r"""Return normal lifetimes from phasor coordinates.

    The normal lifetime of phasor coordinates represents the single lifetime
    equivalent corresponding to the perpendicular projection of the coordinates
    onto the universal semicircle, as defined in [2]_.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    unit_conversion : float, optional
        Product of `frequency` and returned `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    normal_lifetime : ndarray
        Normal lifetime of phasor coordinates.

    See Also
    --------
    :ref:`sphx_glr_tutorials_phasorpy_lifetime_geometry.py`

    Notes
    -----
    The phasor coordinates `real` (:math:`G`) and `imag` (:math:`S`)
    are converted to normal lifetimes `normal_lifetime` (:math:`\tau_{N}`)
    at frequency :math:`f` according to:

    .. math::

        \omega &= 2 \pi f

        G_{N} &= 0.5 \cdot (1 + \cos{\arctan{\frac{S}{G - 0.5}}})

        \tau_{N} &= \sqrt{\frac{1 - G_{N}}{\omega^{2} \cdot G_{N}}}

    References
    ----------
    .. [2] Silberberg M, and Grecco H.
      `pawFLIM: reducing bias and uncertainty to enable lower photon
      count in FLIM experiments
      <https://doi.org/10.1088/2050-6120/aa72ab>`_.
      *Methods Appl Fluoresc*, 5(2): 024016 (2017)

    Examples
    --------
    The normal lifetimes of phasor coordinates with a real component of 0.5
    are independent of the imaginary component:

    >>> phasor_to_normal_lifetime(
    ...     0.5, [0.5, 0.45], frequency=80
    ... )  # doctest: +NUMBER
    array([1.989, 1.989])

    """
    omega = numpy.asarray(frequency, dtype=numpy.float64, copy=True)
    omega *= math.pi * 2.0 * unit_conversion
    return _phasor_to_normal_lifetime(  # type: ignore[no-any-return]
        real, imag, omega, **kwargs
    )


def lifetime_to_frequency(
    lifetime: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
) -> NDArray[numpy.float64]:
    r"""Return optimal frequency for resolving single component lifetime.

    Parameters
    ----------
    lifetime : array_like
        Single component lifetime.
    unit_conversion : float, optional, default: 1e-3
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.

    Returns
    -------
    frequency : ndarray
        Optimal laser pulse or modulation frequency for resolving `lifetime`.

    Notes
    -----
    The optimal frequency :math:`f` to resolve a single component lifetime
    :math:`\tau` is
    (:ref:`Redford & Clegg 2005 <redford-clegg-2005>`. Eq. B.6):

    .. math::

        \omega &= 2 \pi f

        \omega^2 &= \frac{1 + \sqrt{3}}{2 \tau^2}

    Examples
    --------
    Measurements of a lifetime near 4 ns should be made at 47 MHz,
    near 1 ns at 186 MHz:

    >>> lifetime_to_frequency([4.0, 1.0])  # doctest: +NUMBER
    array([46.5, 186])

    """
    t = numpy.reciprocal(lifetime, dtype=numpy.float64)
    t *= 0.18601566519848653 / unit_conversion
    return t


def lifetime_from_frequency(
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
) -> NDArray[numpy.float64]:
    r"""Return single component lifetime best resolved at frequency.

    Parameters
    ----------
    frequency : array_like
        Laser pulse or modulation frequency.
    unit_conversion : float, optional, default: 1e-3
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.

    Returns
    -------
    lifetime : ndarray
        Single component lifetime best resolved at `frequency`.

    Notes
    -----
    The lifetime :math:`\tau` that is best resolved at frequency :math:`f` is
    (:ref:`Redford & Clegg 2005 <redford-clegg-2005>`. Eq. B.6):

    .. math::

        \omega &= 2 \pi f

        \tau^2 &=  \frac{1 + \sqrt{3}}{2 \omega^2}

    Examples
    --------
    Measurements at frequencies of 47 and 186 MHz are best for measuring
    lifetimes near 4 and 1 ns respectively:

    >>> lifetime_from_frequency([46.5, 186])  # doctest: +NUMBER
    array([4, 1])

    """
    t = numpy.reciprocal(frequency, dtype=numpy.float64)
    t *= 0.18601566519848653 / unit_conversion
    return t


def lifetime_fraction_to_amplitude(
    lifetime: ArrayLike, fraction: ArrayLike, *, axis: int = -1
) -> NDArray[numpy.float64]:
    r"""Return pre-exponential amplitude from fractional intensity.

    Parameters
    ----------
    lifetime : array_like
        Lifetime components.
    fraction : array_like
        Fractional intensities of lifetime components.
        Fractions are normalized to sum to 1.
    axis : int, optional
        Axis over which to compute pre-exponential amplitudes.
        The default is the last axis (-1).

    Returns
    -------
    amplitude : ndarray
        Pre-exponential amplitudes.
        The product of `amplitude` and `lifetime` sums to 1 along `axis`.

    See Also
    --------
    phasorpy.lifetime.lifetime_fraction_from_amplitude

    Notes
    -----
    The pre-exponential amplitude :math:`a` of component :math:`j` with
    lifetime :math:`\tau` and fractional intensity :math:`\alpha` is:

    .. math::

        a_{j} = \frac{\alpha_{j}}{\tau_{j} \cdot \sum_{j} \alpha_{j}}

    Examples
    --------
    >>> lifetime_fraction_to_amplitude(
    ...     [4.0, 1.0], [1.6, 0.4]
    ... )  # doctest: +NUMBER
    array([0.2, 0.2])

    """
    t = numpy.asarray(fraction, dtype=numpy.float64, copy=True)
    t /= numpy.sum(t, axis=axis, keepdims=True)
    numpy.true_divide(t, lifetime, out=t)
    return t


def lifetime_fraction_from_amplitude(
    lifetime: ArrayLike, amplitude: ArrayLike, *, axis: int = -1
) -> NDArray[numpy.float64]:
    r"""Return fractional intensity from pre-exponential amplitude.

    Parameters
    ----------
    lifetime : array_like
        Lifetime of components.
    amplitude : array_like
        Pre-exponential amplitudes of lifetime components.
    axis : int, optional
        Axis over which to compute fractional intensities.
        The default is the last axis (-1).

    Returns
    -------
    fraction : ndarray
        Fractional intensities, normalized to sum to 1 along `axis`.

    See Also
    --------
    phasorpy.lifetime.lifetime_fraction_to_amplitude

    Notes
    -----
    The fractional intensity :math:`\alpha` of component :math:`j` with
    lifetime :math:`\tau` and pre-exponential amplitude :math:`a` is:

    .. math::

        \alpha_{j} = \frac{a_{j} \tau_{j}}{\sum_{j} a_{j} \tau_{j}}

    Examples
    --------
    >>> lifetime_fraction_from_amplitude(
    ...     [4.0, 1.0], [1.0, 1.0]
    ... )  # doctest: +NUMBER
    array([0.8, 0.2])

    """
    t: NDArray[numpy.float64]
    t = numpy.multiply(amplitude, lifetime, dtype=numpy.float64)
    t /= numpy.sum(t, axis=axis, keepdims=True)
    return t


def phasor_semicircle(
    samples: int = 101, /
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    r"""Return equally spaced phasor coordinates on universal semicircle.

    Parameters
    ----------
    samples : int, optional, default: 101
        Number of coordinates to return.

    Returns
    -------
    real : ndarray
        Real component of phasor coordinates on universal semicircle.
    imag : ndarray
        Imaginary component of phasor coordinates on universal semicircle.

    Raises
    ------
    ValueError
        The number of `samples` is smaller than 1.

    Notes
    -----
    If more than one sample, the first and last phasor coordinates returned
    are ``(0, 0)`` and ``(1, 0)``.
    The center coordinate, if any, is ``(0.5, 0.5)``.

    The universal semicircle is composed of the phasor coordinates of
    single lifetime components, where the relation of polar coordinates
    (phase :math:`\phi` and modulation :math:`M`) is:

    .. math::

        M = \cos{\phi}

    Examples
    --------
    Calculate three phasor coordinates on universal semicircle:

    >>> phasor_semicircle(3)  # doctest: +NUMBER
    (array([0, 0.5, 1]), array([0.0, 0.5, 0]))

    """
    if samples < 1:
        raise ValueError(f'{samples=} < 1')
    arange = numpy.linspace(math.pi, 0.0, samples)
    real = numpy.cos(arange)
    real += 1.0
    real *= 0.5
    imag = numpy.sin(arange)
    imag *= 0.5
    return real, imag


def phasor_semicircle_intersect(
    real0: ArrayLike,
    imag0: ArrayLike,
    real1: ArrayLike,
    imag1: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return intersection of line through phasors with universal semicircle.

    Return the phasor coordinates of the two intersections of the universal
    semicircle with the line between two phasor coordinates.
    Return NaN if the line does not intersect the semicircle.

    Parameters
    ----------
    real0 : array_like
        Real component of first set of phasor coordinates.
    imag0 : array_like
        Imaginary component of first set of phasor coordinates.
    real1 : array_like
        Real component of second set of phasor coordinates.
    imag1 : array_like
        Imaginary component of second set of phasor coordinates.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real0 : ndarray
        Real component of first intersect of phasors with semicircle.
    imag0 : ndarray
        Imaginary component of first intersect of phasors with semicircle.
    real1 : ndarray
        Real component of second intersect of phasors with semicircle.
    imag1 : ndarray
        Imaginary component of second intersect of phasors with semicircle.

    Examples
    --------
    Calculate two intersects of a line through two phasor coordinates
    with the universal semicircle:

    >>> phasor_semicircle_intersect(0.2, 0.25, 0.6, 0.25)  # doctest: +NUMBER
    (0.066, 0.25, 0.933, 0.25)

    The line between two phasor coordinates may not intersect the semicircle
    at two points:

    >>> phasor_semicircle_intersect(0.2, 0.0, 0.6, 0.25)  # doctest: +NUMBER
    (nan, nan, 0.817, 0.386)

    """
    return _intersect_semicircle_line(  # type: ignore[no-any-return]
        real0, imag0, real1, imag1, **kwargs
    )


def phasor_at_harmonic(
    real: ArrayLike,
    harmonic: ArrayLike,
    other_harmonic: ArrayLike,
    /,
    **kwargs: Any,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    r"""Return phasor coordinates on universal semicircle at other harmonics.

    Return phasor coordinates at any harmonic from the real component of
    phasor coordinates of a single exponential lifetime at a certain harmonic.
    The input and output phasor coordinates lie on the universal semicircle.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates of single exponential lifetime
        at `harmonic`.
    harmonic : array_like
        Harmonic of `real` coordinate. Must be integer >= 1.
    other_harmonic : array_like
        Harmonic for which to return phasor coordinates. Must be integer >= 1.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real_other : ndarray
        Real component of phasor coordinates at `other_harmonic`.
    imag_other : ndarray
        Imaginary component of phasor coordinates at `other_harmonic`.

    Notes
    -----
    The phasor coordinates
    :math:`g_{n}` (`real_other`) and :math:`s_{n}` (`imag_other`)
    of a single exponential lifetime at harmonic :math:`n` (`other_harmonic`)
    is calculated from the real part of the phasor coordinates
    :math:`g_{m}` (`real`) at harmonic :math:`m` (`harmonic`) according to
    (:ref:`Torrado, Malacrida, & Ranjit. 2022 <torrado-2022>`. Eq. 25):

    .. math::

        g_{n} &= \frac{m^2 \cdot g_{m}}{n^2 + (m^2-n^2) \cdot g_{m}}

        s_{n} &= \sqrt{G_{n} - g_{n}^2}

    This function is equivalent to the following operations:

    .. code-block:: python

        phasor_from_lifetime(
            frequency=other_harmonic,
            lifetime=phasor_to_apparent_lifetime(
                real, sqrt(real - real * real), frequency=harmonic
            )[0],
        )

    Examples
    --------
    The phasor coordinates at higher harmonics are approaching the origin:

    >>> phasor_at_harmonic(0.5, 1, [1, 2, 4, 8])  # doctest: +NUMBER
    (array([0.5, 0.2, 0.05882, 0.01538]), array([0.5, 0.4, 0.2353, 0.1231]))

    """
    harmonic = numpy.asarray(harmonic, dtype=numpy.int32)
    if numpy.any(harmonic < 1):
        raise ValueError('invalid harmonic')

    other_harmonic = numpy.asarray(other_harmonic, dtype=numpy.int32)
    if numpy.any(other_harmonic < 1):
        raise ValueError('invalid other_harmonic')

    return _phasor_at_harmonic(  # type: ignore[no-any-return]
        real, harmonic, other_harmonic, **kwargs
    )


def polar_to_apparent_lifetime(
    phase: ArrayLike,
    modulation: ArrayLike,
    /,
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return apparent single lifetimes from polar coordinates.

    Parameters
    ----------
    phase : array_like
        Angular component of polar coordinates.
    modulation : array_like
        Radial component of polar coordinates.
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    unit_conversion : float, optional
        Product of `frequency` and returned `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    phase_lifetime : ndarray
        Apparent single lifetime from `phase`.
    modulation_lifetime : ndarray
        Apparent single lifetime from `modulation`.

    See Also
    --------
    phasorpy.lifetime.polar_from_apparent_lifetime

    Notes
    -----
    The polar coordinates `phase` (:math:`\phi`) and `modulation` (:math:`M`)
    are converted to apparent single lifetimes
    `phase_lifetime` (:math:`\tau_{\phi}`) and
    `modulation_lifetime` (:math:`\tau_{M}`) at frequency :math:`f`
    according to:

    .. math::

        \omega &= 2 \pi f

        \tau_{\phi} &= \omega^{-1} \cdot \tan{\phi}

        \tau_{M} &= \omega^{-1} \cdot \sqrt{1 / M^2 - 1}

    Examples
    --------
    The apparent single lifetimes from phase and modulation are equal
    only if the polar coordinates lie on the universal semicircle:

    >>> polar_to_apparent_lifetime(
    ...     math.pi / 4, numpy.hypot([0.5, 0.45], [0.5, 0.45]), frequency=80
    ... )  # doctest: +NUMBER
    (array([1.989, 1.989]), array([1.989, 2.411]))

    """
    omega = numpy.asarray(frequency, dtype=numpy.float64, copy=True)
    omega *= math.pi * 2.0 * unit_conversion
    return _polar_to_apparent_lifetime(  # type: ignore[no-any-return]
        phase, modulation, omega, **kwargs
    )


def polar_from_apparent_lifetime(
    phase_lifetime: ArrayLike,
    modulation_lifetime: ArrayLike | None,
    /,
    frequency: ArrayLike,
    *,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    r"""Return polar coordinates from apparent single lifetimes.

    Parameters
    ----------
    phase_lifetime : ndarray
        Apparent single lifetime from phase.
    modulation_lifetime : ndarray, optional
        Apparent single lifetime from modulation.
        If None, `modulation_lifetime` is same as `phase_lifetime`.
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    unit_conversion : float, optional
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    phase : ndarray
        Angular component of polar coordinates.
    modulation : ndarray
        Radial component of polar coordinates.

    See Also
    --------
    phasorpy.lifetime.polar_to_apparent_lifetime

    Notes
    -----
    The apparent single lifetimes `phase_lifetime` (:math:`\tau_{\phi}`)
    and `modulation_lifetime` (:math:`\tau_{M}`) are converted to polar
    coordinates `phase` (:math:`\phi`) and `modulation` (:math:`M`) at
    frequency :math:`f` according to:

    .. math::

        \omega &= 2 \pi f

        \phi & = \arctan(\omega \tau_{\phi})

        M &= 1 / \sqrt{1 + (\omega \tau_{M})^2}

    Examples
    --------
    If the apparent single lifetimes from phase and modulation are equal,
    the polar coordinates lie on the universal semicircle, else inside:

    >>> polar_from_apparent_lifetime(
    ...     1.9894, [1.9894, 2.4113], frequency=80.0
    ... )  # doctest: +NUMBER
    (array([0.7854, 0.7854]), array([0.7071, 0.6364]))

    """
    omega = numpy.asarray(frequency, dtype=numpy.float64, copy=True)
    omega *= math.pi * 2.0 * unit_conversion
    if modulation_lifetime is None:
        return _polar_from_single_lifetime(  # type: ignore[no-any-return]
            phase_lifetime, omega, **kwargs
        )
    return _polar_from_apparent_lifetime(  # type: ignore[no-any-return]
        phase_lifetime, modulation_lifetime, omega, **kwargs
    )


def phasor_from_fret_donor(
    frequency: ArrayLike,
    donor_lifetime: ArrayLike,
    *,
    fret_efficiency: ArrayLike = 0.0,
    donor_fretting: ArrayLike = 1.0,
    donor_background: ArrayLike = 0.0,
    background_real: ArrayLike = 0.0,
    background_imag: ArrayLike = 0.0,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return phasor coordinates of FRET donor channel.

    Calculate phasor coordinates of a FRET (Förster Resonance Energy Transfer)
    donor channel as a function of frequency, donor lifetime, FRET efficiency,
    fraction of donors undergoing FRET, and background fluorescence.

    The phasor coordinates of the donor channel contain fractions of:

    - donor not undergoing energy transfer
    - donor quenched by energy transfer
    - background fluorescence

    Parameters
    ----------
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    donor_lifetime : array_like
        Lifetime of donor without FRET in ns.
    fret_efficiency : array_like, optional, default 0
        FRET efficiency in range [0, 1].
    donor_fretting : array_like, optional, default 1
        Fraction of donors participating in FRET. Range [0, 1].
    donor_background : array_like, optional, default 0
        Weight of background fluorescence in donor channel
        relative to fluorescence of donor without FRET.
        A weight of 1 means the fluorescence of background and donor
        without FRET are equal.
    background_real : array_like, optional, default 0
        Real component of background fluorescence phasor coordinate
        at `frequency`.
    background_imag : array_like, optional, default 0
        Imaginary component of background fluorescence phasor coordinate
        at `frequency`.
    unit_conversion : float, optional
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real : ndarray
        Real component of donor channel phasor coordinates.
    imag : ndarray
        Imaginary component of donor channel phasor coordinates.

    See Also
    --------
    phasorpy.lifetime.phasor_from_fret_acceptor
    :ref:`sphx_glr_tutorials_api_phasorpy_fret.py`
    :ref:`sphx_glr_tutorials_applications_phasorpy_fret_efficiency.py`

    Examples
    --------
    Compute the phasor coordinates of a FRET donor channel at three
    FRET efficiencies:

    >>> phasor_from_fret_donor(
    ...     frequency=80,
    ...     donor_lifetime=4.2,
    ...     fret_efficiency=[0.0, 0.3, 1.0],
    ...     donor_fretting=0.9,
    ...     donor_background=0.1,
    ...     background_real=0.11,
    ...     background_imag=0.12,
    ... )  # doctest: +NUMBER
    (array([0.1766, 0.2737, 0.1466]), array([0.3626, 0.4134, 0.2534]))

    """
    omega = numpy.asarray(frequency, dtype=numpy.float64, copy=True)
    omega *= math.pi * 2.0 * unit_conversion
    return _phasor_from_fret_donor(  # type: ignore[no-any-return]
        omega,
        donor_lifetime,
        fret_efficiency,
        donor_fretting,
        donor_background,
        background_real,
        background_imag,
        **kwargs,
    )


def phasor_from_fret_acceptor(
    frequency: ArrayLike,
    donor_lifetime: ArrayLike,
    acceptor_lifetime: ArrayLike,
    *,
    fret_efficiency: ArrayLike = 0.0,
    donor_fretting: ArrayLike = 1.0,
    donor_bleedthrough: ArrayLike = 0.0,
    acceptor_bleedthrough: ArrayLike = 0.0,
    acceptor_background: ArrayLike = 0.0,
    background_real: ArrayLike = 0.0,
    background_imag: ArrayLike = 0.0,
    unit_conversion: float = 1e-3,
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return phasor coordinates of FRET acceptor channel.

    Calculate phasor coordinates of a FRET (Förster Resonance Energy Transfer)
    acceptor channel as a function of frequency, donor and acceptor lifetimes,
    FRET efficiency, fraction of donors undergoing FRET, fraction of directly
    excited acceptors, fraction of donor fluorescence in acceptor channel,
    and background fluorescence.

    The phasor coordinates of the acceptor channel contain fractions of:

    - acceptor sensitized by energy transfer
    - directly excited acceptor
    - donor bleedthrough
    - background fluorescence

    Parameters
    ----------
    frequency : array_like
        Laser pulse or modulation frequency in MHz.
    donor_lifetime : array_like
        Lifetime of donor without FRET in ns.
    acceptor_lifetime : array_like
        Lifetime of acceptor in ns.
    fret_efficiency : array_like, optional, default 0
        FRET efficiency in range [0, 1].
    donor_fretting : array_like, optional, default 1
        Fraction of donors participating in FRET. Range [0, 1].
    donor_bleedthrough : array_like, optional, default 0
        Weight of donor fluorescence in acceptor channel
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence from donor and fully sensitized
        acceptor are equal.
        The background in the donor channel does not bleed through.
    acceptor_bleedthrough : array_like, optional, default 0
        Weight of fluorescence from directly excited acceptor
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence from directly excited acceptor
        and fully sensitized acceptor are equal.
    acceptor_background : array_like, optional, default 0
        Weight of background fluorescence in acceptor channel
        relative to fluorescence of fully sensitized acceptor.
        A weight of 1 means the fluorescence of background and fully
        sensitized acceptor are equal.
    background_real : array_like, optional, default 0
        Real component of background fluorescence phasor coordinate
        at `frequency`.
    background_imag : array_like, optional, default 0
        Imaginary component of background fluorescence phasor coordinate
        at `frequency`.
    unit_conversion : float, optional
        Product of `frequency` and `lifetime` units' prefix factors.
        The default is 1e-3 for MHz and ns, or Hz and ms.
        Use 1.0 for Hz and s.
    **kwargs
        Optional `arguments passed to numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    real : ndarray
        Real component of acceptor channel phasor coordinates.
    imag : ndarray
        Imaginary component of acceptor channel phasor coordinates.

    See Also
    --------
    phasorpy.lifetime.phasor_from_fret_donor
    :ref:`sphx_glr_tutorials_api_phasorpy_fret.py`

    Examples
    --------
    Compute the phasor coordinates of a FRET acceptor channel at three
    FRET efficiencies:

    >>> phasor_from_fret_acceptor(
    ...     frequency=80,
    ...     donor_lifetime=4.2,
    ...     acceptor_lifetime=3.0,
    ...     fret_efficiency=[0.0, 0.3, 1.0],
    ...     donor_fretting=0.9,
    ...     donor_bleedthrough=0.1,
    ...     acceptor_bleedthrough=0.1,
    ...     acceptor_background=0.1,
    ...     background_real=0.11,
    ...     background_imag=0.12,
    ... )  # doctest: +NUMBER
    (array([0.1996, 0.05772, 0.2867]), array([0.3225, 0.3103, 0.4292]))

    """
    omega = numpy.asarray(frequency, dtype=numpy.float64, copy=True)
    omega *= math.pi * 2.0 * unit_conversion
    return _phasor_from_fret_acceptor(  # type: ignore[no-any-return]
        omega,
        donor_lifetime,
        acceptor_lifetime,
        fret_efficiency,
        donor_fretting,
        donor_bleedthrough,
        acceptor_bleedthrough,
        acceptor_background,
        background_real,
        background_imag,
        **kwargs,
    )
