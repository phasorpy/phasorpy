"""Experimental functions.

The ``phasorpy.experimental`` module provides functions related to phasor
analysis for evaluation.
The functions may be removed or moved to other modules in future releases.

"""

from __future__ import annotations

__all__ = [
    'anscombe_transform',
    'anscombe_transform_inverse',
    'signal_from_dho',
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

from ._phasorpy import (
    _anscombe,
    _anscombe_inverse,
    _anscombe_inverse_approx,
    _signal_from_dho,
)


def anscombe_transform(
    data: ArrayLike,
    /,
    **kwargs: Any,
) -> NDArray[Any]:
    r"""Return Anscombe variance-stabilizing transformation.

    The Anscombe transformation normalizes the standard deviation of noisy,
    Poisson-distributed data.
    It can be used to transform unnormalized phasor coordinates to
    approximate standard Gaussian distributions.

    Parameters
    ----------
    data : array_like
        Noisy Poisson-distributed data to be transformed.
    **kwargs
        Optional arguments passed to `numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    ndarray
        Anscombe-transformed data with variance of approximately 1.

    Notes
    -----
    The Anscombe transformation according to [1]_:

    .. math::

        z = 2 \cdot \sqrt{x + 3 / 8}

    References
    ----------
    .. [1] Anscombe FJ.
       `The transformation of Poisson, binomial and negative-binomial data
       <https://doi.org/10.2307/2332343>`_.
       *Biometrika*, 35(3-4): 246-254 (1948)

    Examples
    --------
    >>> z = anscombe_transform(numpy.random.poisson(10, 10000))
    >>> numpy.allclose(numpy.std(z), 1.0, atol=0.1)
    True

    """
    return _anscombe(data, **kwargs)  # type: ignore[no-any-return]


def anscombe_transform_inverse(
    data: ArrayLike,
    /,
    *,
    approx: bool = False,
    **kwargs: Any,
) -> NDArray[Any]:
    r"""Return inverse Anscombe transformation.

    Parameters
    ----------
    data : array_like
        Anscombe-transformed data.
    approx : bool, optional, default: False
        Return approximation of exact unbiased inverse.
    **kwargs
        Optional arguments passed to `numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    ndarray
        Inverse Anscombe-transformed data.

    Notes
    -----
    The inverse Anscombe transformation according to [1]_:

    .. math::

        x = (z / 2.0)^2 - 3 / 8

    The approximate inverse Anscombe transformation according to [2]_ and [3]_:

    .. math::

        x = 1/4 \cdot z^2
          + 1/4 \cdot \sqrt{3/2} \cdot z^{-1}
          - 11/8 \cdot z^{-2}
          + 5/8 \cdot \sqrt{3/2} \cdot z^{-3}
          - 1/8

    References
    ----------
    .. [2] Makitalo M, and Foi A.
       `A closed-form approximation of the exact unbiased inverse of the
       Anscombe variance-stabilizing transformation
       <https://doi.org/10.1109/TIP.2011.2121085>`_.
       *IEEE Trans Image Process*, 20(9): 2697-2698 (2011)

    .. [3] Makitalo M, and Foi A.
       `Optimal inversion of the generalized Anscombe transformation for
       Poisson-Gaussian noise
       <https://doi.org/10.1109/TIP.2012.2202675>`_,
       *IEEE Trans Image Process*, 22(1): 91-103 (2013)

    Examples
    --------
    >>> x = numpy.random.poisson(10, 100)
    >>> x2 = anscombe_transform_inverse(anscombe_transform(x))
    >>> numpy.allclose(x, x2, atol=1e-3)
    True

    """
    if approx:
        return _anscombe_inverse_approx(  # type: ignore[no-any-return]
            data, **kwargs
        )
    return _anscombe_inverse(data, **kwargs)  # type: ignore[no-any-return]


def signal_from_dho(
    wavelength: ArrayLike,
    origin: ArrayLike,
    sigma: ArrayLike,
    hr_factor: ArrayLike,
    vib_frequency: ArrayLike,
    *,
    scale: ArrayLike = 1.0,
    offset: ArrayLike = 0.0,
    absorption: bool = False,
    **kwargs: Any,
) -> NDArray[Any]:
    r"""Return normalized fluorescence emission or absorption at wavelengths.

    Using the area-normalized Displaced Harmonic Oscillator (DHO) model
    to approximate the absorption or fluorescence emission spectrum of a
    fluorophore with a single vibrational mode.

    Parameters
    ----------
    wavelength : array_like
        Wavelength at which to calculate emission intensity in nm.
    origin : array_like
        Center wavelength of 0->0 electronic origin transition in nm.
        Typically in the range 400 to 700 nm.
    sigma : array_like
        Gaussian spectral broadening factor in :math:`cm^{-1}`.
        Typically in the range 200 to 600 :math:`cm^{-1}`.
    hr_factor : array_like
        Huang-Rhys structural coupling/displacement parameter (dimensionless).
        Typically in the range 0.1 to 2.0.
    vib_frequency : array_like
        Vibrational spacing frequency in :math:`cm^{-1}`.
        Typically in the range 1000 to 1600 :math:`cm^{-1}`.
    scale : array_like, optional, default: 1.0
        Factor multiplied to normalized DHO intensity.
    offset : array_like, optional, default: 0.0
        Offset added after scaling.
    absorption: bool, optional
        If True, return absorption intensity instead of fluorescence emission.
    **kwargs
        Optional arguments passed to `numpy universal functions
        <https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs>`_.

    Returns
    -------
    intensity : ndarray
        Absorption or fluorescence emission intensities.

    See Also
    --------
    :ref:`sphx_glr_tutorials_misc_phasorpy_apps.py`

    Notes
    -----
    The intensity in wavenumber space :math:`I(\nu)` is calculated from the
    :math:`0\rightarrow0` electronic origin :math:`\nu_0 = 10^7 / \lambda_0`,
    the Gaussian spectral broadening factor :math:`\sigma`,
    the Huang-Rhys factor :math:`S`,
    and the vibrational frequency :math:`\nu_{\text{vib}}`
    using the DHO model with :math:`N = 6` vibronic terms:

    .. math::

        I(\nu) = \frac{1}{\sigma\sqrt{2\pi}}
        \sum_{n=0}^{N} \frac{S^n e^{-S}}{n!}
        \exp\left( - \frac{(\nu - \nu_n)^2}{2\sigma^2} \right)

    where :math:`\nu_n = \nu_0 - n \cdot \nu_{\text{vib}}` for emission
    (red-shifted sidebands) and
    :math:`\nu_n = \nu_0 + n \cdot \nu_{\text{vib}}` for absorption
    (blue-shifted sidebands).

    Calculations are performed in wavenumber :math:`\nu` space and
    transformed to wavelength :math:`\lambda` space using the Jacobian:

    .. math::

        I(\lambda) = I(\nu) \cdot \left| \frac{d\nu}{d\lambda} \right| =
        I(\nu) \cdot \frac{10^7}{\lambda^2}

    The returned signal can be affine-transformed for measured spectra:

    .. math::

        I_{\text{out}}(\lambda) = I(\lambda) \cdot \text{scale} + \text{offset}

    Examples
    --------
    Approximate the fluorescence emission spectrum of Fluorescein:

    >>> signal_from_dho(
    ...     wavelength=numpy.linspace(450, 650, 100),
    ...     origin=518,
    ...     sigma=500,
    ...     hr_factor=0.4,
    ...     vib_frequency=1200,
    ... )
    array([..., 0.0001434, 0.0001329, 0.0001228])

    """
    # TODO: move this function to phasorpy.spectral?
    # TODO: find citation for DHO model and add to docstring
    return _signal_from_dho(  # type: ignore[no-any-return]
        wavelength,
        origin,
        sigma,
        hr_factor,
        vib_frequency,
        scale,
        offset,
        absorption,
        **kwargs,
    )
