"""Cluster phasor coordinates.

The `phasorpy.cluster` module provides functions to:

- cluster phasor coordinates using Gaussian Mixture Model (GMM):
    - :py:func:`phasor_cluster_gmm`

"""

from __future__ import annotations

__all__ = ['phasor_cluster_gmm']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike

import numbers
from collections.abc import Sequence

import numpy
from sklearn.mixture import GaussianMixture


def phasor_cluster_gmm(
    real: ArrayLike,
    imag: ArrayLike,
    scaling: float = 2 * numpy.sqrt(2),
    /,
    *,
    clusters: int = 1,
    skip_axis: int | Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return elliptic clusters in phasor coordinates using GMM.

    Fit a Gaussian Mixture Model (GMM) to the provided phasor coordinates and
    extract the parameters of ellipses that represent each cluster according
    to [1]_.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    scaling: float, default = 2*sqrt(2)
        Scaling factor for the radii of major and minor axes. By default, it is set
        to 2√2, which corresponds to the scaling of eigenvalues for a 95% confidence
        ellipse (2σ).
    clusters : int, optional
        Number of Gaussian distributions to fit to phasor coordinates.
        Defaults to 1.
    skip_axis : int or sequence of int, optional
        Axes to skip in the data (useful for multi-dimensional arrays).
    **kwargs
        Additional keyword arguments passed to
        `sklearn.mixture.GaussianMixture`.
        Common options include:

        - covariance_type : {'full', 'tied', 'diag', 'spherical'}
        - max_iter : int, maximum number of EM iterations
        - random_state : int, for reproducible results

    Returns
    -------
    center_real : tuple of float
        Real component of ellipse centers.
    center_imag : tuple of float
        Imaginary component of ellipse centers.
    radius_major : tuple of float
        Major radii of ellipses.
    radius_minor : tuple of float
        Minor radii of ellipses.
    angle : tuple of float
        Rotation angles of major axes in radians, within range [0, π].

    Raises
    ------
    ValueError
        If the array shapes of `real` and `imag` do not match.
        If `clusters` is not a positive integer.

    Notes
    -----
    The radii represent the 95% confidence intervals of the Gaussian
    distributions, scaled by 2.0 * sqrt(2.0) * sqrt(eigenvalues).

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_cluster.py`

    References
    ----------
    .. [1] Vallmitjana A, Torrado B, and Gratton E.
      `Phasor-based image segmentation: machine learning clustering techniques
      <https://doi.org/10.1364/BOE.422766>`_.
      *Biomed Opt Express*, 12(6): 3410-3422 (2021).

    Examples
    --------
    Recover the clusters from a synthetic distribution of phasor coordinates
    with two clusters:

    >>> real1, imag1 = numpy.random.multivariate_normal(
    ...     [2, 3], [[0.3, 0.1], [0.1, 0.2]], 100
    ... ).T
    >>> real2, imag2 = numpy.random.multivariate_normal(
    ...     [5, 6], [[0.2, -0.1], [-0.1, 0.3]], 100
    ... ).T
    >>> real = numpy.concatenate([real1, real2])
    >>> imag = numpy.concatenate([imag1, imag2])
    >>> (center_real, center_imag, radius_major, radius_minor, angle) = (
    ...     phasor_cluster_gmm(real, imag, clusters=2)
    ... )

    """

    real = numpy.asarray(real)
    imag = numpy.asarray(imag)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if not isinstance(clusters, numbers.Integral) or clusters < 1:
        raise ValueError(f"{clusters=} of type {type(clusters)}")
    clusters = int(clusters)

    skip_axis, _ = _parse_skip_axis(skip_axis, real.ndim)

    if skip_axis:
        real = numpy.mean(real, axis=skip_axis, keepdims=True)
        imag = numpy.mean(imag, axis=skip_axis, keepdims=True)

    coords = numpy.column_stack((real.ravel(), imag.ravel()))

    valid_data = ~numpy.isnan(coords).any(axis=1)
    coords = coords[valid_data]

    if coords.shape[0] < clusters:
        raise ValueError(
            f'Not enough points ({coords.shape[0]}) for requested components ({clusters})'
        )

    kwargs.pop('n_components', None)

    gmm = GaussianMixture(n_components=clusters, **kwargs)
    gmm.fit(coords)

    center_real = []
    center_imag = []
    radius_major = []
    radius_minor = []
    angle = []

    for i in range(clusters):
        center_real.append(float(gmm.means_[i, 0]))
        center_imag.append(float(gmm.means_[i, 1]))

        if gmm.covariance_type == 'full':
            cov = gmm.covariances_[i]
        elif gmm.covariance_type == 'tied':
            cov = gmm.covariances_
        elif gmm.covariance_type == 'diag':
            cov = numpy.diag(gmm.covariances_[i])
        else:  # 'spherical'
            cov = numpy.eye(2) * gmm.covariances_[i]

        eigenvalues, eigenvectors = numpy.linalg.eigh(cov[:2, :2])

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        major_vector = eigenvectors[:, 0]
        current_angle = numpy.arctan2(major_vector[1], major_vector[0])

        if current_angle < 0:
            current_angle += numpy.pi

        angle.append(float(current_angle))

        radius_major.append(scaling * numpy.sqrt(eigenvalues[0]))
        radius_minor.append(scaling * numpy.sqrt(eigenvalues[1]))

    return (
        tuple(center_real),
        tuple(center_imag),
        tuple(radius_major),
        tuple(radius_minor),
        tuple(angle),
    )


# Added here, or should it be imported from phasor.py where
# there's an exact copy of the function?
def _parse_skip_axis(
    skip_axis: int | Sequence[int] | None,
    /,
    ndim: int,
    prepend_axis: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return axes to skip and not to skip.

    This helper function is used to validate and parse `skip_axis`
    parameters.

    Parameters
    ----------
    skip_axis : int or sequence of int, optional
        Axes to skip. If None, no axes are skipped.
    ndim : int
        Dimensionality of array in which to skip axes.
    prepend_axis : bool, optional
        Prepend one dimension and include in `skip_axis`.

    Returns
    -------
    skip_axis
        Ordered, positive values of `skip_axis`.
    other_axis
        Axes indices not included in `skip_axis`.

    Raises
    ------
    IndexError
        If any `skip_axis` value is out of bounds of `ndim`.

    Examples
    --------
    >>> _parse_skip_axis((1, -2), 5)
    ((1, 3), (0, 2, 4))

    >>> _parse_skip_axis((1, -2), 5, True)
    ((0, 2, 4), (1, 3, 5))

    """
    if ndim < 0:
        raise ValueError(f'invalid {ndim=}')
    if skip_axis is None:
        if prepend_axis:
            return (0,), tuple(range(1, ndim + 1))
        return (), tuple(range(ndim))
    if not isinstance(skip_axis, Sequence):
        skip_axis = (skip_axis,)
    if any(i >= ndim or i < -ndim for i in skip_axis):
        raise IndexError(f'skip_axis={skip_axis} out of range for {ndim=}')
    skip_axis = sorted(int(i % ndim) for i in skip_axis)
    if prepend_axis:
        skip_axis = [0] + [i + 1 for i in skip_axis]
        ndim += 1
    other_axis = tuple(i for i in range(ndim) if i not in skip_axis)
    return tuple(skip_axis), other_axis
