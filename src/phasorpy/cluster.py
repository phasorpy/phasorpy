"""Cluster phasor coordinates.

The `phasorpy.cluster` module provides functions to:

- cluster phasor coordinates using Gaussian Mixture Model (GMM):
    - :py:func:`phasor_cluster_gmm`

"""

from __future__ import annotations

__all__ = ['phasor_cluster_gmm']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import math
from collections.abc import Sequence

import numpy
from sklearn.mixture import GaussianMixture

from ._utils import parse_skip_axis


def phasor_cluster_gmm(
    real: NDArray[numpy.float64],
    imag: NDArray[numpy.float64],
    sigma: float = 2.0,
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
    sigma: float, default = 2.0
        Scaling factor for the radii of major and minor axes. By default, it is set
        to 2, which corresponds to the scaling of eigenvalues for a 95 per cent confidence
        ellipse.
    clusters : int, optional
        Number of Gaussian distributions to fit to phasor coordinates.
        Defaults to 1.
    skip_axis : int or sequence of int, optional
        Axes to skip in the data (useful for multi-dimensional arrays).
    **kwargs
        Additional keyword arguments passed to
        :py:class:`sklearn.mixture.GaussianMixture`.

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
        Rotation angles of major axes in radians, within range [0, pi].

    Raises
    ------
    ValueError
        If the array shapes of `real` and `imag` do not match.
        If `clusters` is not a positive integer.

    Notes
    -----
    The radii represent the 95 per cent confidence intervals of the Gaussian
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
    real = numpy.asarray(real, dtype=numpy.float64)
    imag = numpy.asarray(imag, dtype=numpy.float64)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')

    skip_axis, _ = parse_skip_axis(skip_axis, real.ndim)

    if skip_axis:
        real = numpy.mean(real, axis=skip_axis, keepdims=True)
        imag = numpy.mean(imag, axis=skip_axis, keepdims=True)

    coords: NDArray[numpy.float64] = numpy.column_stack(
        (real.ravel(), imag.ravel())
    )

    valid_data = ~numpy.isnan(coords).any(axis=1)
    coords = coords[valid_data]

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
        current_angle = math.atan2(major_vector[1], major_vector[0])

        if current_angle < 0:
            current_angle += math.pi

        angle.append(float(current_angle))

        radius_major.append(sigma * math.sqrt(2 * eigenvalues[0]))
        radius_minor.append(sigma * math.sqrt(2 * eigenvalues[1]))

    return (
        tuple(center_real),
        tuple(center_imag),
        tuple(radius_major),
        tuple(radius_minor),
        tuple(angle),
    )
