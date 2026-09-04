# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""Cluster phasor coordinates.

The ``phasorpy.cluster`` module provides functions to:

- fit elliptical clusters to phasor coordinates using a
  Gaussian Mixture Model (GMM):

  - :py:func:`phasor_cluster_gmm`

- assign phasor coordinates to clusters using k-means clustering:

  - :py:func:`phasor_cluster_kmeans`

"""

from __future__ import annotations

__all__ = ['phasor_cluster_gmm', 'phasor_cluster_kmeans']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, Literal, NDArray, Sequence

import math

import numpy


def phasor_cluster_gmm(
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    sigma: float = 2.0,
    clusters: int = 1,
    sort: Literal['polar', 'phasor', 'area'] | None = None,
    **kwargs: Any,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    """Return elliptical clusters in phasor coordinates using GMM.

    Fit a Gaussian Mixture Model (GMM) to the provided phasor coordinates and
    extract the parameters of ellipses that represent each cluster according
    to [1]_.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    sigma : float, optional, default: 2
        Scaling factor for radii of major and minor axes.
        The default 2.0 is commonly used for visualization of confidence
        ellipses (~98.2%).
    clusters : int, optional, default: 1
        Number of Gaussian distributions to fit to phasor coordinates.
    sort : {'polar', 'phasor', 'area'}, optional
        Sorting method for output clusters.
        By default, use 'polar' sorting.

        - 'polar': Sort by polar coordinates (phase, then modulation).
        - 'phasor': Sort by phasor coordinates (imaginary, then real).
        - 'area': Sort by inverse area of ellipse (-major * minor).

    **kwargs
        Optional arguments passed to
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
        If `sigma` is not positive.
        If `clusters` is less than 1.
        If the array shapes of `real` and `imag` do not match.
        If the number of valid (non-NaN) data points is less than `clusters`.
        If `sort` is not a valid sorting method.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_cluster.py`

    References
    ----------
    .. [1] Vallmitjana A, Torrado B, and Gratton E.
       `Phasor-based image segmentation: machine learning clustering techniques
       <https://doi.org/10.1364/BOE.422766>`_.
       *Biomed Opt Express*, 12(6): 3410-3422 (2021)

    Examples
    --------
    Recover the clusters from a synthetic distribution of phasor coordinates
    with two clusters:

    >>> real1, imag1 = numpy.random.multivariate_normal(
    ...     [0.2, 0.3], [[3e-3, 1e-3], [1e-3, 2e-3]], 100
    ... ).T
    >>> real2, imag2 = numpy.random.multivariate_normal(
    ...     [0.4, 0.5], [[2e-3, -1e-3], [-1e-3, 3e-3]], 100
    ... ).T
    >>> real = numpy.concatenate([real1, real2])
    >>> imag = numpy.concatenate([imag1, imag2])
    >>> center_real, center_imag, radius_major, radius_minor, angle = (
    ...     phasor_cluster_gmm(real, imag, clusters=2)
    ... )
    >>> center_real  # doctest: +SKIP
    (0.2, 0.4)

    """
    from sklearn.mixture import GaussianMixture

    if sigma <= 0.0:
        msg = f'{sigma=} <= 0'
        raise ValueError(msg)
    sigma = float(sigma)

    if clusters < 1:
        msg = f'{clusters=} < 1'
        raise ValueError(msg)

    coords = numpy.stack([real, imag], axis=-1).reshape((-1, 2))

    valid_data = ~numpy.isnan(coords).any(axis=1)
    coords = coords[valid_data]

    if coords.shape[0] < clusters:
        msg = f'number of valid data points ({coords.shape[0]}) < {clusters=}'
        raise ValueError(msg)

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

        match gmm.covariance_type:
            case 'full':
                cov = gmm.covariances_[i]
            case 'tied':
                cov = gmm.covariances_
            case 'diag':
                cov = numpy.diag(gmm.covariances_[i])
            case 'spherical':
                cov = numpy.eye(2) * gmm.covariances_[i]
            case _:
                msg = f'unknown {gmm.covariance_type=!r}'
                raise ValueError(msg)

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

    argsort = _argsort_clusters(
        center_real,
        center_imag,
        sort,
        area=[
            -major * minor
            for major, minor in zip(radius_major, radius_minor, strict=True)
        ],
    )

    return (
        tuple(center_real[i] for i in argsort),
        tuple(center_imag[i] for i in argsort),
        tuple(radius_major[i] for i in argsort),
        tuple(radius_minor[i] for i in argsort),
        tuple(angle[i] for i in argsort),
    )


def phasor_cluster_kmeans(
    mean: ArrayLike | None,
    real: ArrayLike,
    imag: ArrayLike,
    /,
    *,
    clusters: int = 1,
    sort: Literal['polar', 'phasor', 'size'] | None = None,
    **kwargs: Any,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    NDArray[numpy.intp],
]:
    """Return k-means clusters of phasor coordinates.

    Partition phasor coordinates into `clusters` groups using k-means
    clustering, assigning each phasor coordinate to the cluster with the
    nearest center.

    Parameters
    ----------
    mean : array_like or None
        Intensity of phasor coordinates.
        Must be same shape as `real`. Values must not be negative.
        If not None, phasor coordinates are weighted by intensity, such that
        coordinates of brighter pixels contribute more to the clusters.
        If None, all phasor coordinates contribute equally.
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    clusters : int, optional, default: 1
        Number of clusters to partition phasor coordinates into.
    sort : {'polar', 'phasor', 'size'}, optional
        Sorting method for output clusters.
        By default, use 'polar' sorting.

        - 'polar': Sort by polar coordinates (phase, then modulation).
        - 'phasor': Sort by phasor coordinates (imaginary, then real).
        - 'size': Sort by decreasing number of coordinates in cluster.

    **kwargs
        Optional arguments passed to :py:class:`sklearn.cluster.KMeans`.

        Common options include:

        - init : {'k-means++', 'random'}, method of initialization
        - n_init : int, number of initializations to perform
        - max_iter : int, maximum number of iterations
        - random_state : int, for reproducible results

    Returns
    -------
    center_mean : tuple of float
        Intensity centers of clusters.
        NaN if `mean` is None.
    center_real : tuple of float
        Real component of cluster centers.
    center_imag : tuple of float
        Imaginary component of cluster centers.
    labels : ndarray
        Zero-based index of cluster each phasor coordinate belongs to.
        Same shape as `real` and `imag`.
        Values are -1 where `mean`, `real`, or `imag` are NaN.

    Raises
    ------
    ValueError
        If `clusters` is less than 1.
        If the array shapes of `mean`, `real`, and `imag` do not match.
        If the number of valid (non-NaN) data points is less than `clusters`.
        If `mean` contains negative values, or sums to zero.
        If `sort` is not a valid sorting method.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_cluster.py`

    Notes
    -----
    Unlike :py:func:`phasor_cluster_gmm`, k-means clustering assigns each
    phasor coordinate to exactly one cluster. The clusters are separated by
    straight lines, which may not follow the elliptical shape of phasor
    distributions.

    Examples
    --------
    Partition phasor coordinates into two clusters and return the cluster
    centers and the cluster index of each coordinate:

    >>> center_mean, center_real, center_imag, labels = phasor_cluster_kmeans(
    ...     None, [0.1, 0.2, 0.5, 0.6], [0.1, 0.2, 0.5, 0.6], clusters=2
    ... )
    >>> center_real  # doctest: +NUMBER
    (0.15, 0.55)
    >>> labels
    array([0, 0, 1, 1])

    Phasor coordinates that are NaN are not assigned to any cluster:

    >>> phasor_cluster_kmeans(
    ...     None, [0.1, numpy.nan, 0.6], [0.1, 0.2, 0.6], clusters=2
    ... )[3]
    array([ 0, -1,  1])

    Weight phasor coordinates by intensity, moving the cluster centers
    towards the coordinates of brighter pixels:

    >>> center_mean, center_real, center_imag, labels = phasor_cluster_kmeans(
    ...     [1.0, 3.0, 3.0, 1.0],
    ...     [0.1, 0.2, 0.5, 0.6],
    ...     [0.1, 0.2, 0.5, 0.6],
    ...     clusters=2,
    ... )
    >>> center_mean  # doctest: +NUMBER
    (2.0, 2.0)
    >>> center_real  # doctest: +NUMBER
    (0.175, 0.525)

    """
    from sklearn.cluster import KMeans

    if clusters < 1:
        msg = f'{clusters=} < 1'
        raise ValueError(msg)

    real = numpy.asarray(real)
    imag = numpy.asarray(imag)

    if real.shape != imag.shape:
        msg = f'{real.shape=} != {imag.shape=}'
        raise ValueError(msg)

    coords = numpy.stack([real, imag], axis=-1).reshape((-1, 2))
    valid_data = ~numpy.isnan(coords).any(axis=1)

    weight = None
    if mean is not None:
        mean = numpy.asarray(mean)
        if mean.shape != real.shape:
            msg = f'{mean.shape=} != {real.shape=}'
            raise ValueError(msg)
        weight = mean.astype(numpy.float64).reshape(-1)
        valid_data &= ~numpy.isnan(weight)

    size = int(valid_data.sum())

    if size < clusters:
        msg = f'number of valid data points ({size}) < {clusters=}'
        raise ValueError(msg)

    if weight is not None:
        weight = weight[valid_data]
        if weight.min() < 0.0:
            msg = f'mean.min()={weight.min()} < 0'
            raise ValueError(msg)
        if weight.sum() <= 0.0:
            msg = f'mean.sum()={weight.sum()} <= 0'
            raise ValueError(msg)

    kwargs.pop('n_clusters', None)

    kmeans = KMeans(n_clusters=clusters, **kwargs)
    index = kmeans.fit_predict(coords[valid_data], sample_weight=weight)

    counts = numpy.bincount(index, minlength=clusters)
    if weight is None:
        center_mean = [math.nan] * clusters
    else:
        intensity_sums = numpy.bincount(
            index, weights=weight, minlength=clusters
        )
        center_mean = [float(value) for value in intensity_sums / counts]
    center_real = [float(value) for value in kmeans.cluster_centers_[:, 0]]
    center_imag = [float(value) for value in kmeans.cluster_centers_[:, 1]]

    argsort = _argsort_clusters(
        center_real,
        center_imag,
        sort,
        size=[-int(n) for n in counts],
    )

    relabel = numpy.empty(clusters, dtype=numpy.intp)
    relabel[argsort] = numpy.arange(clusters)

    labels = numpy.full(valid_data.size, -1, dtype=numpy.intp)
    labels[valid_data] = relabel[index]

    return (
        tuple(center_mean[i] for i in argsort),
        tuple(center_real[i] for i in argsort),
        tuple(center_imag[i] for i in argsort),
        labels.reshape(real.shape),
    )


def _argsort_clusters(
    center_real: Sequence[float],
    center_imag: Sequence[float],
    sort: str | None,
    /,
    **methods: Sequence[float],
) -> list[int]:
    """Return indices that sort clusters by specified method.

    Parameters
    ----------
    center_real : sequence of float
        Real component of cluster centers.
    center_imag : sequence of float
        Imaginary component of cluster centers.
        Must be same length as `center_real`.
    sort : str or None
        Sorting method: 'polar', 'phasor', or any name in `methods`.
        By default, use 'polar' sorting.
    **methods : sequence of float
        Additional sorting methods, mapping method name to the values by
        which clusters are sorted in increasing order.
        Must be same length as `center_real`.

    Returns
    -------
    list of int
        Indices of clusters in sorted order.

    Raises
    ------
    ValueError
        If `sort` is not a valid sorting method.

    """
    match sort:
        case 'polar' | None:

            def sort_key(i: int) -> Any:
                return (
                    math.atan2(center_imag[i], center_real[i]),
                    math.hypot(center_real[i], center_imag[i]),
                )

        case 'phasor':

            def sort_key(i: int) -> Any:
                return center_imag[i], center_real[i]

        case str() as method if method in methods:
            values = methods[method]

            def sort_key(i: int) -> Any:
                return values[i]

        case _:
            names = ', '.join(repr(s) for s in ('polar', 'phasor', *methods))
            msg = f'{sort=!r} not in {{{names}}}'
            raise ValueError(msg)

    if len(center_real) < 2:
        return list(range(len(center_real)))
    return sorted(range(len(center_real)), key=sort_key)
