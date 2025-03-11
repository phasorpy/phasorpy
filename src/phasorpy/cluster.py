"""Clustering techniques."""

from __future__ import annotations

__all__ = ['phasor_cluster_gmm']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

from typing import List, Tuple  # Needed to correct some mypy issues

import numpy
from sklearn.mixture import GaussianMixture


def phasor_cluster_gmm(
    real: ArrayLike,
    imag: ArrayLike,
    clusters: int,
    **kwargs: Any,
) -> tuple[list[Any], list[Any], list[float], list[float], list[Any]]:
    """Return clusters from phasor coordinates using Gaussian Mixture Model.

    This function is implemented given the results shown in [1]_,
    which shows the power of GMM applied in these cases.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    clusters : int, optional
        The numbers of clusters present in a sample, i.e. fluorophores
        that will be present, or are thought to be present.
    **kwargs : dict, optional
        Additional keyword arguments passed to
        :py:class:`sklearn.mixture.GaussianMixture`.

    Returns
    -------
    centers_real : array_like, shape (clusters,)
        Real coordinates of ellipses centers.
    centers_imag : array_like, shape (clusters,)
        Imaginary coordinates of ellipses centers.
    radius : array_like, shape (clusters,)
        Radii of ellipses along semi-major axis.
    radius_minor : array_like, shape (clusters,)
        Radii of ellipses along semi-minor axis.
    angle : array_like, optional, shape (clusters,)
        Rotation angles of semi-major axes of ellipses in radians.

    Raises
    ------
    ValueError
        The array shapes of `real` and `imag` do not match.
        clusters is not an integer.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_cluster.py`

    References
    ----------
    .. [1] Vallmitjana A, Torrado B & Gratton E. `Phasor-based image segmentation:
      machine learning clustering techniques
      <https://doi.org/10.1364/BOE.422766>`_.
      *Biomed. Opt. Express 12(6), 3410–3422 (2021).

    Examples
    --------
    Create a phasor from lifetimes, set the component fractions, and frequency:

    >>> component_lifetimes = [1.0, 8.0]
    >>> component_fractions = [0.7, 0.3]
    >>> frequency = 80.0  # MHz

    >>> from phasorpy.phasor import phasor_from_lifetime

    >>> real, imag = phasor_from_lifetime(
    ...     frequency, component_lifetimes, component_fractions
    ... )

    Create synthetic data from phasor coordinates:

    >>> real, imag = numpy.random.multivariate_normal(
    ...     (0.6, 0.4), [[3e-3, -1e-3], [-1e-3, 1e-3]], (256, 256)
    ... ).T

    Set the number of clusters and get the coordinates for the center
    and radii of the ellipses.

    >>> clusters = 2
    >>> centers_real, centers_imag, radius, radius_minor, angles = (
    ...     phasor_cluster_gmm(real, imag, clusters=clusters)
    ... )

    """

    real = numpy.asarray(real)
    imag = numpy.asarray(imag)

    if real.shape != imag.shape:
        raise ValueError(f'{real.shape=} != {imag.shape=}')
    if not isinstance(clusters, int):
        raise ValueError(
            f"clusters is expected to be an integer, but got {clusters=} of type {type(clusters)}"
        )

    # Reshape and stack the coordinate
    X = numpy.column_stack((real.ravel(), imag.ravel()))

    # Remove any NaN values if present
    X = X[~numpy.isnan(X).any(axis=1)]

    # Fit Gaussian Mixture Model
    gm = GaussianMixture(n_components=clusters, n_init=10)
    gm.fit(X)

    # Generate the empty arrays
    angles = numpy.array([]).reshape(0, 1)
    radio = numpy.array([]).reshape(0, 2)

    for n in range(clusters):
        cov = gm.covariances_[n][:2, :2]
        eigenvalues, eigenvectors = numpy.linalg.eigh(cov)

        # Get the index of the largest eigenvalue (major axis)
        idx = numpy.argmax(eigenvalues)

        # Use the eigenvector corresponding to the largest eigenvalue
        major_axis = eigenvectors[:, idx]

        # Calculate angle using arctan2
        angle = numpy.arctan2(major_axis[1], major_axis[0])

        # Ensure the angle is in the range [0, π]
        if angle < 0:
            angle += numpy.pi

        # Scale eigenvalues for confidence ellipse
        eigenvalues = 2.0 * numpy.sqrt(2.0) * numpy.sqrt(eigenvalues)

        angles = numpy.vstack([angles, [angle]])
        radio = numpy.vstack([radio, eigenvalues])

    # Generate the empty arrays
    centers_real = numpy.array([]).reshape(0, 1)
    centers_imag = numpy.array([]).reshape(0, 1)

    for n in range(clusters):
        centers_real = numpy.vstack([centers_real, [gm.means_[n, 0]]])
        centers_imag = numpy.vstack([centers_imag, [gm.means_[n, 1]]])

    centers_real = centers_real.T
    centers_imag = centers_imag.T

    angles_list = [i for fil in angles for i in fil]  # flattens the list
    centers_real_list = [
        i for fil in centers_real for i in fil
    ]  # flattens the list
    centers_imag_list = [
        i for fil in centers_imag for i in fil
    ]  # flattens the list

    radius = []
    radius_minor = []

    for n in range(len(radio)):
        radius.append(float(radio[n, 1]))
        radius_minor.append(float(radio[n, 0]))

    return (
        centers_real_list,
        centers_imag_list,
        radius,
        radius_minor,
        angles_list,
    )
