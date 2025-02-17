
"""Clustering techniques."""

from __future__ import annotations

__all__ = ['gaussian_mixture_model']

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._typing import Any, ArrayLike, NDArray

import numpy

import math
from sklearn.mixture import GaussianMixture

def gaussian_mixture_model(
    real: ArrayLike,
    imag: ArrayLike,
    n_components: int,
    **kwargs: Any,
) -> NDArray[numpy.bool_]:
    """Returns centers, radii and angles of clusters from 
     the application of GMM.

    Parameters
    ----------
    real : array_like
        Real component of phasor coordinates.
    imag : array_like
        Imaginary component of phasor coordinates.
    n_components : int, optional
        The numbers of components present in a sample, i.e. fluorophores
        that will be present, or are thought to be present.

    Returns
    -------
    centers_real : array_like, shape (n_components,)
        Real coordinates of ellipses centers.
    centers_imag : array_like, shape (n_components,)
        Imaginary coordinates of ellipses centers.
    radio : array_like, shape (n_components,)
        Radii of ellipses along semi-major axis.
    radius_minor : array_like, shape (n_components,)
        Radii of ellipses along semi-minor axis.
    angle : array_like, optional, shape (n_components,)
        Rotation angles of semi-major axes of ellipses in radians.


    #TODO: Raises, See Also (if there's some), and Examples
    Raises
    ------
    ValueError
        The array shapes of `real` and `imag` do not match.
        The array shapes of `center*`, `radius*`, or `angle` have more than
        one dimension.

    See Also
    --------
    :ref:`sphx_glr_tutorials_api_phasorpy_cursors.py`



    Examples
    --------
    Create a :

    >>> component_lifetimes = [1.0, 8.0]
    >>> component_fractions = [0.7, 0.3]


    Create masks for two elliptic cursors with different radii:

    >>> mask_from_elliptic_cursor(
    ...     [0.2, 0.5],
    ...     [0.4, 0.5],
    ...     [0.2, 0.5],
    ...     [0.4, 0.5],
    ...     radius=[0.1, 0.05],
    ...     radius_minor=[0.15, 0.1],
    ...     angle=[math.pi, math.pi / 2],
    ... )
    array([[ True, False],
           [False,  True]])

    """

    real = numpy.asarray(real)
    imag = numpy.asarray(imag)

    real_reshaped = [ i for fil in real for i in fil] #flattens the list
    imag_reshaped = [ i for fil in imag for i in fil] #flattens the list

    nan_found = 0

    # Checks if there is a nan in any of the input arrays.
    if (any(numpy.isnan(real_reshaped)) or any(numpy.isnan(imag_reshaped))):

        nan_found = 1

        real_reshaped = numpy.nan_to_num(real_reshaped) #sklearn does not like nan's at all.
        imag_reshaped = numpy.nan_to_num(imag_reshaped)
        
        # It has to be number of components + one,
        # I could not make it work around the nan's,
        # and it creates another cluster on zero
        # in order to solve that problem.
        n_components = n_components + 1

    X = numpy.column_stack([real_reshaped,imag_reshaped])

    # Fit Gaussian Mixture Model
    gm = GaussianMixture(n_components= n_components, n_init=10, **kwargs)
    gm.fit(X)

    #Generate the empty arrays
    angles = numpy.array([]).reshape(0, 1)
    radius = numpy.array([]).reshape(0, 2)

    for n in range(n_components):
        cov = gm.covariances_[n][:2, :2]
        eigenvalues, eigenvectors = numpy.linalg.eigh(cov)

        u = eigenvectors[0] / numpy.linalg.norm(eigenvectors[0])

        angle = numpy.arctan2(u[1], u[0])
        angle = 180 * angle / numpy.pi  # convert to degrees
        eigenvalues = 2.0 * numpy.sqrt(2.0) * numpy.sqrt(eigenvalues)
        angle = math.radians(angle) #convert to radians
        
        angles = numpy.vstack([angles, [angle]])
        radius = numpy.vstack([radius, eigenvalues]) 

    #Undoing what was done to avoid working with nan's
    for n in range(n_components):
        magnitude = numpy.linalg.norm(gm.means_[n,:])

        
        if (nan_found):        
            if (magnitude == 0):
                index_del = n
                # Variable only defined if there was a nan.

    #Generate the empty arrays
    centers_real = numpy.array([]).reshape(0, 1)
    centers_imag = numpy.array([]).reshape(0, 1)

    for n in range(n_components):

        if (not nan_found) or (nan_found and n != index_del): 
            centers_real = numpy.vstack([centers_real, [gm.means_[n, 0]]])
            centers_imag = numpy.vstack([centers_imag, [gm.means_[n, 1]]])

    centers_real = centers_real.T
    centers_imag = centers_imag.T
    
    if (nan_found):
        angles = numpy.delete(angles, index_del, axis=0)  # Deletes row

    angles = [ i for fil in angles for i in fil] #flattens the list
    centers_real = [ i for fil in centers_real for i in fil] #flattens the list
    centers_imag = [ i for fil in centers_imag for i in fil] #flattens the list

    radio = []
    radius_minor = []

    for n in range(len(radius)):
        if (not nan_found) or (nan_found and n != index_del): 
            radio.append(float(radius[n, 1]))
            radius_minor.append(float(radius[n, 0]))

    return centers_real, centers_imag, radio, radius_minor, angles
