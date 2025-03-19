"""Tests for the phasorpy.cluster module."""

import numpy
import pytest
from phasorpy.cluster import phasor_cluster_gmm
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
)

@pytest.mark.parametrize("clusters", [1, 2, 3])
def test_phasor_cluster_gmm_basic(clusters):
    numpy.random.seed(42)
    real1, imag1 = numpy.random.multivariate_normal([2, 3], [[0.3, 0.1], [0.1, 0.2]], 100).T
    real2, imag2 = numpy.random.multivariate_normal([5, 6], [[0.2, -0.1], [-0.1, 0.3]], 100).T
    real = numpy.concatenate([real1, real2])
    imag = numpy.concatenate([imag1, imag2])
    
    centers_real, centers_imag, radius_major, radius_minor, angles = phasor_cluster_gmm(real, imag, clusters=clusters)
    
    assert len(centers_real) == clusters
    assert len(centers_imag) == clusters
    assert len(radius_major) == clusters
    assert len(radius_minor) == clusters
    assert len(angles) == clusters


def test_phasor_cluster_gmm_invalid_shapes():
    real = numpy.array([1, 2, 3])
    imag = numpy.array([1, 2])  # Different shape
    with pytest.raises(ValueError, match="real.shape"): 
        phasor_cluster_gmm(real, imag)

def test_phasor_cluster_gmm_too_few_points():
    real = numpy.array([1.0])
    imag = numpy.array([1.0])
    with pytest.raises(ValueError, match="Not enough points"):
        phasor_cluster_gmm(real, imag, clusters=2)

def test_phasor_cluster_gmm_nan_values():
    real = numpy.array([1.0, numpy.nan, 2.0])
    imag = numpy.array([1.0, 2.0, numpy.nan])

    with pytest.raises(ValueError, match=r'Not enough points \(1\) for components \(1\)'):
        phasor_cluster_gmm(real, imag, clusters=1)


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diag", "spherical"])
def test_phasor_cluster_gmm_covariance(covariance_type):

    real = numpy.array([1.0, 2.0, 3.0])
    imag = numpy.array([1.0, 2.0, 3.0])

    centers_real, centers_imag, radius_major, radius_minor, angles = phasor_cluster_gmm(
        real, imag, clusters=2, covariance_type=covariance_type
    )

    assert len(centers_real) == 2
    assert len(centers_imag) == 2

    if covariance_type == "full":
        assert isinstance(radius_major, tuple) and len(radius_major) == 2
    elif covariance_type == "tied":
        assert isinstance(radius_major, tuple) and len(radius_major) == 2
    elif covariance_type == "diag":
        assert isinstance(radius_major, tuple) and len(radius_major) == 2
    elif covariance_type == "spherical":
        assert isinstance(radius_major, tuple) and len(radius_major) == 2

@pytest.mark.parametrize(
    "real, imag, match",
    [
        (numpy.array([1, 2, 3]), numpy.array([1, 2]), "real.shape"),  # Mismatched shapes
        (numpy.array([1.0]), numpy.array([1.0]), "Not enough points"),  # Too few points
        (numpy.array([1.0, numpy.nan, 2.0]), numpy.array([1.0, 2.0, numpy.nan]), "Not enough points"),  # NaN handling
    ],
)
def test_phasor_cluster_gmm_exceptions(real, imag, match):
    with pytest.raises(ValueError, match=match):
        phasor_cluster_gmm(real, imag, clusters=2)


