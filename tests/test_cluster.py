"""Tests for the phasorpy.cluster module."""

import numpy
import pytest

from phasorpy.cluster import phasor_cluster_gmm


@pytest.mark.parametrize('clusters', [1, 2, 3])
def test_phasor_cluster_gmm_basic(clusters):
    numpy.random.seed(42)
    real1, imag1 = numpy.random.multivariate_normal(
        [2, 3], [[0.3, 0.1], [0.1, 0.2]], 100
    ).T
    real2, imag2 = numpy.random.multivariate_normal(
        [5, 6], [[0.2, -0.1], [-0.1, 0.3]], 100
    ).T
    real = numpy.concatenate([real1, real2])
    imag = numpy.concatenate([imag1, imag2])

    centers_real, centers_imag, radius_major, radius_minor, angles = (
        phasor_cluster_gmm(real, imag, clusters=clusters)
    )

    assert len(centers_real) == clusters
    assert len(centers_imag) == clusters
    assert len(radius_major) == clusters
    assert len(radius_minor) == clusters
    assert len(angles) == clusters


def test_phasor_cluster_gmm_invalid_shapes():
    with pytest.raises(ValueError):
        phasor_cluster_gmm([1, 2, 3], [1, 2])


@pytest.mark.parametrize(
    'covariance_type', ['full', 'tied', 'diag', 'spherical']
)
def test_phasor_cluster_gmm_covariance(covariance_type):
    centers_real, centers_imag, radius_major, radius_minor, angles = (
        phasor_cluster_gmm(
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            clusters=2,
            covariance_type=covariance_type,
        )
    )
    assert len(centers_real) == 2
    assert len(centers_imag) == 2
    if covariance_type == 'full':
        assert isinstance(radius_major, tuple) and len(radius_major) == 2
    elif covariance_type == 'tied':
        assert isinstance(radius_major, tuple) and len(radius_major) == 2
    elif covariance_type == 'diag':
        assert isinstance(radius_major, tuple) and len(radius_major) == 2
    elif covariance_type == 'spherical':
        assert isinstance(radius_major, tuple) and len(radius_major) == 2


@pytest.mark.parametrize(
    'real, imag',
    [
        ([1, 2, 3], [1, 2]),
        ([1.0], [1.0]),
        ([1.0, numpy.nan, 2.0], [1.0, 2.0, numpy.nan]),
    ],
)
def test_phasor_cluster_gmm_exceptions(real, imag):
    with pytest.raises(ValueError):
        phasor_cluster_gmm(real, imag, clusters=2)


@pytest.mark.parametrize(
    'real, imag',
    [
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
    ],
)
def test_phasor_cluster_gmm_column_stack(real, imag):
    centers_real, centers_imag, *_ = phasor_cluster_gmm(real, imag, clusters=1)
    assert len(centers_real) == 1
    assert len(centers_imag) == 1


# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, unreachable, redundant-expr"
