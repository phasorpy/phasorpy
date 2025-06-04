"""Tests for the phasorpy.cluster module."""

import numpy
import pytest
from numpy.testing import assert_allclose

from phasorpy.cluster import phasor_cluster_gmm

numpy.random.seed(42)


@pytest.mark.parametrize('clusters', [1, 2, 3])
def test_phasor_cluster_gmm_basic(clusters):
    real1, imag1 = numpy.random.multivariate_normal(
        [0.2, 0.3], [[3e-3, 1e-3], [1e-3, 2e-3]], 2**15
    ).T
    real2, imag2 = numpy.random.multivariate_normal(
        [0.4, 0.5], [[2e-3, -1e-3], [-1e-3, 3e-3]], 2**14
    ).T
    real = numpy.concatenate([real1, real2])
    imag = numpy.concatenate([imag1, imag2])
    centers_real, centers_imag, radius_major, radius_minor, angle = (
        phasor_cluster_gmm(real, imag, clusters=clusters)
    )
    assert len(centers_real) == clusters
    assert len(centers_imag) == clusters
    assert len(radius_major) == clusters
    assert len(radius_minor) == clusters
    assert len(angle) == clusters
    if clusters == 2:
        # TODO: custers are not returned in any particular order
        assert_allclose(centers_real, [0.4, 0.2], atol=0.01)
        assert_allclose(centers_imag, [0.5, 0.3], atol=0.01)
        assert_allclose(radius_major, [0.17, 0.17], atol=0.01)
        assert_allclose(radius_minor, [0.105, 0.105], atol=0.01)
        assert_allclose(angle, [2.11, 0.54], atol=0.2)


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
