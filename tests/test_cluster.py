# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""Test the phasorpy.cluster module."""

import numpy
import pytest
from numpy.testing import assert_allclose

from phasorpy._typing import ArrayLike
from phasorpy.cluster import phasor_cluster_gmm

rng = numpy.random.default_rng(42)


@pytest.mark.parametrize('clusters', [1, 2, 3])
@pytest.mark.parametrize('sort', ['polar', 'phasor', 'area'])
def test_phasor_cluster_gmm_basic(clusters: int, sort: str) -> None:
    """Test phasor_cluster_gmm function with basic cases."""
    real1, imag1 = rng.multivariate_normal(
        [0.2, 0.3], [[3e-3, 1e-3], [1e-3, 1e-3]], 2**15
    ).T
    real2, imag2 = rng.multivariate_normal(
        [0.3, 0.5], [[1e-3, -0.5e-3], [-0.5e-3, 1e-3]], 2**14
    ).T
    real = numpy.concatenate([real1, real2])
    imag = numpy.concatenate([imag1, imag2])
    center_real, center_imag, radius_major, radius_minor, angle = (
        phasor_cluster_gmm(real, imag, clusters=clusters, sort=sort)  # type: ignore[arg-type]
    )
    assert len(center_real) == clusters
    assert len(center_imag) == clusters
    assert len(radius_major) == clusters
    assert len(radius_minor) == clusters
    assert len(angle) == clusters
    if clusters == 2:
        assert_allclose(center_real, [0.2, 0.3], atol=0.01)
        assert_allclose(center_imag, [0.3, 0.5], atol=0.01)
        assert_allclose(radius_major, [0.165, 0.108], atol=0.02)
        assert_allclose(radius_minor, [0.068, 0.063], atol=0.02)
        assert_allclose(angle, [0.396, 2.369], atol=0.2)


def test_phasor_cluster_gmm_invalid_shapes() -> None:
    """Test phasor_cluster_gmm function with invalid shapes."""
    # shape mismatch
    with pytest.raises(ValueError):
        phasor_cluster_gmm([1, 2, 3], [1, 2])

    # invalid sort method
    with pytest.raises(ValueError):
        phasor_cluster_gmm([1, 2], [1, 2], clusters=2, sort='invalid')  # type: ignore[arg-type]

    # clusters < 1
    with pytest.raises(ValueError):
        phasor_cluster_gmm([1, 2, 3], [1, 2, 3], clusters=0)

    with pytest.raises(ValueError):
        phasor_cluster_gmm([1, 2, 3], [1, 2, 3], clusters=-1)

    # insufficient data points for clusters
    with pytest.raises(ValueError):
        phasor_cluster_gmm([1, 2], [1, 2], clusters=3)


def test_phasor_cluster_gmm_invalid_sigma() -> None:
    """Test phasor_cluster_gmm function with invalid sigma."""
    with pytest.raises(ValueError):
        phasor_cluster_gmm([1, 2], [1, 2], sigma=-1.0)


@pytest.mark.parametrize(
    'covariance_type', ['full', 'tied', 'diag', 'spherical']
)
def test_phasor_cluster_gmm_covariance(covariance_type: str) -> None:
    """Test phasor_cluster_gmm function with different covariance types."""
    center_real, center_imag, radius_major, _radius_minor, _angles = (
        phasor_cluster_gmm(
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
            clusters=2,
            covariance_type=covariance_type,
        )
    )
    assert len(center_real) == 2
    assert len(center_imag) == 2
    assert isinstance(radius_major, tuple)
    assert len(radius_major) == 2


@pytest.mark.parametrize(
    ('real', 'imag'),
    [
        ([1, 2, 3], [1, 2]),
        ([1.0], [1.0]),
        ([1.0, numpy.nan, 2.0], [1.0, 2.0, numpy.nan]),
    ],
)
def test_phasor_cluster_gmm_exceptions(
    real: ArrayLike, imag: ArrayLike
) -> None:
    """Test phasor_cluster_gmm function raises exceptions on invalid input."""
    with pytest.raises(ValueError):
        phasor_cluster_gmm(real, imag, clusters=2)


@pytest.mark.parametrize(
    ('real', 'imag'),
    [
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
        ([1, 2, 3, 4], [1, 2, 3, 4]),
    ],
)
def test_phasor_cluster_gmm_column_stack(
    real: ArrayLike, imag: ArrayLike
) -> None:
    """Test phasor_cluster_gmm function with column stack input."""
    center_real, center_imag, *_ = phasor_cluster_gmm(real, imag, clusters=1)
    assert len(center_real) == 1
    assert len(center_imag) == 1
