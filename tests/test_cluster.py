# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""Test the phasorpy.cluster module."""

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from phasorpy._typing import ArrayLike
from phasorpy.cluster import phasor_cluster_gmm, phasor_cluster_kmeans

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

    # invalid sort method, also with a single cluster
    with pytest.raises(ValueError):
        phasor_cluster_gmm([1, 2], [1, 2], clusters=2, sort='invalid')  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        phasor_cluster_gmm([1, 2], [1, 2], clusters=1, sort='invalid')  # type: ignore[arg-type]

    # sorting method of other cluster function
    with pytest.raises(ValueError):
        phasor_cluster_gmm([1, 2], [1, 2], clusters=2, sort='size')  # type: ignore[arg-type]

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


@pytest.mark.parametrize('clusters', [1, 2, 3])
@pytest.mark.parametrize('sort', ['polar', 'phasor', 'size'])
def test_phasor_cluster_kmeans_basic(clusters: int, sort: str) -> None:
    """Test phasor_cluster_kmeans function with basic cases."""
    real1, imag1 = rng.multivariate_normal(
        [0.2, 0.3], [[3e-3, 1e-3], [1e-3, 1e-3]], 2**15
    ).T
    real2, imag2 = rng.multivariate_normal(
        [0.3, 0.5], [[1e-3, -0.5e-3], [-0.5e-3, 1e-3]], 2**14
    ).T
    real = numpy.concatenate([real1, real2])
    imag = numpy.concatenate([imag1, imag2])
    center_mean, center_real, center_imag, labels = phasor_cluster_kmeans(
        None,
        real,
        imag,
        clusters=clusters,
        sort=sort,  # type: ignore[arg-type]
        random_state=42,
    )
    assert len(center_mean) == clusters
    assert len(center_real) == clusters
    assert len(center_imag) == clusters
    assert numpy.isnan(center_mean).all()  # mean is None
    assert labels.shape == real.shape
    assert labels.min() == 0
    assert labels.max() == clusters - 1
    if clusters == 2:
        assert_allclose(center_real, [0.2, 0.3], atol=0.01)
        assert_allclose(center_imag, [0.3, 0.5], atol=0.01)
        assert_allclose(numpy.bincount(labels), [2**15, 2**14], rtol=0.05)


def test_phasor_cluster_kmeans_labels() -> None:
    """Test phasor_cluster_kmeans function assigns nearest cluster."""
    real1, imag1 = rng.multivariate_normal(
        [0.6, 0.2], [[1e-4, 0], [0, 1e-4]], 100
    ).T
    real2, imag2 = rng.multivariate_normal(
        [0.2, 0.6], [[1e-4, 0], [0, 1e-4]], 1000
    ).T
    real = numpy.concatenate([real1, real2])
    imag = numpy.concatenate([imag1, imag2])

    # 'polar' sorting: small cluster has lower phase, hence comes first
    _, center_real, center_imag, labels = phasor_cluster_kmeans(
        None, real, imag, clusters=2, random_state=42
    )
    assert_allclose(center_real, [0.6, 0.2], atol=0.01)
    assert_allclose(center_imag, [0.2, 0.6], atol=0.01)
    assert_allclose(numpy.bincount(labels), [100, 1000])
    assert_array_equal(labels, [0] * 100 + [1] * 1000)

    # 'size' sorting: large cluster comes first
    _, center_real, center_imag, labels = phasor_cluster_kmeans(
        None, real, imag, clusters=2, sort='size', random_state=42
    )
    assert_allclose(center_real, [0.2, 0.6], atol=0.01)
    assert_allclose(center_imag, [0.6, 0.2], atol=0.01)
    assert_array_equal(labels, [1] * 100 + [0] * 1000)

    # labels index the returned, sorted centers
    distance = numpy.hypot(
        real[:, None] - numpy.asarray(center_real),
        imag[:, None] - numpy.asarray(center_imag),
    )
    assert_array_equal(distance.argmin(axis=1), labels)


def test_phasor_cluster_kmeans_shape() -> None:
    """Test phasor_cluster_kmeans function preserves shape of input."""
    real = rng.normal([[0.2], [0.6]], 0.01, (2, 100)).reshape(4, 5, 10)
    imag = rng.normal([[0.3], [0.5]], 0.01, (2, 100)).reshape(4, 5, 10)
    *_, labels = phasor_cluster_kmeans(
        None, real, imag, clusters=2, random_state=42
    )
    assert labels.shape == (4, 5, 10)
    assert labels.dtype.kind == 'i'

    # scalar input
    center_mean, center_real, center_imag, labels = phasor_cluster_kmeans(
        2.0, 0.5, 0.3
    )
    assert center_mean == (2.0,)
    assert center_real == (0.5,)
    assert center_imag == (0.3,)
    assert labels.shape == ()
    assert labels[()] == 0


def test_phasor_cluster_kmeans_nan() -> None:
    """Test phasor_cluster_kmeans function with NaN coordinates."""
    _, center_real, center_imag, labels = phasor_cluster_kmeans(
        None,
        [0.1, numpy.nan, 0.5, 0.6, 0.2],
        [0.1, 0.2, 0.5, 0.6, numpy.nan],
        clusters=2,
        random_state=42,
    )
    assert_allclose(center_real, [0.1, 0.55], atol=1e-6)
    assert_allclose(center_imag, [0.1, 0.55], atol=1e-6)
    assert_array_equal(labels, [0, -1, 1, 1, -1])

    # all NaN
    with pytest.raises(ValueError):
        phasor_cluster_kmeans(
            None, [numpy.nan, numpy.nan], [numpy.nan, numpy.nan]
        )


def test_phasor_cluster_kmeans_weighted() -> None:
    """Test phasor_cluster_kmeans function with intensity weighting."""
    real = [0.1, 0.2, 0.5, 0.6]
    imag = [0.1, 0.2, 0.5, 0.6]
    unweighted = phasor_cluster_kmeans(None, real, imag, clusters=2)

    # uniform intensity is equivalent to no weighting, except center_mean
    for mean in ([1.0] * 4, [3.0] * 4):
        center_mean, center_real, center_imag, labels = phasor_cluster_kmeans(
            mean, real, imag, clusters=2
        )
        assert_allclose(center_mean, [mean[0]] * 2)
        assert_allclose(center_real, unweighted[1])
        assert_allclose(center_imag, unweighted[2])
        assert_array_equal(labels, unweighted[3])

    # centers are intensity-weighted means of coordinates in each cluster
    center_mean, center_real, center_imag, labels = phasor_cluster_kmeans(
        [1.0, 3.0, 3.0, 1.0], real, imag, clusters=2
    )
    assert_allclose(center_mean, [2.0, 2.0], atol=1e-6)
    assert_allclose(center_real, [0.175, 0.525], atol=1e-6)
    assert_allclose(center_imag, [0.175, 0.525], atol=1e-6)
    assert_array_equal(labels, [0, 0, 1, 1])

    # coordinates with NaN intensity are not assigned to any cluster
    *_, labels = phasor_cluster_kmeans(
        [1.0, numpy.nan, 1.0, 1.0], real, imag, clusters=2
    )
    assert_array_equal(labels, [0, -1, 1, 1])

    # integer intensity, multidimensional shape, and sorting
    coords = numpy.asarray(real).reshape(2, 2)
    _, center_real, _, labels = phasor_cluster_kmeans(
        numpy.asarray([[1, 3], [3, 1]]),
        coords,
        coords,
        clusters=2,
        sort='size',
    )
    assert labels.shape == (2, 2)
    assert_allclose(sorted(center_real), [0.175, 0.525], atol=1e-6)


def test_phasor_cluster_kmeans_weighted_shifts_centers() -> None:
    """Test phasor_cluster_kmeans intensity weighting shifts centers."""
    real1, imag1 = rng.multivariate_normal(
        [0.2, 0.3], [[1e-3, 0], [0, 1e-3]], 2**12
    ).T
    real2, imag2 = rng.multivariate_normal(
        [0.5, 0.5], [[1e-3, 0], [0, 1e-3]], 2**12
    ).T
    real = numpy.concatenate([real1, real2])
    imag = numpy.concatenate([imag1, imag2])
    # brighter towards larger real coordinates
    mean = numpy.exp(4.0 * real)

    _, center_real, _, _ = phasor_cluster_kmeans(
        None, real, imag, clusters=2, n_init=10, random_state=42
    )
    _, weighted_real, _, _ = phasor_cluster_kmeans(
        mean, real, imag, clusters=2, n_init=10, random_state=42
    )
    assert numpy.all(numpy.asarray(weighted_real) > numpy.asarray(center_real))


def test_phasor_cluster_kmeans_kwargs() -> None:
    """Test phasor_cluster_kmeans function with sklearn arguments."""
    real = [0.1, 0.11, 0.5, 0.51]
    imag = [0.1, 0.11, 0.5, 0.51]
    _, center_real, _, labels = phasor_cluster_kmeans(
        None, real, imag, clusters=2, init='random', n_init=10, random_state=42
    )
    assert_allclose(center_real, [0.105, 0.505], atol=1e-6)
    assert_array_equal(labels, [0, 0, 1, 1])

    # n_clusters is ignored in favor of clusters
    _, center_real, *_ = phasor_cluster_kmeans(
        None, real, imag, clusters=2, n_clusters=3, random_state=42
    )
    assert len(center_real) == 2


def test_phasor_cluster_kmeans_exceptions() -> None:
    """Test phasor_cluster_kmeans function raises exceptions."""
    # shape mismatch
    with pytest.raises(ValueError):
        phasor_cluster_kmeans(None, [1, 2, 3], [1, 2])

    # invalid sort method, also with a single cluster
    with pytest.raises(ValueError):
        phasor_cluster_kmeans(None, [1, 2], [1, 2], clusters=2, sort='invalid')  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        phasor_cluster_kmeans(None, [1, 2], [1, 2], clusters=1, sort='invalid')  # type: ignore[arg-type]

    # sorting method of other cluster function
    with pytest.raises(ValueError):
        phasor_cluster_kmeans(None, [1, 2], [1, 2], clusters=2, sort='area')  # type: ignore[arg-type]

    # clusters < 1
    with pytest.raises(ValueError):
        phasor_cluster_kmeans(None, [1, 2, 3], [1, 2, 3], clusters=0)

    with pytest.raises(ValueError):
        phasor_cluster_kmeans(None, [1, 2, 3], [1, 2, 3], clusters=-1)

    # insufficient data points for clusters
    with pytest.raises(ValueError):
        phasor_cluster_kmeans(None, [1, 2], [1, 2], clusters=3)

    with pytest.raises(ValueError):
        phasor_cluster_kmeans(
            None, [1.0, numpy.nan, 2.0], [1.0, 2.0, numpy.nan], clusters=2
        )

    # mean shape mismatch
    with pytest.raises(ValueError):
        phasor_cluster_kmeans([1, 2, 3], [1, 2], [1, 2])

    # negative mean
    with pytest.raises(ValueError):
        phasor_cluster_kmeans([1.0, -1.0], [1, 2], [1, 2])

    # mean sums to zero
    with pytest.raises(ValueError):
        phasor_cluster_kmeans([0.0, 0.0], [1, 2], [1, 2])

    # all mean is NaN
    with pytest.raises(ValueError):
        phasor_cluster_kmeans([numpy.nan] * 2, [1, 2], [1, 2])
