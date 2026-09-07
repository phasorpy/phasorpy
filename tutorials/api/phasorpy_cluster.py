# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""
Clusters
========

The :py:mod:`phasorpy.cluster` module provides functions to automatically
find clusters in phasor coordinates, either by fitting ellipses using a
Gaussian mixture model, or by assigning every phasor coordinate to a cluster
using k-means clustering.

"""

# %%
# Import required modules, functions, and classes:

import numpy

from phasorpy.cluster import phasor_cluster_gmm, phasor_cluster_kmeans
from phasorpy.color import CATEGORICAL
from phasorpy.plot import PhasorPlot

# %%
# Synthetic phasor coordinates
# ----------------------------
#
# Create a synthetic distribution of phasor coordinates containing two
# partially overlapping clusters of different size and shape. The second
# cluster is twenty times brighter than the first:

rng = numpy.random.default_rng(42)

real1, imag1 = rng.multivariate_normal(
    [0.40, 0.33], [[2.4e-3, -4.0e-4], [-4.0e-4, 9.0e-4]], 4000
).T
real2, imag2 = rng.multivariate_normal(
    [0.56, 0.29], [[1.8e-3, -3.0e-4], [-3.0e-4, 7.0e-4]], 2000
).T

real = numpy.concatenate([real1, real2])
imag = numpy.concatenate([imag1, imag2])
mean = numpy.concatenate(
    [rng.normal(10.0, 2.0, 4000), rng.normal(200.0, 40.0, 2000)]
)

# %%
# Plot the distribution as a two-dimensional histogram:

plot = PhasorPlot(title='Synthetic phasor coordinates')
plot.hist2d(real, imag, cmap='Greys')
plot.show()

# %%
# Gaussian mixture model
# ----------------------
#
# The :py:func:`phasorpy.cluster.phasor_cluster_gmm` function fits a Gaussian
# mixture model to the phasor coordinates and returns the parameters of
# ellipses describing the clusters:

gmm_real, gmm_imag, radius, radius_minor, angle = phasor_cluster_gmm(
    real, imag, clusters=2, random_state=42
)

print(f'centers: {gmm_real[0]:.3f}, {gmm_imag[0]:.3f}')
print(f'         {gmm_real[1]:.3f}, {gmm_imag[1]:.3f}')

# %%
# Both clustering functions start from a random initialization. Arguments
# such as ``random_state`` are passed to the underlying scikit-learn
# estimators and are used throughout this tutorial to obtain reproducible
# results.

# %%
# The ellipses have the same parameters as elliptical cursors and can be
# plotted with :py:meth:`phasorpy.plot.PhasorPlot.cursor`:

plot = PhasorPlot(title='Elliptical clusters')
plot.hist2d(real, imag, cmap='Greys')
plot.cursor(
    gmm_real,
    gmm_imag,
    radius=radius,
    radius_minor=radius_minor,
    angle=angle,
    color=CATEGORICAL[:2],
)
for re, im, color in zip(gmm_real, gmm_imag, CATEGORICAL[:2], strict=True):
    plot.plot(
        re,
        im,
        'o',
        color=color,
        markersize=7,
    )
plot.show()

# %%
# K-means clustering
# ------------------
#
# Instead of describing clusters by ellipses, the
# :py:func:`phasorpy.cluster.phasor_cluster_kmeans` function partitions the
# phasor coordinates into a fixed number of clusters, assigning each phasor
# coordinate to the cluster with the nearest center.
#
# Pass ``None`` as the first argument to let all phasor coordinates
# contribute equally:

center_mean, center_real, center_imag, labels = phasor_cluster_kmeans(
    None, real, imag, clusters=2, random_state=42
)

# %%
# The returned ``labels`` array has the same shape as the phasor coordinates
# and contains the index of the cluster each coordinate belongs to.
# Use it to plot the phasor coordinates in the color of their cluster:

plot = PhasorPlot(title='K-means clusters')
for index, color in enumerate(CATEGORICAL[:2]):
    plot.plot(
        real[labels == index],
        imag[labels == index],
        color=color,
        markersize=1,
        alpha=0.5,
        label=f'Cluster {index}',
    )
for re, im, color in zip(
    center_real, center_imag, CATEGORICAL[:2], strict=True
):
    plot.plot(
        re,
        im,
        'o',
        color=color,
        markeredgecolor='k',
        markeredgewidth=1.5,
        markersize=7,
    )
plot.show()

# %%
# Phasor coordinates that are NaN, for example, after filtering with
# :py:func:`phasorpy.filter.phasor_threshold`, are not assigned to any
# cluster and are labeled -1:

print(
    phasor_cluster_kmeans(
        None, [0.56, numpy.nan, 0.40], [0.29, 0.20, 0.33], clusters=2
    )[3]
)

# %%
# Intensity weighting
# -------------------
#
# Pass the intensity of the phasor coordinates as the first argument to
# weight them, such that the coordinates of brighter pixels contribute more
# to the clusters. The first returned value is then the average intensity of
# each cluster:

weighted_mean, weighted_real, weighted_imag, weighted_labels = (
    phasor_cluster_kmeans(mean, real, imag, clusters=2, random_state=42)
)

for index in range(2):
    print(f'cluster {index}')
    print(
        f'  unweighted center: {center_real[index]:.3f}, '
        f'{center_imag[index]:.3f}'
    )
    print(
        f'  weighted center:   {weighted_real[index]:.3f}, '
        f'{weighted_imag[index]:.3f}'
    )
    print(f'  average intensity: {weighted_mean[index]:.1f}')
print(f'reassigned coordinates: {(labels != weighted_labels).sum()}')

# %%
# Weighting moves the cluster centers towards the coordinates of brighter
# pixels. Since the clusters are separated by the perpendicular bisector
# between their centers, coordinates near the cluster boundary are
# reassigned:

reassigned = labels != weighted_labels

plot = PhasorPlot(title='Effect of intensity weighting')
plot.plot(real, imag, color='0.8', markersize=1)
plot.plot(
    real[reassigned],
    imag[reassigned],
    color=CATEGORICAL[2],
    markersize=1,
    label='reassigned',
)
for index, color in enumerate(CATEGORICAL[:2]):
    plot.plot(
        center_real[index],
        center_imag[index],
        'o',
        color=color,
        markeredgecolor='k',
        markeredgewidth=1.5,
        markersize=7,
        label=f'Unweighted center {index}',
    )
    plot.plot(
        weighted_real[index],
        weighted_imag[index],
        'x',
        color=color,
        markersize=9,
        markeredgewidth=2,
        label=f'Weighted center {index}',
    )
plot.show()

# %%
# Intensity weighting is not available for
# :py:func:`phasorpy.cluster.phasor_cluster_gmm`, since Gaussian mixture
# models are fit to unweighted coordinates.

# %%
# Comparing the methods
# ---------------------
#
# Both methods find the same two clusters, but describe them differently.
# The Gaussian mixture model returns ellipses, which may overlap and leave
# coordinates outside of any cluster, while k-means assigns every coordinate
# to exactly one cluster along a straight boundary:

plot = PhasorPlot(title='Gaussian mixture model and k-means')
for index, color in enumerate(CATEGORICAL[:2]):
    plot.plot(
        real[labels == index],
        imag[labels == index],
        color=color,
        markersize=1,
        alpha=0.3,
    )
plot.cursor(
    gmm_real,
    gmm_imag,
    radius=radius,
    radius_minor=radius_minor,
    angle=angle,
    color='k',
)
plot.show()

# %%
# The clusters returned by both functions are sorted, by default by their
# polar coordinates. Use the ``sort`` parameter to select another ordering,
# for example, to keep cluster indices and colors consistent across datasets.

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 3
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, assignment"
# sphinx_gallery_end_ignore
