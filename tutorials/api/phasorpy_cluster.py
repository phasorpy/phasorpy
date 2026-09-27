# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""
Clusters
========

Find clusters in distributions of phasor coordinates.

The :py:mod:`phasorpy.cluster` module provides functions to automatically
find clusters in distributions of phasor coordinates, either by fitting
ellipses using a Gaussian mixture model, or by assigning every phasor
coordinate to a cluster using k-means clustering.

"""

# %%
# Import required modules, functions, and classes:

import numpy

from phasorpy.cluster import phasor_cluster_gmm, phasor_cluster_kmeans
from phasorpy.phasor import phasor_center
from phasorpy.plot import PhasorPlot

rng = numpy.random.default_rng(42)  # initialize random number generator

# %%
# Distribution of phasor coordinates
# ----------------------------------
#
# Create a synthetic distribution of phasor coordinates from two partially
# overlapping Gaussian distributions of different size and shape.
# The first is 20 times brighter than the second:

real0_mean, imag0_mean = 0.56, 0.29
real1_mean, imag1_mean = 0.40, 0.33

mean0 = rng.normal(200.0, 40.0, 2000)
real0, imag0 = rng.multivariate_normal(
    [real0_mean, imag0_mean], [[1.8e-3, -3.0e-4], [-3.0e-4, 7.0e-4]], 2000
).T

mean1 = rng.normal(10.0, 2.0, 4000)
real1, imag1 = rng.multivariate_normal(
    [real1_mean, imag1_mean], [[2.4e-3, -4.0e-4], [-4.0e-4, 9.0e-4]], 4000
).T

mean = numpy.concatenate([mean0, mean1])
real = numpy.concatenate([real0, real1])
imag = numpy.concatenate([imag0, imag1])

colors = ['tab:blue', 'tab:red']

# %%
# Plot the distribution as a two-dimensional histogram, together with the
# mean coordinates of the two distributions for reference:

plot = PhasorPlot(title='Distribution of phasor coordinates')
plot.hist2d(real, imag, cmap='Greys')
plot.plot(real0_mean, imag0_mean, color=colors[0], label='Distribution 0 mean')
plot.plot(real1_mean, imag1_mean, color=colors[1], label='Distribution 1 mean')
plot.show()

# %%
# Gaussian mixture model
# ----------------------
#
# The :py:func:`phasorpy.cluster.phasor_cluster_gmm` function fits a Gaussian
# mixture model to the phasor coordinates and returns the parameters of
# ellipses describing the clusters:

gmm_real, gmm_imag, radius_major, radius_minor, angle = phasor_cluster_gmm(
    real, imag, clusters=2, sigma=2, random_state=42
)

print(f'cluster 0: {gmm_real[0]:.3f}, {gmm_imag[0]:.3f}')
print(f'cluster 1: {gmm_real[1]:.3f}, {gmm_imag[1]:.3f}')

# %%
# The ``sigma`` parameter controls the size of the ellipses (``sigma=2``
# corresponds to ~98.2% confidence).
#
# Clustering functions start from a random initialization. Arguments
# such as ``random_state`` are passed to the underlying scikit-learn
# estimators and are used throughout this tutorial to obtain reproducible
# results.
#
# The ellipses have the same parameters as elliptical cursors and can be
# plotted with :py:meth:`phasorpy.plot.PhasorPlot.cursor`:

plot = PhasorPlot(title='Gaussian mixture model')
plot.hist2d(real, imag, cmap='Greys')
plot.cursor(
    gmm_real,
    gmm_imag,
    radius=radius_major,
    radius_minor=radius_minor,
    angle=angle,
    color=colors,
    label=['Cluster 0', 'Cluster 1'],
)
for re, im, color in zip(gmm_real, gmm_imag, colors, strict=True):
    plot.plot(re, im, color=color)
plot.show()

# %%
# K-means
# -------
#
# Instead of describing clusters by ellipses, the
# :py:func:`phasorpy.cluster.phasor_cluster_kmeans` function partitions the
# phasor coordinates into a fixed number of clusters, assigning each phasor
# coordinate to the cluster with the nearest center:

center_real, center_imag, labels = phasor_cluster_kmeans(
    real, imag, clusters=2, random_state=42
)

# %%
# The returned ``labels`` array has the same shape as the phasor coordinates
# and contains the index of the cluster each coordinate belongs to.
# Use it to plot the phasor coordinates in the color of their cluster:

plot = PhasorPlot(title='K-means')
for index in range(2):
    plot.plot(
        real[labels == index],
        imag[labels == index],
        color=colors[index],
        markersize=1,
        alpha=0.5,
    )
    plot.plot(
        center_real[index],
        center_imag[index],
        color=colors[index],
        markeredgecolor='black',
        markeredgewidth=1.5,
        label=f'Cluster {index}',
    )
plot.show()

# %%
# Phasor coordinates that are NaN, for example, after filtering with
# :py:func:`phasorpy.filter.phasor_threshold`, are not assigned to any
# cluster and are labeled -1:

print(
    phasor_cluster_kmeans(
        [0.56, numpy.nan, 0.40], [0.29, 0.20, 0.33], clusters=2
    )[2]
)

# %%
# Intensity-weighted centers
# --------------------------
#
# The cluster centers returned by both functions are unweighted. To obtain
# intensity-weighted centers, apply
# :py:func:`phasorpy.phasor.phasor_center` to the coordinates of each
# cluster:

for index in range(2):
    weighted = phasor_center(
        mean[labels == index], real[labels == index], imag[labels == index]
    )
    print(f'cluster {index}')
    print(f'  unweighted: {center_real[index]:.3f}, {center_imag[index]:.3f}')
    print(f'  weighted:   {float(weighted[1]):.3f}, {float(weighted[2]):.3f}')

# %%
# Compare methods
# ---------------
#
# Both clustering methods find two clusters, but describe them differently.
# The Gaussian mixture model returns ellipses, which may overlap and leave
# coordinates outside of any cluster, while k-means assigns every coordinate
# to exactly one cluster along a straight boundary:

plot = PhasorPlot(title='Gaussian mixture model vs k-means')
for index in range(2):
    plot.plot(
        real[labels == index],
        imag[labels == index],
        color=colors[index],
        markersize=1,
        alpha=0.3,
        label='K-means' if index == 0 else None,
    )
plot.cursor(
    gmm_real,
    gmm_imag,
    radius=radius_major,
    radius_minor=radius_minor,
    angle=angle,
    color=colors,
    label=['GMM', '_nolegend_'],
)
plot.show()

# %%
# The clusters returned by both functions are sorted, by default by their
# polar coordinates. Use the ``sort`` parameter to select another ordering,
# for example, to keep cluster indices and colors consistent across datasets.

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = -1
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, assignment"
# sphinx_gallery_end_ignore
