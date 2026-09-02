# Copyright (c) PhasorPy Contributors
# SPDX-License-Identifier: MIT
# See LICENSE.txt file in the project root for details.

"""
Find clusters
=============

The :py:mod:`phasorpy.cluster` module provides functions to find clusters
of phasor coordinates, for example, using Gaussian mixture models or
k-means clustering.

"""

# %%
# Import required modules, functions, and classes:

from phasorpy.cluster import phasor_cluster_gmm, phasor_cluster_kmeans
from phasorpy.color import CATEGORICAL
from phasorpy.cursor import mask_from_elliptic_cursor, pseudo_color
from phasorpy.datasets import fetch
from phasorpy.filter import phasor_threshold
from phasorpy.io import signal_from_lsm
from phasorpy.phasor import phasor_from_signal
from phasorpy.plot import PhasorPlot, plot_image

# %%
# Load a hyperspectral dataset used throughout this tutorial and calculate
# phasor coordinates at the first harmonic and filter out pixels with low intensity:

signal = signal_from_lsm(fetch('paramecium.lsm'))
mean, real, imag = phasor_from_signal(signal, axis=0)
_, real, imag = phasor_threshold(mean, real, imag, mean_min=1)

# %%
# The phasor coordinates of this dataset form two distinct clusters:

plot = PhasorPlot(allquadrants=True, title='Hyperspectral phasor plot')
plot.hist2d(real, imag, cmap='Greys')
plot.show()

# %%
# Gaussian mixture model
# ----------------------
#
# The :py:func:`phasorpy.cluster.phasor_cluster_gmm` function fits a Gaussian
# mixture model to the phasor coordinates and returns the parameters of
# ellipses describing the clusters:

center_real, center_imag, radius, radius_minor, angle = phasor_cluster_gmm(
    real, imag, clusters=2
)

# %%
# Plot the ellipses in distinct colors:

plot = PhasorPlot(allquadrants=True, title='Elliptical clusters')
plot.hist2d(real, imag, cmap='Greys')
plot.cursor(
    center_real,
    center_imag,
    radius=radius,
    radius_minor=radius_minor,
    angle=angle,
    color=CATEGORICAL[:2],
)
plot.show()

# %%
# Regions of interest in the phasor space can be defined using the ellipses
# returned by the Gaussian mixture model. The parameters of these ellipses
# can be passed to :py:func:`phasorpy.cursor.mask_from_elliptic_cursor` to
# create elliptical masks:

elliptic_masks = mask_from_elliptic_cursor(
    real,
    imag,
    center_real,
    center_imag,
    radius=radius,
    radius_minor=radius_minor,
    angle=angle,
)

# %%
# Plot a pseudo-color image, composited from the elliptical cursor masks and
# the mean intensity image.

pseudo_color_image = pseudo_color(*elliptic_masks, intensity=mean)

plot_image(
    pseudo_color_image, title='Pseudo-color image from elliptical cursors'
)

# %%
# K-means clustering
# ------------------
#
# Instead of describing clusters by ellipses, the
# :py:func:`phasorpy.cluster.phasor_cluster_kmeans` function partitions the
# phasor coordinates into a fixed number of clusters, assigning each phasor
# coordinate to the cluster with the nearest center:

center_real, center_imag, labels = phasor_cluster_kmeans(
    real, imag, clusters=2
)

# %%
# Plot the phasor coordinates in the color of the cluster they belong to,
# and mark the cluster centers. Phasor coordinates that are NaN are not
# assigned to any cluster and are labeled -1:

plot = PhasorPlot(allquadrants=True, title='K-means clusters')
for index, color in enumerate(CATEGORICAL[:2]):
    plot.plot(
        real[labels == index],
        imag[labels == index],
        color=color,
        markersize=1,
        alpha=0.5,
        label=f'Cluster {index}',
    )
plot.plot(center_real, center_imag, marker='x', color='k', markersize=10)
plot.show()

# %%
# Since every phasor coordinate is assigned to a cluster, the cluster labels
# can be used directly to plot a pseudo-color image:

pseudo_color_image = pseudo_color(labels == 0, labels == 1, intensity=mean)

plot_image(
    pseudo_color_image, title='Pseudo-color image from k-means clusters'
)

# %%
# The clusters returned by both functions are sorted, by default by their
# polar coordinates. Use the ``sort`` parameter to select another ordering,
# for example, to keep cluster indices and colors consistent across datasets.

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 4
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type"
# sphinx_gallery_end_ignore
