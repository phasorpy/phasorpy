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
from phasorpy.filter import phasor_filter_median, phasor_threshold
from phasorpy.io import signal_from_imspector_tiff, signal_from_lsm
from phasorpy.lifetime import phasor_calibrate
from phasorpy.phasor import phasor_from_signal
from phasorpy.plot import PhasorPlot, plot_image

# %%
# Gaussian mixture model
# ----------------------
#
# Load a hyperspectral dataset and calculate phasor coordinates at the first
# harmonic and filter out pixels with low intensity:

signal = signal_from_lsm(fetch('paramecium.lsm'))
mean, real, imag = phasor_from_signal(signal, axis=0)
_, real, imag = phasor_threshold(mean, real, imag, mean_min=1)

# %%
# The phasor coordinates of this dataset form two distinct clusters:

plot = PhasorPlot(allquadrants=True, title='Hyperspectral phasor plot')
plot.hist2d(real, imag, cmap='Greys')
plot.show()

# %%
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
# Load a time-correlated single photon counting (TCSPC) dataset of a
# zebrafish embryo, and calculate, calibrate, and filter phasor coordinates
# at the first harmonic:

signal = signal_from_imspector_tiff(fetch('Embryo.tif'))
frequency = signal.attrs['frequency']
reference_signal = signal_from_imspector_tiff(fetch('Fluorescein_Embryo.tif'))

mean, real, imag = phasor_from_signal(signal, axis=0)
reference = phasor_from_signal(reference_signal, axis=0)

real, imag = phasor_calibrate(
    real, imag, *reference, frequency=frequency, lifetime=4.2
)
mean, real, imag = phasor_filter_median(mean, real, imag, size=3, repeat=2)
mean, real, imag = phasor_threshold(mean, real, imag, mean_min=1)

# %%
# Instead of describing clusters by ellipses, the
# :py:func:`phasorpy.cluster.phasor_cluster_kmeans` function partitions the
# phasor coordinates into a fixed number of clusters, assigning each phasor
# coordinate to the cluster with the nearest center:

_, center_real, center_imag, labels = phasor_cluster_kmeans(
    None, real, imag, clusters=3, n_init=10, random_state=42
)

# %%
# K-means clustering starts from a random initialization and may converge to
# different solutions when clusters are not well separated. Arguments such as
# ``n_init`` and ``random_state`` are passed to
# :py:class:`sklearn.cluster.KMeans` and are used here to obtain reproducible
# results.

# %%
# Plot the phasor coordinates in the color of the cluster they belong to,
# and mark the cluster centers. Phasor coordinates that are NaN are not
# assigned to any cluster and are labeled -1:

plot = PhasorPlot(frequency=frequency, title='K-means clusters')
for index, color in enumerate(CATEGORICAL[:3]):
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
# can be used directly to mask regions of interest and to plot a pseudo-color
# image:

pseudo_color_image = pseudo_color(
    labels == 0, labels == 1, labels == 2, intensity=mean
)

plot_image(
    pseudo_color_image, title='Pseudo-color image from k-means clusters'
)

# %%
# Intensity weighting
# -------------------
#
# Passing ``None`` as the first argument, as above, lets all phasor
# coordinates contribute equally to the clusters, regardless of the number of
# photons detected at each pixel. Pass the mean intensity image instead to
# weight the phasor coordinates, such that the coordinates of brighter pixels
# contribute more:

center_mean, weighted_real, weighted_imag, weighted_labels = (
    phasor_cluster_kmeans(
        mean, real, imag, clusters=3, n_init=10, random_state=42
    )
)

for index in range(3):
    print(f'cluster {index}')
    print(
        f'  unweighted center: {center_real[index]:.4f}, '
        f'{center_imag[index]:.4f}'
    )
    print(
        f'  weighted center:   {weighted_real[index]:.4f}, '
        f'{weighted_imag[index]:.4f}'
    )
    print(f'  mean intensity:    {center_mean[index]:.4f}')
print(f'reassigned coordinates: {(labels != weighted_labels).sum()}')

# %%
# When weighting by intensity, the cluster centers are the phasor centers of
# the coordinates assigned to each cluster, as calculated by
# :py:func:`phasorpy.phasor.phasor_center`. The returned ``center_mean`` is
# the mean intensity of each cluster.

# %%
# The clusters returned by both functions are sorted, by default by their
# polar coordinates. Use the ``sort`` parameter to select another ordering,
# for example, to keep cluster indices and colors consistent across datasets.

# sphinx_gallery_start_ignore
# sphinx_gallery_thumbnail_number = 4
# mypy: allow-untyped-defs, allow-untyped-calls
# mypy: disable-error-code="arg-type, assignment"
# sphinx_gallery_end_ignore
